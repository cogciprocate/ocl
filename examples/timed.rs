//! Timed kernel and buffer tests / benchmarks / examples.
//!
//! Manipulate the consts below to fiddle with parameters. To create longer
//! running tests, increase `DATASET_SIZE`, and the `*_ITERS` consts. The
//! other consts can be anything at all.
//!
//! Due to buggy and/or intentionally crippled drivers, this example may not
//! work on NVIDIA hardware. Until NVIDIA's implementation is corrected this
//! example will likely fail on that platform.

extern crate ocl;
extern crate time;
// [TODO]: Bring this back once `Instant` stabilizes:
// use std::time::Instant;


use ocl::{util, core, ProQue, Buffer, EventList};

const DATASET_SIZE: usize = 2 << 12;

const KERNEL_RUN_ITERS: i32 = 800;
const BUFFER_READ_ITERS: i32 = 20;
const KERNEL_AND_BUFFER_ITERS: i32 = 1000;

const SCALAR: f32 = 1.0;
const INIT_VAL_RANGE: (f32, f32) = (100.0, 200.0);

const PRINT_SOME_RESULTS: bool = true;
const RESULTS_TO_PRINT: usize = 5;


fn main() {
    // Define a kernel:
    let src = r#"
        __kernel void add(
                    __global float const* const source,
                    __private float scalar,
                    __global float* const result)
        {
            uint idx = get_global_id(0);
            result[idx] = source[idx] + scalar;
        }
    "#;

    // Create an all-in-one context, program, and command queue:
    let ocl_pq = ProQue::builder().src(src).dims([DATASET_SIZE]).build().unwrap();

    // Create init and result buffers and vectors:
    let vec_init = util::scrambled_vec(INIT_VAL_RANGE, ocl_pq.dims().to_len());
    let buffer_init = Buffer::new(ocl_pq.queue(), Some(core::MEM_READ_WRITE |
        core::MEM_COPY_HOST_PTR), ocl_pq.dims().clone(), Some(&vec_init)).unwrap();

    let mut vec_result = vec![0.0f32; DATASET_SIZE];
    let buffer_result = Buffer::<f32>::new(ocl_pq.queue(), None,
        ocl_pq.dims(), None).unwrap();

    // Create a kernel with arguments matching those in the kernel:
    let mut kern = ocl_pq.create_kernel("add").unwrap()
        .gws(ocl_pq.dims().clone())
        .arg_buf_named("source", Some(&buffer_init))
        .arg_scl(SCALAR)
        .arg_buf(&buffer_result);


    // ##################################################
    // ##################### KERNEL #####################
    // ##################################################

    print!("\n");
    println!("Enqueuing {} kernel runs... ", KERNEL_RUN_ITERS);

    // Start kernel timer
    let kern_start = time::get_time();

    // Enqueue kernel the first time:
    kern.enq().unwrap();

    // Set kernel source buffer to the same as result:
    kern.set_arg_buf_named("source", Some(&buffer_result)).unwrap();

    // Enqueue kernel for additional iterations:
    for _ in 0..(KERNEL_RUN_ITERS - 1) {
        kern.enq().unwrap();
    }

    // Wait for all kernels to run:
    ocl_pq.queue().finish();

    // Print elapsed time for kernels:
    print_elapsed("total elapsed", kern_start);

    // ##################################################
    // ##################### BUFFER #####################
    // ##################################################

    print!("\n");
    println!("Enqueuing {} buffer reads... ", BUFFER_READ_ITERS);

    // Start kernel timer
    let buffer_start = time::get_time();

    // Read results from the device into buffer's local vector:
    for _ in 0..BUFFER_READ_ITERS {
        buffer_result.cmd().read(&mut vec_result).enq().unwrap()
    }

    print_elapsed("queue unfinished", buffer_start);
    ocl_pq.queue().finish();
    print_elapsed("queue finished", buffer_start);

    verify_results(&vec_init, &vec_result, KERNEL_RUN_ITERS);

    // ##################################################
    // ########### KERNEL & BUFFER BLOCKING #############
    // ##################################################

    print!("\n");
    println!("Enqueuing {} blocking kernel buffer sequences... ", KERNEL_AND_BUFFER_ITERS);

    let kern_buf_start = time::get_time();

    for _ in 0..(KERNEL_AND_BUFFER_ITERS) {
        kern.enq().unwrap();
        buffer_result.cmd().read(&mut vec_result).enq().unwrap();
    }

    print_elapsed("queue unfinished", kern_buf_start);
    ocl_pq.queue().finish();
    print_elapsed("queue finished", kern_buf_start);

    verify_results(&vec_init, &vec_result, KERNEL_AND_BUFFER_ITERS + KERNEL_RUN_ITERS);

    // ##################################################
    // ######### KERNEL & BUFFER NON-BLOCKING ###########
    // ##################################################

    print!("\n");
    println!("Enqueuing {} non-blocking kernel buffer sequences... ", KERNEL_AND_BUFFER_ITERS);

    let kern_buf_start = time::get_time();

    let mut kern_events = EventList::new();
    let mut buf_events = EventList::new();

    for _ in 0..KERNEL_AND_BUFFER_ITERS {
        kern.cmd().ewait(&buf_events).enew(&mut kern_events).enq().unwrap();
        buffer_result.cmd().read(&mut vec_result).ewait(&kern_events)
            .enew(&mut buf_events).enq().unwrap();
    }

    print_elapsed("queue unfinished", kern_buf_start);
    ocl_pq.queue().finish();
    print_elapsed("queue finished", kern_buf_start);

    kern_events.wait().unwrap();
    kern_events.clear_completed().unwrap();
    buf_events.wait().unwrap();
    buf_events.clear_completed().unwrap();

    verify_results(&vec_init, &vec_result,
        KERNEL_AND_BUFFER_ITERS + KERNEL_AND_BUFFER_ITERS + KERNEL_RUN_ITERS);

    // ##################################################
    // ############# CAUTION IS OVERRATED ###############
    // ##################################################

    print!("\n");
    println!("Enqueuing {} oh-fuck-it kernel buffer sequences... ", KERNEL_AND_BUFFER_ITERS);

    let kern_buf_start = time::get_time();

    for _ in 0..KERNEL_AND_BUFFER_ITERS {
        kern.cmd().enew(&mut kern_events).enq().unwrap();
        unsafe { buffer_result.cmd().read_async(&mut vec_result).enew(&mut buf_events).enq().unwrap() }
    }

    print_elapsed("queue unfinished", kern_buf_start);
    ocl_pq.queue().finish();
    print_elapsed("queue finished", kern_buf_start);

    kern_events.wait().unwrap();
    buf_events.wait().unwrap();

    verify_results(&vec_init, &vec_result,
        KERNEL_AND_BUFFER_ITERS + KERNEL_AND_BUFFER_ITERS + KERNEL_AND_BUFFER_ITERS + KERNEL_RUN_ITERS);
}


// [KEEP]: Convert back to this once `Instant` stabilizes:
// fn print_elapsed(title: &str, start: Instant) {
//     let time_elapsed = time::get_time() - start;
//     // let time_elapsed = time::get_time().duration_since(start);
//     let elapsed_ms = time_elapsed.subsec_nanos() / 1000000;
//     let separator = if title.len() > 0 { ": " } else { "" };
//     println!("    {}{}: {}.{:03}", title, separator, time_elapsed.as_secs(), elapsed_ms);
// }
// [/KEEP]

fn print_elapsed(title: &str, start: time::Timespec) {
    let time_elapsed = time::get_time() - start;
    let elapsed_ms = time_elapsed.num_milliseconds();
    let separator = if title.len() > 0 { ": " } else { "" };
    println!("    {}{}: {}.{:03}", title, separator, time_elapsed.num_seconds(), elapsed_ms);
}



fn verify_results(vec_init: &Vec<f32>, vec_result: &Vec<f32>, iters: i32) {
    print!("\nVerifying result values... ");
    if PRINT_SOME_RESULTS { print!("(printing {})\n", RESULTS_TO_PRINT); }

    // let margin_of_error = iters as f32 / 100000.0;
    let margin_of_error = 0.1 as f32;

    for idx in 0..DATASET_SIZE {
        let correct = vec_init[idx] + (iters as f32 * SCALAR);
        assert!((correct - vec_result[idx]).abs() < margin_of_error,
            "    INVALID RESULT[{}]: init: {}, correct: {}, margin: {}, result: {}",
            idx, vec_init[idx], correct, margin_of_error, vec_result[idx]);

        if PRINT_SOME_RESULTS && (idx % (DATASET_SIZE / RESULTS_TO_PRINT)) == 0  {
            println!("    [{}]: init: {}, correct: {}, result: {}", idx, vec_init[idx],
                correct, vec_result[idx]);
        }
    }

    if PRINT_SOME_RESULTS { print!("\n"); }
    println!("All result values are correct.");
}
