#![feature(time2)]
extern crate ocl;
use std::time::Instant;

use ocl::{ProQue, SimpleDims, Buffer, EventList};

const DATASET_SIZE: usize = 5000000;
const SCALAR: f32 = 1.0;
const KERNEL_RUN_ITERS: i32 = 8000;
const BUFFER_READ_ITERS: i32 = 200;
const KERNEL_AND_BUFFER_ITERS: i32 = 1000;
const PRINT_SOME_RESULTS: bool = true;
const RESULTS_TO_PRINT: usize = 5;
const INIT_VAL_RANGE: (f32, f32) = (100.0, 200.0);

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

    // Set our work dimensions / data set size to something arbitrary:
    let dims = SimpleDims::One(DATASET_SIZE);


    // Create an all-in-one context, program, and command queue:
    let ocl_pq = ProQue::builder().src(src).build().unwrap();

    // Create init and result buffers:
    let buffer_init: Buffer<f32> = Buffer::with_vec_scrambled(
         INIT_VAL_RANGE, &dims, &ocl_pq.queue());
    let mut buffer_result: Buffer<f32> = Buffer::with_vec(&dims, &ocl_pq.queue());

    // Create a kernel with arguments matching those in the kernel:
    let mut kern = ocl_pq.create_kernel("add", dims.work_dims()).unwrap()
        .arg_buf_named("source", Some(&buffer_init))
        .arg_scl(SCALAR)
        .arg_buf(&buffer_result);


    // ##################################################
    // ##################### KERNEL #####################
    // ##################################################

    print!("\n");
    println!("Enqueuing {} kernel runs... ", KERNEL_RUN_ITERS);

    // Start kernel timer
    let kern_start = Instant::now();

    // Enqueue kernel the first time:
    kern.enqueue(None, None);

    // Set kernel source buffer to the same as result:
    kern.set_arg_buf_named("source", Some(&buffer_result));

    // Enqueue kernel for additional iterations:
    for _ in 0..(KERNEL_RUN_ITERS - 1) {
        kern.enqueue(None, None);
    }

    // Wait for all kernels to run:
    ocl_pq.queue().finish();
    
    // Print elapsed time for kernels:
    print_elapsed("", kern_start);

    // ##################################################
    // ##################### BUFFER #####################
    // ##################################################

    print!("\n");
    println!("Enqueuing {} buffer reads... ", BUFFER_READ_ITERS);

    // Start kernel timer
    let buffer_start = Instant::now();

    // Read results from the device into buffer's local vector:
    for _ in 0..BUFFER_READ_ITERS {
        buffer_result.fill_vec().ok();
    }

    // Print elapsed time for buffer reads:
    print_elapsed("", buffer_start);

    // ##################################################
    // ########### KERNEL & BUFFER BLOCKING #############
    // ##################################################

    print!("\n");
    println!("Enqueuing {} blocking kernel buffer sequences... ", KERNEL_AND_BUFFER_ITERS);

    let kern_buf_start = Instant::now();

    for _ in 0..(KERNEL_AND_BUFFER_ITERS) {
        kern.enqueue(None, None);
        buffer_result.fill_vec().ok();
    }

    ocl_pq.queue().finish();
    
    print_elapsed("", kern_buf_start);

    // ##################################################
    // ######### KERNEL & BUFFER NON-BLOCKING ###########
    // ##################################################

    print!("\n");
    println!("Enqueuing {} non-blocking kernel buffer sequences... ", KERNEL_AND_BUFFER_ITERS);

    let kern_buf_start = Instant::now();

    // let mut el = EventList::new();

    for _ in 0..(KERNEL_AND_BUFFER_ITERS) {
        // kern.enqueue(Some(&el), Some(&mut el));
        // buffer_result.fill_vec_async(Some(&el), Some(&mut el));
    }

    ocl_pq.queue().finish();
    
    print_elapsed("", kern_buf_start);

    // ##################################################
    // ##################### VERIFY #####################
    // ################################################## 

    print!("\nVerifying result values... ");
    if PRINT_SOME_RESULTS { print!("(printing {})\n\n", RESULTS_TO_PRINT); }

    for idx in 0..DATASET_SIZE {
        let correct = buffer_init[idx] + (KERNEL_RUN_ITERS as f32 * SCALAR);
        // let correct = buffer_init[i] + SCALAR;
        assert!((correct - buffer_result[idx]) < 0.1, 
            "    init: {}, correct: {}, result: {}", buffer_init[idx], correct, buffer_result[idx]);

        if PRINT_SOME_RESULTS && (idx % (DATASET_SIZE / RESULTS_TO_PRINT)) == 0  {
            println!("    [{}]: init: {}, correct: {}, result: {}", idx, buffer_init[idx],
                correct, buffer_result[idx]);
        }
    }

    if PRINT_SOME_RESULTS { print!("\n"); }
    println!("All result values are correct.");

    // if PRINT_SOME_RESULTS { println!("The value at index [{}] is '{}'!", 9001, buffer_result[9001]); }
}


fn print_elapsed(title: &str, start: Instant) {
    let time_elapsed = Instant::now().duration_from_earlier(start);
    let elapsed_ms = time_elapsed.subsec_nanos() / 1000000;
    println!("{}elapsed: {}.{:03}", title, time_elapsed.as_secs(), elapsed_ms);
}
