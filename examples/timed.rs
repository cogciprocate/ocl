#![feature(time2)]
extern crate ocl;
use std::time::Instant;

use ocl::{ProQue, SimpleDims, Buffer};

const DATASET_SIZE: usize = 5000000;
const SCALAR: f32 = 1.0;
const KERNEL_ITERS: i32 = 1;
const BUFFER_READ_ITERS: i32 = 1;
const PRINT_SOME_RESULTS: bool = false;

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
    let ocl_pq = ProQue::builder().src(src).build().unwrap();

    // Set our work dimensions / data set size to something arbitrary:
    let dims = SimpleDims::One(DATASET_SIZE);

    // Create init and result buffers:
    let buffer_init: Buffer<f32> = Buffer::with_vec_scrambled(
         (100.0, 200.0), &dims, &ocl_pq.queue());
    let mut buffer_result: Buffer<f32> = Buffer::with_vec(&dims, &ocl_pq.queue());

    // Create a kernel with arguments matching those in the kernel:
    let mut kern = ocl_pq.create_kernel("add", dims.work_dims()).unwrap()
        .arg_buf_named("source", Some(&buffer_init))
        .arg_scl(SCALAR)
        .arg_buf(&buffer_result);

    // // Keep track of inital values:
    // let init_vals = buffer.vec().unwrap().clone();

    // #################################
    // ############ KERNEL #############
    // #################################

    // Start kernel timer
    let kern_start = Instant::now();

    // Enqueue kernel the first time:
    kern.enqueue(None, None);

    // Set kernel source buffer to the same as result:
    kern.set_arg_buf_named("source", Some(&buffer_result));

    // Enqueue kernel for additional iterations:
    for i in 0..(KERNEL_ITERS - 1) {
        kern.enqueue(None, None);
    }
    
    print_elapsed("Kernel:     ", kern_start);

    // #################################
    // ############ BUFFER #############
    // #################################

    // Start kernel timer
    let buffer_start = Instant::now();

    // Read results from the device into buffer's local vector:
    for _ in 0..BUFFER_READ_ITERS {
        buffer_result.fill_vec();
    }

    print_elapsed("Buffer Read:", buffer_start);

    // #################################
    // ############ VERIFY #############
    // ################################# 

    print!("Verifying result values... ");

    for idx in 0..DATASET_SIZE {
        let correct = buffer_init[idx] + (KERNEL_ITERS as f32 * SCALAR);
        // let correct = buffer_init[i] + SCALAR;
        assert!((correct - buffer_result[idx]) < 0.001, 
            "init: {}, correct: {}, result: {}", buffer_init[idx], correct, buffer_result[idx]);

        if PRINT_SOME_RESULTS && idx < DATASET_SIZE && idx > DATASET_SIZE - 20 {
            println!("init: {}, correct: {}, result: {}", buffer_init[idx], correct, buffer_result[idx]);
        }
    }
    println!("All result values are correct.");

    if PRINT_SOME_RESULTS { println!("The value at index [{}] is '{}'!", 9001, buffer_result[9001]); }
}


fn print_elapsed(title: &str, start: Instant) {
    let time_elapsed = Instant::now().duration_from_earlier(start);
    let elapsed_ms = time_elapsed.subsec_nanos() / 1000000;
    println!("{} elapsed: {}.{:03}", title, time_elapsed.as_secs(), elapsed_ms);
}
