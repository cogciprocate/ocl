#![feature(time2)]
extern crate ocl;
use std::time::Instant;

use ocl::{ProQue, SimpleDims, Buffer};

const SCALAR: f32 = 1.0;
const KERNEL_ITERS: i32 = 3;
const BUFFER_READ_ITERS: i32 = 1;

fn main() {
    // Define a kernel:
    let src = r#"
        __kernel void multiply(
                    __global float* buffer, 
                    __private float scalar) 
        {
            uint idx = get_global_id(0);
            buffer[idx] += scalar;
        }
    "#;

    // Create an all-in-one context, program, and command queue:
    let ocl_pq = ProQue::builder().src(src).build().unwrap();

    // Set our work dimensions / data set size to something arbitrary:
    let dims = SimpleDims::One(5000000);

    // Create a 'Buffer' with a built-in vector and initialize it with random 
    // floats between 0.0 and 20.0:
    let mut buffer: Buffer<f32> = Buffer::with_vec_scrambled(
         (100.0, 200.0), &dims, &ocl_pq.queue());

    // Create a kernel with arguments matching those in the kernel:
    let kern = ocl_pq.create_kernel("multiply", dims.work_dims()).unwrap()
        .arg_buf(&buffer)
        .arg_scl(SCALAR);

    // Keep track of inital values:
    let init_vals = buffer.vec().unwrap().clone();

    // #################################
    // ############ KERNEL #############
    // #################################

    // Start kernel timer
    let kern_start = Instant::now();

    // Enqueue kernel a bunch:
    for _ in 0..KERNEL_ITERS {
        kern.enqueue(None, None);
    }
    
    print_elapsed("Kernel:     ", kern_start);

    // #################################
    // ############ BUFFER #############
    // #################################

    // Start kernel timer
    let buffer_start = Instant::now();

    // // Read results from the device into buffer's local vector:
    // for _ in 0..BUFFER_READ_ITERS {
    //     buffer.fill_vec();
    // }
    buffer.fill_vec();

    print_elapsed("Buffer Read:", buffer_start);

    // #################################
    // ############ VERIFY #############
    // ################################# 

    for (&init, &result) in init_vals.iter().zip(buffer.iter()) {
        let correct = init + (KERNEL_ITERS as f32 * SCALAR);
        assert!((correct - result) < 0.0001, 
            "init: {}, correct: {}, result: {}", init, correct, result);
    }

    println!("The value at index [{}] is '{}'!", 9001, buffer[9001]);  
}


fn print_elapsed(title: &str, start: Instant) {
    let time_elapsed = Instant::now().duration_from_earlier(start);
    let elapsed_ms = time_elapsed.subsec_nanos() / 1000000;
    println!("{} elapsed: {}.{:03}", title, time_elapsed.as_secs(), elapsed_ms);
}
