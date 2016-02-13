#![feature(time2)]
extern crate ocl;
use std::time::Instant;

use ocl::{ProQue, SimpleDims, Buffer};

fn main() {
    // Define a kernel:
    let src = r#"
        __kernel void multiply(__global float* buffer, float coeff) {
            buffer[get_global_id(0)] *= coeff;
        }
    "#;

    // Create an all-in-one context, program, and command queue:
    let ocl_pq = ProQue::builder().src(src).build().unwrap();

    // Set our work dimensions / data set size to something arbitrary:
    let dims = SimpleDims::One(5000000);

    // Create a 'Buffer' with a built-in vector and initialize it with random 
    // floats between 0.0 and 20.0:
    let mut buffer: Buffer<f32> = Buffer::with_vec_scrambled(
         (0.0, 20.0), &dims, &ocl_pq.queue());

    // Create a kernel with arguments matching those in the kernel:
    let kern = ocl_pq.create_kernel("multiply", dims.work_dims()).unwrap()
        .arg_buf(&buffer)
        .arg_scl(1.00001f32);

    // Start timer
    let start = Instant::now();

    // Number of times to run kernel:
    let iters = 10000;

    // Enqueue kernel a bunch:
    for _ in 0..iters {
        kern.enqueue(None, None);
    }

    // Read results from the device into buffer's local vector:
    buffer.fill_vec();

    // Print a result:
    println!("The value at index [{}] is '{}'!", 200007, buffer[200007]);   

    print_elapsed(start);
}


fn print_elapsed(start: Instant) {
    let time_elapsed = Instant::now().duration_from_earlier(start);
    let elapsed_ms = time_elapsed.subsec_nanos() / 1000000;
    println!("Elapsed time: {}.{:03}", time_elapsed.as_secs(), elapsed_ms);
}
