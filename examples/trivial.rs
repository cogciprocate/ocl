extern crate ocl;
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
    let dims = SimpleDims::One(500000);

    // Create a 'Buffer' with a built-in vector and initialize it with random 
    // floats between 0.0 and 20.0:
    let mut buffer: Buffer<f32> = Buffer::with_vec_scrambled(
         (0.0, 20.0), &dims, &ocl_pq.queue());

    // Create a kernel with arguments matching those in the kernel:
    let kern = ocl_pq.create_kernel("multiply", dims.work_dims()).unwrap()
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    // // Enqueue kernel:
    kern.enqueue(None, None);

    // // Read results from the device into buffer's local vector:
    buffer.fill_vec().unwrap();

    // // Print a result:
    println!("The value at index [{}] is '{}'!", 200007, buffer[200007]);
}
