extern crate ocl;
use ocl::{ProQue, SimpleDims, Buffer};

fn main() {
    let src = r#"
        __kernel void multiply(__global float* buffer, float coeff) {
            buffer[get_global_id(0)] *= coeff;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(SimpleDims::One(500000))
        .build().unwrap();   

    let mut buffer: Buffer<f32> = Buffer::with_vec_scrambled(
         (0.1, 1.0), &pro_que.dims(), &pro_que.queue());

    let kernel = pro_que.create_kernel("multiply")
        .arg_buf(&buffer)
        .arg_scl(100.0f32);

    kernel.enqueue();
    buffer.fill_vec();

    println!("The buffer element at [{}] is '{}'", 200007, buffer[200007]);
}


//
//
// Expanded version:
//
//

#[allow(dead_code)]
fn main_explained_and_checked() {
    // Define some program source code:
    let src = r#"
        __kernel void multiply(__global float* buffer, float coeff) {
            buffer[get_global_id(0)] *= coeff;
        }
    "#;

    // Create an all-in-one context, program, command queue, and work / buffer
    // dimensions:
    let ocl_pq = ProQue::builder()
        .src(src)
        .dims(SimpleDims::One(500000))
        .build().unwrap();

    // Create a `Buffer` with a built-in `Vec` and initialize it with random 
    // floats between 0.0 and 20.0:
    let mut buffer: Buffer<f32> = Buffer::with_vec_scrambled(
         (0.0, 20.0), &ocl_pq, &ocl_pq.queue());

    // Declare a value to multiply our buffer's contents by:
    let scalar = 10.0f32;

    // Create a kernel with arguments matching those in the source above:
    let kernel = ocl_pq.create_kernel("multiply")
        .arg_buf(&buffer)
        .arg_scl(scalar);

    // Choose an element to keep track of:
    let element_idx = 200007;
    let element_original_value = buffer[element_idx];

    // Run the kernel (the optional arguments are for event lists):
    kernel.enqueue();

    // Read results from the device into our buffer's built-in vector:
    buffer.fill_vec();

    // Verify and print a result:
    let element_final_value = buffer[element_idx];
    assert!((element_final_value - (element_original_value * scalar)).abs() < 0.0001);
    println!("The value at index [{}] was '{}' and is now '{}'!", 
        element_idx, element_original_value, element_final_value);
}
