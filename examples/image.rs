extern crate ocl;
use ocl::{Context, ProQue, ProgramBuilder, SimpleDims};

static KERNEL_SRC: &'static str = r#"
        __kernel void multiply_by_scalar(
                    __global float const* const src,
                    __private float const coeff,
                    __global float* const res)
        {
            uint const idx = get_global_id(0);

            res[idx] = src[idx] * coeff;
        }
    "#;

fn main() {
    let image_dims = SimpleDims::Two(200, 100);

    // Create a context with the first avaliable platform and default device type:
    let ocl_cxt = Context::new(None, None).unwrap();

    // Create a program/queue with the first available device: 
    let ocl_pq = ProQue::builder().src(KERNEL_SRC).build().expect("ProQue build");

    // // Set up our work dimensions / data set size:
    // let dims = SimpleDims::One(data_set_size);

    // // Create a 'Buffer' (a local vector + a remote buffer) as a data source:
    // let source_buffer: Buffer<f32> = 
    //  Buffer::with_vec_scrambled(0.0f32, 20.0f32, &dims, &ocl_pq.queue());

    // // Create another empty buffer for results:
    // let mut result_buffer: Buffer<f32> = Buffer::with_vec(&dims, &ocl_pq.queue());

    // // Create a kernel with three arguments corresponding to those in the kernel:
    // let kernel = ocl_pq.create_kernel("multiply_by_scalar", dims.work_size())
    //  .arg_buf(&source_buffer)
    //  .arg_buf(&mut result_buffer)
    // ;

    // // Enqueue kernel depending on and creating no events:
    // kernel.enqueue(None, None);

    // // Read results from the device into the buffer's vector:
    // result_buffer.fill_vec();

    // // Check results and print the first 20:
    // for idx in 0..data_set_size {
    //  // Check:
    //  assert_eq!(result_buffer[idx], source_buffer[idx] * coeff);

    //  // Print:
    //  if idx < RESULTS_TO_PRINT { 
    //      println!("source_buffer[idx]: {}, coeff: {}, result_buffer[idx]: {}",
    //      source_buffer[idx], coeff, result_buffer[idx]); 
    //  }
    // }
}
