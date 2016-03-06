extern crate ocl;

use ocl::{util, core, ProQue, Buffer};
// use ocl::traits::{BufferExtras};

// Number of results to print out:
const RESULTS_TO_PRINT: usize = 20;

// Our arbitrary data set size and coefficent:
const DATA_SET_SIZE: usize = 900000;
const COEFF: f32 = 5432.1;

// Our kernel source code:
static KERNEL_SRC: &'static str = r#"
    __kernel void multiply_by_scalar(
                __private float const coeff,
                __global float const* const src,
                __global float* const res)
    {
        uint const idx = get_global_id(0);

        res[idx] = src[idx] * coeff;
    }
"#;


fn main() {
    // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims([DATA_SET_SIZE])
        .build().expect("Build ProQue");

    // Set up our work dimensions / data set size with an array or tuple:
    // let dims = [DATA_SET_SIZE];

    // Create a 'Buffer' (a device buffer + a local vector) as a data source
    // and initialize it with random floats between 0.0 and 20.0:
    // let source_buffer: Buffer<f32> = 
    //     Buffer::with_vec_scrambled((0.0, 20.0), ocl_pq.dims(), &ocl_pq.queue());

    let vec_source = util::scrambled_vec((0.0, 20.0), ocl_pq.dims().to_len().unwrap());
    let source_buffer = Buffer::newer_new(ocl_pq.queue(), Some(core::MEM_READ_WRITE | 
        core::MEM_COPY_HOST_PTR), ocl_pq.dims().clone(), Some(&vec_source)).unwrap();

    // Create another empty buffer for results (using ocl_pq for fun):
    let mut vec_result = vec![0.0f32; DATA_SET_SIZE];
    let mut result_buffer: Buffer<f32> = ocl_pq.create_buffer();

    // Create a kernel with three arguments corresponding to those in the kernel:
    let kern = ocl_pq.create_kernel("multiply_by_scalar")
        .arg_scl(COEFF)
        .arg_buf(&source_buffer)
        .arg_buf(&mut result_buffer);

    // Enqueue kernel:
    kern.enqueue();

    // Read results from the device into result_buffer's local vector:
    // result_buffer.fill_vec();
    unsafe { result_buffer.cmd().read_async(&mut vec_result).enq().unwrap() }

    // Check results and print the first 20:
    for idx in 0..DATA_SET_SIZE {
        assert_eq!(vec_result[idx], vec_source[idx] * COEFF);

        if idx < RESULTS_TO_PRINT { 
            println!("source[{idx}]: {}, \tcoeff: {}, \tresult[{idx}]: {}",
            vec_source[idx], COEFF, vec_result[idx], idx = idx); 
        }
    }
}
