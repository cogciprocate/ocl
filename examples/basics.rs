extern crate ocl;

use ocl::{util, ProQue, Buffer, MemFlags};

// Number of results to print out:
const RESULTS_TO_PRINT: usize = 20;

// Our arbitrary data set size (about a million) and coefficent:
const DATA_SET_SIZE: usize = 2 << 20;
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
        .dims(DATA_SET_SIZE)
        .build().expect("Build ProQue");

    // Create a temporary init vector and the source buffer. Initialize them
    // with random floats between 0.0 and 20.0:
    let vec_source = util::scrambled_vec((0.0, 20.0), ocl_pq.dims().to_len());
    let source_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write().copy_host_ptr())
        .dims(ocl_pq.dims().clone())
        .host_data(&vec_source)
        .build().unwrap();

    // Create an empty vec and buffer (the quick way) for results. Note that
    // there is no need to initialize the buffer as we did above because we
    // will be writing to the entire buffer first thing, overwriting any junk
    // data that may be there.
    let mut vec_result = vec![0.0f32; DATA_SET_SIZE];
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer().unwrap();

    // Create a kernel with arguments corresponding to those in the kernel:
    let kern = ocl_pq.create_kernel("multiply_by_scalar").unwrap()
        .arg_scl(COEFF)
        .arg_buf(&source_buffer)
        .arg_buf(&result_buffer);

    println!("Kernel global work size: {:?}", kern.get_gws());

    // Enqueue kernel:
    kern.enq().unwrap();

    // Read results from the device into result_buffer's local vector:
    result_buffer.read(&mut vec_result).enq().unwrap();

    // Check results and print the first 20:
    for idx in 0..DATA_SET_SIZE {
        if idx < RESULTS_TO_PRINT {
            println!("source[{idx}]: {:.03}, \t coeff: {}, \tresult[{idx}]: {}",
            vec_source[idx], COEFF, vec_result[idx], idx = idx);
        }
        assert_eq!(vec_source[idx] * COEFF, vec_result[idx]);
    }
}
