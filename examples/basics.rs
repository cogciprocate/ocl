extern crate ocl;

// Number of results to print out:
const RESULTS_TO_PRINT: usize = 20;

// Our arbitrary data set size and coefficent:
const DATA_SET_SIZE: usize = 900000;
const COEFF: f32 = 5432.1;

// Our kernel source code:
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
    use ocl::{ProQue, SimpleDims, Buffer};

    // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
    let ocl_pq = ProQue::builder().src(KERNEL_SRC).build().expect("ProQue build");

    // Set up our work dimensions / data set size:
    let dims = SimpleDims::One(DATA_SET_SIZE);

    // Create a 'Buffer' (a device buffer + a local vector) as a data source
    // and initialize it with random floats between 0.0 and 20.0:
    let source_buffer: Buffer<f32> = 
        Buffer::with_vec_scrambled((0.0, 20.0), &dims, &ocl_pq.queue());

    // Create another empty buffer for results:
    let mut result_buffer: Buffer<f32> = Buffer::with_vec(&dims, &ocl_pq.queue());

    // Create a kernel with three arguments corresponding to those in the kernel:
    let kernel = ocl_pq.create_kernel("multiply_by_scalar", dims.work_dims())
        .arg_buf(&source_buffer)
        .arg_scl(COEFF)
        .arg_buf(&mut result_buffer);

    // Enqueue kernel depending on and creating no events:
    kernel.enqueue(None, None);

    // Read results from the device into result_buffer's local vector:
    result_buffer.fill_vec();

    // Check results and print the first 20:
    for idx in 0..DATA_SET_SIZE {
        assert_eq!(result_buffer[idx], source_buffer[idx] * COEFF);

        if idx < RESULTS_TO_PRINT { 
            println!("source[{idx}]: {}, \tcoeff: {}, \tresult[{idx}]: {}",
            source_buffer[idx], COEFF, result_buffer[idx], idx = idx); 
        }
    }
}
