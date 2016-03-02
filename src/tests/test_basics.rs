
use standard::{Buffer, SimpleDims, ProQue};

const PRINT_DEBUG: bool = false;

#[test]
fn test_basics() {
    // Set our data set size and coefficent to arbitrary values:
    let data_set_size = 900000;
    let coeff = 5432.1;

    let kernel_src = r#"
        __kernel void multiply_by_scalar(
                    __global float const* const src,
                    __private float const coeff,
                    __global float* const res)
        {
            uint const idx = get_global_id(0);

            res[idx] = src[idx] * coeff;
        }
    "#;


    let ocl_pq = ProQue::builder().src(kernel_src).build().expect("ProQue build");

    // Set up our work dimensions / data set size:
    let dims = SimpleDims::One(data_set_size);

    // Create an buffer (a local array + a remote buffer) as a data source:
    let source_buffer = Buffer::with_vec_scrambled((0.0f32, 20.0f32), &dims, &ocl_pq.queue());

    // Create another empty buffer for results:
    let mut result_buffer = Buffer::<f32>::with_vec(&dims, &ocl_pq.queue());

    // Create a kernel with three arguments corresponding to those in the kernel:
    let kernel = ocl_pq.create_kernel("multiply_by_scalar")
        .gws(dims.clone())
        .arg_buf(&source_buffer)
        .arg_scl(coeff)
        .arg_buf(&mut result_buffer)
    ;

    // Enqueue kernel depending on and creating no events:
    kernel.enqueue();

    // Read results:
    result_buffer.fill_vec();

    // Check results and print the first 20:
    for idx in 0..data_set_size {
        // Check:
        assert_eq!(result_buffer[idx], source_buffer[idx] * coeff);

        // Print:
        if PRINT_DEBUG && (idx < 20) { 
            println!("source_buffer[idx]: {}, coeff: {}, result_buffer[idx]: {}",
            source_buffer[idx], coeff, result_buffer[idx]); 
        }
    }
}
