extern crate ocl;
extern crate ocl_extras;

use ocl::{ProQue, Buffer, MemFlags};

// Number of results to print out:
const RESULTS_TO_PRINT: usize = 20;

// Our arbitrary data set size (about a million) and coefficent:
const WORK_SIZE: usize = 1 << 20;
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


fn basics() -> ocl::Result<()> {
    // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(WORK_SIZE)
        .build().expect("Build ProQue");

    // Create a temporary init vector and the source buffer. Initialize them
    // with random floats between 0.0 and 20.0:
    let vec_source = ocl_extras::scrambled_vec((0.0, 20.0), ocl_pq.dims().to_len());
    let source_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(WORK_SIZE)
        .copy_host_slice(&vec_source)
        .build()?;

    // Create an empty vec and buffer (the quick way) for results. Note that
    // there is no need to initialize the buffer as we did above because we
    // will be writing to the entire buffer first thing, overwriting any junk
    // data that may be there.
    let mut vec_result = vec![0.0f32; WORK_SIZE];
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer()?;

    // Create a kernel with arguments corresponding to those in the kernel.
    // Just for fun, one argument will be 'named':
    let kern = ocl_pq.kernel_builder("multiply_by_scalar")
        .arg(COEFF)
        .arg(None::<&Buffer<f32>>)
        .arg_named("result", None::<&Buffer<f32>>)
        .build()?;

    // Set our named argument. The Option<_> wrapper is, well... optional:
    kern.set_arg("result", &result_buffer)?;
    // We can also set arguments (named or not) by index. Just for
    // demonstration, we'll set one using an option:
    kern.set_arg(0, &COEFF)?;
    kern.set_arg(1, Some(&source_buffer))?;
    kern.set_arg(2, &result_buffer)?;

    println!("Kernel global work size: {:?}", kern.default_global_work_size());

    // Enqueue kernel:
    unsafe { kern.enq()?; }

    // Read results from the device into result_buffer's local vector:
    result_buffer.read(&mut vec_result).enq()?;

    // Check results and print the first 20:
    for idx in 0..WORK_SIZE {
        if idx < RESULTS_TO_PRINT {
            println!("source[{idx}]: {:.03}, \t coeff: {}, \tresult[{idx}]: {}",
            vec_source[idx], COEFF, vec_result[idx], idx = idx);
        }
        assert_eq!(vec_source[idx] * COEFF, vec_result[idx]);
    }

    Ok(())
}

pub fn main() {
    match basics() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}