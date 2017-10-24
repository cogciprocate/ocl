
extern crate futures;

// use futures::Future;
use ::{util, ProQue, Buffer, MemFlags, Event, EventList};
use async::BufferSink;

// Number of results to print out:
const RESULTS_TO_PRINT: usize = 20;

// Our arbitrary data set size (about a million) and coefficent:
const DATA_SET_SIZE: usize = 1 << 20;
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


#[test]
fn buffer_sink() {
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(DATA_SET_SIZE)
        .build().expect("Build ProQue");

    let source_buffer = Buffer::<f32>::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write().alloc_host_ptr())
        .dims(ocl_pq.dims().clone())
        .build().unwrap();

    let mut vec_result = vec![0.0f32; DATA_SET_SIZE];
    let result_buffer: Buffer<f32> = ocl_pq.create_buffer().unwrap();

    let kern = ocl_pq.create_kernel("multiply_by_scalar").unwrap()
        .arg_scl(COEFF)
        .arg_buf(&source_buffer)
        .arg_buf(&result_buffer);

    let queue = ocl_pq.queue().clone();
    let buffer_sink = unsafe {
        BufferSink::new(source_buffer.clone(), queue, 0, DATA_SET_SIZE).unwrap()
    };

    let source_data = util::scrambled_vec((0.0, 20.0), ocl_pq.dims().to_len());

    {
        let mut write_guard = buffer_sink.clone().write().wait().unwrap();
        write_guard.copy_from_slice(&source_data);
    }

    buffer_sink.flush(None::<EventList>, None::<&mut Event>).unwrap().wait().unwrap();

    println!("Kernel global work size: {:?}", kern.get_gws());

    unsafe { kern.enq().unwrap(); }

    result_buffer.read(&mut vec_result).enq().unwrap();

    // Check results and print the first 20:
    for idx in 0..DATA_SET_SIZE {
        if idx < RESULTS_TO_PRINT {
            println!("source[{idx}]: {:.03}, \t coeff: {}, \tresult[{idx}]: {}",
            source_data[idx], COEFF, vec_result[idx], idx = idx);
        }
        assert_eq!(source_data[idx] * COEFF, vec_result[idx]);
    }
}
