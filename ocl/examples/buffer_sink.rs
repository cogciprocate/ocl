
extern crate futures;
extern crate ocl;
extern crate ocl_extras;

use std::thread::{JoinHandle, Builder as ThreadBuilder};
use futures::Future;
use ocl::{ProQue, Buffer, MemFlags};
use ocl::async::{BufferSink, WriteGuard};

// Our arbitrary data set size (about a million) and coefficent:
const WORK_SIZE: usize = 1 << 20;
const COEFF: i32 = 321;

const THREAD_COUNT: usize = 32;

// Our kernel source code:
static KERNEL_SRC: &'static str = r#"
    __kernel void multiply_by_scalar(
            __private int const coeff,
            __global int const* const src,
            __global int* const res)
    {
        uint const idx = get_global_id(0);
        res[idx] = src[idx] * coeff;
    }
"#;


fn buffer_sink() -> ocl::Result<()> {
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(WORK_SIZE)
        .build().expect("Build ProQue");

    let source_buffer = Buffer::<i32>::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write().alloc_host_ptr())
        .len(WORK_SIZE)
        .build()?;

    let mut vec_result = vec![0i32; WORK_SIZE];
    let result_buffer: Buffer<i32> = ocl_pq.create_buffer()?;

    let kern = ocl_pq.kernel_builder("multiply_by_scalar")
        .arg(COEFF)
        .arg(&source_buffer)
        .arg(&result_buffer)
        .build()?;
    assert_eq!(kern.default_global_work_size().to_len(), WORK_SIZE);

    let buffer_sink = unsafe {
        BufferSink::from_buffer(source_buffer.clone(), Some(ocl_pq.queue().clone()), 0,
            WORK_SIZE)?
    };
    // let source_data = ocl_extras::scrambled_vec((0, 20), ocl_pq.dims().to_len());
    let source_datas: Vec<_> = (0..THREAD_COUNT).map(|_| {
        ocl_extras::scrambled_vec((0, 20), ocl_pq.dims().to_len())
    }).collect();
    let mut threads = Vec::<JoinHandle<()>>::with_capacity(THREAD_COUNT * 2);

    for i in 0..THREAD_COUNT {
        let writer_0 = buffer_sink.clone().write();
        threads.push(ThreadBuilder::new().name(format!("thread_{}", i)).spawn(move || {
            let mut write_guard = writer_0.wait().unwrap();
            write_guard.copy_from_slice(&[0i32; WORK_SIZE]);
            let buffer_sink: BufferSink<_> = WriteGuard::release(write_guard).into();
            buffer_sink.flush().enq().unwrap().wait().unwrap();
        })?);

        let source_data = source_datas[i].clone();

        let writer_1 = buffer_sink.clone().write();
        threads.push(ThreadBuilder::new().name(format!("thread_{}", i)).spawn(move || {
            let mut write_guard = writer_1.wait().unwrap();
            write_guard.copy_from_slice(&source_data);
            let buffer_sink: BufferSink<_> = WriteGuard::release(write_guard).into();
            buffer_sink.flush().enq().unwrap().wait().unwrap();
        })?);

        unsafe { kern.enq()?; }

        result_buffer.read(&mut vec_result).enq()?;

        // // Check results:
        // for (&src, &res) in source_data.iter().zip(vec_result.iter()) {
        //     assert_eq!(src * COEFF, res);
        // }
    }

    for thread in threads {
        thread.join().unwrap();
    }
    Ok(())
}

pub fn main() {
    match buffer_sink() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}