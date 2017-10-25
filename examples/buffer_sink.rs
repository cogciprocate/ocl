#![allow(unused_imports)]

extern crate ocl;
extern crate futures;

use std::thread::{self, JoinHandle, Builder as ThreadBuilder};
use futures::Future;
use ocl::{util, ProQue, Buffer, MemFlags, Event, EventList};
use ocl::async::{BufferSink, WriteGuard};

// Our arbitrary data set size (about a million) and coefficent:
const DATA_SET_SIZE: usize = 1 << 20;
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


fn main() {
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims(DATA_SET_SIZE)
        .build().expect("Build ProQue");

    let source_buffer = Buffer::<i32>::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write().alloc_host_ptr())
        .dims(ocl_pq.dims().clone())
        .build().unwrap();

    let mut vec_result = vec![0i32; DATA_SET_SIZE];
    let result_buffer: Buffer<i32> = ocl_pq.create_buffer().unwrap();

    let kern = ocl_pq.create_kernel("multiply_by_scalar").unwrap()
        .arg_scl(COEFF)
        .arg_buf(&source_buffer)
        .arg_buf(&result_buffer);
    assert_eq!(kern.get_gws().to_len(), DATA_SET_SIZE);

    let buffer_sink = unsafe {
        BufferSink::from_buffer(source_buffer.clone(), ocl_pq.queue().clone(), 0, DATA_SET_SIZE).unwrap()
    };
    // let source_data = util::scrambled_vec((0, 20), ocl_pq.dims().to_len());
    let source_datas: Vec<_> = (0..THREAD_COUNT).map(|_| {
        util::scrambled_vec((0, 20), ocl_pq.dims().to_len())
    }).collect();
    let mut threads = Vec::<JoinHandle<()>>::with_capacity(THREAD_COUNT * 2);

    for i in 0..THREAD_COUNT {
        // buffer_sink.clone().write().wait().unwrap()
        //     .copy_from_slice(&[0i32; DATA_SET_SIZE]);
        // buffer_sink.flush(None, None::<&mut Event>).unwrap().wait().unwrap();
        let writer_0 = buffer_sink.clone().write();
        threads.push(ThreadBuilder::new().name(format!("thread_{}", i)).spawn(move || {
            let mut write_guard = writer_0.wait().unwrap();
            write_guard.copy_from_slice(&[0i32; DATA_SET_SIZE]);
            let buffer_sink: BufferSink<_> = WriteGuard::release(write_guard).into();
            buffer_sink.flush().enq().unwrap().wait().unwrap();
        }).unwrap());

        let source_data = source_datas[i].clone();

        // buffer_sink.clone().write().wait().unwrap()
        //     .copy_from_slice(&source_data);
        // buffer_sink.flush(None, None::<&mut Event>).unwrap().wait().unwrap();
        let writer_1 = buffer_sink.clone().write();
        threads.push(ThreadBuilder::new().name(format!("thread_{}", i)).spawn(move || {
            let mut write_guard = writer_1.wait().unwrap();
            write_guard.copy_from_slice(&source_data);
            let buffer_sink: BufferSink<_> = WriteGuard::release(write_guard).into();
            buffer_sink.flush().enq().unwrap().wait().unwrap();
        }).unwrap());

        unsafe { kern.enq().unwrap(); }

        result_buffer.read(&mut vec_result).enq().unwrap();

        // // Check results:
        // for (&src, &res) in source_data.iter().zip(vec_result.iter()) {
        //     assert_eq!(src * COEFF, res);
        // }
    }

    for thread in threads {
        thread.join().unwrap();
    }
}



