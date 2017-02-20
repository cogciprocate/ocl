//! Use a thread pool to offload host pre- and post-processing on multiple
//! asynchronous tasks.
//!
//!
//!

extern crate futures;
extern crate futures_cpupool;
extern crate chrono;
extern crate ocl;
#[macro_use] extern crate colorify;

use std::cell::Cell;
use std::collections::VecDeque;
use futures::{stream, Stream};
use futures::future::{Future};
use futures_cpupool::CpuPool;
use ocl::{Platform, Device, Context, Queue, Program, Buffer, Kernel, Event};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::aliases::ClFloat4;


static KERN_SRC: &'static str = r#"
    __kernel void add(
        __global float4* in,
        float4 values,
        __global float4* out)
    {
        uint idx = get_global_id(0);
        out[idx] = in[idx] + values;
    }
"#;


fn fmt_duration(duration: chrono::Duration) -> String {
    let el_sec = duration.num_seconds();
    let el_ms = duration.num_milliseconds() - (el_sec * 1000);
    format!("{}.{} seconds", el_sec, el_ms)
}


pub fn main() {
    let start_time = chrono::Local::now();

    let platform = Platform::default();
    println!("Platform: {}", platform.name());

    let device = Device::first(platform);
    println!("Device: {} {}", device.vendor(), device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    let queue_flags = Some(CommandQueueProperties::new().out_of_order());
    let write_queue = Queue::new(&context, device, queue_flags).unwrap();
    let read_queue = Queue::new(&context, device, queue_flags).unwrap();
    let kern_queue = Queue::new(&context, device, queue_flags).unwrap();

    let thread_pool = CpuPool::new_num_cpus();
    let task_count = 12;
    let redundancy_count = 2000;
    let mut offloads = VecDeque::with_capacity(task_count);

    println!("Creating and enqueuing tasks...");

    for task_id in 0..task_count {
        let work_size = 2 << 14;

        let write_buf_flags = Some(MemFlags::new().read_only().host_write_only());
        let read_buf_flags = Some(MemFlags::new().write_only().host_read_only());

        // Create a write and read buffer:
        let write_buf: Buffer<ClFloat4> = Buffer::new(write_queue.clone(),
            write_buf_flags, work_size, None).unwrap();
        let read_buf: Buffer<ClFloat4> = Buffer::new(read_queue.clone(),
            read_buf_flags, work_size, None).unwrap();

        // Create program and kernel:
        let program = Program::builder()
            .devices(device)
            .src(KERN_SRC)
            .build(&context).unwrap();

        let kern = Kernel::new("add", &program, kern_queue.clone()).unwrap()
            .gws(work_size)
            .arg_buf(&write_buf)
            .arg_vec(ClFloat4(100., 100., 100., 100.))
            .arg_buf(&read_buf);

        // (0) INIT: Fill buffer with -999's just to ensure the upcoming
        // write misses nothing:
        let mut fill_event = Event::empty();
        write_buf.cmd().fill(ClFloat4(-999., -999., -999., -999.), None).enew(&mut fill_event).enq().unwrap();

        // (1) WRITE: Map the buffer and write 50's to the entire buffer, then
        // unmap to 'flush' data to the device:
        let mut future_write_data = write_buf.cmd().map().flags(MapFlags::new().write_invalidate_region())
            .ewait(&fill_event)
            .enq_async().unwrap();

        let write_unmap_event = future_write_data.create_unmap_event().unwrap().clone();

        let write = future_write_data.and_then(move |mut data| {
            for _ in 0..redundancy_count {
                for val in data.iter_mut() {
                    *val = ClFloat4(50., 50., 50., 50.);
                }
            }

            println!("Mapped write complete (task: {}). ", task_id);

            Ok(task_id)
        });

        let spawned_write = thread_pool.spawn(write);

        // (2) KERNEL: Run kernel: Add 100 to everything (total should now be 150):
        let mut kern_event = Event::empty();

        kern.cmd()
            .enew(&mut kern_event)
            .ewait(&write_unmap_event)
            .enq().unwrap();

        // (3) READ: Read results and verify that the write and kernel have
        // both completed successfully:
        let future_read_data = read_buf.cmd().map().flags(MapFlags::new().read())
            .ewait(&kern_event)
            .enq_async().unwrap();

        let read = future_read_data.and_then(move |data| {
                let mut val_count = 0usize;

                for _ in 0..redundancy_count {
                    for val in data.iter() {
                        let correct_val = ClFloat4(150., 150., 150., 150.);
                        if *val != correct_val {
                            return Err(format!("Result value mismatch: {:?} != {:?}", val, correct_val).into())
                        }
                        val_count += 1;
                    }
                }

                println!("Mapped read and verify complete (task: {}). ", task_id);

                Ok(val_count)
            });

        let spawned_read = thread_pool.spawn(read);
        // Presumably this could be either `join` or `and_then`:
        let offload = spawned_write.join(spawned_read);

        offloads.push_back(offload);
    }

    println!("Running tasks...");
    let create_duration = chrono::Local::now() - start_time;
    let correct_val_count = Cell::new(0usize);

    // Finish things up (basically a thread join):
    stream::futures_unordered(offloads).for_each(|(task_id, val_count)| {
        correct_val_count.set(correct_val_count.get() + val_count);
        println!("Task: {} has completed.", task_id);
        Ok(())
    }).wait().unwrap();

    let run_duration = chrono::Local::now() - start_time - create_duration;
    let total_duration = chrono::Local::now() - start_time;

    printlnc!(yellow_bold: "All {} (float4) result values are correct! \n\
        Durations => | Create/Enqueue: {} | Run: {} | Total: {} |",
        correct_val_count.get() / redundancy_count, fmt_duration(create_duration),
        fmt_duration(run_duration), fmt_duration(total_duration));
}







