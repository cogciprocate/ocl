//! Cyclical asynchronous pipelined processing.
//!
//! For many tasks, a single, cyclical pipeline is necessary (think graphics).
//! This examples shows how to fully saturate the OpenCL device and as many
//! CPU threads as necessary while ensuring that tasks that depend on a
//! previous task completing do not begin before they should. Combine this
//! approach with a temporal dependency graph such as the `CommandGraph` in
//! the `ocl-extras` crate to create arbitrarily complex cyclical task 'webs'
//! that fully saturate all available resources.
//!
//! The dependency chain in this example is simple and linear:
//!
//! Map-Write-Unmap -> Kernel -> Map-Read/Verify-Unmap
//!
//! The Map-Write must only wait upon the previous iteration's Kernel to run.
//! The Kernel must wait on the **current** iteration's Map-Write as well as
//! the **previous** iteration's Map-read. And the Map-read only depends on
//! the current iteration's Kernel.
//!
//! OpenCL events are used to synchronize commands amongst themselves and with
//! the processing chores being run on the CPU pool. A sync channel is used to
//! synchronize with the main thread and to control how many tasks may be
//! in-flight at the same time (see notes below about the size of that
//! channel/buffer).
//!
//! Each command only needs to wait for the completion of the command(s)
//! immediately before and immediately after it. This is a key part of the
//! design of the aforementioned `CommandGraph` but we're managing the events
//! manually here for demonstration.
//!


#![allow(unused_imports)]

extern crate chrono;
extern crate futures;
extern crate futures_cpupool;
extern crate ocl;
#[macro_use] extern crate colorify;

use std::thread;
use std::sync::mpsc;
use chrono::{Duration, Local};
use futures::{Future, Join};
use futures_cpupool::{CpuPool, CpuFuture};
use ocl::{Platform, Device, Context, Queue, Program, Kernel, Event, EventList, Buffer};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::aliases::ClInt4;

// Size of buffers and kernel work size:
const WORK_SIZE: usize = 1 << 25;

// Number of times to run the loop:
const TASK_ITERS: usize = 10;

// The size of the pipeline channel/buffer/queue/whatever (minimum 2). This
// has the effect of increasing the number of threads in use at any one time.
// It does not necessarily mean that those threads will be able to do work
// yet. Because the task in this example has only two CPU-side processing
// stages, raising this number above the number of stages has no effect on the
// overall performance but could induce extra latency if this example were
// processing input. If more (CPU-bound) steps were added, a larger queue
// would mean more in-flight tasks at a time and therefore more stages being
// processed concurrently.
const MAX_CONCURRENT_TASK_COUNT: usize = 2;

pub static KERN_SRC: &'static str = r#"
    __kernel void add(
        __global int4* in,
        __private int4 values,
        __global int4* out)
    {
        uint idx = get_global_id(0);
        out[idx] = in[idx] + values;
    }
"#;

pub fn fmt_duration(duration: Duration) -> String {
    let el_sec = duration.num_seconds();
    let el_ms = duration.num_milliseconds() - (el_sec * 1000);
    format!("{}.{} seconds", el_sec, el_ms)
}

pub fn main() {
    let start_time = Local::now();

    let platform = Platform::default();
    printlnc!(peach: "Platform: {}", platform.name());
    let device = Device::first(platform);
    printlnc!(peach_bold: "Device: {} {}", device.vendor(), device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    // Note that for unmap commands, the buffers will each use a dedicated
    // queue to avoid any chance of a deadlock. All other commands will use a
    // common queue.
    let queue_flags = Some(CommandQueueProperties::new().out_of_order());
    let write_unmap_queue = Queue::new(&context, device, queue_flags).unwrap();
    let read_unmap_queue = Queue::new(&context, device, queue_flags).unwrap();
    let common_queue = Queue::new(&context, device, queue_flags).unwrap();

    // Allocating host memory allows the OpenCL runtime to use special pinned
    // memory which considerably improves the transfer performance of map
    // operations for devices that do not already use host memory (GPUs,
    // etc.). Adding read and write only specifiers also allows for other
    // optimizations.
    let write_buf_flags = MemFlags::new().alloc_host_ptr().read_only().host_write_only();
    let read_buf_flags = MemFlags::new().alloc_host_ptr().write_only().host_read_only();

    // Create write and read buffers:
    let write_buf: Buffer<ClInt4> = Buffer::builder()
        .context(&context)
        .flags(write_buf_flags)
        .dims(WORK_SIZE)
        .build().unwrap();

    let read_buf: Buffer<ClInt4> = Buffer::builder()
        .context(&context)
        .flags(read_buf_flags)
        .dims(WORK_SIZE)
        .build().unwrap();

    // Create program and kernel:
    let program = Program::builder()
        .devices(device)
        .src(KERN_SRC)
        .build(&context).unwrap();

    let kern = Kernel::new("add", &program).unwrap()
        .gws(WORK_SIZE)
        .arg_buf(&write_buf)
        .arg_vec(ClInt4(100, 100, 100, 100))
        .arg_buf(&read_buf);

    // Thread pool for offloaded tasks.
    let thread_pool = CpuPool::new_num_cpus();

    // A channel with room to keep a pre-specified number of tasks in-flight.
    let (tx, rx) = mpsc::sync_channel::<Option<CpuFuture<_, _>>>(MAX_CONCURRENT_TASK_COUNT - 2);
    // let (tx, rx) = mpsc::sync_channel::<Option<Join<_, _>>>(MAX_CONCURRENT_TASK_COUNT - 2);

    // Create a thread to handle the stream of work. If this were graphics,
    // this thread could represent the processing being done after a 'finish'
    // call and before a frame being drawn to the screen.
    let completion_thread = thread::spawn(move || {
        let mut task_i = 0usize;

        loop {
            match rx.recv().unwrap() {
                Some(task) => {
                    task.wait().unwrap();
                    println!("Task {} complete.", task_i);

                    task_i += 1;
                    continue;
                },
                None => break,
            }
        }

        printlnc!(white_bold: "All {} futures complete.", task_i);
    });

    // Our events for synchronization:
    let mut write_unmap_event;
    let mut kernel_event = Some(Event::empty());
    let mut read_unmap_event = None::<Event>;
    let mut kernel_wait_list = EventList::with_capacity(2);

    // // (0) INIT: Fill buffer with -999's just to ensure the upcoming
    // // write misses nothing:
    // write_buf.cmd().fill(ClInt4(-999, -999, -999, -999), None)
    //     .enew_opt(kernel_event.as_mut()).enq().unwrap();

    // kernel_event.as_ref().unwrap().wait_for().unwrap();


    // Run our main loop. This could run indefinitely if we had a source of input.
    for task_iter in 0..TASK_ITERS {
        // (1) WRITE: Map the buffer and write 50's to the entire buffer, then
        //     unmap to actually move data to the device. The `map` will use
        //     the common queue and the `unmap` will automatically use the
        //     dedicated queue passed to the buffer during creation (unless we
        //     specify otherwise).
        let mut future_write_data = write_buf.cmd().map()
            .queue(&common_queue)
            .flags(MapFlags::new().write_invalidate_region())
            .ewait_opt(kernel_event.as_ref())
            .enq_async().unwrap();

        // Set the write unmap completion event which will be set to complete
        // (triggered) after the CPU-side processing is complete:
        write_unmap_event = Some(future_write_data.create_unmap_event().unwrap().clone());
        let write_queue_copy = write_unmap_queue.clone();

        let write = future_write_data.and_then(move |mut data| {
            printlnc!(teal_bold: "* Mapped write starting (iter: {}) ...", task_iter);

            for val in data.iter_mut() {
                *val = ClInt4(50, 50, 50, 50);
            }

            printlnc!(teal_bold: "* Mapped write complete (iter: {})", task_iter);

            // Normally we could just let `data` (a `MemMap`) fall out of
            // scope and it would unmap itself. Since we need to specify a
            // special dedicated queue to avoid deadlocks in this case, we
            // call it explicitly.
            data.unmap().queue(&write_queue_copy).enq()?;

            Ok(task_iter)
        });

        // let write_spawned = thread_pool.spawn(write);

        printlnc!(teal: "Mapped write enqueued (iter: {})", task_iter);

        // (2) KERNEL: Run kernel: Add 100 to everything (total should now be
        //    150). Note that the events that this kernel depends on are
        //    linked to the *unmap*, not the map commands of the preceding
        //    read and writes.
        kernel_wait_list.clear();
        if let Some(ref write_ev) = write_unmap_event { kernel_wait_list.push(write_ev.clone()); };
        if let Some(ref read_ev) = read_unmap_event { kernel_wait_list.push(read_ev.clone()); };
        kernel_event = Some(Event::empty());

        // Enqueues the kernel. Since we did not specify a default queue upon
        // creation (for no particular reason) we must specify it here.
        kern.cmd()
            .queue(&common_queue)
            .enew_opt(kernel_event.as_mut())
            .ewait(&kernel_wait_list[..])
            .enq().unwrap();

        printlnc!(magenta: "Kernel enqueued (iter: {})", task_iter);

        // (3) READ: Read results and verify that the write and kernel have
        //     both completed successfully. The `map` will use
        //     the common queue and the `unmap` will use the dedicated queue
        //     passed to the buffer during creation.
        let mut future_read_data = read_buf.cmd().map()
            .queue(&common_queue)
            .flags(MapFlags::new().read())
            .ewait_opt(kernel_event.as_ref())
            .enq_async().unwrap();

        // Set the read unmap completion event:
        read_unmap_event = Some(future_read_data.create_unmap_event().unwrap().clone());
        let read_queue_copy = read_unmap_queue.clone();

        let read = future_read_data.and_then(move |mut data| {
                let mut val_count = 0usize;

                printlnc!(lime_bold: "* Mapped read/verify starting (iter: {}) ...", task_iter);

                for (idx, val) in data.iter().enumerate() {
                    let correct_val = ClInt4(150, 150, 150, 150);
                    if *val != correct_val {
                        return Err(format!("Result value mismatch: {:?} != {:?} @ [{}]", val, correct_val, idx).into());
                    }
                    val_count += 1;
                }

                printlnc!(lime_bold: "* Mapped read/verify complete (iter: {})", task_iter);

                // Explicitly enqueuing the unmap with our dedicated queue.
                data.unmap().queue(&read_queue_copy).enq()?;

                Ok(val_count)
            });

        printlnc!(lime: "Mapped read enqueued (iter: {})", task_iter);

        // let write_spawned = thread_pool.spawn(write);
        // let read_spawned = thread_pool.spawn(read);
        let join = thread_pool.spawn(write.join(read));
        // let join = write_spawned.join(read_spawned);

        printlnc!(orange: "Read and write tasks spawned and running (iter: {}) ...", task_iter);

        // This places our already spawned and running task into the queue for
        // later collection by our completion thread.
        tx.send(Some(join)).unwrap();


        // [DEBUG: TEMPORARY]:
        // write_unmap_event.as_ref().unwrap().wait_for().unwrap();
        // kernel_event.as_ref().unwrap().wait_for().unwrap();
        // read_unmap_event.as_ref().unwrap().wait_for().unwrap();
    }

    tx.send(None).unwrap();
    completion_thread.join().unwrap();
    let total_duration = chrono::Local::now() - start_time;

    printlnc!(yellow_bold: "All result values are correct! \n\
        Duration => | Total: {} |", fmt_duration(total_duration));
}