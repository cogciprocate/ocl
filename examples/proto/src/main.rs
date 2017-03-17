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


#![allow(unused_imports, unused_variables)]
// #![feature(conservative_impl_trait, unboxed_closures)]

extern crate chrono;
extern crate futures;
extern crate futures_cpupool;
extern crate ocl;
#[macro_use] extern crate colorify;

use std::fmt::Debug;
use std::thread::{self, JoinHandle};
use std::sync::mpsc::{self, Receiver};
use chrono::{Duration, DateTime, Local, Timelike};
use futures::{Future, Join, AndThen, BoxFuture};
use futures_cpupool::{CpuPool, CpuFuture};
use ocl::{Platform, Device, Context, Queue, Program, Kernel, Event, EventList, Buffer, OclPrm,
    FutureMemMap, MemMap, RwVec};
use ocl::traits::{IntoMarker, IntoRawList};
use ocl::async::{Error as AsyncError, Result as AsyncResult, PendingRwGuard, RwGuard};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::prm::Int4;
use ocl::ffi::{cl_event, c_void};

// Size of buffers and kernel work size:
const WORK_SIZE: usize = 1 << 24;

// Initial value and addend for this example:
const INIT_VAL: i32 = 50;
const SCALAR_ADDEND: i32 = 100;

// Number of times to run the loop:
const TASK_ITERS: i32 = 10;

// The size of the pipeline channel/buffer/queue/whatever (minimum 2). This
// has the effect of increasing the number of threads in use at any one time.
// It does not necessarily mean that those threads will be able to do work
// yet. Because the task in this example has only three CPU-side processing
// stages, raising this number above that has no effect on the overall
// performance but could induce extra latency if this example were processing
// input. If more (CPU-bound) steps were added, a larger queue would mean more
// in-flight tasks at a time and therefore more stages being processed
// concurrently. Note that regardless of where this is set, an unlimited
// number of things may be happening concurrently on the OpenCL device(s).
const MAX_CONCURRENT_TASK_COUNT: usize = 3;

static mut START_TIME: Option<DateTime<Local>> = None;

// A kernel that makes a career out of adding values.
pub static KERN_SRC: &'static str = r#"
    __kernel void add_slowly(
        __global int4* in,
        __private int addend,
        __global int4* out)
    {
        uint const idx = get_global_id(0);

        float4 const inflated_val = (float4)(addend) * (float4)(255.0);
        int4 sum = (int4)(0);

        for (int i = 0; i < addend; i++) {
            sum += convert_int4((inflated_val / (float4)(255.0)) / (float4)(addend));
        }

        out[idx] = in[idx] + sum;
    }
"#;

/// Returns a duration formatted into a sec.millisec string.
pub fn fmt_duration(duration: Duration) -> String {
    let el_sec = duration.num_seconds();
    let el_mus = duration.num_microseconds().unwrap() - (el_sec * 1000000);
    format!("{}.{:06}", el_sec, el_mus)
}

/// Returns a timestamp with the number of microseconds since the `START_TIME`
/// global time.
pub fn timestamp() -> String {
    fmt_duration(chrono::Local::now() - unsafe { START_TIME.unwrap() })
}

/// Returns a thread hooked up to the provided receiver which simply waits for
/// completion of each `CpuFuture` sent until none remain.
pub fn completion_thread<T, E>(rx: Receiver<Option<CpuFuture<T, E>>>)
        -> JoinHandle<()>
        where T: Send + 'static, E: Send + Debug + 'static
{
    thread::spawn(move || {
        let mut task_i = 0usize;

        loop {
            match rx.recv().unwrap() {
                Some(task) => {
                    task.wait().unwrap();
                    println!("Task {} complete (t: {}s)", task_i, timestamp());

                    task_i += 1;
                    continue;
                },
                None => break,
            }
        }

        printlnc!(white_bold: "All {} futures complete.", task_i);
    })
}


/// 0. Fill-Junk
/// ============
///
/// Fill buffer with -999's just to ensure the upcoming write misses nothing:
pub fn fill_junk(src_buf: &Buffer<Int4>, common_queue: &Queue,
        kernel_event: Option<&Event>,
        write_init_event: Option<&Event>,
        fill_event: &mut Option<Event>,
        task_iter: i32)
{
    // These just print status messages...
    extern "C" fn _print_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        printlnc!(peach_bold: "* Fill starting \t\t(iter: {}, t: {}s) ...",
            task_iter as usize, timestamp());
    }
    extern "C" fn _print_complete(_: cl_event, _: i32, task_iter : *mut c_void) {
        printlnc!(peach_bold: "* Fill complete \t\t(iter: {}, t: {}s)",
            task_iter as usize, timestamp());
    }

    // Clear the wait list and push the previous iteration's kernel event
    // and the previous iteration's write init (unmap) event if they are set.
    let wait_list = [&kernel_event, &write_init_event].into_raw_list();

    // Create a marker so we can print the status message:
    let fill_wait_marker = wait_list.to_marker(&common_queue).unwrap();

    if let Some(ref marker) = fill_wait_marker {
        unsafe { marker.set_callback(_print_starting, task_iter as *mut c_void).unwrap(); }
    } else {
        _print_starting(0 as cl_event, 0, task_iter as *mut c_void);
    }

    *fill_event = Some(Event::empty());

    src_buf.cmd().fill(Int4::new(-999, -999, -999, -999), None)
        .queue(common_queue)
        .ewait(&wait_list)
        .enew_opt(fill_event.as_mut())
        .enq().unwrap();

    unsafe { fill_event.as_ref().unwrap()
        .set_callback(_print_complete, task_iter as *mut c_void).unwrap(); }
}


/// 1. Map-Write-Init
/// =================
///
/// Map the buffer and write 50's to the entire buffer, then
/// unmap to actually move data to the device. The `map` will use
/// the common queue and the `unmap` will automatically use the
/// dedicated queue passed to the buffer during creation (unless we
/// specify otherwise).
pub fn write_init(src_buf: &Buffer<Int4>, common_queue: &Queue,
        write_init_unmap_queue: Queue,
        fill_event: Option<&Event>,
        verify_init_event: Option<&Event>,
        write_init_event: &mut Option<Event>,
        write_val: i32, task_iter: i32)
        // -> AndThen<FutureMemMap<Int4>, AsyncResult<i32>,
        //     impl FnOnce(MemMap<Int4>) -> AsyncResult<i32>>
        -> BoxFuture<i32, AsyncError>
{
    extern "C" fn _write_complete(_: cl_event, _: i32, task_iter : *mut c_void) {
        printlnc!(teal_bold: "* Write init complete \t\t(iter: {}, t: {}s)",
            task_iter as usize, timestamp());
    }

    // Clear the wait list and push the previous iteration's verify init event
    // and the current iteration's fill event if they are set.
    let wait_list = [&verify_init_event, &fill_event].into_raw_list();

    let mut future_write_data = src_buf.cmd().map()
        .queue(common_queue)
        // .queue(&write_init_unmap_queue)
        .flags(MapFlags::new().write_invalidate_region())
        .ewait(&wait_list)
        .enq_async().unwrap();

    // Set the write unmap completion event which will be set to complete
    // (triggered) after the CPU-side processing is complete and the data is
    // transferred to the device:
    *write_init_event = Some(future_write_data.create_unmap_event().unwrap().clone());

    unsafe { write_init_event.as_ref().unwrap().set_callback(_write_complete,
        task_iter as *mut c_void).unwrap(); }

    future_write_data.and_then(move |mut data| {
        printlnc!(teal_bold: "* Write init starting \t\t(iter: {}, t: {}s) ...",
            task_iter, timestamp());

        for val in data.iter_mut() {
            *val = Int4::new(write_val, write_val, write_val, write_val);
        }

        // Normally we could just let `data` (a `MemMap`) fall out of
        // scope and it would unmap itself. Since we need to specify a
        // special dedicated queue to avoid deadlocks in this case, we
        // call it explicitly.
        data.unmap().queue(&write_init_unmap_queue).enq()?;

        Ok(task_iter)
    }).boxed()
}


/// 2. Read-Verify-Init
/// ===================
/// Read results and verify that the initial mapped write has completed
/// successfully. This will use the common queue for the read and a dedicated
/// queue for the verification completion event (used to signal the next
/// command in the chain).
pub fn verify_init(src_buf: &Buffer<Int4>, dst_vec: &RwVec<Int4>, common_queue: &Queue,
        verify_init_queue: &Queue,
        write_init_event: Option<&Event>,
        verify_init_event: &mut Option<Event>,
        correct_val: i32, task_iter: i32)
        // -> AndThen<PendingRwGuard<Int4>, AsyncResult<i32>,
        //     impl FnOnce(RwGuard<Int4>) -> AsyncResult<i32>>
        -> BoxFuture<i32, AsyncError>
{
    extern "C" fn _verify_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        printlnc!(blue_bold: "* Verify init starting \t\t(iter: {}, t: {}s) ...",
            task_iter as usize, timestamp());
    }

    // Clear the wait list and push the previous iteration's read verify
    // completion event (if it exists) and the current iteration's write unmap
    // event.
    let wait_list = [&verify_init_event.as_ref(), &write_init_event].into_raw_list();

    // // Create a marker so we can print the status message:
    // let verify_init_wait_marker = wait_list.to_marker(&verify_init_queue).unwrap();

    // // Attach a status message printing callback to what approximates the
    // // verify_init wait (start-time) event:
    // unsafe { verify_init_wait_marker.as_ref().unwrap()
    //     .set_callback(_verify_starting, task_iter as *mut c_void).unwrap(); }

    let mut future_read_data = src_buf.cmd().read(dst_vec)
        .queue(common_queue)
        .ewait(&wait_list)
        .enq_async().unwrap();

    // Attach a status message printing callback to what approximates the
    // verify_init start-time event:
    unsafe { future_read_data.command_trigger_event()
        .set_callback(_verify_starting, task_iter as *mut c_void).unwrap(); }

    // Create an empty event ready to hold the new verify_init event, overwriting any old one.
    *verify_init_event = Some(future_read_data.create_drop_event(verify_init_queue)
        .unwrap().clone());

    // The future which will actually verify the initial value:
    future_read_data.and_then(move |data| {
        // println!("Attempting to lock RwVec for reading...");
        // let data = rw_vec.lock().wait().unwrap();
        // println!("RwVec read lock established.");
        let mut val_count = 0;

        for (idx, val) in data.iter().enumerate() {
            let cval = Int4::new(correct_val, correct_val, correct_val, correct_val);
            if *val != cval {
                return Err(format!("Verify init: Result value mismatch: {:?} != {:?} @ [{}]", val, cval, idx).into());
            }
            val_count += 1;
        }

        printlnc!(blue_bold: "* Verify init complete \t\t(iter: {}, t: {}s)",
            task_iter, timestamp());

        Ok(val_count)
    }).boxed()
}


/// 3. Kernel-Add
/// =============
///
/// Enqueues a kernel which adds a value to each element in the input buffer.
///
/// The `Kernel complete ...` message is sometimes delayed slightly (a few
/// microseconds) due to the time it takes the callback to trigger.
pub fn kernel_add(kern: &Kernel, common_queue: &Queue,
        verify_add_event: Option<&Event>,
        write_init_event: Option<&Event>,
        kernel_event: &mut Option<Event>,
        task_iter: i32)
{
    // These just print status messages...
    extern "C" fn _print_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        printlnc!(magenta_bold: "* Kernel starting \t\t(iter: {}, t: {}s) ...",
            task_iter as usize, timestamp());
    }
    extern "C" fn _print_complete(_: cl_event, _: i32, task_iter : *mut c_void) {
        printlnc!(magenta_bold: "* Kernel complete \t\t(iter: {}, t: {}s)",
            task_iter as usize, timestamp());
    }

    // Clear the wait list and push the previous iteration's read unmap event
    // and the current iteration's write unmap event if they are set.
    let wait_list = [&verify_add_event, &write_init_event].into_raw_list();

    // Create a marker so we can print the status message:
    let kernel_wait_marker = wait_list.to_marker(&common_queue).unwrap();

    // Attach a status message printing callback to what approximates the
    // kernel wait (start-time) event:
    unsafe { kernel_wait_marker.as_ref().unwrap()
        .set_callback(_print_starting, task_iter as *mut c_void).unwrap(); }

    // Create an empty event ready to hold the new kernel event, overwriting any old one.
    *kernel_event = Some(Event::empty());

    // Enqueues the kernel. Since we did not specify a default queue upon
    // creation (for no particular reason) we must specify it here. Also note
    // that the events that this kernel depends on are linked to the *unmap*,
    // not the map commands of the preceding read and writes.
    kern.cmd()
        .queue(common_queue)
        .ewait(&wait_list)
        .enew_opt(kernel_event.as_mut())
        .enq().unwrap();

    // Attach a status message printing callback to the kernel completion event:
    unsafe { kernel_event.as_ref().unwrap().set_callback(_print_complete,
        task_iter as *mut c_void).unwrap(); }
}


/// 4. Map-Verify-Add
/// =================
/// Read results and verify that the write and kernel have both
/// completed successfully. The `map` will use the common queue and the
/// `unmap` will use a dedicated queue to avoid deadlocks.
///
/// This occasionally shows as having begun a few microseconds before the
/// kernel has completed but that's just due to the slight callback delay on
/// the kernel completion event.
pub fn verify_add(dst_buf: &Buffer<Int4>, common_queue: &Queue,
        verify_add_unmap_queue: Queue,
        wait_event: Option<&Event>,
        verify_add_event: &mut Option<Event>,
        correct_val: i32, task_iter: i32)
        // -> AndThen<FutureMemMap<Int4>, AsyncResult<i32>,
        //     impl FnOnce(MemMap<Int4>) -> AsyncResult<i32>>
        -> BoxFuture<i32, AsyncError>
{
    extern "C" fn _verify_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        printlnc!(lime_bold: "* Verify add starting \t\t(iter: {}, t: {}s) ...",
            task_iter as usize, timestamp());
    }

    unsafe { wait_event.as_ref().unwrap()
        .set_callback(_verify_starting, task_iter as *mut c_void).unwrap(); }

    let mut future_read_data = dst_buf.cmd().map()
        .queue(common_queue)
        .flags(MapFlags::new().read())
        .ewait_opt(wait_event)
        .enq_async().unwrap();

    // Set the read unmap completion event:
    *verify_add_event = Some(future_read_data.create_unmap_event().unwrap().clone());

    future_read_data.and_then(move |mut data| {
        let mut val_count = 0;

        for (idx, val) in data.iter().enumerate() {
            // let cval = Int4::new(correct_val, correct_val, correct_val, correct_val);
            let cval = Int4::splat(correct_val);
            if *val != cval {
                return Err(format!("Verify add: Result value mismatch: {:?} != {:?} @ [{}]", 
                    val, cval, idx).into());
            }
            val_count += 1;
        }

        printlnc!(lime_bold: "* Verify add complete \t\t(iter: {}, t: {}s)",
            task_iter, timestamp());

        // Explicitly enqueue the unmap with our dedicated queue,
        // otherwise the common queue would be used and could cause
        // a deadlock:
        data.unmap().queue(&verify_add_unmap_queue).enq()?;

        Ok(val_count)
    }).boxed()
}


/// Main
/// ====
///
/// Repeatedly:
///   0. fills with garbage,
///   1. writes a start value,
///   2. verifies the write,
///   3. adds a value,
///   4. and verifies the sum.
///
pub fn main() {
    let platform = Platform::default();
    printlnc!(dark_grey_bold: "Platform: {}", platform.name());
    let device = Device::first(platform);
    printlnc!(dark_grey_bold: "Device: {} {}", device.vendor(), device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    // For unmap commands, the buffers will each use a dedicated queue to
    // avoid any chance of a deadlock. All other commands will use an
    // unordered common queue.
    let queue_flags = Some(CommandQueueProperties::new().out_of_order());
    let common_queue = Queue::new(&context, device, queue_flags).unwrap();
    let write_init_unmap_queue = Queue::new(&context, device, queue_flags).unwrap();
    let verify_init_queue = Queue::new(&context, device, queue_flags).unwrap();
    let verify_add_unmap_queue = Queue::new(&context, device, queue_flags).unwrap();

    // Allocating host memory allows the OpenCL runtime to use special pinned
    // memory which considerably improves the transfer performance of map
    // operations for devices that do not already use host memory (GPUs,
    // etc.). Adding read and write only specifiers also allows for other
    // optimizations.
    //
    // let src_buf_flags = MemFlags::new().alloc_host_ptr().read_only().host_write_only();
    let src_buf_flags = MemFlags::new().alloc_host_ptr().read_only();
    let dst_buf_flags = MemFlags::new().alloc_host_ptr().write_only().host_read_only();

    // Create write and read buffers:
    let src_buf: Buffer<Int4> = Buffer::builder()
        .context(&context)
        .flags(src_buf_flags)
        .dims(WORK_SIZE)
        .build().unwrap();

    let dst_buf: Buffer<Int4> = Buffer::builder()
        .context(&context)
        .flags(dst_buf_flags)
        .dims(WORK_SIZE)
        .build().unwrap();

    // Create program and kernel:
    let program = Program::builder()
        .devices(device)
        .src(KERN_SRC)
        .build(&context).unwrap();

    let kern = Kernel::new("add_slowly", &program).unwrap()
        .gws(WORK_SIZE)
        .arg_buf(&src_buf)
        .arg_scl(SCALAR_ADDEND)
        .arg_buf(&dst_buf);

    // A lockable vector for non-map reads.
    let rw_vec: RwVec<Int4> = RwVec::from(vec![Default::default(); WORK_SIZE]);

    // Thread pool for offloaded tasks.
    let thread_pool = CpuPool::new_num_cpus();

    // A channel with room to keep a pre-specified number of tasks in-flight.
    let (tx, rx) = mpsc::sync_channel::<Option<CpuFuture<_, _>>>(MAX_CONCURRENT_TASK_COUNT - 2);

    // Create a thread to handle the stream of work. If this were graphics,
    // this thread could represent the processing being done after a 'finish'
    // call and before a frame being drawn to the screen.
    let completion_thread = completion_thread(rx);

    // Our events for synchronization.
    let mut fill_event = None;
    let mut write_init_event = None;
    let mut verify_init_event: Option<Event> = None;
    let mut kernel_event = None;
    let mut verify_add_event = None;

    unsafe { START_TIME = Some(Local::now()); }
    printlnc!(white_bold: "Starting cycles (t: {}s) ...", timestamp());

    // Our main loop. Could run indefinitely if we had a stream of input.
    for task_iter in 0..TASK_ITERS {
        let ival = INIT_VAL + task_iter;
        let tval = ival + SCALAR_ADDEND;

        // 0. Fill-Junk
        // ============
        fill_junk(&src_buf, &common_queue,
            write_init_event.as_ref(),
            kernel_event.as_ref(),
            &mut fill_event,
            task_iter);

        // 1. Map-Write-Init
        // ============
        let write_init = write_init(&src_buf, &common_queue,
            write_init_unmap_queue.clone(),
            fill_event.as_ref(),
            verify_init_event.as_ref(),
            &mut write_init_event,
            ival, task_iter);

        // 2. Read-Verify-Init
        // ============
        let verify_init = verify_init(&src_buf, &rw_vec, &common_queue,
            &verify_init_queue,
            write_init_event.as_ref(),
            &mut verify_init_event,
            ival, task_iter);

        // 3. Kernel-Add
        // =============
        kernel_add(&kern, &common_queue,
            verify_add_event.as_ref(),
            write_init_event.as_ref(),
            &mut kernel_event,
            task_iter);

        // 3. Map-Verify-Add
        // =================
        let verify_add = verify_add(&dst_buf, &common_queue,
            verify_add_unmap_queue.clone(),
            kernel_event.as_ref(),
            &mut verify_add_event,
            tval, task_iter);

        printlnc!(orange: "All commands for iteration {} enqueued    (t: {}s)",
            task_iter, timestamp());

        // let join = write_init.join3(verify_init, verify_add);
        // let join_spawned = thread_pool.spawn(join);

        // // This places our already spawned and running task into the queue for
        // // later collection by our completion thread. This call will block if
        // // the queue is full, preventing us from unnecessarily queuing up
        // // cycles too far in advance.
        // tx.send(Some(join_spawned)).unwrap();


        ////// [DEBUG]:
            fill_event.as_ref().unwrap().wait_for().unwrap();
            write_init.wait().unwrap();
            verify_init.wait().unwrap();
            kernel_event.as_ref().unwrap().wait_for().unwrap();
            verify_add.wait().unwrap();

            // let join = write_init.join3(verify_init, verify_add);
            // let join_spawned = thread_pool.spawn(join);
            // tx.send(Some(join_spawned)).unwrap();

            continue;

            tx.send(None::<CpuFuture<(), AsyncResult<()>>>).unwrap();
        ///////
    }

    tx.send(None).unwrap();
    completion_thread.join().unwrap();

    printlnc!(yellow_bold: "All result values are correct! \n\
        Duration => | Total: {} seconds |", timestamp());
}