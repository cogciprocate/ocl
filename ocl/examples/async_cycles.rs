//! Cyclical asynchronous pipelined processing.
//!
//! For many tasks, a single, cyclical pipeline is necessary (think graphics).
//! This examples shows how to saturate the OpenCL device and as many CPU
//! threads as necessary while ensuring that tasks that depend on a previous
//! task completing do not begin before they should. Combine this approach
//! with a temporal dependency graph such as the `CommandGraph` in the
//! `ocl-extras` crate to create arbitrarily complex cyclical task 'webs' that
//! fully saturate all available resources.
//!
//! OpenCL events are used to synchronize commands amongst themselves and with
//! the processing chores being run on the CPU pool. A sync channel is used to
//! synchronize with the main thread and to control how many tasks may be
//! in-flight at the same time (see notes below about the size of that
//! channel/buffer).
//!
//! Each command only needs to wait for the completion of the command(s)
//! immediately before and immediately after it. This is a key part of the
//! design of the aforementioned `CommandGraph` but here we're managing the
//! events manually for demonstration.
//!

// #![feature(conservative_impl_trait, unboxed_closures)]

extern crate chrono;
extern crate futures;
extern crate futures_cpupool;
extern crate ocl;
#[macro_use]
extern crate colorify;

use chrono::{DateTime, Duration, Local};
use futures::Future;
use futures_cpupool::{CpuFuture, CpuPool};
use ocl::{
    error::{Error as OclError, Result as OclResult},
    ffi::{c_void, cl_event},
    flags::{CommandQueueProperties, MemFlags},
    prm::Int4,
    traits::IntoRawEventArray,
    Buffer, Context, Device, Event, Kernel, Platform, Program, Queue, RwVec,
};
use std::{
    fmt::Debug,
    sync::mpsc::{self, Receiver},
    thread::{self, JoinHandle},
};

// Size of buffers and kernel work size:
const WORK_SIZE: usize = 1 << 22;

// Initial value and addend for this example:
const INIT_VAL: i32 = 50;
const SCALAR_ADDEND: i32 = 100;

// Number of times to run the loop:
const TASK_ITERS: i32 = 10;

// The size of the pipeline channel/buffer/queue/whatever (minimum 2). This
// has the effect of increasing the number of threads in use at any one time.
// It does not necessarily mean that those threads will be able to do work
// yet. Because the task in this example has only three CPU-side processing
// stages, only two of which can act concurrently, raising this number above
// two has no effect on the overall performance but could induce extra latency
// if this example were processing input. If more (CPU-bound) steps were
// added, a larger queue would mean more in-flight tasks at a time and
// therefore more stages being processed concurrently. Note that regardless of
// where this is set, an unlimited number of things may be happening
// concurrently on the OpenCL device(s).
const MAX_CONCURRENT_TASK_COUNT: usize = 4;

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
pub fn completion_thread<T, E>(rx: Receiver<Option<CpuFuture<T, E>>>) -> JoinHandle<()>
where
    T: Send + 'static,
    E: Send + Debug + 'static,
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
                }
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
pub fn fill_junk(
    src_buf: &Buffer<Int4>,
    common_queue: &Queue,
    verify_init_event: Option<&Event>,
    kernel_event: Option<&Event>,
    fill_event: &mut Option<Event>,
    task_iter: i32,
) -> OclResult<()> {
    // These just print status messages...
    extern "C" fn _print_starting(_: cl_event, _: i32, task_iter: *mut c_void) {
        printlnc!(peach_bold: "* Fill starting \t\t(iter: {}, t: {}s) ...",
            task_iter as usize, timestamp());
    }
    extern "C" fn _print_complete(_: cl_event, _: i32, task_iter: *mut c_void) {
        printlnc!(peach_bold: "* Fill complete \t\t(iter: {}, t: {}s)",
            task_iter as usize, timestamp());
    }

    // Clear the wait list and push the previous iteration's kernel event
    // and the previous iteration's write init (unmap) event if they are set.
    let wait_list = [&kernel_event, &verify_init_event].into_raw_array();

    // Create a marker so we can print the status message:
    let fill_wait_marker = wait_list.to_marker(&common_queue)?;

    if let Some(ref marker) = fill_wait_marker {
        unsafe {
            marker.set_callback(_print_starting, task_iter as *mut c_void)?;
        }
    } else {
        _print_starting(0 as cl_event, 0, task_iter as *mut c_void);
    }

    *fill_event = Some(Event::empty());

    src_buf
        .cmd()
        .fill(Int4::new(-999, -999, -999, -999), None)
        .queue(common_queue)
        .ewait(&wait_list)
        .enew(fill_event.as_mut())
        .enq()?;

    unsafe {
        fill_event
            .as_ref()
            .unwrap()
            .set_callback(_print_complete, task_iter as *mut c_void)?;
    }
    Ok(())
}

/// 1. Map-Write-Init
/// =================
///
/// Map the buffer and write 50's to the entire buffer, then
/// unmap to actually move data to the device. The `map` will use
/// the common queue and the `unmap` will automatically use the
/// dedicated queue passed to the buffer during creation (unless we
/// specify otherwise).
pub fn write_init(
    src_buf: &Buffer<Int4>,
    common_queue: &Queue,
    write_init_unmap_queue: Queue,
    fill_event: Option<&Event>,
    verify_init_event: Option<&Event>,
    write_init_event: &mut Option<Event>,
    write_val: i32,
    task_iter: i32,
) -> OclResult<Box<dyn Future<Item = i32, Error = OclError> + Send>> {
    extern "C" fn _write_complete(_: cl_event, _: i32, task_iter: *mut c_void) {
        printlnc!(teal_bold: "* Write init complete \t\t(iter: {}, t: {}s)",
            task_iter as usize, timestamp());
    }

    *write_init_event = Some(Event::empty());

    let future_write_data = unsafe {
        src_buf
            .cmd()
            .map()
            .queue(common_queue)
            .write_invalidate()
            .enq_async()?
            // Set up the wait list to use when unmapping. As this is an
            // invalidating write, only the unmap command actually guarantee
            // any data has been moved (the map command does little more than
            // return a reference to our mapped memory when using the
            // `MAP_WRITE_INVALIDATE_REGION` flag).
            .ewait_unmap([&verify_init_event, &fill_event])
            // Set the write unmap completion event which will be set to
            // complete (triggered) after the CPU-side processing is complete
            // and the data is transferred to the device:
            .enew_unmap(write_init_event.as_mut().unwrap())
            // Specify a dedicated queue for the unmap, otherwise the common
            // queue would be used and could cause a deadlock:
            .with_unmap_queue(write_init_unmap_queue)
    };

    unsafe {
        write_init_event
            .as_ref()
            .unwrap()
            .set_callback(_write_complete, task_iter as *mut c_void)?;
    }

    Ok(Box::new(future_write_data.and_then(move |mut data| {
        printlnc!(teal_bold: "* Write init starting \t\t(iter: {}, t: {}s) ...",
            task_iter, timestamp());

        for val in data.iter_mut() {
            *val = Int4::new(write_val, write_val, write_val, write_val);
        }

        Ok(task_iter)
    })))
}

/// 2. Read-Verify-Init
/// ===================
///
/// Read results and verify that the initial mapped write has completed
/// successfully. This will use the common queue for the read and a dedicated
/// queue for the verification completion event (used to signal the next
/// command in the chain).
pub fn verify_init(
    src_buf: &Buffer<Int4>,
    dst_vec: &RwVec<Int4>,
    common_queue: &Queue,
    verify_init_queue: &Queue,
    write_init_event: Option<&Event>,
    verify_init_event: &mut Option<Event>,
    correct_val: i32,
    task_iter: i32,
) -> OclResult<Box<dyn Future<Item = i32, Error = OclError> + Send>> {
    extern "C" fn _verify_starting(_: cl_event, _: i32, task_iter: *mut c_void) {
        printlnc!(blue_bold: "* Verify init starting \t\t(iter: {}, t: {}s) ...",
            task_iter as usize, timestamp());
    }

    // Clear the wait list and push the previous iteration's read verify
    // completion event (if it exists) and the current iteration's write unmap
    // event.
    let wait_list = [&write_init_event, &verify_init_event.as_ref()].into_raw_array();

    let mut future_read_data = src_buf
        .cmd()
        .read(dst_vec)
        .queue(common_queue)
        .ewait(&wait_list)
        .enq_async()?;

    // Attach a status message printing callback to what approximates the
    // verify_init start-time event:
    unsafe {
        future_read_data
            .lock_event()
            .unwrap()
            .set_callback(_verify_starting, task_iter as *mut c_void)?;
    }

    // Create a release event which is triggered when the read guard is dropped.
    *verify_init_event = Some(
        future_read_data
            .create_release_event(verify_init_queue)?
            .clone(),
    );

    // The future which will actually verify the initial value:
    Ok(Box::new(future_read_data.and_then(move |data| {
        let mut val_count = 0;

        for (idx, val) in data.iter().enumerate() {
            let cval = Int4::new(correct_val, correct_val, correct_val, correct_val);
            if *val != cval {
                return Err(format!(
                    "Verify init: Result value mismatch: {:?} != {:?} @ [{}]",
                    val, cval, idx
                )
                .into());
            }
            val_count += 1;
        }

        printlnc!(blue_bold: "* Verify init complete \t\t(iter: {}, t: {}s)",
            task_iter, timestamp());

        Ok(val_count)
    })))
}

/// 3. Kernel-Add
/// =============
///
/// Enqueues a kernel which adds a value to each element in the input buffer.
///
/// The `Kernel complete ...` message is sometimes delayed slightly (a few
/// microseconds) due to the time it takes the callback to trigger.
pub fn kernel_add(
    kern: &Kernel,
    common_queue: &Queue,
    verify_add_event: Option<&Event>,
    write_init_event: Option<&Event>,
    kernel_event: &mut Option<Event>,
    task_iter: i32,
) -> OclResult<()> {
    // These just print status messages...
    extern "C" fn _print_starting(_: cl_event, _: i32, task_iter: *mut c_void) {
        printlnc!(magenta_bold: "* Kernel starting \t\t(iter: {}, t: {}s) ...",
            task_iter as usize, timestamp());
    }
    extern "C" fn _print_complete(_: cl_event, _: i32, task_iter: *mut c_void) {
        printlnc!(magenta_bold: "* Kernel complete \t\t(iter: {}, t: {}s)",
            task_iter as usize, timestamp());
    }

    // Clear the wait list and push the previous iteration's read unmap event
    // and the current iteration's write unmap event if they are set.
    let wait_list = [&verify_add_event, &write_init_event].into_raw_array();

    // Create a marker so we can print the status message:
    let kernel_wait_marker = wait_list.to_marker(&common_queue)?;

    // Attach a status message printing callback to what approximates the
    // kernel wait (start-time) event:
    unsafe {
        kernel_wait_marker
            .as_ref()
            .unwrap()
            .set_callback(_print_starting, task_iter as *mut c_void)?;
    }

    // Create an empty event ready to hold the new kernel event, overwriting any old one.
    *kernel_event = Some(Event::empty());

    // Enqueues the kernel. Since we did not specify a default queue upon
    // creation (for no particular reason) we must specify it here. Also note
    // that the events that this kernel depends on are linked to the *unmap*,
    // not the map commands of the preceding read and writes.
    unsafe {
        kern.cmd()
            .queue(common_queue)
            .ewait(&wait_list)
            .enew(kernel_event.as_mut())
            .enq()?;
    }

    // Attach a status message printing callback to the kernel completion event:
    unsafe {
        kernel_event
            .as_ref()
            .unwrap()
            .set_callback(_print_complete, task_iter as *mut c_void)?;
    }
    Ok(())
}

/// 4. Map-Verify-Add
/// =================
///
/// Read results and verify that the write and kernel have both
/// completed successfully. The `map` will use the common queue and the
/// `unmap` will use a dedicated queue to avoid deadlocks.
///
/// This occasionally shows as having begun a few microseconds before the
/// kernel has completed but that's just due to the slight callback delay on
/// the kernel completion event.
pub fn verify_add(
    dst_buf: &Buffer<Int4>,
    common_queue: &Queue,
    verify_add_unmap_queue: Queue,
    wait_event: Option<&Event>,
    verify_add_event: &mut Option<Event>,
    correct_val: i32,
    task_iter: i32,
) -> OclResult<Box<dyn Future<Item = i32, Error = OclError> + Send>> {
    extern "C" fn _verify_starting(_: cl_event, _: i32, task_iter: *mut c_void) {
        printlnc!(lime_bold: "* Verify add starting \t\t(iter: {}, t: {}s) ...",
            task_iter as usize, timestamp());
    }

    unsafe {
        wait_event
            .as_ref()
            .unwrap()
            .set_callback(_verify_starting, task_iter as *mut c_void)?;
    }

    *verify_add_event = Some(Event::empty());

    let future_read_data = unsafe {
        dst_buf
            .cmd()
            .map()
            .queue(common_queue)
            .read()
            .ewait(wait_event)
            .enq_async()?
            // Set the read unmap completion event:
            .enew_unmap(verify_add_event.as_mut().unwrap())
            // Specify a dedicated queue for the unmap, otherwise the common
            // queue would be used and could cause a deadlock:
            .with_unmap_queue(verify_add_unmap_queue)
    };

    Ok(Box::new(future_read_data.and_then(move |data| {
        let mut val_count = 0;
        let cval = Int4::splat(correct_val);

        for (idx, val) in data.iter().enumerate() {
            if *val != cval {
                return Err(format!(
                    "Verify add: Result value mismatch: {:?} != {:?} @ [{}]",
                    val, cval, idx
                )
                .into());
            }
            val_count += 1;
        }

        printlnc!(lime_bold: "* Verify add complete \t\t(iter: {}, t: {}s)",
            task_iter, timestamp());

        Ok(val_count)
    })))
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
pub fn async_cycles() -> OclResult<()> {
    let platform = Platform::default();
    printlnc!(dark_grey_bold: "Platform: {}", platform.name()?);
    let device = Device::first(platform)?;
    printlnc!(dark_grey_bold: "Device: {} {}", device.vendor()?, device.name()?);

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // For unmap commands, the buffers will each use a dedicated queue to
    // avoid any chance of a deadlock. All other commands will use an
    // unordered common queue.
    let queue_flags = Some(CommandQueueProperties::new().out_of_order());
    let common_queue = Queue::new(&context, device, queue_flags)
        .or_else(|_| Queue::new(&context, device, None))?;
    let write_init_unmap_queue = Queue::new(&context, device, queue_flags)
        .or_else(|_| Queue::new(&context, device, None))?;
    let verify_init_queue = Queue::new(&context, device, queue_flags)
        .or_else(|_| Queue::new(&context, device, None))?;
    let verify_add_unmap_queue = Queue::new(&context, device, queue_flags)
        .or_else(|_| Queue::new(&context, device, None))?;

    // Allocating host memory allows the OpenCL runtime to use special pinned
    // memory which considerably improves the transfer performance of map
    // operations for devices that do not already use host memory (GPUs,
    // etc.). Adding read and write only specifiers also allows for other
    // optimizations.
    let src_buf_flags = MemFlags::new().alloc_host_ptr().read_only();
    let dst_buf_flags = MemFlags::new()
        .alloc_host_ptr()
        .write_only()
        .host_read_only();

    // Create write and read buffers:
    let src_buf: Buffer<Int4> = Buffer::builder()
        .context(&context)
        .flags(src_buf_flags)
        .len(WORK_SIZE)
        .build()?;

    let dst_buf: Buffer<Int4> = Buffer::builder()
        .context(&context)
        .flags(dst_buf_flags)
        .len(WORK_SIZE)
        .build()?;

    // Create program and kernel:
    let program = Program::builder()
        .devices(device)
        .src(KERN_SRC)
        .build(&context)?;

    let kern = Kernel::builder()
        .name("add_slowly")
        .program(&program)
        .global_work_size(WORK_SIZE)
        .arg(&src_buf)
        .arg(SCALAR_ADDEND)
        .arg(&dst_buf)
        .build()?;

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

    unsafe {
        START_TIME = Some(Local::now());
    }
    printlnc!(white_bold: "Starting cycles (t: {}s) ...", timestamp());

    // Our main loop. Could run indefinitely if we had a stream of input.
    for task_iter in 0..TASK_ITERS {
        let ival = INIT_VAL + task_iter;
        let tval = ival + SCALAR_ADDEND;

        // 0. Fill-Junk
        // ============
        fill_junk(
            &src_buf,
            &common_queue,
            verify_init_event.as_ref(),
            kernel_event.as_ref(),
            &mut fill_event,
            task_iter,
        )?;

        // 1. Map-Write-Init
        // ============
        let write_init = write_init(
            &src_buf,
            &common_queue,
            write_init_unmap_queue.clone(),
            fill_event.as_ref(),
            verify_init_event.as_ref(),
            &mut write_init_event,
            ival,
            task_iter,
        )?;

        // 2. Read-Verify-Init
        // ============
        let verify_init = verify_init(
            &src_buf,
            &rw_vec,
            &common_queue,
            &verify_init_queue,
            write_init_event.as_ref(),
            &mut verify_init_event,
            ival,
            task_iter,
        )?;

        // 3. Kernel-Add
        // =============
        kernel_add(
            &kern,
            &common_queue,
            verify_add_event.as_ref(),
            write_init_event.as_ref(),
            &mut kernel_event,
            task_iter,
        )?;

        // 4. Map-Verify-Add
        // =================
        let verify_add = verify_add(
            &dst_buf,
            &common_queue,
            verify_add_unmap_queue.clone(),
            kernel_event.as_ref(),
            &mut verify_add_event,
            tval,
            task_iter,
        )?;

        printlnc!(orange: "All commands for iteration {} enqueued    (t: {}s)",
            task_iter, timestamp());

        let join = write_init.join3(verify_init, verify_add);
        let join_spawned = thread_pool.spawn(join);

        // This places our already spawned and running task into the queue for
        // later collection by our completion thread. This call will block if
        // the queue is full, preventing us from unnecessarily queuing up
        // cycles too far in advance.
        tx.send(Some(join_spawned)).unwrap();
    }

    tx.send(None).unwrap();
    completion_thread.join().unwrap();

    printlnc!(yellow_bold: "All result values are correct! \n\
        Duration => | Total: {} seconds |", timestamp());

    Ok(())
}

pub fn main() {
    match async_cycles() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}
