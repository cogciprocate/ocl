//! async.rs
//!
//! Only tests the default platform (all devices).
//!
//! Does not panic on errors (use `device_check.rs` example for platform bugs).

// #![allow(unused_imports, unused_variables, unused_mut)]

use std::thread;
use futures::{Future};
use core::Status;
use ::{Platform, Device, Context, Queue, Program, Kernel, Event, Buffer, RwVec};
use ::traits::IntoRawEventArray;
use ::error::{Error as OclError, Result as OclResult};
use ::flags::{MemFlags, CommandQueueProperties};
use ::prm::Int4;
use ::ffi::{cl_event, c_void};

// Size of buffers and kernel work size:
//
// NOTE: Intel platform drivers may intermittently crash and error with
// `DEVICE_NOT_AVAILABLE` on older hardware if this number is too low. Use AMD
// drivers.
const WORK_SIZE: usize = 1 << 12;

// Initial value and addend for this example:
const INIT_VAL: i32 = 50;
const SCALAR_ADDEND: i32 = 100;

// The number of tasks to run concurrently.
const TASK_ITERS: i32 = 16;

const PRINT: bool = false;


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


/// 0. Fill-Junk
/// ============
///
/// Fill buffer with -999's just to ensure the upcoming write misses nothing:
pub fn fill_junk(
        src_buf: &Buffer<Int4>,
        common_queue: &Queue,
        kernel_event: Option<&Event>,
        verify_init_event: Option<&Event>,
        fill_event: &mut Option<Event>,
        task_iter: i32)
{
    // These just print status messages...
    extern "C" fn _print_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Fill starting        \t(iter: {}) ...", task_iter as usize); }
    }
    extern "C" fn _print_complete(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Fill complete        \t(iter: {})", task_iter as usize); }
    }

    // Clear the wait list and push the previous iteration's kernel event
    // and the previous iteration's write init (unmap) event if they are set.
    let wait_list = [kernel_event, verify_init_event].into_raw_array();

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
        .enew(fill_event.as_mut())
        .enq().unwrap();

    unsafe { fill_event.as_ref().unwrap()
        .set_callback(_print_complete, task_iter as *mut c_void).unwrap(); }
}


/// 1. Write-Init
/// =================
///
/// Map the buffer and write 50's to the entire buffer, then
/// unmap to actually move data to the device. The `map` will use
/// the common queue and the `unmap` will automatically use the
/// dedicated queue passed to the buffer during creation (unless we
/// specify otherwise).
pub fn write_init(
        src_buf: &Buffer<Int4>,
        rw_vec: &RwVec<Int4>,
        common_queue: &Queue,
        write_init_release_queue_0: &Queue,
        // write_init_release_queue_1: &Queue,
        fill_event: Option<&Event>,
        verify_init_event: Option<&Event>,
        write_init_event: &mut Option<Event>,
        write_val: i32, task_iter: i32)
        -> Box<Future<Item=i32, Error=OclError> + Send>
{
    extern "C" fn _write_complete(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Write init complete  \t(iter: {})", task_iter as usize); }
    }

    // // Clear the wait list and push the previous iteration's verify init event
    // // and the current iteration's fill event if they are set.
    // let wait_marker = [verify_init_event, fill_event].into_marker(common_queue).unwrap();

    // println!("###### write_init() events (task_iter: [{}]): \n\
    //  ######     'verify_init_event': {:?}, \n\
    //  ######     'fill_event': {:?} \n\
    //  ######       -> 'wait_marker': {:?}",
    //  task_iter, verify_init_event, fill_event, wait_marker);

    let wait_list = [verify_init_event, fill_event];

    let mut future_guard = rw_vec.clone().write();
    // future_guard.set_wait_event(wait_marker.as_ref().unwrap().clone());
    future_guard.set_lock_wait_events(wait_list);
    let release_event = future_guard.create_release_event(write_init_release_queue_0)
        .unwrap().clone();

    // println!("######     'release_event' (generate): {:?}", release_event);

    let future_write_vec = future_guard.and_then(move |mut data| {
        if PRINT { println!("* Write init starting  \t(iter: {}) ...", task_iter); }

        for val in data.iter_mut() {
            *val = Int4::splat(write_val);
        }

        Ok(())
    });

    let mut future_write_buffer = src_buf.cmd().write(rw_vec)
        .queue(common_queue)
        .ewait(&release_event)
        .enq_async().unwrap();

    // Set the write unmap completion event which will be set to complete
    // (triggered) after the CPU-side processing is complete and the data is
    // transferred to the device:
    *write_init_event = Some(future_write_buffer.create_release_event(write_init_release_queue_0)
        .unwrap().clone());

    // println!("######     'release_event' ('write_init_event'): {:?}", write_init_event);

    unsafe { write_init_event.as_ref().unwrap().set_callback(_write_complete,
        task_iter as *mut c_void).unwrap(); }

    let future_drop_guard = future_write_buffer.and_then(move |_| Ok(()));

    Box::new(future_write_vec.join(future_drop_guard).map(move |(_, _)| task_iter))
}


/// 2. Verify-Init
/// ===================
///
/// Read results and verify that the initial mapped write has completed
/// successfully. This will use the common queue for the read and a dedicated
/// queue for the verification completion event (used to signal the next
/// command in the chain).
pub fn verify_init(
        src_buf: &Buffer<Int4>,
        rw_vec: &RwVec<Int4>,
        common_queue: &Queue,
        verify_init_queue: &Queue,
        write_init_event: Option<&Event>,
        verify_init_event: &mut Option<Event>,
        correct_val: i32, task_iter: i32)
        -> Box<Future<Item=i32, Error=OclError> + Send>
{
    extern "C" fn _verify_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Verify init starting \t(iter: {}) ...", task_iter as usize); }
    }

    // Clear the wait list and push the previous iteration's read verify
    // completion event (if it exists) and the current iteration's write unmap
    // event.
    let wait_list = [&verify_init_event.as_ref(), &write_init_event].into_raw_array();

    let mut future_read_data = src_buf.cmd().read(rw_vec)
        .queue(common_queue)
        .ewait(&wait_list)
        .enq_async().unwrap();

    // Attach a status message printing callback to what approximates the
    // verify_init start-time event:
    unsafe { future_read_data.lock_event().unwrap().set_callback(
        _verify_starting, task_iter as *mut c_void).unwrap(); }

    // Create an empty event ready to hold the new verify_init event, overwriting any old one.
    *verify_init_event = Some(future_read_data.create_release_event(verify_init_queue)
        .unwrap().clone());

    // The future which will actually verify the initial value:
    Box::new(future_read_data.and_then(move |data| {
        let mut val_count = 0;

        for (idx, val) in data.iter().enumerate() {
            let cval = Int4::new(correct_val, correct_val, correct_val, correct_val);
            if *val != cval {
                return Err(format!("Verify init: Result value mismatch: {:?} != {:?} @ [{}] \
                    for task iter: [{}].", val, cval, idx, task_iter).into());
            }
            val_count += 1;
        }

        if PRINT { println!("* Verify init complete \t(iter: {})", task_iter); }

        Ok(val_count)
    }))
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
        task_iter: i32)
{
    // These just print status messages...
    extern "C" fn _print_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Kernel starting      \t(iter: {}) ...", task_iter as usize); }
    }
    extern "C" fn _print_complete(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Kernel complete      \t(iter: {})", task_iter as usize); }
    }

    // Clear the wait list and push the previous iteration's read unmap event
    // and the current iteration's write unmap event if they are set.
    let wait_list = [&verify_add_event, &write_init_event].into_raw_array();

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
    unsafe {
        kern.cmd()
            .queue(common_queue)
            .ewait(&wait_list)
            .enew(kernel_event.as_mut())
            .enq().unwrap();
    }

    // Attach a status message printing callback to the kernel completion event:
    unsafe { kernel_event.as_ref().unwrap().set_callback(_print_complete,
        task_iter as *mut c_void).unwrap(); }
}


/// 4. Verify-Add
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
        rw_vec: &RwVec<Int4>,
        common_queue: &Queue,
        verify_add_unmap_queue: &Queue,
        wait_event: Option<&Event>,
        verify_add_event: &mut Option<Event>,
        correct_val: i32, task_iter: i32)
        -> Box<Future<Item=i32, Error=OclError> + Send>
{
    extern "C" fn _verify_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Verify add starting  \t(iter: {}) ...", task_iter as usize); }
    }

    let mut future_read_data = dst_buf.cmd().read(rw_vec)
        .queue(common_queue)
        .ewait(wait_event)
        .enq_async().unwrap();

    // Attach a status message printing callback to what approximates the
    // verify_init start-time event:
    unsafe { future_read_data.lock_event().unwrap().set_callback(
        _verify_starting, task_iter as *mut c_void).unwrap(); }

    // Create an empty event ready to hold the new verify_init event, overwriting any old one.
    *verify_add_event = Some(future_read_data.create_release_event(verify_add_unmap_queue)
        .unwrap().clone());

    Box::new(future_read_data.and_then(move |mut data| {
        let mut val_count = 0;

        for (idx, val) in data.iter().enumerate() {
            let cval = Int4::splat(correct_val);
            if *val != cval {
                return Err(format!("Verify add: Result value mismatch: {:?} != {:?} @ [{}] \
                    for task iter: [{}].", val, cval, idx, task_iter).into());
            }
            val_count += 1;
        }

        // This is just for shits:
        for val in data.iter_mut() {
            *val = Int4::splat(0);
        }

        if PRINT { println!("* Verify add complete  \t(iter: {})", task_iter); }

        Ok(val_count)
    }))
}

/// Creates an out-of-order queue or a shorter error message if unsupported.
fn create_queue(context: &Context, device: Device, flags: Option<CommandQueueProperties>)
        -> OclResult<Queue>
{
    Queue::new(context, device, flags.clone()).or_else(|err| {
        match err.api_status() {
            Some(Status::CL_INVALID_VALUE) => Err("Device does not support out of order queues.".into()),
            _ => Err(err.into()),
        }
    })
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
//
// Ignores errors due to platform driver bugs.
//
#[test]
pub fn rw_vec() {
    // if cfg!(not(feature = "async_block")) { panic!("'async_block' disabled!"); }

    // let platform = Platform::default();
    // println!("Platform: {}", platform.name());
    // let device = Device::by_idx_wrap(&platform, 2);
    // println!("Device: {} {}", device.vendor(), device.name());

    for platform in Platform::list() {
        for device in Device::list_all(platform).unwrap() {
            println!("Device: {} {}", device.vendor().unwrap(), device.name().unwrap());

            let context = Context::builder()
                .platform(platform)
                .devices(device)
                .build().unwrap();

            // For unmap commands, the buffers will each use a dedicated queue
            // to avoid any chance of a deadlock. All other commands will use
            // an unordered common queue. If the `async_block` feature is
            // enabled, use in-order queues (since by blocking, everything
            // will be synchronous anyway).
            let queue_flags = if cfg!(feature = "async_block") {
                None
            } else {
                Some(CommandQueueProperties::new().out_of_order())
            };

            let common_queue = create_queue(&context, device, queue_flags).unwrap();
            let write_init_unmap_queue_0 = create_queue(&context, device, queue_flags).unwrap();
            // let write_init_unmap_queue_1 = create_queue(&context, device, queue_flags).unwrap();
            let verify_init_queue = create_queue(&context, device, queue_flags).unwrap();
            let verify_add_unmap_queue = create_queue(&context, device, queue_flags).unwrap();

            // Allocating host memory allows the OpenCL runtime to use special pinned
            // memory which considerably improves the transfer performance of map
            // operations for devices that do not already use host memory (GPUs,
            // etc.). Adding read and write only specifiers also allows for other
            // optimizations.
            let src_buf_flags = MemFlags::new().alloc_host_ptr().read_only();
            let dst_buf_flags = MemFlags::new().alloc_host_ptr().write_only().host_read_only();

            // Create write and read buffers:
            let src_buf: Buffer<Int4> = Buffer::builder()
                .context(&context)
                .flags(src_buf_flags)
                .len(WORK_SIZE)
                .build().unwrap();

            let dst_buf: Buffer<Int4> = Buffer::builder()
                .context(&context)
                .flags(dst_buf_flags)
                .len(WORK_SIZE)
                .build().unwrap();

            // Create program and kernel:
            let program = Program::builder()
                .devices(device)
                .src(KERN_SRC)
                .build(&context).unwrap();

            let kern = Kernel::builder()
                .program(&program)
                .name("add_slowly")
                .global_work_size(WORK_SIZE)
                .arg_buf(&src_buf)
                .arg_scl(&SCALAR_ADDEND)
                .arg_buf(&dst_buf)
                .build().unwrap();

            // A lockable vector for reads and writes:
            let rw_vec: RwVec<Int4> = RwVec::from(vec![Default::default(); WORK_SIZE]);

            // A place to store our threads:
            let mut threads = Vec::with_capacity(TASK_ITERS as usize);

            // Our events for synchronization.
            let mut fill_event = None;
            let mut write_init_event = None;
            let mut verify_init_event = None;
            let mut kernel_event = None;
            let mut verify_add_event = None;

            if PRINT { println!("Starting cycles ..."); }

            // Our main loop. Could run indefinitely if we had a stream of input.
            for task_iter in 0..TASK_ITERS {
                let ival = INIT_VAL + task_iter;
                let tval = ival + SCALAR_ADDEND;

                // 0. Fill-Junk
                // ============
                fill_junk(&src_buf, &common_queue,
                    verify_init_event.as_ref(),
                    kernel_event.as_ref(),
                    &mut fill_event,
                    task_iter);

                // 1. Write-Init
                // ============
                let write_init = write_init(&src_buf, &rw_vec, &common_queue,
                    &write_init_unmap_queue_0,
                    // &write_init_unmap_queue_1,
                    fill_event.as_ref(),
                    verify_init_event.as_ref(),
                    &mut write_init_event,
                    ival, task_iter);

                // 2. Verify-Init
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

                // 4. Verify-Add
                // =================
                let verify_add = verify_add(&dst_buf, &rw_vec, &common_queue,
                    &verify_add_unmap_queue,
                    kernel_event.as_ref(),
                    &mut verify_add_event,
                    tval, task_iter);

                if PRINT { println!("All commands for iteration {} enqueued", task_iter); }

                let task = write_init.join3(verify_init, verify_add);
                // let task = write_init.join(verify_add);

                threads.push(thread::Builder::new()
                        .name(format!("task_iter_[{}]", task_iter).into())
                        .spawn(move ||
                {
                    if PRINT { println!("Waiting on task iter [{}]...", task_iter); }
                    match task.wait() {
                        Ok(res) => {
                            if PRINT { println!("Task iter [{}] complete with result: {:?}", task_iter, res); }
                            true
                        },
                        Err(err) => {
                            if PRINT { println!("\n############## ERROR (task iter: [{}]) \
                                ############## \n{:?}\n", task_iter, err); }
                            false
                        },
                    }
                }).unwrap());
            }

            let mut all_correct = true;

            for thread in threads {
                // thread.join().unwrap();
                match thread.join() {
                    Ok(res) => {
                        if PRINT { println!("Thread result: {:?}", res); }
                        if !res { all_correct = false; }
                    },
                    Err(err) => panic!("{:?}", err),
                }
            }

            if all_correct {
                println!("All result values are correct.");
            } else {
                // panic!("Errors found!");
                println!("Errors found!");
            }
        }
    }
}