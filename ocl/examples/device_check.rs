//! Checks all platforms and devices for driver bugs.
//!
//! Originally designed to isolate a severe glitch on a particular device and
//! grew into a general purpose device stress tester.
//!
//! Far from complete but will check for a few possible problems.
//!
//! - Buffer without `CL_ALLOC_HOST_PTR` [FIXME: list flags]
//! - FIXME: Finish this
//!
//!
//! FIXME: List future plans here
//!
//!
//!
//!

extern crate futures;
extern crate rand;
extern crate ocl;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate colorify;

use std::fmt::{Debug};
use futures::{Future};
use ocl::{core, Platform, Device, Context, Queue, Program, Buffer, Kernel, OclPrm,
    Event, EventList, MemMap, RwVec};
use ocl::error::{Error as OclError, Result as OclResult};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::traits::{IntoRawEventArray};
use ocl::prm::{Float4, Int4};
use ocl::core::{Event as EventCore, Status};
use ocl::ffi::{cl_event, c_void};
use rand::{Rng, SeedableRng, rngs::SmallRng};

// The number of tasks to run concurrently.
const TASK_ITERS: i32 = 16;

const PRINT: bool = false;


#[derive(Debug, Clone)]
pub struct Kern {
    pub name: &'static str,
    pub op_add: bool,
}


#[derive(Debug, Clone)]
pub struct Vals<T: OclPrm> {
    pub type_str: &'static str,
    pub zero: T,
    pub addend: T,
    pub range: (T, T),
    pub use_source_vec: bool,
}

#[derive(Debug, Clone)]
pub struct Misc {
    pub work_size_range: (u32, u32),
    pub alloc_host_ptr: bool,
}

#[derive(Debug, Clone)]
pub struct Switches<T: OclPrm> {
    pub config_name: &'static str,
    pub kern: Kern,
    pub val: Vals<T>,
    pub misc: Misc,

    pub map_write: bool,
    pub map_read: bool,
    pub async_write: bool,
    pub async_read: bool,
    pub alloc_source_vec: bool,
    pub event_callback: bool,
    pub queue_out_of_order: bool,
    pub futures: bool,
}

lazy_static! {
    pub static ref CONFIG_MAPPED_WRITE_OOO_ASYNC: Switches<Float4> = Switches {
        config_name: "Out of Order | Async-Future ",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        val: Vals {
            type_str: "float4",
            zero: Float4::new(0., 0., 0., 0.),
            addend: Float4::new(50., 50., 50., 50.),
            range: (Float4::new(-200., -200., -200., -200.), Float4::new(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((1 << 24) - 1, 1 << 24),
            work_size_range: (1 << 12, 1 << 21),
            alloc_host_ptr: false,
        },
        map_write: true,
        map_read: false,
        async_write: true,
        async_read: true,
        alloc_source_vec: false,
        queue_out_of_order: true,
        event_callback: false,
        futures: true,
    };

    pub static ref CONFIG_MAPPED_WRITE_OOO_ASYNC_AHP: Switches<Float4> = Switches {
        config_name: "Out of Order | Async-Future | Alloc Host Ptr",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        val: Vals {
            type_str: "float4",
            zero: Float4::new(0., 0., 0., 0.),
            addend: Float4::new(50., 50., 50., 50.),
            range: (Float4::new(-200., -200., -200., -200.), Float4::new(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((1 << 24) - 1, 1 << 24),
            work_size_range: (1 << 12, 1 << 21),
            alloc_host_ptr: true,
        },
        map_write: true,
        map_read: false,
        async_write: true,
        async_read: true,
        alloc_source_vec: false,
        queue_out_of_order: true,
        event_callback: false,
        futures: true,
    };

    pub static ref CONFIG_MAPPED_READ_OOO_ASYNC_CB: Switches<Float4> = Switches {
        config_name: "In-Order | Async-Future ",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        val: Vals {
            type_str: "float4",
            zero: Float4::new(0., 0., 0., 0.),
            addend: Float4::new(50., 50., 50., 50.),
            range: (Float4::new(-200., -200., -200., -200.), Float4::new(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((1 << 24) - 1, 1 << 24),
            work_size_range: (1 << 12, 1 << 21),
            alloc_host_ptr: false,
        },
        map_write: false,
        map_read: true,
        async_write: true,
        async_read: true,
        alloc_source_vec: true,
        queue_out_of_order: true,
        event_callback: false,
        futures: true,
    };

    pub static ref CONFIG_MAPPED_WRITE_INO_ASYNC_CB: Switches<Float4> = Switches {
        config_name: "In-Order | Async-Future ",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        val: Vals {
            type_str: "float4",
            zero: Float4::new(0., 0., 0., 0.),
            addend: Float4::new(50., 50., 50., 50.),
            range: (Float4::new(-200., -200., -200., -200.), Float4::new(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((1 << 24) - 1, 1 << 24),
            work_size_range: (1 << 12, 1 << 21),
            alloc_host_ptr: false,
        },
        map_write: true,
        map_read: false,
        async_write: true,
        async_read: true,
        alloc_source_vec: false,
        queue_out_of_order: false,
        event_callback: true,
        futures: true,
    };

    pub static ref CONFIG_MAPPED_WRITE_OOO_ELOOP: Switches<Float4> = Switches {
        config_name: "Out of Order | NonBlocking",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        val: Vals {
            type_str: "float4",
            zero: Float4::new(0., 0., 0., 0.),
            addend: Float4::new(50., 50., 50., 50.),
            range: (Float4::new(-200., -200., -200., -200.), Float4::new(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((1 << 24) - 1, 1 << 24),
            work_size_range: (1 << 12, 1 << 21),
            alloc_host_ptr: false,
        },
        map_write: true,
        map_read: false,
        async_write: true,
        async_read: true,
        alloc_source_vec: false,
        queue_out_of_order: true,
        event_callback: false,
        futures: true,
    };

    pub static ref CONFIG_MAPPED_WRITE_OOO_ELOOP_CB: Switches<Float4> = Switches {
        config_name: "Out of Order | NonBlocking | Callback",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        val: Vals {
            type_str: "float4",
            zero: Float4::new(0., 0., 0., 0.),
            addend: Float4::new(50., 50., 50., 50.),
            range: (Float4::new(-200., -200., -200., -200.), Float4::new(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((1 << 24) - 1, 1 << 24),
            work_size_range: (1 << 14, 1 << 21),
            alloc_host_ptr: false,
        },
        map_write: true,
        map_read: false,
        async_write: true,
        async_read: true,
        alloc_source_vec: false,
        queue_out_of_order: true,
        event_callback: true,
        futures: true,
    };

    pub static ref CONFIG_THREADS: Switches<Int4> = Switches {
        config_name: "Out of Order | NonBlocking | Callback",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        val: Vals {
            type_str: "int4",
            zero: Int4::new(0, 0, 0, 0),
            addend: Int4::new(50, 50, 50, 50),
            range: (Int4::new(-200, -200, -200, -200), Int4::new(-200, -200, -200, -200)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((1 << 24) - 1, 1 << 24),
            work_size_range: (1 << 14, (1 << 14) + 1),
            alloc_host_ptr: false,
        },
        map_write: true,
        map_read: false,
        async_write: true,
        async_read: true,
        alloc_source_vec: false,
        queue_out_of_order: true,
        event_callback: true,
        futures: true,
    };
}

fn gen_kern_src(kernel_name: &str, type_str: &str, simple: bool, add: bool) -> String {
    let op = if add { "+" } else { "-" };

    if simple {
        format!(r#"__kernel void {kn}(
                __global {ts}* in,
                {ts} values,
                __global {ts}* out)
            {{
                uint idx = get_global_id(0);
                out[idx] = in[idx] {op} values;
            }}"#
            ,
            kn=kernel_name, op=op, ts=type_str
        )
    } else {
        format!(r#"__kernel void {kn}(
                __global {ts}* in_0,
                __global {ts}* in_1,
                __global {ts}* in_2,
                {ts} values,
                __global {ts}* out)
            {{
                uint idx = get_global_id(0);
                out[idx] = in_0[idx] {op} in_1[idx] {op} in_2[idx] {op} values;
            }}"#
            ,
            kn=kernel_name, op=op, ts=type_str
        )
    }
}


fn create_queue(device: Device, context: &Context, flags: Option<CommandQueueProperties>)
        -> OclResult<Queue> {
    Queue::new(&context, device, flags.clone()).or_else(|err| {
        // match *err.kind() {
        //     OclCoreErrorKind::Status { status: Status::CL_INVALID_VALUE, .. } => {
        //         Err("Device does not support out of order queues.".into())
        //     },
        //     _ => Err(err.into()),
        // }
        match err.api_status() {
            Some(Status::CL_INVALID_VALUE) => Err("Device does not support out of order queues.".into()),
            _ => Err(err.into()),
        }
    })
}


pub fn create_queues(device: Device, context: &Context, out_of_order: bool)
        -> OclResult<(Queue, Queue, Queue)>
{
    let ooo_flag = if out_of_order {
        CommandQueueProperties::new().out_of_order()
    } else {
        CommandQueueProperties::empty()
    };

    let flags = Some( ooo_flag | CommandQueueProperties::new().profiling());

    let write_queue = create_queue(device, context, flags.clone())?;
    let kernel_queue = create_queue(device, context, flags.clone())?;
    let read_queue = create_queue(device, context, flags.clone())?;

    Ok((write_queue, kernel_queue, read_queue))
}


fn wire_callback(wire_callback: bool, context: &Context, map_event: &Event) -> Option<Event> {
    if wire_callback {
        unsafe {
            let user_map_event = EventCore::user(context).unwrap();
            let unmap_target_ptr = user_map_event.clone().into_raw();
            map_event.set_callback(core::_complete_user_event, unmap_target_ptr).unwrap();
            Some(Event::from(user_map_event))
        }
    } else {
        None
    }
}

fn check_failure<T: OclPrm + Debug>(idx: usize, tar: T, src: T) -> OclResult<()> {
    if tar != src {
        let fail_reason = format!(colorify!(red_bold:
            "VALUE MISMATCH AT INDEX [{}]: {:?} != {:?}"),
            idx, tar, src);

        Err(fail_reason.into())
    } else {
        Ok(())
    }
}


fn print_result(operation: &str, result: OclResult<()>) {
    match result {
        Ok(_) => {
            printc!(white: "    {}  ", operation);
            printc!(white: "<");
            printc!(green_bold: "success");
            printc!(white: ">");
        },
        Err(reason) => {
            println!("    {}", reason);
            printc!(white: "    {}  ", operation);
            printc!(white: "<");
            printc!(red_bold: "failure");
            printc!(white: ">");

        }
    }

    print!("\n");
}

pub fn check(device: Device, context: &Context, rng: &mut SmallRng, cfg: Switches<Float4>)
        -> OclResult<()>
{
    let work_size_range = cfg.misc.work_size_range.0..cfg.misc.work_size_range.1;
    let work_size = rng.gen_range(work_size_range);

    // Create queues:
    let (write_queue, kernel_queue, read_queue) =
        create_queues(device, &context, cfg.queue_out_of_order)?;

    let ahp_flag = if cfg.misc.alloc_host_ptr {
        MemFlags::new().alloc_host_ptr()
    } else {
        MemFlags::empty()
    };

    // Create buffers:
    // let write_buf_flags = Some(MemFlags::read_only() | MemFlags::host_write_only() | ahp_flag);
    let write_buf_flags = ahp_flag.read_only().host_write_only();
    // let read_buf_flags = Some(MemFlags::write_only() | MemFlags::host_read_only() | ahp_flag);
    let read_buf_flags = ahp_flag.write_only().host_read_only();

    let source_buf = Buffer::<Float4>::builder()
        .queue(write_queue.clone())
        .flags(write_buf_flags)
        .len(work_size)
        .build()?;

    let target_buf = Buffer::<Float4>::builder()
        .queue(read_queue.clone())
        .flags(read_buf_flags)
        .len(work_size)
        .build()?;

    // Generate kernel source:
    let kern_src = gen_kern_src(cfg.kern.name, cfg.val.type_str, true, cfg.kern.op_add);
    // println!("{}\n", kern_src);

    let program = Program::builder()
        .devices(device)
        .src(kern_src)
        .build(context)?;

    let kern = Kernel::builder()
        .program(&program)
        .name(cfg.kern.name)
        .queue(kernel_queue)
        .global_work_size(work_size)
        .arg(&source_buf)
        .arg(cfg.val.addend)
        .arg(&target_buf)
        .build()?;


    let source_vec = if cfg.alloc_source_vec {
        // let source_vec = util::scrambled_vec(rand_val_range, work_size);
        vec![cfg.val.range.0; work_size as usize]
    } else {
        Vec::with_capacity(0)
    };

    // Extra wait list for certain scenarios:
    let wait_events = EventList::with_capacity(8);

    //#########################################################################
    //############################## WRITE ####################################
    //#########################################################################
    // Create write event then enqueue write:
    let mut write_event = Event::empty();

    if cfg.map_write {
        //###################### cfg.MAP_WRITE ############################

        let mut mapped_mem = if cfg.futures {
            let future_mem = unsafe {
                source_buf.cmd().map()
                    .flags(MapFlags::new().write_invalidate_region())
                    // .flags(MapFlags::write())
                    .ewait(&wait_events)
                    // .enew(&mut map_event)
                    .enq_async()?
            };

            // if let Some(tar_ev) = wire_callback(cfg.event_callback, context, &mut map_event) {
            //     map_event = tar_ev;
            // }

            // // Print map event status:
            // printlnc!(dark_grey: "    Map Event Status (PRE-wait) : {:?} => {:?}",
            //     map_event, core::event_status(&map_event)?);

            /////// TODO: ADD THIS AS AN OPTION?:
            // // Wait for queue completion:
            // source_buf.default_queue().flush();
            // source_buf.default_queue().finish().unwrap();

            // Wait for event completion:
            future_mem.wait()?
        } else {
            let mut map_event = Event::empty();

            let new_mm = unsafe {
                let mm_core = core::enqueue_map_buffer::<Float4, _, _, _>(
                    &write_queue,
                    source_buf.as_core(),
                    !cfg.async_write,
                    MapFlags::new().write_invalidate_region(),
                    // MapFlags::write(),
                    0,
                    source_buf.len(),
                    Some(&wait_events),
                    Some(&mut map_event),
                )?;

                MemMap::new(mm_core, source_buf.len(), None, None, source_buf.as_core().clone(),
                    write_queue.clone(), /*source_buf.is_mapped()
                        .expect("Buffer unable to be mapped").clone()*/)
            };

            if let Some(tar_ev) = wire_callback(cfg.event_callback, context, &mut map_event) {
                map_event = tar_ev;
            }

            // ///////// Print pre-wait map event status:
            // printlnc!(dark_grey: "    Map Event Status (PRE-wait) : {:?} => {:?}",
            //     map_event, core::event_status(&map_event)?);

            // ///////// NO EFFECT:
            // wait_events.clear()?;
            // wait_events.push(map_event);
            // map_event = Event::empty();
            // core::enqueue_marker_with_wait_list(source_buf.default_queue(),
            //     Some(&wait_events), Some(&mut map_event),
            //     Some(&source_buf.default_queue().device_version()))?;

            /////// TODO: ADD THIS AS AN OPTION:
            // // Wait for queue completion:
            // source_buf.default_queue().flush();
            // source_buf.default_queue().finish().unwrap();

            // Wait for event completion:
            // while !map_event.is_complete()? {}
            map_event.wait_for()?;

            new_mm
        };

        // ///////// Print post-wait map event status:
        // printlnc!(dark_grey: "    Map Event Status (POST-wait): {:?} => {:?}",
        //     map_event, core::event_status(&map_event)?);

        if cfg.alloc_source_vec && cfg.val.use_source_vec {
            //############### cfg.USE_SOURCE_VEC ######################
            for (map_val, vec_val) in mapped_mem.iter_mut().zip(source_vec.iter()) {
                *map_val = *vec_val;
            }
        } else {
            //############## !(cfg.USE_SOURCE_VEC) ####################
            for val in mapped_mem.iter_mut() {
                *val = cfg.val.range.0;
            }

            // ////////// Early verify:
            // for (idx, val) in mapped_mem.iter().enumerate() {
            //     if *val != cfg.val.range.0 {
            //         return Err(format!("Early map write verification failed at index: {}.", idx)
            //             .into());
            //     }
            // }
            // //////////
        }

        mapped_mem.enqueue_unmap(None, None::<&Event>, Some(&mut write_event))?;

    } else {
        //#################### !(cfg.MAP_WRITE) ###########################
        // Ensure the source vec has been allocated:
        assert!(cfg.alloc_source_vec);

        source_buf.write(&source_vec)
            .enew(&mut write_event)
            .enq()?;
    }

    //#########################################################################
    //#################### INSERT WRITE EVENT CALLBACK ########################
    //#########################################################################
    if let Some(tar_event) = wire_callback(cfg.event_callback, context, &mut write_event) {
        write_event = tar_event;
    }

    //#########################################################################
    //############################## KERNEL ###################################
    //#########################################################################
    // Create kernel event then enqueue kernel:
    let mut kern_event = Event::empty();

    unsafe {
        kern.cmd()
            .ewait(&write_event)
            .enew(&mut kern_event)
            .enq()?;
    }

    //#########################################################################
    //################### INSERT KERNEL EVENT CALLBACK ########################
    //#########################################################################
    if let Some(tar_event) = wire_callback(cfg.event_callback, context, &mut kern_event) {
        kern_event = tar_event;
    }

    //#########################################################################
    //############################### READ ####################################
    //#########################################################################

    // Create read event then enqueue read:
    let mut read_event = Event::empty();

    let mut target_vec = None;
    let mut target_map = None;

    if cfg.map_read {
        //###################### cfg.MAP_READ #############################
        unsafe {
            let mm_core = core::enqueue_map_buffer::<Float4, _, _, _>(
                &read_queue,
                target_buf.as_core(),
                false,
                MapFlags::new().read(),
                0,
                target_buf.len(),
                Some(&kern_event),
                Some(&mut read_event),
            )?;

            target_map = Some(MemMap::new(mm_core, source_buf.len(), None, None,
                source_buf.as_core().clone(), read_queue.clone(), /*target_buf.is_mapped()
                    .expect("Buffer unable to be mapped").clone()*/));
        }
    } else {
        //##################### !(cfg.MAP_READ) ###########################
        let mut tvec = vec![cfg.val.zero; work_size as usize];

        unsafe { target_buf.cmd().read(&mut tvec)
            .ewait(&kern_event)
            .enew(&mut read_event)
            .block(true)
            .enq()?; }

        target_vec = Some(tvec);
    };

    //#########################################################################
    //#################### INSERT READ EVENT CALLBACK #########################
    //#########################################################################
    if let Some(tar_event) = wire_callback(cfg.event_callback, context, &mut read_event) {
        read_event = tar_event;
    }

    //#########################################################################
    //########################## VERIFY RESULTS ###############################
    //#########################################################################
    // Wait for completion:
    read_event.wait_for()?;

    if cfg.alloc_source_vec && cfg.val.use_source_vec {
        if cfg.map_read {
            for (idx, (&tar, &src)) in target_map.unwrap().iter().zip(source_vec.iter()).enumerate() {
                check_failure(idx, tar, src + cfg.val.addend)?;
            }
        } else {
            for (idx, (&tar, &src)) in target_vec.unwrap().iter().zip(source_vec.iter()).enumerate() {
                check_failure(idx, tar, src + cfg.val.addend)?;
            }
        }
    } else {
        if cfg.map_read {
            for (idx, &tar) in target_map.unwrap().iter().enumerate() {
                check_failure(idx, tar, cfg.val.range.0 + cfg.val.addend)?;
            }
        } else {
            for (idx, &tar) in target_vec.unwrap().iter().enumerate() {
                check_failure(idx, tar, cfg.val.range.0 + cfg.val.addend)?;
            }
        }
    }

    Ok(())
}

pub fn fill_junk(
        src_buf: &Buffer<Int4>,
        common_queue: &Queue,
        kernel_event: Option<&Event>,
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
    let wait_list = [kernel_event].into_raw_array();

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

pub fn vec_write_async(
        src_buf: &Buffer<Int4>,
        rw_vec: &RwVec<Int4>,
        common_queue: &Queue,
        write_release_queue: &Queue,
        fill_event: Option<&Event>,
        write_event: &mut Option<Event>,
        write_val: i32, task_iter: i32)
        -> Box<dyn Future<Item=i32, Error=OclError> + Send>
{
    extern "C" fn _write_complete(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Write init complete  \t(iter: {})", task_iter as usize); }
    }

    let mut future_guard = rw_vec.clone().write();
    // let wait_list = [fill_event].into_raw_array();
    future_guard.set_lock_wait_events(fill_event);
    let release_event = future_guard.create_release_event(write_release_queue).unwrap().clone();

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

    *write_event = Some(future_write_buffer.create_release_event(write_release_queue)
        .unwrap().clone());


    unsafe { write_event.as_ref().unwrap().set_callback(_write_complete,
        task_iter as *mut c_void).unwrap(); }

    let future_drop_guard = future_write_buffer.and_then(move |_| Ok(()));

    Box::new(future_write_vec.join(future_drop_guard).map(move |(_, _)| task_iter))
}

pub fn kernel_add(
        kern: &Kernel,
        common_queue: &Queue,
        verify_add_event: Option<&Event>,
        write_init_event: Option<&Event>,
        kernel_event: &mut Option<Event>,
        task_iter: i32)
{
    extern "C" fn _print_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Kernel starting      \t(iter: {}) ...", task_iter as usize); }
    }
    extern "C" fn _print_complete(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Kernel complete      \t(iter: {})", task_iter as usize); }
    }

    let wait_list = [&verify_add_event, &write_init_event].into_raw_array();
    let kernel_wait_marker = wait_list.to_marker(&common_queue).unwrap();

    unsafe { kernel_wait_marker.as_ref().unwrap()
        .set_callback(_print_starting, task_iter as *mut c_void).unwrap(); }

    *kernel_event = Some(Event::empty());

    unsafe {
        kern.cmd()
            .queue(common_queue)
            .ewait(&wait_list)
            .enew(kernel_event.as_mut())
            .enq().unwrap();
    }

    unsafe { kernel_event.as_ref().unwrap().set_callback(_print_complete,
        task_iter as *mut c_void).unwrap(); }
}

pub fn map_read_async(dst_buf: &Buffer<Int4>, common_queue: &Queue,
        verify_add_unmap_queue: Queue, wait_event: Option<&Event>,
        verify_add_event: &mut Option<Event>, correct_val: i32,
        task_iter: i32) -> Box<dyn Future<Item=i32, Error=OclError> + Send>
{
    extern "C" fn _verify_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        printlnc!(lime_bold: "* Verify add starting \t\t(iter: {}) ...",
            task_iter as usize);
    }

    unsafe { wait_event.as_ref().unwrap()
        .set_callback(_verify_starting, task_iter as *mut c_void).unwrap(); }

    let mut future_read_data = unsafe {
        dst_buf.cmd().map()
            .queue(common_queue)
            .flags(MapFlags::new().read())
            .ewait(wait_event)
            .enq_async().unwrap()
    };

    *verify_add_event = Some(future_read_data.create_unmap_event().unwrap().clone());

    Box::new(future_read_data.and_then(move |mut data| {
        let mut val_count = 0;

        for (idx, val) in data.iter().enumerate() {
            let cval = Int4::splat(correct_val);
            if *val != cval {
                return Err(format!("Verify add: Result value mismatch: {:?} != {:?} @ [{}]",
                    val, cval, idx).into());
            }
            val_count += 1;
        }

        printlnc!(lime_bold: "* Verify add complete \t\t(iter: {})",
            task_iter);

        data.unmap().queue(&verify_add_unmap_queue).enq()?;

        Ok(val_count)
    }))
}

pub fn vec_read_async(dst_buf: &Buffer<Int4>, rw_vec: &RwVec<Int4>, common_queue: &Queue,
        verify_add_release_queue: &Queue, kernel_event: Option<&Event>,
        verify_add_event: &mut Option<Event>, correct_val: i32, task_iter: i32)
        -> Box<dyn Future<Item=i32, Error=OclError> + Send>
{
    extern "C" fn _verify_starting(_: cl_event, _: i32, task_iter : *mut c_void) {
        if PRINT { println!("* Verify add starting  \t(iter: {}) ...", task_iter as usize); }
    }

    let mut future_read_data = dst_buf.cmd().read(rw_vec)
        .queue(common_queue)
        .ewait(kernel_event)
        .enq_async().unwrap();

    // Attach a status message printing callback to what approximates the
    // verify_init start-time event:
    unsafe { future_read_data.lock_event().unwrap().set_callback(
        _verify_starting, task_iter as *mut c_void).unwrap(); }

    // Create an empty event ready to hold the new verify_init event, overwriting any old one.
    *verify_add_event = Some(future_read_data.create_release_event(verify_add_release_queue)
        .unwrap().clone());

    Box::new(future_read_data.and_then(move |data| {
        let mut val_count = 0;

        for (idx, val) in data.iter().enumerate() {
            let cval = Int4::splat(correct_val);
            if *val != cval {
                return Err(format!("Result value @ idx[{}]: {:?} \n    should be == {:?}\
                    (task iter: [{}]).", idx, val, cval, task_iter).into());
            }
            val_count += 1;
        }

        if PRINT { println!("* Verify add complete  \t(iter: {})", task_iter); }

        Ok(val_count)
    }))
}

pub fn check_async(device: Device, context: &Context, rng: &mut SmallRng, cfg: Switches<Int4>)
        -> OclResult<()>
{
    use std::thread;

    let work_size_range = cfg.misc.work_size_range.0..cfg.misc.work_size_range.1;
    let work_size = rng.gen_range(work_size_range);

    // // Create queues:
    // let queue_flags = Some(CommandQueueProperties::new().out_of_order());
    // let common_queue = Queue::new(&context, device, queue_flags).or_else(|_|
    //     Queue::new(&context, device, None)).unwrap();
    // let write_queue = Queue::new(&context, device, queue_flags).or_else(|_|
    //     Queue::new(&context, device, None)).unwrap();
    // let read_queue = Queue::new(&context, device, queue_flags).or_else(|_|
    //     Queue::new(&context, device, None)).unwrap();

    // Create queues:
    let queue_flags = Some(CommandQueueProperties::new().out_of_order());
    let common_queue = create_queue(device, context, queue_flags)?;
    let write_queue = create_queue(device, context, queue_flags)?;
    let read_queue = create_queue(device, context, queue_flags)?;

    let ahp_flag = if cfg.misc.alloc_host_ptr {
        MemFlags::new().alloc_host_ptr()
    } else {
        MemFlags::empty()
    };

    // Create buffers:
    // let write_buf_flags = Some(MemFlags::read_only() | MemFlags::host_write_only() | ahp_flag);
    let write_buf_flags = ahp_flag.read_only().host_write_only();
    // let read_buf_flags = Some(MemFlags::write_only() | MemFlags::host_read_only() | ahp_flag);
    let read_buf_flags = ahp_flag.write_only().host_read_only();

    let src_buf = Buffer::<Int4>::builder()
        .queue(write_queue.clone())
        .flags(write_buf_flags)
        .len(work_size)
        .build()?;

    let tar_buf = Buffer::<Int4>::builder()
        .queue(read_queue.clone())
        .flags(read_buf_flags)
        .len(work_size)
        .build()?;

    // Generate kernel source:
    let kern_src = gen_kern_src(cfg.kern.name, cfg.val.type_str, true, cfg.kern.op_add);

    let program = Program::builder()
        .devices(device)
        .src(kern_src)
        .build(context)?;

    let kern = Kernel::builder()
        .program(&program)
        .name(cfg.kern.name)
        .queue(common_queue.clone())
        .global_work_size(work_size)
        .arg(&src_buf)
        .arg(&cfg.val.addend)
        .arg(&tar_buf)
        .build()?;


    // A lockable vector for reads and writes:
    let rw_vec: RwVec<Int4> = RwVec::from(vec![Default::default(); work_size as usize]);

    // Our events for synchronization.
    let mut fill_event = None;
    let mut write_event = None;
    let mut kernel_event = None;
    let mut read_event = None;

    if PRINT { println!("Starting cycles ..."); }

    let mut threads = Vec::with_capacity(TASK_ITERS as usize);

    // Our main loop. Could run indefinitely if we had a stream of input.
    for task_iter in 0..TASK_ITERS {
        let ival = cfg.val.zero[0] + task_iter;
        let tval = ival + cfg.val.addend[0];

        fill_junk(
            &src_buf,
            &common_queue,
            kernel_event.as_ref(),
            &mut fill_event,
            task_iter);

        let write = vec_write_async(
            &src_buf,
            &rw_vec,
            &common_queue,
            &write_queue,
            fill_event.as_ref(),
            &mut write_event,
            ival,
            task_iter);

        kernel_add(
            &kern,
            &common_queue,
            read_event.as_ref(),
            write_event.as_ref(),
            &mut kernel_event,
            task_iter);

        ////// KEEP:
        // let read = map_read_async(
        //     &tar_buf,
        //     &common_queue,
        //     read_queue.clone(),
        //     kernel_event.as_ref(),
        //     &mut read_event,
        //     tval,
        //     task_iter);

        let read = vec_read_async(
            &tar_buf,
            &rw_vec,
            &common_queue,
            &read_queue,
            kernel_event.as_ref(),
            &mut read_event,
            tval,
            task_iter);

        if PRINT { println!("All commands for iteration {} enqueued", task_iter); }

        let task = write.join(read);

        threads.push(thread::Builder::new()
                .name(format!("task_iter_[{}]", task_iter).into())
                .spawn(move ||
        {
            if PRINT { println!("Waiting on task iter [{}]...", task_iter); }
            match task.wait() {
                Ok(res) => {
                    if PRINT { println!("Task iter [{}] complete with result: {:?}", task_iter, res); }
                    Ok(res)
                },
                Err(err) => {
                    Err(format!("[{}] ERROR: {:?}", task_iter, err))
                },
            }
        }).unwrap());
    }

    let mut all_correct = true;

    for thread in threads {
        match thread.join() {
            Ok(res) => {
                match res {
                    Ok(res) => if PRINT { println!("Thread result: {:?}", res) },
                    Err(err) => {
                        println!("{}", err);
                        all_correct = false;
                    },
                }
            },
            Err(err) => panic!("{:?}", err),
        }
    }

    if all_correct {
        Ok(())
    } else {
        Err("\nErrors found. Your device/platform does not properly support asynchronous\n\
            multi-threaded operation. It is recommended that you enable the `async_block`\n\
            feature when compiling this library for use with the device and platform combination \n\
            listed just above (https://doc.rust-lang.org/book/conditional-compilation.html).\n".into())
    }
}



pub fn device_check() -> OclResult<()> {
    let mut rng = SmallRng::from_entropy();

    for (p_idx, platform) in Platform::list().into_iter().enumerate() {
    // for &platform in &[Platform::default()] {
        let devices = Device::list_all(&platform).unwrap();

        for (d_idx, device) in devices.into_iter().enumerate() {
            printlnc!(blue: "Platform [{}]: {}", p_idx, platform.name()?);
            printlnc!(teal: "Device [{}]: {} {}", d_idx, device.vendor()?, device.name()?);

            if device.is_available().unwrap() {

                let context = Context::builder()
                    .platform(platform)
                    .devices(device)
                    .build().unwrap();

                let result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_OOO_ASYNC.clone());
                print_result("Out-of-order MW/Async-CB:     ", result);

                let result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_OOO_ASYNC_AHP.clone());
                print_result("Out-of-order MW/Async-CB+AHP: ", result);

                let result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_READ_OOO_ASYNC_CB.clone());
                print_result("Out-of-order MW/ASync+CB/MR:  ", result);

                let result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_INO_ASYNC_CB.clone());
                print_result("In-order MW/ASync+CB:         ", result);

                let result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_OOO_ELOOP.clone());
                print_result("Out-of-order MW/ELOOP:        ", result);

                let result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_OOO_ELOOP_CB.clone());
                print_result("Out-of-order MW/ELOOP+CB:     ", result);

                let result = check_async(device, &context, &mut rng,
                    CONFIG_THREADS.clone());
                print_result("In-order RwVec Multi-thread:  ", result);

            } else {
                printlnc!(red: "    [UNAVAILABLE]");
            }
        }
    }

    printlnc!(light_grey: "All checks complete.");
    Ok(())
}


pub fn main() {
    match device_check() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}