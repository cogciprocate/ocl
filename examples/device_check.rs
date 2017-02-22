//! Checks all platforms and devices for driver bugs.
//!
//! Originally designed to isolate a severe glitch on a particular device and
//! grew into a general purpose device stress tester.
//!
//! Far from complete but will check for a few possible problems.
//!
//! - Buffer without `CL_ALLOC_HOST_PTR` [FIXME: list flags]
//! - [FIXME]: Finish this
//!
//!
//! [FIXME]: List future plans here
//!
//!
//!
//!

extern crate libc;
extern crate futures;
extern crate rand;
extern crate chrono;
extern crate ocl;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate colorify;

use std::fmt::{Debug};
use futures::{Future};
use rand::{XorShiftRng};
use rand::distributions::{IndependentSample, Range as RandRange};
use ocl::{core, Platform, Device, Context, Queue, Program, Buffer, Kernel, OclPrm,
    Event, EventList, MemMap};
use ocl::async::Result as AsyncResult;
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::aliases::ClFloat4;
use ocl::core::Event as EventCore;


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


pub fn create_queues(device: Device, context: &Context, out_of_order: bool)
        -> (Queue, Queue, Queue)
{
    let ooo_flag = if out_of_order {
        CommandQueueProperties::new().out_of_order()
    } else {
        CommandQueueProperties::empty()
    };

    let flags = Some( ooo_flag | CommandQueueProperties::new().profiling());

    let write_queue = Queue::new(&context, device, flags.clone()).unwrap();
    let kernel_queue = Queue::new(&context, device, flags.clone()).unwrap();
    let read_queue = Queue::new(&context, device, flags).unwrap();

    (write_queue, kernel_queue, read_queue)
}


fn wire_callback(wire_callback: bool, context: &Context, map_event: &Event) -> Option<Event> {
    if wire_callback {
        unsafe {
            let user_map_event = EventCore::user(context).unwrap();
            let unmap_target_ptr = user_map_event.clone().into_raw();
            map_event.set_callback(Some(core::_complete_user_event), unmap_target_ptr).unwrap();
            Some(Event::from(user_map_event))
        }
    } else {
        None
    }
}


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
    pub vals: Vals<T>,
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
    // pub static ref CONFIG_READ_WRITE_F32: Switches<f32> = Switches {
    //     config_name: "| Write | Read | f32 |",
    //     kernel_name: "add_values",
    //     type_str: "float",
    //     op_add: true,
    //     zero_val: 0.0f32,
    //     addend: 100.0f32,
    //     val_range: (-2000.0, 2000.0f32),
    //     queue_ordering: flags::QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
    //     work_size_range: (2 << 19, 2 << 22),
    //     async_write: false,
    //     async_read: false,
    //     map_write: false,
    //     map_read: false,
    //     event_callback: false,
    //     use_source_vec: true,
    // };

    // pub static ref CONFIG_MAPPED_WRITE_CLFLOAT4_CALLBACK: Switches<ClFloat4> = Switches {
    //     config_name: "Buffer | Mapped Write | Read | ClFloat4 | Callback",
    //     kernel_name: "add_values",
    //     type_str: "float4",
    //     op_add: true,
    //     zero_val: ClFloat4(0., 0., 0., 0.),
    //     addend: ClFloat4(50., 50., 50., 50.),
    //     val_range: (ClFloat4(-200., -200., -200., -200.), ClFloat4(-200., -200., -200., -200.)),
    //     queue_ordering: CommandQueueProperties::empty(),
    //     work_size_range: (2 << 19, 2 << 22),
    //     async_write: true,
    //     async_read: true,
    //     map_write: true,
    //     map_read: false,
    //     event_callback: true,
    //     use_source_vec: true,
    // };

    pub static ref CONFIG_MAPPED_WRITE_OOO_ASYNC: Switches<ClFloat4> = Switches {
        config_name: "Out of Order | Async-Future ",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        vals: Vals {
            type_str: "float4",
            zero: ClFloat4(0., 0., 0., 0.),
            addend: ClFloat4(50., 50., 50., 50.),
            range: (ClFloat4(-200., -200., -200., -200.), ClFloat4(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((2 << 24) - 1, 2 << 24),
            work_size_range: (2 << 21, 2 << 22),
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

    pub static ref CONFIG_MAPPED_WRITE_OOO_ASYNC_AHP: Switches<ClFloat4> = Switches {
        config_name: "Out of Order | Async-Future | Alloc Host Ptr",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        vals: Vals {
            type_str: "float4",
            zero: ClFloat4(0., 0., 0., 0.),
            addend: ClFloat4(50., 50., 50., 50.),
            range: (ClFloat4(-200., -200., -200., -200.), ClFloat4(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((2 << 24) - 1, 2 << 24),
            work_size_range: (2 << 21, 2 << 22),
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

    pub static ref CONFIG_MAPPED_READ_OOO_ASYNC_CB: Switches<ClFloat4> = Switches {
        config_name: "In-Order | Async-Future ",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        vals: Vals {
            type_str: "float4",
            zero: ClFloat4(0., 0., 0., 0.),
            addend: ClFloat4(50., 50., 50., 50.),
            range: (ClFloat4(-200., -200., -200., -200.), ClFloat4(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((2 << 24) - 1, 2 << 24),
            work_size_range: (2 << 21, 2 << 22),
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

    pub static ref CONFIG_MAPPED_WRITE_INO_ASYNC_CB: Switches<ClFloat4> = Switches {
        config_name: "In-Order | Async-Future ",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        vals: Vals {
            type_str: "float4",
            zero: ClFloat4(0., 0., 0., 0.),
            addend: ClFloat4(50., 50., 50., 50.),
            range: (ClFloat4(-200., -200., -200., -200.), ClFloat4(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((2 << 24) - 1, 2 << 24),
            work_size_range: (2 << 21, 2 << 22),
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

    pub static ref CONFIG_MAPPED_WRITE_OOO_ELOOP: Switches<ClFloat4> = Switches {
        config_name: "Out of Order | NonBlocking",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        vals: Vals {
            type_str: "float4",
            zero: ClFloat4(0., 0., 0., 0.),
            addend: ClFloat4(50., 50., 50., 50.),
            range: (ClFloat4(-200., -200., -200., -200.), ClFloat4(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((2 << 24) - 1, 2 << 24),
            work_size_range: (2 << 21, 2 << 22),
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

    pub static ref CONFIG_MAPPED_WRITE_OOO_ELOOP_CB: Switches<ClFloat4> = Switches {
        config_name: "Out of Order | NonBlocking | Callback",
        kern: Kern {
            name: "add_values",
            op_add: true,
        },
        vals: Vals {
            type_str: "float4",
            zero: ClFloat4(0., 0., 0., 0.),
            addend: ClFloat4(50., 50., 50., 50.),
            range: (ClFloat4(-200., -200., -200., -200.), ClFloat4(-200., -200., -200., -200.)),
            use_source_vec: false,
        },
        misc: Misc {
            // work_size_range: ((2 << 24) - 1, 2 << 24),
            work_size_range: (2 << 21, 2 << 22),
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




pub fn check(device: Device, context: &Context, rng: &mut XorShiftRng, cfg: Switches<ClFloat4>)
        -> AsyncResult<()>
{
    let work_size_range = RandRange::new(cfg.misc.work_size_range.0, cfg.misc.work_size_range.1);
    let work_size = work_size_range.ind_sample(rng);

    // Create queues:
    let (write_queue, kernel_queue, read_queue) = create_queues(device, &context, cfg.queue_out_of_order);

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

    let source_buf = Buffer::<ClFloat4>::builder()
        .queue(write_queue.clone())
        .flags(write_buf_flags)
        .dims(work_size)
        .build()?;

    let target_buf = Buffer::<ClFloat4>::builder()
        .queue(read_queue.clone())
        .flags(read_buf_flags)
        .dims(work_size)
        .build()?;

    // Generate kernel source:
    let kern_src = gen_kern_src(cfg.kern.name, cfg.vals.type_str, true, cfg.kern.op_add);
    // println!("{}\n", kern_src);

    let program = Program::builder()
        .devices(device)
        .src(kern_src)
        .build(context)?;

    let kern = Kernel::new(cfg.kern.name, &program)?
        .queue(kernel_queue)
        .gws(work_size)
        .arg_buf(&source_buf)
        .arg_scl(cfg.vals.addend)
        .arg_buf(&target_buf);


    let source_vec = if cfg.alloc_source_vec {
        // let source_vec = util::scrambled_vec(rand_val_range, work_size);
        vec![cfg.vals.range.0; work_size as usize]
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
            let future_mem = source_buf.cmd().map()
                .flags(MapFlags::new().write_invalidate_region())
                // .flags(MapFlags::write())
                .ewait(&wait_events)
                // .enew(&mut map_event)
                .enq_async()?;

            // if let Some(tar_ev) = wire_callback(cfg.event_callback, context, &mut map_event) {
            //     map_event = tar_ev;
            // }

            // // Print map event status:
            // printlnc!(dark_grey: "    Map Event Status (PRE-wait) : {:?} => {:?}",
            //     map_event, core::event_status(&map_event)?);

            /////// [TODO]: ADD THIS AS AN OPTION?:
            // // Wait for queue completion:
            // source_buf.default_queue().flush();
            // source_buf.default_queue().finish().unwrap();

            // Wait for event completion:
            future_mem.wait()?
        } else {
            let mut map_event = Event::empty();

            let new_mm = unsafe {
                let mm_core = core::enqueue_map_buffer::<ClFloat4, _, _, _>(
                    &write_queue,
                    source_buf.core(),
                    !cfg.async_write,
                    MapFlags::new().write_invalidate_region(),
                    // MapFlags::write(),
                    0,
                    source_buf.len(),
                    Some(&wait_events),
                    Some(&mut map_event),
                )?;

                MemMap::new(mm_core, source_buf.len(), None, source_buf.core().clone(),
                    write_queue.core().clone())
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

            /////// [TODO]: ADD THIS AS AN OPTION:
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

        if cfg.alloc_source_vec && cfg.vals.use_source_vec {
            //############### cfg.USE_SOURCE_VEC ######################
            for (map_val, vec_val) in mapped_mem.iter_mut().zip(source_vec.iter()) {
                *map_val = *vec_val;
            }
        } else {
            //############## !(cfg.USE_SOURCE_VEC) ####################
            for val in mapped_mem.iter_mut() {
                *val = cfg.vals.range.0;
            }

            // ////////// Early verify:
            // for (idx, val) in mapped_mem.iter().enumerate() {
            //     if *val != cfg.vals.range.0 {
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

    kern.cmd()
        .ewait(&write_event)
        .enew(&mut kern_event)
        .enq()?;

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
            let mm_core = core::enqueue_map_buffer::<ClFloat4, _, _, _>(
                &read_queue,
                target_buf.core(),
                false,
                MapFlags::new().read(),
                0,
                target_buf.len(),
                Some(&kern_event),
                Some(&mut read_event),
            )?;

            target_map = Some(MemMap::new(mm_core, source_buf.len(), None,
                source_buf.core().clone(), read_queue.core().clone()));
        }
    } else {
        //##################### !(cfg.MAP_READ) ###########################
        let mut tvec = vec![cfg.vals.zero; work_size as usize];

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

    if cfg.alloc_source_vec && cfg.vals.use_source_vec {
        if cfg.map_read {
            for (idx, (&tar, &src)) in target_map.unwrap().iter().zip(source_vec.iter()).enumerate() {
                check_failure(idx, tar, src + cfg.vals.addend)?;
            }
        } else {
            for (idx, (&tar, &src)) in target_vec.unwrap().iter().zip(source_vec.iter()).enumerate() {
                check_failure(idx, tar, src + cfg.vals.addend)?;
            }
        }
    } else {
        if cfg.map_read {
            for (idx, &tar) in target_map.unwrap().iter().enumerate() {
                check_failure(idx, tar, cfg.vals.range.0 + cfg.vals.addend)?;
            }
        } else {
            for (idx, &tar) in target_vec.unwrap().iter().enumerate() {
                check_failure(idx, tar, cfg.vals.range.0 + cfg.vals.addend)?;
            }
        }
    }

    Ok(())
}


fn check_failure<T: OclPrm + Debug>(idx: usize, tar: T, src: T) -> AsyncResult<()> {
    if tar != src {
        let fail_reason = format!(colorify!(red_bold:
            "VALUE MISMATCH AT INDEX [{}]: {:?} != {:?}"),
            idx, tar, src);

        Err(fail_reason.into())
    } else {
        Ok(())
    }
}


fn print_result(operation: &str, result: AsyncResult<()>) {
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


pub fn main() {
    let mut rng = rand::weak_rng();

    for platform in Platform::list() {
    // for &platform in &[Platform::default()] {

        let devices = Device::list_all(&platform).unwrap();

        for device in devices {
            printlnc!(blue: "Platform: {}", platform.name());
            printlnc!(teal: "Device: {} {}", device.vendor(), device.name());

            if device.is_available().unwrap() {

                let context = Context::builder()
                    .platform(platform)
                    .devices(device)
                    .build().unwrap();

                // // Check current device using in-order queues:
                // let in_order_result = check(device, &context, &mut rng,
                //     CONFIG_MAPPED_WRITE_CLFLOAT4.clone());
                // print_result("In-order:             ", in_order_result);

                // // Check current device using out-of-order queues:
                // let out_of_order_result = check(device, &context, &mut rng,
                //     CONFIG_MAPPED_WRITE_CLFLOAT4_OOO.clone());
                // print_result("Out-of-order:         ", out_of_order_result);

                let out_of_order_result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_OOO_ASYNC.clone());
                print_result("Out-of-order MW/Async-CB:     ", out_of_order_result);

                let out_of_order_result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_OOO_ASYNC_AHP.clone());
                print_result("Out-of-order MW/Async-CB+AHP: ", out_of_order_result);

                let in_order_result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_READ_OOO_ASYNC_CB.clone());
                print_result("Out-of-order MW/ASync+CB/MR:  ", in_order_result);

                let in_order_result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_INO_ASYNC_CB.clone());
                print_result("In-order MW/ASync+CB:         ", in_order_result);

                let in_order_result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_OOO_ELOOP.clone());
                print_result("Out-of-order MW/ELOOP:        ", in_order_result);

                let in_order_result = check(device, &context, &mut rng,
                    CONFIG_MAPPED_WRITE_OOO_ELOOP_CB.clone());
                print_result("Out-of-order MW/ELOOP+CB:     ", in_order_result);

            } else {
                printlnc!(red: "    [UNAVAILABLE]");
            }
        }
    }

    printlnc!(light_grey: "All checks complete.");
}
