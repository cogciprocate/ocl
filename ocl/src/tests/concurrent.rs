//! Checks to make sure that the platform can be used concurrently.
//!
//! This generally fails on NVidia hardware.

#![allow(unused_imports, unused_variables, dead_code, unused_mut)]

use std::thread::{self, JoinHandle};
use std::sync::mpsc;
use std::time::Duration;
use crate::tests::rand::{self, Rng};
use crate::core::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ProgramInfo,
    ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo, EventInfo, ProfilingInfo};
use crate::standard::{Platform, Device, Context, Queue, Buffer, Program, Kernel, EventList};
use crate::error::Result as OclResult;

static SRC: &'static str = r#"
    __kernel void add(__global float* buffer, float addend) {
        buffer[get_global_id(0)] += addend;
    }
"#;

const THREAD_COUNT: u32 = 2;

#[test]
fn concurrent() {
    let mut rng = rand::weak_rng();
    let data_set_size = 1 << 10;
    let dims = [data_set_size];
    let mut threads = Vec::with_capacity(THREAD_COUNT as usize);

    println!("Listing platforms {} times...", THREAD_COUNT);

    for i in 0..THREAD_COUNT {
        let thread_name = format!("[thread_{}]", i);

        let th = thread::Builder::new().name(thread_name.clone()).spawn(move || {
            let platforms = Platform::list();
        }).expect(&format!("Error creating {}", &thread_name));

        threads.push(th);
    }


    for th in threads.into_iter() {
        let th_name = String::from(th.thread().name().unwrap_or(""));
        if let Err(e) = th.join() { panic!("Error joining thread: '{:?}'", th_name); }
    }

    println!("Donesky.");
}

// UNUSED
fn main_from_example() -> OclResult<()> {
    let mut rng = rand::weak_rng();
    let data_set_size = 1 << 10;
    let dims = [data_set_size];
    let mut threads = Vec::new();

    let platforms = Platform::list();

    println!("Looping through avaliable platforms ({}):", platforms.len());

    // Loop through each avaliable platform:
    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];
        println!("Platform[{}]: {} ({})", p_idx, platform.name()?, platform.vendor()?);

        let devices = Device::list_all(platform).unwrap();

        // Loop through each device:
        for device_idx in 0..devices.len() {
            // Choose a device at random:
            // let dev_idx = rng.gen_range(0, devices.len());

            let device = devices[device_idx];
            println!("Device[{}]: {} ({})", device_idx, device.name()?, device.vendor()?);

            // Make a context to share around:
            let context = Context::builder().build().unwrap();
            let program = Program::builder().src(SRC).devices(device)
                .build(&context).unwrap();

            // Make a few different queues for the hell of it:
            // let queueball = vec![Queue::new_by_device_index(&context, None),
            //  Queue::new_by_device_index(&context, None),
            //  Queue::new_by_device_index(&context, None)];

            // Make a few different queues for the hell of it:
            let queueball = vec![Queue::new(&context, device, None).unwrap(),
                Queue::new(&context, device, None).unwrap(),
                Queue::new(&context, device, None).unwrap()];

            print!("    Spawning threads... ");

            for i in 0..5 {
                let thread_name = format!("{}:[D{}.I{}]", threads.len(), device_idx, i);

                // Clone all the shared stuff for use by just this thread.
                // You could wrap all of these in an Arc<Mutex<_>> and share
                // them that way but it would be totally redundant as they
                // each contain reference counted pointers at their core.
                // You could pass them around on channels but it would be
                // inconvenient and more costly.
                let context_th = context.clone();
                let program_th = program.clone();
                let dims_th = dims.clone();
                let queueball_th = queueball.clone();

                // [FIXME] Create some channels to swap around buffers, queues, and kernels.
                // let (queue_tx, queue_rx) = mpsc::channel();
                // let (buffer_tx, buffer_rx) = mpsc::channel();
                // let (kernel_tx, kernel_rx) = mpsc::channel();

                print!("{}, ", thread_name);

                let th = thread::Builder::new().name(thread_name.clone()).spawn(move || {
                    // let mut buffer = Buffer::<f32>::with_vec(&dims_th, &queueball_th[0]);
                    // let mut buffer = Buffer::<f32>::new(queueball_th[0].clone(), None,
                    //     &dims_th, None, None).unwrap();
                    let mut buffer = Buffer::<f32>::builder()
                        .queue(queueball_th[0].clone())
                        .len(dims_th)
                        .build().unwrap();
                    let mut vec = vec![0.0f32; buffer.len()];

                    let mut kernel = Kernel::builder()
                        .program(&program_th)
                        .name("add")
                        .queue(queueball_th[0].clone())
                        .global_work_size(&dims_th)
                        .arg(&buffer)
                        .arg(&1000.0f32)
                        .build().unwrap();

                    // Event list isn't really necessary here but hey.
                    let mut event_list = EventList::new();

                    // Change queues around just for fun:
                    unsafe {
                        kernel.cmd().enew(&mut event_list).enq().unwrap();
                        kernel.set_default_queue(queueball_th[1].clone()).enq().unwrap();
                        kernel.cmd().queue(&queueball_th[2]).enq().unwrap();
                    }

                    // Sleep just so the results don't print too quickly.
                    thread::sleep(Duration::from_millis(100));

                    // Basically redundant in this situation.
                    event_list.wait_for().unwrap();

                    // Again, just playing with queues...
                    buffer.set_default_queue(queueball_th[2].clone()).read(&mut vec).enq().unwrap();
                    buffer.read(&mut vec).queue(&queueball_th[1]).enq().unwrap();
                    buffer.read(&mut vec).queue(&queueball_th[0]).enq().unwrap();
                    buffer.read(&mut vec).enq().unwrap();

                    // Print results (won't appear until later):
                    let check_idx = data_set_size / 2;
                    print!("{{{}}}={}, ", &thread_name, vec[check_idx]);
                }).expect("Error creating thread");

                threads.push(th);
            }

            print!("\n");
        }
    }

    print!("\nResults: ");

    for th in threads.into_iter() {
        if let Err(e) = th.join() { println!("Error joining thread: '{:?}'", e); }
    }

    print!("\n");
    Ok(())
}
