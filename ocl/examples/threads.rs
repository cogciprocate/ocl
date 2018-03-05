//! Threading.
//!
//! Much of this is just testing stuff out and not really practical.
//!
//! Due to buggy and/or intentionally crippled drivers, this example may not
//! work on NVIDIA hardware. Until NVIDIA's implementation is corrected this
//! example will likely fail on that platform.
//!
//! [WORK IN PROGRESS]
//!
//! * TODO: Have threads swap stuff around for fun.
//! * TODO: Print the reference counts of each element at various points.

extern crate rand;
extern crate ocl;
#[macro_use] extern crate colorify;

use std::thread::{self, JoinHandle};
use std::time::Duration;
use ocl::{Result as OclResult, Platform, Device, Context, Queue, Buffer, Program, Kernel, EventList};

static SRC: &'static str = r#"
    __kernel void add(__global float* buffer, float addend) {
        buffer[get_global_id(0)] += addend;
    }
"#;

fn threads() -> OclResult<()> {
    let work_size = 1 << 10;
    let mut threads: Vec<JoinHandle<OclResult<String>>> = Vec::new();

    let platforms = Platform::list();

    println!("Looping through avaliable platforms ({}):", platforms.len());

    // Loop through each avaliable platform:
    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];
        printlnc!(green: "\nPlatform[{}]: {} ({})", p_idx, platform.name()?, platform.vendor()?);

        let devices = Device::list_all(platform)?;

        printc!(blue: "DEVICES: {:?}", devices);

        // Loop through each device:
        for device_idx in 0..devices.len() {
            let device = devices[device_idx];
            printlnc!(royal_blue: "\nDevice[{}]: {} ({})", device_idx, device.name()?, device.vendor()?);

            // Make a context to share around:
            let context = Context::builder().platform(*platform).build()?;
            let program = Program::builder().src(SRC).devices(device)
                .build(&context)?;

            // Make a few different queues for the hell of it:
            let queueball = vec![Queue::new(&context, device, None)?,
                Queue::new(&context, device, None)?,
                Queue::new(&context, device, None)?];

            printlnc!(orange: "Spawning threads... ");

            for i in 0..5 {
                let thread_name = format!("{}:[D{}.I{}]", threads.len(), device_idx, i);

                // Clone all the shared stuff for use by just this thread. You
                // could wrap all of these in an Arc<Mutex<_>> and share them
                // that way but it would be totally redundant as they each
                // contain reference counted pointers internally. You could
                // pass them around on channels but it would be inconvenient
                // and more costly.
                let context_th = context.clone();
                let program_th = program.clone();
                let work_size_th = work_size;
                let queueball_th = queueball.clone();

                // [FIXME] Create some channels to swap around buffers, queues, and kernels.
                // let (queue_tx, queue_rx) = mpsc::channel();
                // let (buffer_tx, buffer_rx) = mpsc::channel();
                // let (kernel_tx, kernel_rx) = mpsc::channel();

                print!("{}, ", thread_name);

                let th = thread::Builder::new().name(thread_name.clone()).spawn(move || {
                    // Move these into thread:
                    let _context_th = context_th;
                    let program_th = program_th;
                    let work_size_th = work_size_th;
                    let queueball_th = queueball_th;

                    // let mut buffer = Buffer::<f32>::with_vec(&work_size_th, &queueball_th[0]);
                    let mut buffer = Buffer::<f32>::builder()
                        .queue(queueball_th[0].clone())
                        .len(work_size_th)
                        .build()?;
                    let mut vec = vec![0.0f32; buffer.len()];

                    let mut kernel = Kernel::builder()
                        .program(&program_th)
                        .name("add")
                        .queue(queueball_th[0].clone())
                        .global_work_size(work_size_th)
                        .arg_buf(&buffer)
                        .arg_scl(&1000.0f32)
                        .build()?;

                    // Event list isn't really necessary here but hey.
                    let mut event_list = EventList::new();

                    // Change queues around just for fun:
                    unsafe {
                        kernel.cmd().enew(&mut event_list).enq()?;
                        kernel.set_default_queue(queueball_th[1].clone()).enq()?;
                        kernel.cmd().queue(&queueball_th[2]).enq()?;
                    }

                    // Sleep just so the results don't print too quickly.
                    thread::sleep(Duration::from_millis(100));

                    // Basically redundant in this situation.
                    event_list.wait_for()?;

                    // Again, just playing with queues...
                    buffer.set_default_queue(queueball_th[2].clone()).read(&mut vec).enq()?;
                    buffer.read(&mut vec).queue(&queueball_th[1]).enq()?;
                    buffer.read(&mut vec).queue(&queueball_th[0]).enq()?;
                    buffer.read(&mut vec).enq()?;

                    // Print results (won't appear until later):
                    let check_idx = work_size / 2;
                    Ok(format!("{{{}}}={}, ", &thread_name, vec[check_idx]))
                }).expect("Error creating thread");

                threads.push(th);
            }

            print!("\n");
        }
    }

    printlnc!(orange: "\nResults: ");

    for th in threads.into_iter() {
        match th.join() {
            Ok(r) => print!("{}", r?),
            Err(e) => println!("Error joining thread: '{:?}'", e),
        }
    }

    print!("\n");
    Ok(())
}


pub fn main() {
    match threads() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}