//! Threading.
//!
//! [WORK IN PROGRESS]
//!
//! TODO: Have threads swap stuff around for fun.
//! TODO: Print the reference counts of each element at various points.

#![allow(unused_imports, unused_variables, dead_code)]

extern crate rand;
extern crate ocl;

use std::thread::{self, JoinHandle};
use std::sync::mpsc;
use rand::Rng;
use ocl::{SimpleDims, Platform, Device, Context, Queue, Buffer, Program, Kernel, EventList};
use ocl::core::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo, EventInfo, ProfilingInfo};


static SRC: &'static str = r#"
	__kernel void add(__global float* buffer, float addend) {
        buffer[get_global_id(0)] += addend;
    }
"#;

fn main() {
	let mut rng = rand::weak_rng();
	let data_set_size = 1000;
	let dims = SimpleDims::One(data_set_size);
	let mut threads = Vec::new();

	let platforms = Platform::list();

	println!("Looping through avaliable platforms ({}):", platforms.len());

	// Loop through all avaliable platforms:
    for p_idx in 0..platforms.len() {
    	let platform = &platforms[p_idx];
    	println!("Platform[{}]: {} ({})", p_idx, platform.name(), platform.vendor());

    	let devices = Device::list_all(platform);

    	// Loop through each device an average of 3 times
    	for i in 0..(devices.len() * 2) {
    		// Choose a device at random: 
    		let dev_idx = rng.gen_range(0, devices.len());
    		let device = &devices[dev_idx];
    		println!("Device[{}]: {} ({})", dev_idx, device.name(), device.vendor());

    		// Make a context to share around:
    		let context = Context::new(None, None).unwrap();

			print!("    Spawning threads... ");

			for i in 0..5 {
				let thread_name = format!("{}:[D{}.I{}]", threads.len(), dev_idx, i);

				let context = context.clone();
				let dims = dims.clone();

				// Create some channels to easily swap things around if threads feel like it.
	    		// let (queue_tx, queue_rx) = mpsc::channel();
	    		// let (buffer_tx, buffer_rx) = mpsc::channel();

				print!("{}, ", thread_name);

				let th = thread::Builder::new().name(thread_name.clone()).spawn(move || {
			        let queue = Queue::new(&context, None);
					let mut buffer = Buffer::<f32>::with_vec(&dims, &queue);
					let program = Program::builder().src(SRC).build(&context).unwrap();
					let kernel = Kernel::new("add", &program, &queue, dims.clone()).unwrap()
					        .arg_buf(&buffer)
					        .arg_scl(10.0f32);
					let mut event_list = EventList::new();

					kernel.enqueue_with_events(None, Some(&mut event_list));
					let event = event_list.last_clone().unwrap();
					event_list.wait();

					buffer.fill_vec();

					let check_idx = data_set_size / 2;
					print!("{{{}}}={}, ", &thread_name, buffer[check_idx]);
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
}
