//! Threading.
//!
//! [WORK IN PROGRESS]
//!
//! TODO: Have threads swap stuff around for fun.
//! TODO: Print the reference counts of each element at various points.

#![allow(unused_imports, unused_variables, dead_code, unused_mut)]

extern crate rand;
extern crate ocl;

use std::thread::{self, JoinHandle};
use std::sync::mpsc;
use std::time::Duration;
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

	// Loop through each avaliable platform:
    for p_idx in 0..platforms.len() {
    	let platform = &platforms[p_idx];
    	println!("Platform[{}]: {} ({})", p_idx, platform.name(), platform.vendor());

    	let devices = Device::list_all(platform);

    	// Loop through each device:
    	for device_idx in 0..devices.len() {
    		// Choose a device at random: 
    		// let dev_idx = rng.gen_range(0, devices.len());

    		let device = &devices[device_idx];
    		println!("Device[{}]: {} ({})", device_idx, device.name(), device.vendor());

    		// Make a context to share around:
    		let context = Context::new_by_index_and_type(None, None).unwrap();
    		let program = Program::builder().src(SRC).build(&context).unwrap();

			print!("    Spawning threads... ");

			for i in 0..5 {
				let thread_name = format!("{}:[D{}.I{}]", threads.len(), device_idx, i);

				let context_th = context.clone();
				let program_th = program.clone();
				let dims_th = dims.clone();

				// Create some channels to swap around buffers, queues, and kernels.
	    		// let (queue_tx, queue_rx) = mpsc::channel();
	    		// let (buffer_tx, buffer_rx) = mpsc::channel();
	    		// let (kernel_tx, kernel_rx) = mpsc::channel();

				print!("{}, ", thread_name);

				let th = thread::Builder::new().name(thread_name.clone()).spawn(move || {
			        let queue = Queue::new_by_device_index(&context_th, None);
					let mut buffer = Buffer::<f32>::with_vec(&dims_th, &queue);
					let kernel = Kernel::new("add", &program_th, &queue, dims_th.clone()).unwrap()
					        .arg_buf(&buffer)
					        .arg_scl(1000.0f32);
					let mut event_list = EventList::new();

					kernel.enqueue_with(None, None, Some(&mut event_list)).unwrap();

					thread::sleep(Duration::from_millis(500));

					event_list.wait();
					buffer.fill_vec();

					// Print results (won't appear until later):
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
