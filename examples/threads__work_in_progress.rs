//! Threading.
//!
//! [WORK IN PROGRESS]

#![allow(unused_imports, unused_variables, dead_code)]

extern crate rand;
extern crate ocl;

use std::thread;
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
	// let mut rng = rand::weak_rng();
	// let dims = SimpleDims::One(1000);
	// let threads = Vec::new();

	// let platforms = Platform::list();

	// println!("Looping through avaliable platforms ({}):", platforms.len());

	// // Loop through all avaliable platforms:
 //    for p_idx in 0..platforms.len() {
 //    	let platform = &platforms[p_idx];
 //    	println!("Platform[{}]: {} ({})", p_idx, platform.name(), platform.vendor());

 //    	let devices = Device::list_all(platform);

 //    	// Loop through each device an average of 3 times
 //    	for i in 0..(devices.len() * 3) {
 //    		// Choose a device at random: 
 //    		let dev_idx = rng.gen_range(0, devices.len());
 //    		let device = &devices[dev_idx];
 //    		println!("Device[{}]: {} ({})", dev_idx, device.name(), device.vendor());

 //    		// Make a context to share around:
 //    		let context = Context::new(None, None).unwrap();

 //    		// Create some channels to easily swap a queue, buffer, and 
 //    		// program around if threads feel like it.
 //    		let (queue_tx, queue_rx) = mpsc::channel();
 //    		let (buffer_tx, buffer_rx) = mpsc::channel();


	    	
	// 		let queue = Queue::new(&context, None);
	// 		let buffer = Buffer::<f32>::with_vec(&dims, &queue);


	// 		let program = Program::builder().src(SRC).build(&context).unwrap();
	// 		let kernel = Kernel::new("add", &program, &queue, dims.clone()).unwrap()
	// 		        .arg_buf(&buffer)
	// 		        .arg_scl(10.0f32);
	// 		let mut event_list = EventList::new();

	// 		kernel.enqueue_with_events(None, Some(&mut event_list));
	// 		let event = event_list.last_clone().unwrap();
	// 		event_list.wait();

	// 	}
	// }
}



    

    // let th_win = thread::Builder::new().name("win".to_string()).spawn(move || {
    //     window::MainWindow::open(control_tx, result_rx);
    //     // window::conrod::window_conrod::open(control_tx, result_rx);
    // }).expect("Error creating 'win' thread");

    // let th_vis = thread::Builder::new().name("vis".to_string()).spawn(move || {
    //     interactive::CycleLoop::run(0, control_rx, result_tx);
    // }).expect("Error creating 'vis' thread");

    // if let Err(e) = th_win.join() { println!("th_win.join(): Error: '{:?}'", e); }
    // if let Err(e) = th_vis.join() { println!("th_vin.join(): Error: '{:?}'", e); }


