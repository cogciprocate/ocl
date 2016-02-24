//! [WORK IN PROGRESS] Get information about all the things.
//!
//! Printing algorithm is highly janky (due to laziness -- need to complete
//! for each `*InfoResult` type).

#![allow(unused_imports, unused_variables, dead_code)]

extern crate ocl;

use ocl::{SimpleDims, Platform, Device, Context, Queue, Buffer, Program, Kernel, Event, EventList};
use ocl::core::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo, EventInfo, ProfilingInfo};

const PRINT_DETAILED: bool = true;
// Overrides above for device:
const PRINT_DETAILED_DEVICE: bool = false;

static TAB: &'static str = "    ";
static SRC: &'static str = r#"
	__kernel void multiply(__global float* buffer, float coeff) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
	let dims = SimpleDims::One(1000);
	let platforms = Platform::list();

	println!("Looping through avaliable platforms ({}):", platforms.len());

	// Loop through all avaliable platforms:
    for p_idx in 0..platforms.len() {
    	let platform = &platforms[p_idx];
		print_platform_info(&platform, p_idx);    	

    	let devices = Device::list_all(platform);

    	// Loop through each device
    	for d_idx in 0..devices.len() {
    		let device = &devices[d_idx];		    

	    	let context = Context::builder()
	    		.platform(platform.clone())
	    		.device(device.clone())
	    		.build().unwrap();
	    	let program = Program::builder().src(SRC).build(&context).unwrap();
			let queue = Queue::new(&context, Some(device.clone()));
			let buffer = Buffer::<f32>::new(&dims, &queue);
			// let image = Image::new();
			// let sampler = Sampler::new();
			let kernel = Kernel::new("multiply", &program, &queue, dims.clone()).unwrap()
			        .arg_buf(&buffer)
			        .arg_scl(10.0f32);
			let mut event_list = EventList::new();

			kernel.enqueue_with_events(None, Some(&mut event_list));
			let event = event_list.last_clone().unwrap();
			event_list.wait();
			
			// Print all the things:
			print_device_info(&device, d_idx);
			print_context_info(&context);			
			print_event_info(&event);

			print!("\n");
		}

	}

	// print!("\n");
}


fn print_platform_info(platform: &Platform, p_idx: usize) {
	let devices = Device::list_all(platform);

	print!("\n");

	if PRINT_DETAILED {
		print!("{}", platform);	    		

	} else {
		print!("Platform[{}]: {} ({})", p_idx, platform.name(), platform.vendor());
	}

	print!(" {{ Total Device Count: {} }}", devices.len());
	print!("\n");
	print!("\n");
}


fn print_device_info(device: &Device, d_idx: usize) {
	if PRINT_DETAILED_DEVICE {
		println!("[Device][{}]: {}", d_idx, device);
	} else {
		if !PRINT_DETAILED { print!("{t}", t = TAB); } 
		println!("[Device][{}]: {} ({})", d_idx, device.name(), device.vendor());
	}
}


fn print_context_info(context: &Context) {
	if PRINT_DETAILED {
		println!("{}", context);
	} else {
		println!("{t}{t}[Context]:  ()", t = TAB);
	}
}


fn print_event_info(event: &Event) {
	if PRINT_DETAILED {
		println!("{}", event);
	} else {
		println!("{t}{t}[Event]: {{ Type: {}, Status: {} }}", 
			event.info(EventInfo::CommandType),
			event.info(EventInfo::CommandExecutionStatus),
			t = TAB);
	}
}
