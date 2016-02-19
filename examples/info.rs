//! [WORK IN PROGRESS] Get information about all the things.
//!

#![allow(unused_imports, unused_variables, dead_code)]

extern crate ocl;

use ocl::{SimpleDims, Platform, Device, Context, Queue, Buffer, Program, Kernel, EventList};
use ocl::raw::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo, EventInfo, ProfilingInfo};

const PRINT_DETAILED: bool = false;
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
    	let platform = platforms[p_idx];
    	print!("\n");
    	println!("Platform[{}]: {} ({})", p_idx, platform.name(), platform.vendor());

    	let devices = Device::list_all(platform);

    	// Loop through each device
    	for d_idx in 0..devices.len() {
    		let device = devices[d_idx];

    		if PRINT_DETAILED_DEVICE {
    			println!("Device[{}]: {}", d_idx, device);
    		} else {
	    		println!("{t}Device[{}]: {} ({})", d_idx, device.name(), device.vendor(), t = TAB);
	    	}

	    	let context = Context::new(None, None).unwrap();
			let queue = Queue::new(&context, None);
			let buffer = Buffer::<f32>::new(&dims, &queue);
			// let image = Image::new();
			// let sampler = Sampler::new();
			let program = Program::builder().src(SRC).build(&context).unwrap();
			let device = program.device_ids_raw()[0];
			let kernel = Kernel::new("multiply", &program, &queue, dims.work_dims()).unwrap()
			        .arg_buf(&buffer)
			        .arg_scl(10.0f32);
			let mut event_list = EventList::new();

			kernel.enqueue(None, Some(&mut event_list));
			let event = event_list.last().unwrap().clone();
			event_list.wait();

			if PRINT_DETAILED {
    			println!("{}", event);
    		} else {
	    		println!("{t}{t}Event: {{ Type: {}, Status: {} }}", 
	    			event.info(EventInfo::CommandType),
	    			event.info(EventInfo::CommandExecutionStatus),
	    			t = TAB);
	    	}

			print!("\n");
		}

	}

	// print!("\n");
}
