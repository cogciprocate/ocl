//! [WORK IN PROGRESS] Print information about all the things.
//!
//! Printing info for any of the main types is as simple as 
//! `println("{}", &instance);` as `Display` is implemented for each.
//!
//! Printing algorithm is highly janky (due to laziness -- need to complete
//! for each `*InfoResult` type).
//!
//! 

#![allow(unused_imports, unused_variables, dead_code)]

extern crate ocl;

use ocl::{SimpleDims, Platform, Device, Context, Queue, Buffer, Image, Program, Kernel, Event, EventList};
use ocl::core::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo, EventInfo, ProfilingInfo, OclNum};

const PRINT_DETAILED: bool = true;
// Overrides above:
const PRINT_DETAILED_DEVICE: bool = false;
const PRINT_DETAILED_PROGRAM: bool = false;

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

    	let devices = Device::list_all(platform);

    	// [NOTE]: A new context can also be created for each device if desired.
    	let context = Context::builder()
			.platform(platform.clone())
			.devices(devices.clone())
			.build().unwrap();

		print_platform_info(&platform, p_idx); 
		print_context_info(&context);
		print!("\n");

    	// Loop through each device
    	for d_idx in 0..devices.len() {
    		let device = &devices[d_idx];
	    	
			let queue = Queue::new(&context, Some(device.clone()));
			let buffer = Buffer::<f32>::new(&dims, &queue);
			let image = Image::builder().build(&queue).unwrap();
			// let sampler = Sampler::new();
	    	let program = Program::builder()
	    		.src(SRC)
	    		.devices(vec![device.clone()])
	    		.build(&context).unwrap();
			let kernel = Kernel::new("multiply", &program, &queue, dims.clone()).unwrap()
			        .arg_buf(&buffer)
			        .arg_scl(10.0f32);
			let mut event_list = EventList::new();

			kernel.enqueue_with_events(None, Some(&mut event_list));
			let event = event_list.last_clone().unwrap();
			event_list.wait();
			
			// Print all the things:
			print_device_info(&device, d_idx);
			print_queue_info(&queue);
			print_buffer_info(&buffer);
			print_image_info(&image);
			// print_sampler_info(&sampler);
			print_program_info(&program);
			print_kernel_info(&kernel);
			// print event_list_info(&event_list);
			print_event_info(&event);

			print!("\n");
		}
	}
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
		println!("{t}{t}{}", context, t = TAB);
	}
}


fn print_queue_info(queue: &Queue) {
	if PRINT_DETAILED {
		println!("{}", queue);
	} else {
		println!("{t}{t}{}", queue, t = TAB);
	}
}


fn print_buffer_info<T: OclNum>(buffer: &Buffer<T>) {
	if PRINT_DETAILED {
		println!("{}", buffer);
	} else {
		println!("{t}{t}[Buffer]: {{ Type: {}, Flags: {}, Size: {} }}", 
			buffer.mem_info(MemInfo::Type),
			buffer.mem_info(MemInfo::Flags),
			buffer.mem_info(MemInfo::Size),
			t = TAB,
		);
	}
}


fn print_image_info(image: &Image) {
	println!("{}", image);
}


fn print_sampler_info() {
	unimplemented!();
}


fn print_program_info(program: &Program) {

	if PRINT_DETAILED_PROGRAM {
		println!("{}", program);
	} else {
		if !PRINT_DETAILED { print!("{t}{t}", t = TAB); } 
		println!("[Program]: {{ KernelNames: '{}', NumDevices: {}, ReferenceCount: {}, Context: {} }}", 
			program.info(ProgramInfo::KernelNames),
			program.info(ProgramInfo::NumDevices),
			program.info(ProgramInfo::ReferenceCount),
			program.info(ProgramInfo::Context),
		);
	}
}


fn print_kernel_info(kernel: &Kernel) {
	if PRINT_DETAILED {
		println!("{}", kernel);
	} else {
		println!("{t}{t}{}", kernel, t = TAB);
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


fn print_event_list_info() {
	unimplemented!();
}
