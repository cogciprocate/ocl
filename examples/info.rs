//! Print information about all the things.
//!
//! Printing info for any of the main types is as simple as 
//! `println("{}", &instance);` as `Display` is implemented for each.
//!
//! Printing algorithm is highly janky (due to laziness -- need to complete
//! for each `*InfoResult` type) so lots of stuff isn't formatted correctly
//! (or at all).
//!
//! 

extern crate ocl;
#[macro_use] extern crate colorify;

use ocl::{Platform, Device, Context, Queue, Buffer, Image, Sampler, Program, Kernel, Event, EventList};
use ocl::core::{ProgramInfo, OclPrm};

const PRINT_DETAILED: bool = true;
// Overrides above for device and program:
const PRINT_DETAILED_DEVICE: bool = false;
const PRINT_DETAILED_PROGRAM: bool = false;

static TAB: &'static str = "    ";
static SRC: &'static str = r#"
	__kernel void multiply(__global float* buffer, float coeff) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
	let dims = [2048];
	let platforms = Platform::list();

	println!("Looping through avaliable platforms ({}):", platforms.len());

	// Loop through all avaliable platforms:
    for p_idx in 0..platforms.len() {
    	let platform = &platforms[p_idx];

    	let devices = Device::list_all(platform);

    	// [NOTE]: A new context can also be created for each device if desired.
    	let context = Context::builder()
			.platform(platform.clone())
			.devices(&devices)
			.build().unwrap();

		print_platform_info(&platform);

		for device in devices.iter() {
			print_device_info(device);
		}

    	// Loop through each device
    	for d_idx in 0..devices.len() {
    		let device = devices[d_idx];
	    	
			let queue = Queue::new(&context, device).unwrap();
			let buffer = Buffer::<f32>::new(&queue, None, &dims, None).unwrap();
			let image = Image::<u8>::builder()
				.dims(dims)
				.build(&queue).unwrap();
			let sampler = Sampler::with_defaults(&context).unwrap();
	    	let program = Program::builder()
	    		.src(SRC)
	    		.devices(device)
	    		.build(&context).unwrap();
			let kernel = Kernel::new("multiply", &program, &queue).unwrap()
				.gws(&dims)
		        .arg_buf(&buffer)
		        .arg_scl(10.0f32);
			        
			let mut event_list = EventList::new();
			kernel.cmd().enew(&mut event_list).enq().unwrap();
			event_list.wait().unwrap();

			let mut event = Event::empty();
		    buffer.cmd().write(&vec![0.0; dims[0]]).enew(&mut event).enq().unwrap();
		    event.wait().unwrap();

			// Print all but device (just once):
			if (d_idx == 0) && (p_idx == 0) {
				print_context_info(&context);
				print_queue_info(&queue);
				print_buffer_info(&buffer);
				print_image_info(&image);
				print_sampler_info(&sampler);
				print_program_info(&program);
				print_kernel_info(&kernel);
				print_event_list_info(&event_list);
				print_event_info(&event);
			}
		}
	}
}


fn print_platform_info(platform: &Platform) {
	printc!(blue: "{}", platform);
	let devices = Device::list_all(platform);
	printc!(blue: " {{ Total Device Count: {} }}", devices.len());
	print!("\n");
}


fn print_device_info(device: &Device) {
	if PRINT_DETAILED_DEVICE {
		printlnc!(teal: "{}", device);
	} else {
		if !PRINT_DETAILED { print!("{t}", t = TAB); } 
		printlnc!(teal: "Device (terse) {{ Name: {}, Vendor: {} }}", device.name(), device.vendor());
	}
}


fn print_context_info(context: &Context) {
	printlnc!(purple: "{}", context);
}


fn print_queue_info(queue: &Queue) {
	printlnc!(lime: "{}", queue);
}


fn print_buffer_info<T: OclPrm>(buffer: &Buffer<T>) {
	printlnc!(royal_blue: "{}", buffer);
}


fn print_image_info<S: OclPrm>(image: &Image<S>) {
	printlnc!(peach: "{}", image);
}


fn print_sampler_info(sampler: &Sampler) {
	printlnc!(cyan: "{}", sampler);
}


fn print_program_info(program: &Program) {
	if PRINT_DETAILED_PROGRAM {
		printlnc!(magenta: "{}", program);
	} else {
		if !PRINT_DETAILED { print!("{t}{t}", t = TAB); } 
		printlnc!(magenta: "Program (terse) {{ KernelNames: '{}', NumDevices: {}, ReferenceCount: {}, Context: {} }}", 
			program.info(ProgramInfo::KernelNames),
			program.info(ProgramInfo::NumDevices),
			program.info(ProgramInfo::ReferenceCount),
			program.info(ProgramInfo::Context),
		);
	}
}


fn print_kernel_info(kernel: &Kernel) {
	printlnc!(green: "{}", kernel);
}


fn print_event_list_info(event_list: &EventList) {
	printlnc!(orange: "{:?}", event_list);
}


fn print_event_info(event: &Event) {
	printlnc!(yellow: "{}", event);
}
