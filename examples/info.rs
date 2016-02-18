//! [WORK IN PROGRESS] Get information about all the things.
//!

#![allow(unused_imports, unused_variables, dead_code)]

extern crate ocl;

use ocl::{SimpleDims, Platform, Device, Context, Queue, Buffer, Program, Kernel, EventList};
use ocl::raw::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo, EventInfo, ProfilingInfo};

static TAB: &'static str = "    ";
static SRC: &'static str = r#"
	__kernel void multiply(__global float* buffer, float coeff) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
	let dims = SimpleDims::One(1000);
	let platforms = Platform::list();

    for platform in platforms {
    	println!("{}", platform);

    	let devices = Device::list_all(platform);

    	for device in devices {
	    	println!("{}", device);
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
	}
}
