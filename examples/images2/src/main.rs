use std::io::prelude::*;
use std::fs::File;

extern crate image;
extern crate ocl;
extern crate time;

use std::path::Path;
use ocl::{Context, Queue, Device, Program, Image, Kernel};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType};

fn print_elapsed(title: &str, start: time::Timespec) {
    let time_elapsed = time::get_time() - start;
    let elapsed_ms = time_elapsed.num_milliseconds();
    let separator = if title.len() > 0 { ": " } else { "" };
    println!("    {}{}{}.{:03}", title, separator, time_elapsed.num_seconds(), elapsed_ms);
}


fn read_kernel(loco : &str) -> String {
	let _ = loco;
	let mut f = File::open(loco).unwrap();
	let mut compute_program = String::new();

	f.read_to_string(&mut compute_program).unwrap();
	compute_program
}

fn read_source_image(loco : &str) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
	let dyn = image::open(&Path::new(loco)).unwrap();
	let img = dyn.to_rgba();
	img
}

fn main() {
	let compute_program = read_kernel("src/parallel.cl");

	let context = Context::builder().devices(Device::specifier().type_flags(ocl::flags::DEVICE_TYPE_GPU).first()).build().unwrap();
	let device = context.devices()[0];
	let queue = Queue::new(&context, device).unwrap();

	let program = Program::builder()
		.src(compute_program)
		.devices(device)
		.build(&context)
		.unwrap();

	let img = read_source_image("test.jpg");

	let dims = img.dimensions();
	let mut result : image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(dims.0, dims.1);

	let patch_size = 32;
	let gws = ((dims.0 + patch_size-1)/patch_size, (dims.1 + patch_size - 1)/patch_size);

	let cl_source = Image::<u8>::builder()
		.channel_order(ImageChannelOrder::Rgba)
		.channel_data_type(ImageChannelDataType::UnormInt8)
		.image_type(MemObjectType::Image2d)
		.dims(&dims)
		.flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
		.build_with_data(&queue, &img)
		.unwrap();


	let cl_dest = Image::<u8>::builder()
		.channel_order(ImageChannelOrder::Rgba)
		.channel_data_type(ImageChannelDataType::UnormInt8)
		.image_type(MemObjectType::Image2d)
		.dims(&dims)
		.flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
		.build_with_data(&queue, &result)
		.unwrap();

    let kernel = Kernel::new("rgb2gray", &program, &queue).unwrap()
        .gws(&gws)
        .arg_scl(patch_size)
        .arg_img(&cl_source)
        .arg_img(&cl_dest);

	let start_time = time::get_time();

	kernel.enq().unwrap();

	print_elapsed("queue unfinished", start_time);

	queue.finish();

	print_elapsed("queue finished", start_time);

	cl_dest.read(&mut result).enq().unwrap();
	queue.finish(); // necessary or you will end up with an empty saved image

	result.save(&Path::new("result.png")).unwrap();

}
