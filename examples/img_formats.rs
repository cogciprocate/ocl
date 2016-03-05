extern crate ocl;
use ocl::{Context, Image};
use ocl::enums::MemObjectType;

fn main() {
	let context = Context::builder().build().unwrap();
    let sup_img_formats = Image::supported_formats(&context, ocl::flags::MEM_READ_WRITE, 
        MemObjectType::Image2d).unwrap();

    println!("Image Formats: {:#?}.", sup_img_formats);
}