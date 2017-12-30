#[macro_use] extern crate colorify;
extern crate ocl;
use ocl::{Platform, Device, Context, Image};
use ocl::enums::MemObjectType;

fn main() {
    for (p_idx, platform) in Platform::list().into_iter().enumerate() {
        for (d_idx, device) in Device::list_all(&platform).unwrap().into_iter().enumerate() {
            printlnc!(blue: "Platform [{}]: {}", p_idx, platform.name());
            printlnc!(teal: "Device [{}]: {} {}", d_idx, device.vendor(), device.name());

            let context = Context::builder().platform(platform).devices(device).build().unwrap();

            let sup_img_formats = Image::<u8>::supported_formats(&context, ocl::flags::MEM_READ_WRITE,
                MemObjectType::Image2d).unwrap();

            println!("Image Formats: {:#?}.", sup_img_formats);
        }
    }
}
