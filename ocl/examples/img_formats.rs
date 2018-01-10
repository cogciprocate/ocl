#[macro_use] extern crate colorify;
extern crate ocl;
use ocl::{Result as OclResult, Platform, Device, Context, Image};
use ocl::enums::MemObjectType;

fn img_formats() -> OclResult<()> {
    for (p_idx, platform) in Platform::list().into_iter().enumerate() {
        for (d_idx, device) in Device::list_all(&platform)?.into_iter().enumerate() {
            printlnc!(blue: "Platform [{}]: {}", p_idx, platform.name()?);
            printlnc!(teal: "Device [{}]: {} {}", d_idx, device.vendor()?, device.name()?);

            let context = Context::builder().platform(platform).devices(device).build()?;

            let sup_img_formats = Image::<u8>::supported_formats(&context, ocl::flags::MEM_READ_WRITE,
                MemObjectType::Image2d)?;

            println!("Image Formats: {:#?}.", sup_img_formats);
        }
    }
    Ok(())
}

pub fn main() {
    match img_formats() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}