//! Generates an image (currently a diagonal reddish stripe) then runs it
//! through the kernel, increasing the blue channel for the entire image.
//!
//! Enable saving to disk by setting `SAVE_IMAGES_TO_DISK` to `true` and
//! change the file paths/names if desired.
//!

extern crate image;
extern crate ocl;
#[macro_use]
extern crate colorify;

use ocl::enums::{
    AddressingMode, FilterMode, ImageChannelDataType, ImageChannelOrder, MemObjectType,
};
use ocl::{Context, Device, Image, Kernel, Program, Queue, Sampler};
use std::path::Path;

const SAVE_IMAGES_TO_DISK: bool = false;
static BEFORE_IMAGE_FILE_NAME: &'static str = "before_example_image.png";
static AFTER_IMAGE_FILE_NAME: &'static str = "after_example_image.png";

static KERNEL_SRC: &'static str = r#"
    // Unused... here for comparison purposes:
    __constant sampler_t sampler_const =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_NONE |
        CLK_FILTER_NEAREST;

    __kernel void increase_blue(
                sampler_t sampler_host,
                read_only image2d_t src_image,
                write_only image2d_t dst_image)
    {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));

        float4 pixel = read_imagef(src_image, sampler_host, coord);

        pixel += (float4)(0.0, 0.0, 0.5, 0.0);

        write_imagef(dst_image, coord, pixel);
    }
"#;

/// Generates a diagonal reddish stripe and a grey background.
fn generate_image() -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let img = image::ImageBuffer::from_fn(512, 512, |x, y| {
        let near_midline = (x + y < 536) && (x + y > 488);

        if near_midline {
            image::Rgba([196, 50, 50, 255u8])
        } else {
            image::Rgba([50, 50, 50, 255u8])
        }
    });

    img
}

/// Generates and image then sends it through a kernel and optionally saves.
fn main() {
    println!("Running 'examples/image.rs::main()'...");
    let mut img = generate_image();

    if SAVE_IMAGES_TO_DISK {
        img.save(&Path::new(BEFORE_IMAGE_FILE_NAME)).unwrap();
    }

    let context = Context::builder()
        .devices(Device::specifier().first())
        .build()
        .unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        .build(&context)
        .unwrap();

    let sup_img_formats = Image::<u8>::supported_formats(
        &context,
        ocl::flags::MEM_READ_WRITE,
        MemObjectType::Image2d,
    )
    .unwrap();
    println!("Image formats supported: {}.", sup_img_formats.len());
    // println!("Image Formats: {:#?}.", sup_img_formats);

    let dims = img.dimensions();

    // [NOTE]: When mapping settings from `image` crate, map
    // `ImageChannelOrder` roughly like this:
    // * image::Rgba => ImageChannelOrder::Rgba
    // * image::Rgb  => ImageChannelOrder::Rgb
    // * image::Luma => ImageChannelOrder::Luminance
    // * image::LumaA => Not sure
    //
    // Then just map your primitive type with ImageChannelDataType i.e.:
    // * u8 => ImageChannelDataType::UnormInt8
    // * f32 => ImageChannelDataType::Float
    // * etc.
    //
    // Will probably be some automation for this in the future.
    //
    let src_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(&img)
        .queue(queue.clone())
        .build()
        .unwrap();

    let dst_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_WRITE_ONLY
                | ocl::flags::MEM_HOST_READ_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(&img)
        .queue(queue.clone())
        .build()
        .unwrap();

    // Not sure why you'd bother creating a sampler on the host but here's how:
    let sampler = Sampler::new(&context, false, AddressingMode::None, FilterMode::Nearest).unwrap();

    let kernel = Kernel::builder()
        .program(&program)
        .name("increase_blue")
        .queue(queue.clone())
        .global_work_size(&dims)
        .arg_sampler(&sampler)
        .arg(&src_image)
        .arg(&dst_image)
        .build()
        .unwrap();

    println!("Printing image info:");
    printlnc!(dark_grey: "Source {}", src_image);
    print!("\n");
    printlnc!(dark_grey: "Destination {}", src_image);
    print!("\n");
    println!("Printing the first pixel of the image (each value is a component, RGBA): ");
    printlnc!(dark_grey: "Pixel before: [0..16]: {:?}", &img[(0, 0)]);

    printlnc!(royal_blue: "Attempting to blue-ify the image...");
    unsafe {
        kernel.enq().unwrap();
    }

    dst_image.read(&mut img).enq().unwrap();

    printlnc!(dark_grey: "Pixel after: [0..16]: {:?}", &img[(0, 0)]);

    if SAVE_IMAGES_TO_DISK {
        img.save(&Path::new(AFTER_IMAGE_FILE_NAME)).unwrap();
        printlnc!(lime: "Images saved as: '{}' and '{}'.",
            BEFORE_IMAGE_FILE_NAME, AFTER_IMAGE_FILE_NAME);
    } else {
        printlnc!(orange: "Saving images to disk disabled. \
            Enable by setting 'SAVE_IMAGES_TO_DISK' to 'true'.");
    }
}
