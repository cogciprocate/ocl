//! Generates an image (currently a diagonal reddish stripe) then runs it
//! through the kernel, increasing the blue channel for the entire image.
//!
//! Optionally saves for viewing, fiddle with consts/statics below.

extern crate image;
#[macro_use] extern crate ocl;

use std::path::Path;
use ocl::{Context, Queue, DeviceSpecifier, Program, Image, Sampler, Kernel,
    ImageChannelOrder, ImageChannelDataType, MemObjectType, AddressingMode, FilterMode};

const SAVE_IMAGES_TO_DISK: bool = false;
static BEFORE_IMAGE_FILE_NAME: &'static str = "before_example_image.png";
static AFTER_IMAGE_FILE_NAME: &'static str = "after_example_image.png";

static KERNEL_SRC: &'static str = r#"
    // UNUSED. FOR COMPARISON PURPOSES.
    // USING SAMPLER FROM HOST INSTEAD.
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

/// Mostly stolen from `https://github.com/PistonDevelopers/image`.
///
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

    let context = Context::builder().devices(DeviceSpecifier::Index(0)).build().unwrap();
    let device = context.devices()[0].clone();
    let queue = Queue::new(&context, Some(device.clone()));

    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(vec![device.clone()])
        .build(&context).unwrap();

    let sup_img_formats = Image::supported_formats(&context, ocl::MEM_READ_WRITE, 
        ocl::MemObjectType::Image2d).unwrap();
    println!("Image formats supported: {}.", sup_img_formats.len());
    // println!("Image Formats: {:#?}.", sup_img_formats);

    let dims = img.dimensions();

    // When mapping settings from `image` crate:
    // 
    // Map `ImageChannelOrder` roughly like this:
    // image::Rgba => ImageChannelOrder::Rgba
    // image::Rgb  => ImageChannelOrder::Rgb
    // image::Luma => ImageChannelOrder::Luminance
    // image::LumaA => Not sure
    //
    // Then just map your primitive type with ImageChannelDataType i.e.:
    // u8 => ImageChannelDataType::UnormInt8
    // f32 => ImageChannelDataType::Float
    // etc.
    // 
    // Will probably be some automation for this in the future.
    //
    let src_image = Image::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::MEM_READ_ONLY | ocl::MEM_HOST_WRITE_ONLY | ocl::MEM_COPY_HOST_PTR)
        .build_with_data(&queue, &img).unwrap();

    let dst_image = Image::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::MEM_WRITE_ONLY | ocl::MEM_HOST_READ_ONLY | ocl::MEM_COPY_HOST_PTR)
        .build_with_data(&queue, &img).unwrap();

    // Not sure why you'd bother creating a sampler on the host but here's how:
    let sampler = Sampler::new(&context, true, AddressingMode::None, FilterMode::Nearest).unwrap();

    let kernel = Kernel::new("increase_blue", &program, &queue, &dims).unwrap()
        .arg_smp(&sampler)
        .arg_img(&src_image)
        .arg_img(&dst_image);

    println!("Printing image info:");
    printlnc!(dark_grey: "Source {}", src_image);
    print!("\n");
    printlnc!(dark_grey: "Destination {}", src_image);
    print!("\n");
    println!("Printing the first pixel of the image (each value is a component, RGBA): ");
    printlnc!(dark_grey: "Pixel before: [0..16]: {:?}", &img[(0, 0)]);

    // print!("\n");
    printlnc!(royal_blue: "Attempting to blue-ify the image...");
    kernel.enqueue();

    dst_image.read(&mut img).unwrap();

    // print!("\n");
    printlnc!(dark_grey: "Pixel after: [0..16]: {:?}", &img[(0, 0)]);
    
    if SAVE_IMAGES_TO_DISK {
        img.save(&Path::new(AFTER_IMAGE_FILE_NAME)).unwrap();
        printlnc!(lime: "Images saved as: '{}' and '{}'.", 
            BEFORE_IMAGE_FILE_NAME, AFTER_IMAGE_FILE_NAME);
    } else {
        printlnc!(dark_orange: "Saving images to disk disabled. No files saved.");
    }
}
