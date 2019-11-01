//! Convert a color image (`test.jpg` in this case) to greyscale.
//!
//! Performs this operation twice. First using an "unrolled" global work size
//! (one work item per pixel) then using a "patches" (tesselated / square
//! blocks) method, which computes one square patch per work item (size
//! designated by `patch_size`, below).
//!
//! The image processing done by this example does not need to be manually
//! broken down into "patches" but is shown this way to demonstrate how it
//! would be done if necessary. *Usually*, condensing multiple, repeated
//! operations into a single work item is not idiomatic (or performant) in
//! OpenCL kernels but is often useful in specific cases to improve cache
//! efficiency/performance and for other reasons such as streamlining
//! workgroup reads/writes. This particular example contains operations which
//! each read from independent memory locations and will probably gain nothing
//! from being grouped together into loops (again though, it is still a useful
//! thing to do in certain cases, just not this one).
//!
//!
//! # Image Indexing
//!
//! Due to a lack of clarity in the official OpenCL documentation
//! (particularly documentation regarding how out of range image indexes are
//! or are not clamped) and differences between vendor implementations, the
//! following example shows the best practice for addressing into an image.
//! Put plainly, do not expect clamping to work for *integer* index values
//! whether normalized or not (normalized float indexes should clamp as
//! expected). Always be dilligent in assuring that integer index values are
//! within the bounds of the image buffer.
//!
//!
//! [OPEN QUESTION]: Could this example be refactored / split up to improve clarity?
//!

extern crate find_folder;
extern crate image;
extern crate time;
extern crate fil_ocl as ocl;
#[macro_use] extern crate colorify;

use std::path::Path;
use ocl::{Context, Queue, Device, Program, Image, Kernel};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType};
use find_folder::Search;


fn print_elapsed(title: &str, start: time::Timespec) {
    let time_elapsed = time::get_time() - start;
    let elapsed_ms = time_elapsed.num_milliseconds();
    let separator = if title.len() > 0 { ": " } else { "" };
    println!("    {}{}{}.{:03}", title, separator, time_elapsed.num_seconds(), elapsed_ms);
}


fn read_source_image(loco : &str) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let dyn = image::open(&Path::new(loco)).unwrap();
    let img = dyn.to_rgba();
    img
}


fn main() {
    let compute_program = Search::ParentsThenKids(3, 3)
        .for_folder("cl_src").expect("Error locating 'cl_src'")
        .join("cl/parallel.cl");

    let context = Context::builder().devices(Device::specifier()
        .type_flags(ocl::flags::DEVICE_TYPE_GPU).first()).build().unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src_file(compute_program)
        .devices(device)
        .build(&context)
        .unwrap();

    let img = read_source_image("test.jpg");

    let dims = img.dimensions();

    let cl_source = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .queue(queue.clone())
        .copy_host_slice(&img)
        .build().unwrap();


    // ##################################################
    // #################### UNROLLED ####################
    // ##################################################

    let mut result_unrolled: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(dims.0, dims.1);

    let cl_dest_unrolled = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .queue(queue.clone())
        .copy_host_slice(&result_unrolled)
        .build().unwrap();

    let kernel = Kernel::builder()
        .program(&program)
        .name("rgb2gray_unrolled")
        .queue(queue.clone())
        .global_work_size(&dims)
        .arg(&cl_source)
        .arg(&cl_dest_unrolled)
        .build().unwrap();

    printlnc!(royal_blue: "\nRunning kernel (unrolled)...");
    printlnc!(white_bold: "image dims: {:?}", &dims);
    let start_time = time::get_time();

    unsafe { kernel.enq().unwrap(); }
    print_elapsed("kernel enqueued", start_time);

    queue.finish().unwrap();
    print_elapsed("queue finished", start_time);

    cl_dest_unrolled.read(&mut result_unrolled).enq().unwrap();
    print_elapsed("read finished", start_time);

    result_unrolled.save(&Path::new("result_unrolled.png")).unwrap();


    // ##################################################
    // #################### PATCHES #####################
    // ##################################################
    // As noted above, this method is only shown here for demonstration
    // purposes and is unnecessary in this particular use case.

    let mut result_patches: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(dims.0, dims.1);

    let cl_dest_patches = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .queue(queue.clone())
        .copy_host_slice(&result_patches)
        .build().unwrap();


    let patch_size = 32;

    // The number of `patch_size` squares that fit into the image.
    let gws_patch_count = (dims.0 / patch_size, dims.1 / patch_size);

    let kernel_bulk = Kernel::builder()
        .program(&program)
        .name("rgb2gray_patches")
        .queue(queue.clone())
        .global_work_size(&gws_patch_count)
        .arg(patch_size as i32)
        .arg(&cl_source)
        .arg(&cl_dest_patches)
        .build().unwrap();

    let edge_sizes = (dims.0 % patch_size, dims.1 % patch_size);
    assert_eq!(dims.1 - edge_sizes.1, gws_patch_count.1 * patch_size);
    assert_eq!(dims.0 - edge_sizes.0, gws_patch_count.0 * patch_size);

    let gwo_rght_edge = (dims.0 - edge_sizes.0, 0);
    let gws_rght_edge = (edge_sizes.0, dims.1 - edge_sizes.1);
    let kernel_rght_edge = Kernel::builder()
        .program(&program)
        .name("rgb2gray_unrolled")
        .queue(queue.clone())
        .global_work_offset(&gwo_rght_edge)
        .global_work_size(&gws_rght_edge)
        .arg(&cl_source)
        .arg(&cl_dest_patches)
        .build().unwrap();

    let gwo_bot_edge = (0, dims.1 - edge_sizes.1);
    let gws_bot_edge = (dims.0 - edge_sizes.0, edge_sizes.1);
    let kernel_bot_edge = Kernel::builder()
        .program(&program)
        .name("rgb2gray_unrolled")
        .queue(queue.clone())
        .global_work_offset(&gwo_bot_edge)
        .global_work_size(&gws_bot_edge)
        .arg(&cl_source)
        .arg(&cl_dest_patches)
        .build().unwrap();

    let gwo_corner = (dims.0 - edge_sizes.0, dims.1 - edge_sizes.1);
    let gws_corner = (edge_sizes.0, edge_sizes.1);
    let kernel_corner = Kernel::builder()
        .program(&program)
        .name("rgb2gray_unrolled")
        .queue(queue.clone())
        .global_work_offset(&gwo_corner)
        .global_work_size(&gws_corner)
        .arg(&cl_source)
        .arg(&cl_dest_patches)
        .build().unwrap();

    printlnc!(royal_blue: "\nRunning kernels (patch bulk & patch edges)...");
    printlnc!(white_bold: "image dims: {:?}", &dims);
    printlnc!(white_bold: "bulk dims: {:?} * patch_size {{{}}} = ({}, {})", gws_patch_count, patch_size,
        gws_patch_count.0 * patch_size, gws_patch_count.1 * patch_size);
    printlnc!(white_bold: "edges dims: {:?}", edge_sizes);

    let start_time = time::get_time();

    unsafe {
        kernel_bulk.enq().unwrap();
        kernel_rght_edge.enq().unwrap();
        kernel_bot_edge.enq().unwrap();
        kernel_corner.enq().unwrap();
    }
    print_elapsed("kernels enqueued", start_time);

    queue.finish().unwrap();
    print_elapsed("queue finished", start_time);

    cl_dest_patches.read(&mut result_patches).enq().unwrap();
    print_elapsed("read finished", start_time);

    result_patches.save(&Path::new("result_patches.png")).unwrap();


    // ##################################################
    // ############### COMPARE / VERIFY #################
    // ##################################################

    for (px_unr, px_pat) in result_unrolled.iter().zip(result_patches.iter()) {
        assert_eq!(px_unr, px_pat);
    }
}
