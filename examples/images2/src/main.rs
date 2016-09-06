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

extern crate find_folder;
extern crate image;
extern crate time;
extern crate ocl;
#[macro_use] extern crate colorify;

// use std::io::prelude::*;
// use std::fs::File;
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


// fn read_kernel(loco : &str) -> String {
//  let _ = loco;
//  let mut f = File::open(loco).unwrap();
//  let mut compute_program = String::new();

//  f.read_to_string(&mut compute_program).unwrap();
//  compute_program
// }


fn read_source_image(loco : &str) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let dyn = image::open(&Path::new(loco)).unwrap();
    let img = dyn.to_rgba();
    img
}


fn main() {
    // let compute_program = read_kernel("src/parallel.cl");

    let compute_program = Search::ParentsThenKids(3, 3)
        .for_folder("images2").unwrap().join("src/parallel.cl");

    let context = Context::builder().devices(Device::specifier()
        .type_flags(ocl::flags::DEVICE_TYPE_GPU).first()).build().unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device).unwrap();

    // let program = Program::builder()
    //  .src(compute_program)
    //  .devices(device)
    //  .build(&context)
    //  .unwrap();

    let program = Program::builder()
        .src_file(compute_program)
        .devices(device)
        .build(&context)
        .unwrap();

    let img = read_source_image("test.jpg");

    let dims = img.dimensions();
    let mut result: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(dims.0, dims.1);

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


    // ##################################################
    // #################### UNROLLED ####################
    // ##################################################

    let kernel = Kernel::new("rgb2gray_unrolled", &program, &queue).unwrap()
        .gws(&dims)
        .arg_img(&cl_source)
        .arg_img(&cl_dest);

    printlnc!(royal_blue: "\nRunning kernel (unrolled)...");
    printlnc!(white_bold: "image dims: {:?}", &dims);
    let start_time = time::get_time();

    kernel.enq().unwrap();
    print_elapsed("kernel enqueued", start_time);

    queue.finish();
    print_elapsed("queue finished", start_time);

    cl_dest.read(&mut result).enq().unwrap();
    queue.finish(); // necessary or you will end up with an empty saved image
    print_elapsed("read finished", start_time);

    result.save(&Path::new("result_unrolled.png")).unwrap();


    // ##################################################
    // #################### PATCHES #####################
    // ##################################################

    let patch_size = 32;

    // [NOTE]: The previous calculation created a global work size which was
    // larger than the bounds of the image buffer:
    //
    // let gws = ((dims.0 + patch_size - 1) / patch_size, (dims.1 + patch_size - 1) / patch_size);

    // The part of the image which fits neatly into `patch_size` squares.
    let gws_bulk = (dims.0 / patch_size, dims.1 / patch_size);

    let kernel_bulk = Kernel::new("rgb2gray_patches", &program, &queue).unwrap()
        .gws(&gws_bulk)
        .arg_scl(patch_size)
        .arg_img(&cl_source)
        .arg_img(&cl_dest);

    // [NOTE]: Corners will overlap and be processed twice.
    let gws_edges = (dims.0 % patch_size, dims.1 % patch_size);

    let kernel_edges = Kernel::new("rgb2gray_unrolled", &program, &queue).unwrap()
        .gws(&gws_edges)
        .gwo(&gws_bulk)
        // .arg_scl(patch_size)
        .arg_img(&cl_source)
        .arg_img(&cl_dest);


    printlnc!(royal_blue: "\nRunning kernels (patch bulk & patch edges)...");
    printlnc!(white_bold: "image dims: {:?}", &dims);
    printlnc!(white_bold: "bulk dims: {:?}", gws_bulk);
    printlnc!(white_bold: "edges dims: {:?}", gws_edges);

    let start_time = time::get_time();

    kernel_bulk.enq().unwrap();
    kernel_edges.enq().unwrap();
    print_elapsed("kernels enqueued", start_time);

    queue.finish();
    print_elapsed("queue finished", start_time);

    cl_dest.read(&mut result).enq().unwrap();
    queue.finish(); // necessary or you will end up with an empty saved image
    print_elapsed("read finished", start_time);

    result.save(&Path::new("result_patches.png")).unwrap();
}
