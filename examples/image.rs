//!
//! [UNIMPLEMENTED][WORK IN PROGRESS]
//!

#![allow(unused_imports, unused_variables, dead_code)]

extern crate ocl;
use ocl::{SimpleDims, Context, Queue, DeviceSpecifier, Image, Program, Kernel, ImageFormat,
    ImageChannelOrder, ImageChannelDataType, MemObjectType};

static KERNEL_SRC: &'static str = r#"
    __constant sampler_t sampler = 
        CLK_NORMALIZED_COORDS_FALSE | 
        CLK_ADDRESS_NONE | 
        CLK_FILTER_NEAREST;

    __kernel void addem(
                read_only image2d_t src_image,
                write_only image2d_t dst_image)
    {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));

        float4 pixel = read_imagef(src_image, sampler, coord);

        pixel += (float4)(-1.0, -0.50, -0.00, 0.50);

        write_imagef(dst_image, coord, pixel);
    }
"#;

#[allow(unused_variables)]
fn main() {
    let context = Context::builder().devices(DeviceSpecifier::Index(0)).build().unwrap();
    let device = context.devices()[0].clone();
    let queue = Queue::new(&context, Some(device.clone()));

    let img_formats = Image::supported_formats(&context, ocl::MEM_READ_WRITE, 
        ocl::MemObjectType::Image2d).unwrap();

    println!("Image Formats Avaliable: {}.", img_formats.len());
    // println!("Image Formats: {:#?}.", img_formats);

    let dims = SimpleDims::Two(200, 200);

    // Width * Height * Image Channel Count * Image Channel Size:
    let image_bytes = 200 * 200 * 4 * 1;

    let mut data: Vec<i8> = (0..image_bytes).map(|_| 64).collect();

    let src_image = Image::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::SnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::MEM_READ_ONLY | ocl::MEM_HOST_WRITE_ONLY | ocl::MEM_COPY_HOST_PTR)
        .build_with_data(&queue, &data).unwrap();

    let dst_image = Image::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::SnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::MEM_WRITE_ONLY | ocl::MEM_HOST_READ_ONLY | ocl::MEM_COPY_HOST_PTR)
        .build_with_data(&queue, &data).unwrap();

    println!("{:#}", src_image);

    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(vec![device.clone()])
        .build(&context).unwrap();

    let kernel = Kernel::new("addem", &program, &queue, dims.clone()).unwrap()
        .arg_img(&src_image)
        .arg_img(&dst_image);

    println!("Kernel Info: {:#}", &kernel);

    kernel.enqueue();

    dst_image.read(&mut data).unwrap();

    println!("data: [0..125]: {:?}", &data[0..125]);

}
