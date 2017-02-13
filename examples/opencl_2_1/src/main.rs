#![allow(unused_imports, unused_variables)]

extern crate ocl;

use ocl::{flags, Platform, Device, Context, Queue, Program, Buffer, Kernel};

#[cfg(feature = "opencl_version_2_1")]
static PLATFORM_NAME: &'static str = "Experimental OpenCL 2.1 CPU Only Platform";


fn main() {
    // let src = r#"
    //     __kernel void add(__global float* buffer, float scalar) {
    //         buffer[get_global_id(0)] += scalar;
    //     }
    // "#;

    let il_src: Vec<u8> = vec![
    // Magic number.           Version number: 1.0.
    0x03, 0x02, 0x23, 0x07,    0x00, 0x00, 0x01, 0x00,
    // Generator number: 0.    Bound: 0.
    0x00, 0x00, 0x00, 0x00,    0x00, 0x00, 0x00, 0x00,
    // Reserved word: 0.
    0x00, 0x00, 0x00, 0x00,
    // OpMemoryModel.          Logical.
    0x0e, 0x00, 0x03, 0x00,    0x00, 0x00, 0x00, 0x00,
    // GLSL450.
    0x01, 0x00, 0x00, 0x00];

    #[cfg(feature = "opencl_version_2_1")]
    let platform = Platform::list().into_iter().find(|plat| plat.name() == PLATFORM_NAME)
        .unwrap_or(Platform::default());

    #[cfg(not(feature = "opencl_version_2_1"))]
    let platform = Platform::default();

    let device = Device::first(platform);
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build().unwrap();

    // let program = Program::builder()
    //     .devices(device)
    //     .src(src)
    //     .build(&context).unwrap();
    let program = Program::builder()
        .devices(device)
        .il(il_src)
        .build(&context).unwrap();


    // let queue = Queue::new(&context, device, None).unwrap();
    // let dims = [2 << 20];

    // let mut vec = vec![0.0f32; dims[0]];
    // let buffer = Buffer::<f32>::new(queue.clone(), Some(flags::MEM_READ_WRITE |
    //     flags::MEM_COPY_HOST_PTR), dims, Some(&vec)).unwrap();

    // let kernel = Kernel::new("add", &program, queue.clone()).unwrap()
    //     .gws(&dims)
    //     .arg_buf(&buffer)
    //     .arg_scl(10.0f32);

    // kernel.cmd()
    //     .queue(&queue)
    //     .gwo(kernel.get_gwo())
    //     .gws(&dims)
    //     .lws(kernel.get_lws())
    //     .ewait_opt(None)
    //     .enew_opt(None)
    //     .enq().unwrap();

    // buffer.cmd()
    //     .queue(&queue)
    //     .block(true)
    //     .offset(0)
    //     .read(&mut vec)
    //     .ewait_opt(None)
    //     .enew_opt(None)
    //     .enq().unwrap();

    // println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
}


