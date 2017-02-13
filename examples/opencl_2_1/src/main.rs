extern crate ocl;

use ocl::{flags, Platform, Device, Context, Queue, Program, Buffer, Kernel};

fn main() {

    // let src = r#"
    //     __kernel void add(__global float* buffer, float scalar) {
    //         buffer[get_global_id(0)] += scalar;
    //     }
    // "#;

    let il_src = Vec::<u8>::new();

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims (compare to step 1 above).
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

    let queue = Queue::new(&context, device, None).unwrap();
    let dims = [2 << 20];
    // [NOTE]: At this point we could manually assemble a ProQue by calling:
    // `ProQue::new(context, queue, program, Some(dims))`. One might want to
    // do this when only one program and queue are all that's needed. Wrapping
    // it up into a single struct makes passing it around much simpler.

    // (2) Create a `Buffer`:
    let mut vec = vec![0.0f32; dims[0]];
    let buffer = Buffer::<f32>::new(queue.clone(), Some(flags::MEM_READ_WRITE |
        flags::MEM_COPY_HOST_PTR), dims, Some(&vec)).unwrap();

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = Kernel::new("add", &program, queue.clone()).unwrap()
        .gws(&dims)
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    // (4) Run the kernel (default parameters shown for demonstration purposes):
    kernel.cmd()
        .queue(&queue)
        .gwo(kernel.get_gwo())
        .gws(&dims)
        .lws(kernel.get_lws())
        .ewait_opt(None)
        .enew_opt(None)
        .enq().unwrap();

    // (5) Read results from the device into a vector:
    buffer.cmd()
        .queue(&queue)
        .block(true)
        .offset(0)
        .read(&mut vec)
        .ewait_opt(None)
        .enew_opt(None)
        .enq().unwrap();

    // Print an element:
    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
}


