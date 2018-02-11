extern crate ocl;
use ocl::ProQue;

fn trivial() -> ocl::Result<()> {
    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(1 << 20)
        .build()?;

    let buffer = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que.create_kernel("add")?
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    unsafe { kernel.enq()?; }

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
    Ok(())
}


/// Expanded version with explanations.
///
/// All four functions in this example are identical in all but appearance.
///
/// Continue along to the next few functions after this to see a little bit
/// more about what's going on under the hood.
///
#[allow(dead_code)]
fn trivial_explained() -> ocl::Result<()> {
    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    // (1) Create an all-in-one context, program, command queue, and work /
    // buffer dimensions:
    let pro_que = ProQue::builder()
        .src(src)
        .dims(1 << 20)
        .build()?;

    // (2) Create a `Buffer`:
    let buffer = pro_que.create_buffer::<f32>()?;

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = pro_que.create_kernel("add")?
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    // (4) Run the kernel:
    unsafe { kernel.enq()?; }

    // (5) Read results from the device into a vector:
    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq()?;

    // Print an element:
    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
    Ok(())
}


/// Exploded version! Boom.
///
/// What you saw above uses `ProQue` and other abstractions to greatly reduce
/// the amount of boilerplate and configuration necessary to do basic stuff.
/// Many tasks will require some more configuration and will necessitate
/// either doing away with `ProQue` all together or building it piece by
/// piece.
///
/// Queuing commands for kernels or to read/write from buffers and images also
/// generally requires some more control.
///
/// The following function performs the exact same steps that the above did,
/// with many of the convenience abstractions peeled away.
///
/// See the function below this to take things a step deeper...
///
#[allow(dead_code)]
fn trivial_exploded() -> ocl::Result<()> {
    use ocl::{flags, Platform, Device, Context, Queue, Program,
        Buffer, Kernel, /*Event, EventList*/};

    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims (compare to step 1 above).
    let platform = Platform::default();
    let device = Device::first(platform);
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;
    let program = Program::builder()
        .devices(device)
        .src(src)
        .build(&context)?;
    let queue = Queue::new(&context, device, None)?;
    let dims = 1 << 20;
    // [NOTE]: At this point we could manually assemble a ProQue by calling:
    // `ProQue::new(context, queue, program, Some(dims))`. One might want to
    // do this when only one program and queue are all that's needed. Wrapping
    // it up into a single struct makes passing it around simpler.

    // (2) Create a `Buffer`:
    let mut vec = vec![0.0f32; dims];
    let buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .dims(dims)
        .host_data(&vec)
        .build()?;

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = Kernel::new("add", &program)?
        .queue(queue.clone())
        .gws(dims)
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    // (4) Run the kernel (default parameters shown for demonstration purposes):
    unsafe {
        kernel.cmd()
            .queue(&queue)
            .gwo(kernel.get_gwo())
            .gws(dims)
            .lws(kernel.get_lws())
            .enq()?;
    }

    // (5) Read results from the device into a vector (`::block` not shown):
    buffer.cmd()
        .queue(&queue)
        .offset(0)
        .read(&mut vec)
        .enq()?;

    // Print an element:
    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
    Ok(())
}


/// Falling down the hole...
///
/// This version does the same thing as the others but instead using the
/// `core` module which sports an API equivalent to the OpenCL C API. If
/// you've used OpenCL before, this will look the most familiar to you.
///
/// All 'standard' types such as those used above, `Buffer`, `Kernel` etc,
/// make all of their calls to core just as in the function below...
///
#[allow(dead_code, unused_variables, unused_mut)]
fn trivial_cored() -> ocl::core::Result<()> {
    use std::ffi::CString;
    use ocl::{core, flags};
    use ocl::enums::KernelArg;
    use ocl::builders::ContextProperties;

    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims..
    let platform_id = core::default_platform()?;
    let device_ids = core::get_device_ids(&platform_id, None, None)?;
    let device_id = device_ids[0];
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(Some(&context_properties),
        &[device_id], None, None)?;
    let src_cstring = CString::new(src)?;
    let program = core::create_program_with_source(&context, &[src_cstring])?;
    core::build_program(&program, Some(&[device_id]), &CString::new("")?,
        None, None)?;
    let queue = core::create_command_queue(&context, &device_id, None)?;
    let dims = [1 << 20, 1, 1];

    // (2) Create a `Buffer`:
    let mut vec = vec![0.0f32; dims[0]];
    let buffer = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE |
        flags::MEM_COPY_HOST_PTR, dims[0], Some(&vec))? };

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = core::create_kernel(&program, "add")?;
    core::set_kernel_arg(&kernel, 0, KernelArg::Mem::<f32>(&buffer))?;
    core::set_kernel_arg(&kernel, 1, KernelArg::Scalar(10.0f32))?;

    // (4) Run the kernel:
    unsafe {
        core::enqueue_kernel(&queue, &kernel, 1, None, &dims,
            None, None::<core::Event>, None::<&mut core::Event>)?;
    }

    // (5) Read results from the device into a vector:
    unsafe { core::enqueue_read_buffer(&queue, &buffer, true, 0, &mut vec,
        None::<core::Event>, None::<&mut core::Event>)? };

    // Print an element:
    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
    Ok(())
}


fn main() {
    trivial().unwrap();
    trivial_explained().unwrap();
    trivial_exploded().unwrap();
    trivial_cored().unwrap();
}
