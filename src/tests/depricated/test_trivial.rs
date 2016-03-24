
const IDX: usize = 200007;
const ADDEND: f32 = 10.0;

/// Expanded version with explanations.
///
/// This function is identical to the following two in all but looks :) It
/// only differs from the first function in that it prints an 'original' value
/// (this was cut from the first just for simplicity).
///
/// Continue along to the next few functions after this to see a little bit
/// more about what's going on under the hood.
///
#[allow(dead_code)]
#[test]
fn main_explained() {
    use standard::ProQue;
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    // (1) Create an all-in-one context, program, command queue, and work /
    // buffer dimensions:
    let pro_que = ProQue::builder()
        .src(src)
        .dims([500000])
        .build().unwrap();   

    // (2) Create a `Buffer` with a built-in `Vec`:
    let mut buffer = pro_que.create_buffer::<f32>();

    // For printing purposes:
    let orig_val = buffer[IDX];

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = pro_que.create_kernel("add").expect("[FIXME]: HANDLE ME")
        .arg_buf(&buffer)
        .arg_scl(ADDEND);

    // (4) Run the kernel:
    kernel.enq().expect("[FIXME]: HANDLE ME!");

    // (5) Read results from the device into our buffer's built-in vector:
    buffer.fill_vec();

    // Print an element:
    println!("The value at index [{}] was '{}' and is now '{}'!", 
        IDX, orig_val, buffer[IDX]);
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
#[test]
fn main_exploded() {
    use flags;
    use standard::{Platform, Device, Context, Queue, Program, Buffer, Kernel};

    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims (compare to step 1 above).
    let platform = Platform::default();
    let device = Device::first(&platform);
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build().unwrap();
    let program = Program::builder()
        .devices(&[device.clone()])
        .src(src)
        .build(&context).unwrap();
    let queue = Queue::new(&context, device).unwrap();
    let dims = [500000];
    // [NOTE]: At this point we could manually assemble a ProQue by calling:
    // `ProQue::new(context, queue, program, Some(dims))`. One might want to
    // do this when only one program and queue are all that's needed. Wrapping
    // it up into a single struct makes passing it around much simpler.

    // (2) Create a `Buffer` with a built-in `Vec` (created separately here):
    // [NOTE]: If there were more than one dimension we'd use the product as
    // the length.
    let mut buffer_vec = vec![0.0f32; dims[0]];
    let buffer = Buffer::new(&queue, Some(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR),
        dims[0], Some(&buffer_vec));

    // For verification purposes:
    let orig_val = buffer_vec[IDX];

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = Kernel::new("add", &program, &queue).unwrap()
        .gws(&dims)
        .arg_buf(&buffer)
        .arg_scl(ADDEND);

    // (4) Run the kernel (default parameters shown for demonstration purposes):
    kernel.cmd()
        .queue(&queue)
        .gwo(kernel.get_gwo())
        .gws(&dims)
        .lws(kernel.get_lws())
        .ewait_opt(None)
        .enew_opt(None)
        .enq().unwrap();

    // (5) Read results from the device into our buffer's [no longer] built-in vector:
    buffer.cmd()
        .queue(&queue)
        .block(true)
        .offset(0)
        .read(&mut buffer_vec)
        .ewait_opt(None)
        .enew_opt(None)
        .enq().unwrap(); 
    
    // Print an element:
    println!("The value at index [{}] was '{}' and is now '{}'!", 
        IDX, orig_val, buffer_vec[IDX]);
}


/// Falling down the hole...
///
/// This version does the same thing as the others but instead using the
/// `core` module which sports an API equivalent to the OpenCL C API. If
/// you've used OpenCL before, this will look the most familiar to you.
///
#[allow(dead_code, unused_variables, unused_mut)]
#[test]
fn main_cored() {
    use std::ffi::CString;
    use core::{self, ContextProperties};
    use flags;
    use enums::KernelArg;

    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims..
    let platform_ids = core::get_platform_ids().unwrap();
    let platform_id = platform_ids[0];
    let device_ids = core::get_device_ids(&platform_id, None, None).unwrap();
    let device_id = device_ids[0];
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(&Some(context_properties), 
        &[device_id], None, None).unwrap();
    let src_cstring = CString::new(src).unwrap();
    let program = core::create_program_with_source(&context, &[src_cstring]).unwrap();
    core::build_program(&program, &[device_id], &CString::new("").unwrap(), 
        None, None).unwrap();
    let queue = core::create_command_queue(&context, &device_id).unwrap();
    let dims = [500000, 1, 1usize];

    // (2) Create a `Buffer` with a built-in `Vec` (created separately here):
    // Again, we're cheating on the length calculation.
    let mut buffer_vec = vec![0.0f32; dims[0]];
    let buffer = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE | 
        flags::MEM_COPY_HOST_PTR, dims[0], Some(&buffer_vec)).unwrap() };

    // For verification purposes:
    let orig_val = buffer_vec[IDX];

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = core::create_kernel(&program, "add").unwrap();
    core::set_kernel_arg(&kernel, 0, KernelArg::Mem::<f32>(&buffer)).unwrap();
    core::set_kernel_arg(&kernel, 1, KernelArg::Scalar(&ADDEND)).unwrap();

    // (4) Run the kernel (default parameters shown for elucidation purposes):
    core::enqueue_kernel(&queue, &kernel, 1, None, &dims, 
        None, None::<&core::EventList>, None).unwrap();

    // (5) Read results from the device into our buffer's [no longer] built-in vector:
     unsafe { core::enqueue_read_buffer(&queue, &buffer, true, 0, &mut buffer_vec, 
        None::<&core::EventList>, None).unwrap() };    

    // Print an element:
    println!("The value at index [{}] was '{}' and is now '{}'!", 
        IDX, orig_val, buffer_vec[IDX]);
}
