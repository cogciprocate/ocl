extern crate ocl;
use ocl::ProQue;

fn main() {
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims([500000])
        .build().unwrap();   

    let mut buffer = pro_que.create_buffer::<f32>(true);

    let kernel = pro_que.create_kernel("add")
        .arg_buf(&buffer)
        .arg_scl(100.0f32);

    kernel.enqueue();
    buffer.fill_vec();

    println!("The buffer element at [{}] is '{}'", 200057, buffer[200057]);

    main_explained();
    main_exploded();
    main_cored();

    ////////// See the original file for more //////////
}


/// Expanded version.
///
/// Explanations and some checking. Add `main_explained_and_checked()` to the
/// bottom of `main()` above to run it.
///
#[allow(dead_code)]
fn main_explained() {
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    // (1) Create an all-in-one context, program, command queue, and work / buffer
    // dimensions:
    let pro_que = ProQue::builder()
        .src(src)
        .dims([500000])
        .build().unwrap();   

    // (2) Create a `Buffer` with a built-in `Vec`:
    let mut buffer = pro_que.create_buffer::<f32>(true);

    // For verification purposes:
    let (addend, rand_idx, orig_val ) = (100.0f32, 200057, buffer[200057]);

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = pro_que.create_kernel("add")
        .arg_buf(&buffer)
        .arg_scl(addend);

    // (4) Run the kernel:
    kernel.enqueue();

    // (5) Read results from the device into our buffer's built-in vector:
    buffer.fill_vec();

    // Print an element:
    let final_val = buffer[rand_idx];
    println!("The value at index [{}] was '{}' and is now '{}'!", 
        rand_idx, orig_val, final_val);
}


/// Exploded version! Boom.
///
/// What you saw above uses `ProQue` and other abstractions to greatly reduce
/// the amount of boilerplate and configuration necessary to do basic stuff.
/// Many tasks will require some more configuration and will necessitate either
/// doing away with `ProQue` all together or building it differently. 
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
fn main_exploded() {
    use ocl::{Platform, Device, Context, DeviceSpecifier, Queue, Program,
        Buffer, Kernel, SimpleDims};

    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    // (1) Define which platform and device(s) to use. Create a context,
    // queue, and program then define some dims (compare to step 1 above).
    let platform = Platform::first();
    let device = Device::first(&platform);
    let context = Context::builder()
        .platform(platform)
        .devices(DeviceSpecifier::Single(device.clone()))
        .build().unwrap();
    let queue = Queue::new(&context, device).unwrap();
    let program = Program::builder()
        .devices(&[device.clone()])
        .src(src)
        .build(&context).unwrap();
    let dims = [500000];

    // (2) Create a `Buffer` with a built-in `Vec` (created separately here):
    let physical_len = SimpleDims::from(&dims).padded_len(device.max_wg_size());
    let mut buffer_vec = vec![0.0f32; physical_len];
    let buffer = unsafe { Buffer::new_unchecked(
        ocl::flags::MEM_READ_WRITE | ocl::flags::MEM_COPY_HOST_PTR,
        physical_len, Some(&buffer_vec), &queue) };

    // For verification purposes:
    let (addend, rand_idx, orig_val) = (100.0f32, 200057, buffer_vec[200057]);

    // (3) Create a kernel with arguments matching those in the source above:
    let kernel = Kernel::new("add", &program, &queue).unwrap()
        .gws(&dims)
        .arg_buf(&buffer)
        .arg_scl(addend);

    // (4) Run the kernel (default parameters shown for elucidation purposes):
    kernel.cmd()
        .queue(&queue)
        .gwo(kernel.get_gwo())
        .gws(&dims)
        .lws(kernel.get_lws())
        .wait_opt(None)
        .dest_opt(None)
        .enq().unwrap();

    // (5) Read results from the device into our buffer's [no longer] built-in vector:
    // unsafe { buffer.enqueue_read(Some(&queue), true, 0, &mut buffer_vec, None, None).unwrap(); }

    // unsafe { buffer.cmd().read(&mut buffer_vec).enq().unwrap(); }
    unsafe { buffer.cmd().enq().unwrap(); }

    // Print an element:
    let final_val = buffer_vec[rand_idx];
    println!("The value at index [{}] was '{}' and is now '{}'!", 
        rand_idx, orig_val, final_val);
}


/// Falling down the hole...
///
/// This version does the same thing as the others but instead using the `core`
/// module which sports an API equivalent to the OpenCL C API. If you've used
/// OpenCL before, this will look the most familiar to you.
///
#[allow(dead_code)]
fn main_cored() {

}
