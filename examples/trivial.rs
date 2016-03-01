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
}

////////// See the original file for more //////////

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

    // 1.) Create an all-in-one context, program, command queue, and work / buffer
    // dimensions:
    let pro_que = ProQue::builder()
        .src(src)
        .dims([500000])
        .build().unwrap();   

    // 2.) Create a `Buffer` with a built-in `Vec`:
    let mut buffer = pro_que.create_buffer::<f32>(true);

    let scalar = 10.0f32;
    let element_idx = 200057;
    let element_original_value = buffer[element_idx];

    // 3.) Create a kernel with arguments matching those in the source above:
    let kernel = pro_que.create_kernel("add")
        .arg_buf(&buffer)
        .arg_scl(scalar);

    // 4.) Run the kernel:
    kernel.enqueue();

    // 5.) Read results from the device into our buffer's built-in vector:
    buffer.fill_vec();

    let element_final_value = buffer[element_idx];
    assert!((element_final_value - (element_original_value + scalar)).abs() < 0.0001);
    println!("The value at index [{}] was '{}' and is now '{}'!", 
        element_idx, element_original_value, element_final_value);
}

/// Exploded version! Boom.
///
/// What you see above uses `ProQue` to greatly reduce the amount of
/// boilerplate and configuration necessary to do some basic stuff. Many
/// tasks will require some more configuration and will necessitate
/// Either doing away with `ProQue` all together, or building it
/// differently.
///
/// This function performs the exact same steps that the above does, with
/// some of the convenience abstractions removed.
///
/// See the function below this one to take things even a step deeper...
///
#[allow(dead_code)]
fn main_exploded() {
    use ocl::{Platform, Device, Context, DeviceSpecifier, Queue, Program, Buffer, Kernel};

    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    // 1.) Define which platform and device(s) to use. Create a context,
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

    // 2.) Create a `Buffer` with a built-in `Vec`:
    let mut buffer = Buffer::<f32>::with_vec(&dims, &queue);

    let scalar = 10.0f32;
    let element_idx = 200057;
    let element_original_value = buffer[element_idx];

    // 3.) Create a kernel with arguments matching those in the source above:
    let kernel = Kernel::new("add", &program, &queue).unwrap()
        .gws(&dims)
        .arg_buf(&buffer)
        .arg_scl(scalar)
    ;
    // 4.) Run the kernel:
    kernel.enqueue();

    // 5.) Read results from the device into our buffer's built-in vector:
    buffer.fill_vec();

    let element_final_value = buffer[element_idx];
    assert!((element_final_value - (element_original_value + scalar)).abs() < 0.0001);
    println!("The value at index [{}] was '{}' and is now '{}'!", 
        element_idx, element_original_value, element_final_value);
}

/// Falling down the hole...
///
/// This version does the same thing as the others but instead using the `core`
/// module which sports an API equivalent to the OpenCL C API. If you've used
/// OpenCL before, this will look the most familiar to you.
///
#[allow(dead_code)]
fn main_core() {

}
