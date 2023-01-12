// const IDX: usize = 200007;
const ADDEND: f32 = 10.0;
const DATASET_SIZE: usize = 2 << 20;

use std::ffi::CString;

#[test]
fn buffer_copy_core() {
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let platform_id = crate::default_platform().unwrap();
    let device_ids = crate::get_device_ids(&platform_id, None, None).unwrap();
    let device = device_ids[0];
    let context_properties = crate::ContextProperties::new().platform(platform_id);
    let context = crate::create_context(Some(&context_properties), &[device], None, None).unwrap();

    let src_cstring = CString::new(src).unwrap();
    let program = crate::create_program_with_source(&context, &[src_cstring]).unwrap();
    // ::build_program(&program, Some(&[device]), &CString::new("").unwrap(),
    crate::build_program(
        &program,
        None::<&[()]>,
        &CString::new("").unwrap(),
        None,
        None,
    )
    .unwrap();
    let queue = crate::create_command_queue(&context, &device, Some(crate::QUEUE_PROFILING_ENABLE))
        .unwrap();
    let dims = [DATASET_SIZE, 1, 1usize];

    // Source buffer:
    let mut src_buffer_vec = vec![0.0f32; dims[0]];
    let src_buffer = unsafe {
        crate::create_buffer(
            &context,
            crate::MEM_READ_WRITE | crate::MEM_COPY_HOST_PTR,
            dims[0],
            Some(&src_buffer_vec),
        )
        .unwrap()
    };
    // Dst buffer
    let mut dst_buffer_vec = vec![0.0f32; dims[0]];
    let dst_buffer = unsafe {
        crate::create_buffer(
            &context,
            crate::MEM_READ_WRITE | crate::MEM_COPY_HOST_PTR,
            dims[0],
            Some(&dst_buffer_vec),
        )
        .unwrap()
    };

    // Kernel:
    let kernel = crate::create_kernel(&program, "add").unwrap();
    crate::set_kernel_arg(&kernel, 0, crate::ArgVal::mem(&src_buffer)).unwrap();
    crate::set_kernel_arg(&kernel, 1, crate::ArgVal::scalar(&ADDEND)).unwrap();

    // Run the kernel:
    unsafe {
        crate::enqueue_kernel(
            &queue,
            &kernel,
            1,
            None,
            &dims,
            None,
            None::<crate::Event>,
            None::<&mut crate::Event>,
        )
        .unwrap();
    }

    // Copy src_buffer to dst_buffer:
    let copy_range = (153, 150000);
    crate::enqueue_copy_buffer::<f32, _, _, _>(
        &queue,
        &src_buffer,
        &dst_buffer,
        copy_range.0,
        copy_range.0,
        copy_range.1 - copy_range.0,
        None::<crate::Event>,
        None::<&mut crate::Event>,
    )
    .unwrap();

    // Read results from src_buffer:
    unsafe {
        crate::enqueue_read_buffer(
            &queue,
            &src_buffer,
            true,
            0,
            &mut src_buffer_vec,
            None::<crate::Event>,
            None::<&mut crate::Event>,
        )
        .unwrap()
    };
    // Read results from dst_buffer:
    unsafe {
        crate::enqueue_read_buffer(
            &queue,
            &dst_buffer,
            true,
            0,
            &mut dst_buffer_vec,
            None::<crate::Event>,
            None::<&mut crate::Event>,
        )
        .unwrap()
    };

    for i in 0..dims[0] {
        assert_eq!(src_buffer_vec[i], ADDEND);

        if i >= copy_range.0 && i < copy_range.1 {
            assert_eq!(dst_buffer_vec[i], ADDEND);
        } else {
            assert!(
                dst_buffer_vec[i] == 0.0,
                "dst_vec: {}, idx: {}",
                dst_buffer_vec[i],
                i
            );
        }
    }
}
