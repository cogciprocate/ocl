
const IDX: usize = 200007;
const ADDEND: f32 = 10.0;
const DATASET_SIZE: usize = 2 << 20;

#[test]
fn buffer_copy_core() {
    use std::ffi::CString;
    use core::{self, ContextProperties};
    use flags;
    use enums::KernelArg;

    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let platform_ids = core::get_platform_ids().unwrap();
    let platform_id = platform_ids[0];
    let device_ids = core::get_device_ids(&platform_id, 
        Some(flags::DEVICE_TYPE_ALL), None).unwrap();
    let device_id = device_ids[0];
    let context_properties = ContextProperties::new().platform(platform_id);
    let context = core::create_context(&Some(context_properties), 
        &[device_id], None, None).unwrap();
    let src_cstring = CString::new(src).unwrap();
    let program = core::create_program_with_source(&context, &[src_cstring]).unwrap();
    core::build_program(&program, &[device_id], &CString::new("").unwrap(), 
        None, None).unwrap();
    let queue = core::create_command_queue(&context, &device_id).unwrap();
    let dims = [DATASET_SIZE, 1, 1usize];

    // Source buffer:
    let mut src_buffer_vec = vec![0.0f32; dims[0]];
    let src_buffer = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE | 
        flags::MEM_COPY_HOST_PTR, dims[0], Some(&src_buffer_vec)).unwrap() };
    // Dst buffer
    let mut dst_buffer_vec = vec![0.0f32; dims[0]];
    let dst_buffer = unsafe { core::create_buffer(&context, flags::MEM_READ_WRITE | 
        flags::MEM_COPY_HOST_PTR, dims[0], Some(&dst_buffer_vec)).unwrap() };

    // Kernel:
    let kernel = core::create_kernel(&program, "add").unwrap();
    core::set_kernel_arg(&kernel, 0, KernelArg::Mem::<f32>(&src_buffer)).unwrap();
    core::set_kernel_arg(&kernel, 1, KernelArg::Scalar(&ADDEND)).unwrap();

    // Run the kernel:
    core::enqueue_kernel(&queue, &kernel, 1, None, &dims, 
        None, None::<&core::EventList>, None).unwrap();

    // Copy src_buffer to dst_buffer:
    let copy_range = (153, 150000);
    core::enqueue_copy_buffer::<f32, _>(&queue, &src_buffer, &dst_buffer, 
        copy_range.0, copy_range.0, copy_range.1 - copy_range.0, None::<&core::EventList>,
        None).unwrap();

    // Read results from src_buffer:
    unsafe { core::enqueue_read_buffer(&queue, &src_buffer, true, 0, &mut src_buffer_vec, 
        None::<&core::EventList>, None).unwrap() };
    // Read results from dst_buffer:
    unsafe { core::enqueue_read_buffer(&queue, &dst_buffer, true, 0, &mut dst_buffer_vec, 
        None::<&core::EventList>, None).unwrap() };

    for i in 0..dims[0] {
        assert_eq!(src_buffer_vec[i], ADDEND);

        if i >= copy_range.0 && i < copy_range.1 {
            assert_eq!(dst_buffer_vec[i], ADDEND);
        } else {
            assert!(dst_buffer_vec[i] == 0.0, "dst_vec: {}, idx: {}", dst_buffer_vec[i], i);
        }
    }
}

#[test]
fn buffer_copy_standard() {
    use standard::ProQue;
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims([DATASET_SIZE])
        .build().unwrap();   

    let src_buffer = pro_que.create_buffer::<f32>().unwrap();
    let mut src_vec = vec![0.0f32; src_buffer.len()];
    let dst_buffer = pro_que.create_buffer::<f32>().unwrap();
    let mut dst_vec = vec![0.0f32; dst_buffer.len()];

    let kernel = pro_que.create_kernel("add").unwrap()
        .arg_buf(&src_buffer)
        .arg_scl(ADDEND);

    kernel.enq().expect("[FIXME]: HANDLE ME!");

    // Copy src to dst:
    let copy_range = (IDX, pro_que.dims()[0] - 100);
    src_buffer.cmd().copy(&dst_buffer, copy_range.0, copy_range.1 - copy_range.0).enq().unwrap();

    // Read both buffers from device.
    src_buffer.read(&mut src_vec).enq().unwrap();
    dst_buffer.read(&mut dst_vec).enq().unwrap();

    for i in 0..pro_que.dims()[0] {
        assert_eq!(src_vec[i], ADDEND);

        if i >= copy_range.0 && i < copy_range.1 {
            assert_eq!(dst_vec[i], ADDEND);
        } else {
            assert!(dst_vec[i] == 0.0, "dst_buf: {}, idx: {}", dst_vec[i], i);
        }
    }
}