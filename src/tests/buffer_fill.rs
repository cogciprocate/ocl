use std::ffi::CString;

const DATASET_SIZE: usize = 2 << 20;

#[test]
fn fill() {
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let platform_id = ::default_platform().unwrap();
    let device_ids = ::get_device_ids(&platform_id, None, None).unwrap();
    let device = device_ids[0];
    let context_properties = ::ContextProperties::new().platform(platform_id);
    let context = ::create_context(Some(&context_properties),
        &[device], None, None).unwrap();

    let src_cstring = CString::new(src).unwrap();
    let program = ::create_program_with_source(&context, &[src_cstring]).unwrap();
    ::build_program(&program, &[device], &CString::new("").unwrap(),
        None, None).unwrap();
    let queue = ::create_command_queue(&context, &device, Some(::QUEUE_PROFILING_ENABLE))
        .unwrap();

    // let buffer = pro_que.create_buffer::<f32>().unwrap();
    let buffer = unsafe { ::create_buffer::<f32>(&context, ::MEM_READ_WRITE, DATASET_SIZE, None).unwrap() };

    // buffer.cmd().fill(5.0f32, None).enq().unwrap();
    ::enqueue_fill_buffer::<f32>(&queue, &buffer, 5.0f32, 0, DATASET_SIZE, None, None, None).unwrap();

    let mut vec = vec![0.0f32; DATASET_SIZE];
    // buffer.read(&mut vec).enq().unwrap();
    unsafe { ::enqueue_read_buffer::<f32>(&queue, &buffer, true, 0, &mut vec, None, None).unwrap() };

    assert!(vec.iter().all(|x| *x == 5.0f32));
    // for &ele in vec.iter() {
    //     assert_eq!(ele, 5.0f32);
    // }

    // let kernel = pro_que.create_kernel("add").unwrap()
    //     .arg_buf(&buffer)
    //     .arg_scl(10.0f32);
    //
    // kernel.enq().expect("[FIXME]: HANDLE ME!");
    //
    // let mut vec = vec![0.0f32; buffer.len()];
    // buffer.read(&mut vec).enq().unwrap();
    //
    // for &ele in vec.iter() {
    //     assert_eq!(ele, 15.0f32);
    // }
}
//
// #[test]
// fn fill_with_float4() {
//     let src = r#"
//         __kernel void add_float4(__global float4* buffer, float4 addend) {
//             buffer[get_global_id(0)] += addend;
//         }
//     "#;
//
//     let start_val = ::ClFloat4(9.0f32, 11.0f32, 14.0f32, 18.0f32);
//     // let start_val = ::ClFloat4(5.0f32, 5.0f32, 5.0f32, 5.0f32);
//     let addend = ::ClFloat4(10.0f32, 10.0f32, 10.0f32, 10.0f32);
//     // let final_val = ::ClFloat4(15.0f32, 15.0f32, 15.0f32, 15.0f32);
//     let final_val = start_val + addend;
//
//     let pro_que = ProQue::builder()
//         .src(src)
//         .dims([DATASET_SIZE])
//         .build().unwrap();
//
//     let buffer = pro_que.create_buffer::<::ClFloat4>().unwrap();
//
//     buffer.cmd().fill(start_val, None).enq().unwrap();
//
//     let mut vec = vec![start_val; buffer.len()];
//     buffer.read(&mut vec).enq().unwrap();
//
//     for &ele in vec.iter() {
//         assert_eq!(ele, start_val);
//     }
//
//     let kernel = pro_que.create_kernel("add_float4").unwrap()
//         .arg_buf(&buffer)
//         .arg_scl(addend);
//
//     kernel.enq().unwrap();
//
//     buffer.read(&mut vec).enq().unwrap();
//
//     for &ele in vec.iter() {
//         assert_eq!(ele, final_val);
//     }
// }
