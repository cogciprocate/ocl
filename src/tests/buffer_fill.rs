// extern crate ocl;
use standard::ProQue;

const DATASET_SIZE: usize = 2 << 20;

#[test]
fn fill() {
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims([DATASET_SIZE])
        .build().unwrap();   

    let buffer = pro_que.create_buffer::<f32>().unwrap();

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, 0.0f32);
    }

    let kernel = pro_que.create_kernel("add").unwrap()
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    kernel.enq().expect("[FIXME]: HANDLE ME!");

    buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, 10.0f32);
    }
}