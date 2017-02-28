// extern crate ocl;
use standard::ProQue;

const DATASET_SIZE: usize = 1 << 20;

#[test]
fn fill() {
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(DATASET_SIZE)
        .build().unwrap();

    let buffer = pro_que.create_buffer::<f32>().unwrap();

    buffer.cmd().fill(5.0f32, None).enq().unwrap();

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, 5.0f32);
    }

    let kernel = pro_que.create_kernel("add").unwrap()
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    kernel.enq().expect("[FIXME]: HANDLE ME!");

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, 15.0f32);
    }
}

#[test]
fn fill_with_float4() {
    use aliases::Float4;

    let src = r#"
        __kernel void add_float4(__global float4* buffer, float4 addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let start_val = Float4::new(9.0f32, 11.0f32, 14.0f32, 18.0f32);
    // let start_val = Float4::new(5.0f32, 5.0f32, 5.0f32, 5.0f32);
    let addend = Float4::new(10.0f32, 10.0f32, 10.0f32, 10.0f32);
    // let final_val = Float4::new(15.0f32, 15.0f32, 15.0f32, 15.0f32);
    let final_val = start_val + addend;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(DATASET_SIZE)
        .build().unwrap();

    let buffer = pro_que.create_buffer::<Float4>().unwrap();

    buffer.cmd().fill(start_val, None).enq().unwrap();

    let mut vec = vec![start_val; buffer.len()];
    buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, start_val);
    }

    let kernel = pro_que.create_kernel("add_float4").unwrap()
        .arg_buf(&buffer)
        .arg_scl(addend);

    kernel.enq().unwrap();

    buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, final_val);
    }
}
