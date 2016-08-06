//! [INCOMPLETE]: [TODO]: Test various types with an assortment of operations.
//!

const DATASET_SIZE: usize = 2 << 20;

#[test]
fn test_vector_types() {
    let src = r#"
        __kernel void add_float3(__global float4* buffer, float4 addend) {
            buffer[get_global_id(0)] += addend +
                (float4)(get_global_id(0), get_global_id(0), get_global_id(0), 0.0);
        }
    "#;

    let start_val = ClFloat3::new(9.0, 11.0, 14.0);
    let addend = ClFloat3::from([10.0, 10.0, 10.0f32]);
    let final_val = start_val + addend;

    let pro_que = ProQue::builder()
        .src(src)
        .dims([DATASET_SIZE])
        .build().unwrap();

    let buffer = pro_que.create_buffer::<ClFloat3>().unwrap();

    buffer.cmd().fill(start_val, None).enq().unwrap();

    let mut vec = vec![start_val; buffer.len()];
    buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, start_val);
    }

    let kernel = pro_que.create_kernel("add_float3").unwrap()
        .arg_buf(&buffer)
        .arg_scl(addend);

    kernel.enq().unwrap();

    buffer.read(&mut vec).enq().unwrap();

    let mut i = 0;
    for &ele in vec.iter() {
        assert_eq!(ele, final_val + ClFloat3::new(i as f32, i as f32, i as f32));
        i += 1;
    }
}
