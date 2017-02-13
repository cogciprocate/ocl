//! [INCOMPLETE]: [TODO]: Test various types with an assortment of operations.
//!

use standard::ProQue;
use aliases::ClFloat4;

const DATASET_SIZE: usize = 2 << 20;

#[test]
fn test_vector_types() {

    let src = r#"
        __kernel void add_float3(__global float4* in_buffer, float4 addend, __global float4* out_buffer) {
            uint idx = get_global_id(0);
            out_buffer[idx] = in_buffer[idx] + addend + (float4)(idx, idx, idx, idx);
        }
    "#;

    let start_val = ClFloat4::new(9., 11., 14., 19.);
    let addend = ClFloat4::new(10., 10., 10., 10.0f32);
    let final_val = start_val + addend;

    let pro_que = ProQue::builder()
        .src(src)
        .dims([DATASET_SIZE])
        .build().unwrap();

    let in_buffer = pro_que.create_buffer::<ClFloat4>().unwrap();

    in_buffer.cmd().fill(start_val, None).enq().unwrap();

    let mut vec = vec![start_val; in_buffer.len()];
    in_buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, start_val);
    }

    let out_buffer = pro_que.create_buffer::<ClFloat4>().unwrap();

    let kernel = pro_que.create_kernel("add_float3").unwrap()
        .arg_buf(&in_buffer)
        .arg_vec(addend)
        .arg_buf(&out_buffer);

    kernel.enq().unwrap();

    out_buffer.read(&mut vec).enq().unwrap();

    let mut i = 0;
    for &ele in vec.iter() {
        let i_float = i as f32;
        assert_eq!(ele, final_val + ClFloat4::new(i_float, i_float, i_float, i_float));
        i += 1;
    }
}
