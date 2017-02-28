//! [INCOMPLETE]: [TODO]: Test various types with an assortment of operations.
//!

use standard::ProQue;
use prm::Int4;

const DATASET_SIZE: usize = 1 << 20;

#[test]
fn test_vector_types() {

    let src = r#"
        __kernel void add_int4(__global int4* in_buffer, int4 addend, __global int4* out_buffer) {
            uint idx = get_global_id(0);
            out_buffer[idx] = in_buffer[idx] + addend + (int4)(idx, idx, idx, idx);
        }
    "#;

    let start_val = Int4::new(9, 11, 14, 19);
    let addend = Int4::new(10, 10, 10, 10);
    let final_val = start_val + addend;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(DATASET_SIZE)
        .build().unwrap();

    let in_buffer = pro_que.create_buffer::<Int4>().unwrap();

    in_buffer.cmd().fill(start_val, None).enq().unwrap();

    let mut vec = vec![start_val; in_buffer.len()];
    in_buffer.read(&mut vec).enq().unwrap();

    for &ele in vec.iter() {
        assert_eq!(ele, start_val);
    }

    let out_buffer = pro_que.create_buffer::<Int4>().unwrap();

    let kernel = pro_que.create_kernel("add_int4").unwrap()
        .arg_buf(&in_buffer)
        .arg_vec(addend)
        .arg_buf(&out_buffer);

    kernel.enq().unwrap();

    out_buffer.read(&mut vec).enq().unwrap();

    let mut i = 0i32;
    for &ele in vec.iter() {
        assert_eq!(ele, final_val + Int4::new(i, i, i, i));
        i += 1;
    }
}
