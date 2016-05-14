//! [INCOMPLETE]: [TODO]: Test various types with an assortment of operations.
//! 

use standard::ProQue;
use aliases::cl_float3;

const DATASET_SIZE: usize = 2 << 20;

#[test]
fn test_vector_types() {
	let src = r#"
        __kernel void add(
        	__global float3* data,
        	__global int3* result)
    	{
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims([DATASET_SIZE])
        .build().unwrap();   
}