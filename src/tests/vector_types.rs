//! [INCOMPLETE]: [TODO]: Test various types with an assortment of operations.
//!

// use std::fmt::Debug;
// use std::ops::Add;
use std::ffi::CString;
use ::{/*OclPrm, OclScl, */OclVec, Kernel, Context, CommandQueue, Mem};
// use ::types::vectors::Splat;
use tests::{get_available_contexts};

const DATASET_SIZE: usize = 1 << 14;
const DIMS: [usize; 3] = [DATASET_SIZE, 1, 1usize];

fn kernel<V: OclVec>(context: &Context, src: &str, buffer: &Mem, addend: V) -> Kernel {
    // Create program:
    let src_cstring = CString::new(src).unwrap();
    let program = ::create_program_with_source(&context, &[src_cstring]).unwrap();
    ::build_program(&program, None::<&[()]>, &CString::new("").unwrap(),
        None, None).unwrap();

    // Create kernel:
    let kernel = ::create_kernel(&program, "add").unwrap();
    ::set_kernel_arg(&kernel, 0, ::KernelArg::Mem::<V>(&buffer)).unwrap();
    ::set_kernel_arg(&kernel, 1, ::KernelArg::Vector(addend)).unwrap();

    kernel
}

fn vec_buf<V: OclVec>(context: &Context, len: usize, start_val: V) -> (Vec<V>, Mem) {
    let vec = vec![start_val; len];
    let buf = unsafe { ::create_buffer(context, ::MEM_READ_WRITE |
        ::MEM_COPY_HOST_PTR, len, Some(&vec)).unwrap() };

    (vec, buf)
}

fn create_enqueue_verify<V>(context: &Context, queue: &CommandQueue,
        src: &str, start_val: V, addend: V)
    // where U: OclScl, V: OclVec<Scalar = U> + Splat<Scalar = U> + Add
    where V: OclVec/*, <V as Add>::Output: Debug*/

{
        // Create vec and buffer:
    let (mut vec, buf) = vec_buf(context, DATASET_SIZE, start_val);
    let kernel = kernel(context, src, &buf, addend);

    // Enqueue kernel:
    ::enqueue_kernel(&queue, &kernel, 1, None, &DIMS,
        None, None::<::Event>, None::<&mut ::Event>).unwrap();

    // Read from buffer:
    unsafe { ::enqueue_read_buffer(&queue, &buf, true, 0, &mut vec,
        None::<::Event>, None::<&mut ::Event>).unwrap() };

    let mut iter_v = V::zero();
    // Verify results:
    for &ele in vec.iter() {
        let final_val = start_val + addend + iter_v;
        assert_eq!(ele, final_val);
        iter_v = iter_v + V::one()
    }
}

fn add_float2(context: &Context, queue: &CommandQueue) {
    use ClFloat2;

    let src = r#"
        __kernel void add(__global float2* buffer, float2 addend) {
            int idx = get_global_id(0);

            buffer[idx] += addend +
                (float2)(idx, idx);
        }
    "#;

    let start_val = ClFloat2::new(14.0, 1.0);
    let addend = ClFloat2::splat(10.0f32);

    create_enqueue_verify(context, queue, src, start_val, addend);
}

fn add_float3(context: &Context, queue: &CommandQueue) {
    use ClFloat3;

    let src = r#"
        __kernel void add(__global float3* buffer, float3 addend) {
            int idx = get_global_id(0);

            buffer[idx] += addend +
                (float3)(idx, idx, idx);
        }
    "#;

    let start_val = ClFloat3::new(14.0, 9.0, 1.0);
    let addend = ClFloat3::splat(10.0f32);

    create_enqueue_verify(context, queue, src, start_val, addend);
}


fn add_float4(context: &Context, queue: &CommandQueue) {
    use ClFloat4;

    let src = r#"
        __kernel void add(__global float4* buffer, float4 addend) {
            int idx = get_global_id(0);

            buffer[idx] += addend +
                (float4)(idx, idx, idx, idx);
        }
    "#;

    let start_val = ClFloat4::new(9.0, 11.0, 14.0, 1.0);
    let addend = ClFloat4::from([10.0, 10.0, 10.0, 10.0f32]);

    create_enqueue_verify(context, queue, src, start_val, addend);
}

fn add_float16(context: &Context, queue: &CommandQueue) {
    use ClFloat16;

    let src = r#"
        __kernel void add(__global float16* buffer, float16 addend) {
            int idx = get_global_id(0);

            buffer[idx] += addend +
                (float16)(idx);
        }
    "#;

    let start_val = ClFloat16::new(9.0, 11.0, 14.0, 1.0, 9.0, 11.0, 14.0, 1.0,
        9.0, 11.0, 14.0, 1.0, 9.0, 11.0, 14.0, 1.0);
    let addend = ClFloat16::from([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0f32]);

    create_enqueue_verify(context, queue, src, start_val, addend);
}

fn add_int2(context: &Context, queue: &CommandQueue) {
    use ClInt2;

    let src = r#"
        __kernel void add(__global int2* buffer, int2 addend) {
            int idx = get_global_id(0);

            buffer[idx] += addend +
                (int2)(idx);
        }
    "#;

    let start_val = ClInt2::new(14, 1);
    let addend = ClInt2::from([10, 10i32]);

    create_enqueue_verify(context, queue, src, start_val, addend);
}

fn add_int3(context: &Context, queue: &CommandQueue) {
    use ClInt3;

    let src = r#"
        __kernel void add(__global int3* buffer, int3 addend) {
            int idx = get_global_id(0);

            buffer[idx] += addend +
                (int3)(idx);
        }
    "#;

    let start_val = ClInt3::new(14, 1, 15);
    let addend = ClInt3::from([10, 10, 10i32]);

    create_enqueue_verify(context, queue, src, start_val, addend);
}

fn add_int4(context: &Context, queue: &CommandQueue) {
    use ClInt4;

    let src = r#"
        __kernel void add(__global int4* buffer, int4 addend) {
            int idx = get_global_id(0);

            buffer[idx] += addend +
                (int4)(idx);
        }
    "#;

    let start_val = ClInt4::new(9, 11, 14, 1);
    let addend = ClInt4::from([10, 10, 10, 10i32]);

    create_enqueue_verify(context, queue, src, start_val, addend);
}

fn add_int16(context: &Context, queue: &CommandQueue) {
    use ClInt16;

    let src = r#"
        __kernel void add(__global int16* buffer, int16 addend) {
            int idx = get_global_id(0);

            buffer[idx] += addend +
                (int16)(idx);
        }
    "#;

    let start_val = ClInt16::new(9, 11, 14, 1, 9, 11, 14, 1, 9, 11, 14, 1, 9, 11, 14, 1);
    let addend = ClInt16::from([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10i32]);

    create_enqueue_verify(context, queue, src, start_val, addend);
}

// fn add_char3(context: &Context, queue: &CommandQueue) {
//     use ClChar3;

//     let src = r#"
//         __kernel void add(__global char3* buffer, char3 addend) {
//             int idx = get_global_id(0);

//             buffer[idx] += addend +
//                 (char3)(idx);
//         }
//     "#;

//     let start_val = ClChar3::new(9, 11, 14);
//     let addend = ClChar3::from([10, 10, 10i8]);

//     create_enqueue_verify(context, queue, src, start_val, addend);
// }

// fn add_char16(context: &Context, queue: &CommandQueue) {
//     use ClChar16;

//     let src = r#"
//         __kernel void add(__global char16* buffer, char16 addend) {
//             int idx = get_global_id(0);

//             buffer[idx] += addend +
//                 (char16)(idx);
//         }
//     "#;

//     let start_val = ClChar16::new(9, 11, 14, 1, 9, 11, 14, 1, 9, 11, 14, 1, 9, 11, 14, 1);
//     let addend = ClChar16::from([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10i8]);

//     create_enqueue_verify(context, queue, src, start_val, addend);
// }


#[test]
fn test_vector_types() {
    for (_, device, ref context) in get_available_contexts() {
        // Create queue:
        let queue = ::create_command_queue(context, &device, None).unwrap();

        add_float2(context, &queue);
        add_float3(context, &queue);
        add_float4(context, &queue);
        add_float16(context, &queue);
        add_int2(context, &queue);
        add_int3(context, &queue);
        add_int4(context, &queue);
        add_int16(context, &queue);

        // [TODO]: Implement overflowing adds so that these don't
        // overflow:
        // add_char3(context, &queue);
        // add_char16(context, &queue);
    }
}
