//! Tests for the rectangularly shaped buffer operations: read, write, and copy.
//!
//! Runs both the core function and the 'standard' method call for each.

use core;
// use flags;
use standard::{ProQue, Buffer};
use tests;

const ADDEND: f32 = 10.0;
const DIMS: [usize; 3] = [16, 16, 16];
const TEST_ITERS: i32 = 220;

#[test]
fn buffer_ops_rect() {
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            uint idx = (get_global_id(0) * get_global_size(1) * get_global_size(2)) +
                (get_global_id(1) * get_global_size(2)) +
                get_global_id(2);

            buffer[idx] += addend;
        }

        __kernel void eq(__global float* buffer, float val) {
            uint idx = (get_global_id(0) * get_global_size(1) * get_global_size(2)) +
                (get_global_id(1) * get_global_size(2)) +
                get_global_id(2);

            buffer[idx] = val;
        }
    "#;

    let proque = ProQue::builder()
        .src(src)
        .dims(DIMS)
        .build().unwrap();

    // SRC_BUFFER:
    let mut vec = vec![0.0f32; proque.dims().to_len()];
    // let buf = unsafe { Buffer::new_unchecked(
    //     flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR,
    //     proque.dims().to_len().unwrap(), Some(&vec), proque.queue()) };
    // let buf = Buffer::new(proque.queue().clone(), Some(core::MEM_READ_WRITE |
    //     core::MEM_COPY_HOST_PTR), proque.dims().clone(), Some(&vec), None).unwrap();

    let buf = Buffer::builder()
        .queue(proque.queue().clone())
        .flags(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR)
        .dims(proque.dims().clone())
        .host_data(&vec)
        .build().unwrap();

    let kernel_add = proque.create_kernel("add").unwrap()
        .arg_buf(&buf)
        .arg_scl(ADDEND);


    //========================================================================
    //========================================================================
    //============================ Warm Up Run ===============================
    //========================================================================
    //========================================================================
    // Make sure that pro_que's dims are correct:
    let dims = proque.dims().to_lens().unwrap();
    assert_eq!(DIMS, dims);

    // Verify buffer and vector lengths:
    let len = proque.dims().to_len();
    assert_eq!(buf.len(), len);
    assert_eq!(vec.len(), len);

    // KERNEL RUN #1 -- make sure everything's working normally:
    kernel_add.enq().expect("[FIXME]: HANDLE ME!");
    let mut ttl_runs = 1i32;

    // READ AND VERIFY #1 (LINEAR):
    buf.read(&mut vec).enq().unwrap();

    for idx in 0..proque.dims().to_len() {
        // DEBUG:
        // print!("[{:02}]", vec[i]);
        // if i % 20 == 19 { print!("\n"); }
        assert!(vec[idx] == ADDEND * ttl_runs as f32,
            "vec[{}]: {}", idx, vec[idx]);
    }

    print!("\n");

    // Warm up the verify function:
    tests::verify_vec_rect([0, 0, 0], dims, ADDEND * ttl_runs as f32,
        ADDEND * (ttl_runs - 1) as f32, dims, 1, &vec, ttl_runs, false).unwrap();

    //========================================================================
    //========================================================================
    //=============================== Read ===================================
    //========================================================================
    //========================================================================
    // Buffer origin doesn't matter since it's all the same value host side:
    let buf_origin = [0, 0, 0];
    // Lengths of the two non-major dimensions.
    let row_pitch = dims[0];
    let slc_pitch = dims[0] * dims[1];

    for _ in 0..TEST_ITERS {
        // Generate a random size region and origin point:
        let (vec_origin, read_region) = tests::gen_region_origin(&dims);

        //====================================================================
        //=============== `core::enqueue_read_buffer_rect()` =================
        //====================================================================
        // Reset vec:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf, true,
            [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
            &mut vec, None::<core::Event>, None::<&mut core::Event>).unwrap(); }

        // Run kernel:
        kernel_add.enq().expect("[FIXME]: HANDLE ME!");
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let old_val = ADDEND * (ttl_runs - 1) as f32;

        // Read from the random region into our vec:
        unsafe {
            core::enqueue_read_buffer_rect(proque.queue(), &buf, true, buf_origin, vec_origin,
                read_region.clone(), row_pitch, slc_pitch, row_pitch, slc_pitch, &mut vec,
                None::<core::Event>, None::<&mut core::Event>).unwrap();
        }

        // Verify:
        tests::verify_vec_rect(vec_origin, read_region, cur_val, old_val,
            dims, 1, &vec, ttl_runs, false).unwrap();

        //====================================================================
        //================== `Buffer::cmd().read().rect()` ===================
        //====================================================================
        // Reset vec:
        buf.cmd().read(&mut vec).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
            row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();

        // Run kernel:
        kernel_add.enq().expect("[FIXME]: HANDLE ME!");
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let old_val = ADDEND * (ttl_runs - 1) as f32;

        // Read from the random region into our vec:
        buf.cmd().read(&mut vec).rect(buf_origin, vec_origin, read_region.clone(), row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();

        // Verify:
        tests::verify_vec_rect(vec_origin, read_region, cur_val, old_val,
            dims, 1, &vec, ttl_runs, false).unwrap();
    }

    //========================================================================
    //========================================================================
    //=============================== Write ==================================
    //========================================================================
    //========================================================================
    // Prepare a kernel which will write a single value to the entire buffer
    // and which can be updated on each run (to act as a 'reset').
    let mut kernel_eq = proque.create_kernel("eq").unwrap()
    .arg_buf_named("buf", Some(&buf))
    .arg_scl_named("val", Some(0.0f32));

    // Vector origin doesn't matter for this:
    let vec_origin = [0, 0, 0];
    // Lengths of the two non-major dimensions.
    let row_pitch = dims[0];
    let slc_pitch = dims[0] * dims[1];

    // Reset kernel runs count:
    ttl_runs = 0;

    for _ in 0..TEST_ITERS {
        // Generate a random size region and origin point. For the write test
        // it's the buf origin we care about, not the vec.
        let (buf_origin, read_region) = tests::gen_region_origin(&dims);

        //====================================================================
        //=============== `core::enqueue_write_buffer_rect()` ================
        //====================================================================
        // Set up values. Device buffer will now be init'd one step behind host vec.
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let nxt_val = ADDEND * (ttl_runs + 1) as f32;
        kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enq().expect("[FIXME]: HANDLE ME!");

        // Write `next_val` to all of `vec`. This will be our 'in-region' value:
        for ele in vec.iter_mut() { *ele = nxt_val }

        // Write to the random region:
        core::enqueue_write_buffer_rect(proque.queue(), &buf, false,
            buf_origin, vec_origin, read_region.clone(), row_pitch, slc_pitch,
            row_pitch, slc_pitch, &vec, None::<core::Event>, None::<&mut core::Event>).unwrap();
        // Read the entire buffer back into the vector:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf, true,
            [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
            &mut vec, None::<core::Event>, None::<&mut core::Event>).unwrap(); }
        // Verify that our random region was in fact written correctly:
        tests::verify_vec_rect(buf_origin, read_region, nxt_val, cur_val,
            dims, 1, &vec, ttl_runs, true).unwrap();

        //====================================================================
        //================= `Buffer::cmd().write().rect()` ===================
        //====================================================================
        // Set up values. Device buffer will now be init'd one step behind host vec.
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let nxt_val = ADDEND * (ttl_runs + 1) as f32;
        kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enq().expect("[FIXME]: HANDLE ME!");

        // Write `next_val` to all of `vec`. This will be our 'in-region' value:
        for ele in vec.iter_mut() { *ele = nxt_val }

        // Write to the random region:
        buf.cmd().write(&vec).rect(buf_origin, vec_origin, read_region.clone(), row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();
        // Read the entire buffer back into the vector:
        buf.cmd().read(&mut vec).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
            row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();
        // Verify that our random region was in fact written correctly:
        tests::verify_vec_rect(buf_origin, read_region, nxt_val, cur_val,
            dims, 1, &vec, ttl_runs, true).unwrap();
    }

    //========================================================================
    //========================================================================
    //================================ Copy ==================================
    //========================================================================
    //========================================================================
    // Source Buffer:
    let mut vec_src = vec![0.0f32; proque.dims().to_len()];
    // let buf_src = unsafe { Buffer::new_unchecked(
    //     flags::MEM_READ_ONLY | flags::MEM_HOST_WRITE_ONLY | flags::MEM_COPY_HOST_PTR,
    //     proque.dims().to_len().unwrap(), Some(&vec_src), proque.queue()) };
    // let buf_src = Buffer::new(proque.queue().clone(), Some(core::MEM_READ_WRITE |
    //     core::MEM_COPY_HOST_PTR), proque.dims().clone(), Some(&vec_src), None).unwrap();
    let buf_src = Buffer::builder()
        .queue(proque.queue().clone())
        .flags(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR)
        .dims(proque.dims().clone())
        .host_data(&vec_src)
        .build().unwrap();

    // Destination Buffer:
    let mut vec_dst = vec![0.0f32; proque.dims().to_len()];
    // let buf_dst = unsafe { Buffer::new_unchecked(
    //     flags::MEM_WRITE_ONLY | flags::MEM_HOST_READ_ONLY | flags::MEM_COPY_HOST_PTR,
    //     proque.dims().to_len().unwrap(), Some(&vec_dst), proque.queue()) };
    // let buf_dst = Buffer::new(proque.queue().clone(), Some(core::MEM_READ_WRITE |
    //     core::MEM_COPY_HOST_PTR), proque.dims().clone(), Some(&vec_dst), None).unwrap();
    let buf_dst = Buffer::builder()
        .queue(proque.queue().clone())
        .flags(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR)
        .dims(proque.dims().clone())
        .host_data(&vec_dst)
        .build().unwrap();

    // Source origin doesn't matter for this:
    let src_origin = [0, 0, 0];
    // Lengths of the two non-major dimensions.
    let row_pitch = dims[0];
    let slc_pitch = dims[0] * dims[1];

    // Set our 'eq' kernel's buffer to our dst buffer for reset purposes:
    kernel_eq.set_arg_buf_named("buf", Some(&buf_dst)).unwrap();

    // Reset kernel runs count:
    ttl_runs = 0;

    for _ in 0..TEST_ITERS {
        // Generate a random size region and origin point. For the copy test
        // it's the dst origin we care about, not the src. Each vector+buffer
        // combo now holds the same value.
        let (dst_origin, read_region) = tests::gen_region_origin(&dims);

        //====================================================================
        //=============== `core::enqueue_copy_buffer_rect()` ================
        //====================================================================
        // Set up values. Src buffer will be one step ahead of dst buffer.
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let nxt_val = ADDEND * (ttl_runs + 1) as f32;

        // Reset destination buffer to current val:
        kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enq().expect("[FIXME]: HANDLE ME!");

        // Set all of `vec_src` to equal the 'next' value. This will be our
        // 'in-region' value and will be written to the device before copying.
        for ele in vec_src.iter_mut() { *ele = nxt_val }

        // Write the source vec to the source buf:
        core::enqueue_write_buffer_rect(proque.queue(), &buf_src, true, [0, 0, 0], [0, 0, 0],
            dims, row_pitch, slc_pitch, row_pitch, slc_pitch, &vec_src,
            None::<core::Event>, None::<&mut core::Event>).unwrap();

        // Copy from the source buffer to the random region on the destination buffer:
        core::enqueue_copy_buffer_rect::<f32, _, _, _>(proque.queue(), &buf_src, &buf_dst,
            src_origin, dst_origin, read_region.clone(), row_pitch, slc_pitch,
            row_pitch, slc_pitch, None::<core::Event>, None::<&mut core::Event>).unwrap();
        // Read the entire destination buffer into the destination vec:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf_dst, true,
            [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
            &mut vec_dst, None::<core::Event>, None::<&mut core::Event>).unwrap(); }
        // Verify that our random region was in fact written correctly:
        tests::verify_vec_rect(dst_origin, read_region, nxt_val, cur_val,
            dims, 1, &vec_dst, ttl_runs, true).unwrap();

        //====================================================================
        //================= `Buffer::cmd().copy().rect()` ===================
        //====================================================================
        // Set up values. Src buffer will be one step ahead of dst buffer.
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let nxt_val = ADDEND * (ttl_runs + 1) as f32;

        // Reset destination buffer to current val:
        kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enq().expect("[FIXME]: HANDLE ME!");

        // Set all of `vec_src` to equal the 'next' value. This will be our
        // 'in-region' value and will be written to the device before copying.
        for ele in vec_src.iter_mut() { *ele = nxt_val }

        // Write the source vec to the source buf:
        buf_src.cmd().write(&vec_src).rect([0, 0, 0], [0, 0, 0], dims, row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();

        // Copy from the source buffer to the random region on the destination buffer:
        buf_src.cmd().copy(&buf_dst, None, None).rect(src_origin, dst_origin, read_region.clone(), row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();
        // Read the entire destination buffer into the destination vec:
        buf_dst.cmd().read(&mut vec_dst).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
            row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();
        // Verify that our random region was in fact written correctly:
        tests::verify_vec_rect(dst_origin, read_region, nxt_val, cur_val,
            dims, 1, &vec_dst, ttl_runs, true).unwrap();
    }


    println!("{} total test runs complete.\n", ttl_runs);
}
