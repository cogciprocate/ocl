//! Tests for the rectangularly shaped buffer operations: read, write, and copy.
//!
//! Runs both the core function and the 'standard' method call for each.

use rand::{self, Rng};
use core;
use flags;
use core::OclNum;
use standard::{ProQue, Buffer};

const ADDEND: f32 = 10.0;
const DIMS: [usize; 3] = [12, 12, 12];
const TEST_ITERS: i32 = 220;
const PRINT_ITERS_MAX: i32 = 3;
const PRINT_SLICES_MAX: usize = 12;

fn gen_region_origin(dims: &[usize; 3]) -> ([usize; 3], [usize; 3]) {
    let mut rng = rand::weak_rng();

    let region = [
        rng.gen_range(1, dims[0] + 1),
        rng.gen_range(1, dims[1] + 1),
        rng.gen_range(1, dims[2] + 1),
    ];

    let origin = [
        rng.gen_range(0, (dims[0] - region[0]) + 1),
        rng.gen_range(0, (dims[1] - region[1]) + 1),
        rng.gen_range(0, (dims[2] - region[2]) + 1),
    ];

    (region, origin)
}

fn within_region(coords: [usize; 3], region_ofs: [usize; 3], region_size: [usize; 3]) -> bool {
    let mut within: bool = true;
    for i in 0..3 {
        within &= coords[i] >= region_ofs[i] && coords[i] < (region_ofs[i] + region_size[i]);
    }
    within
}

fn verify_vec_rect<T: OclNum>(origin: [usize; 3], region: [usize; 3], in_region_val: T, 
            out_region_val: T, vec_dims: [usize; 3], vec: &[T], ttl_runs: i32, print: bool) 
{
    let print = print && ttl_runs <= PRINT_ITERS_MAX; 
    let slices_to_print = PRINT_SLICES_MAX;

    if print {
        println!("Verifying run: '{}', origin: {:?}, region: {:?}, vec_dims: {:?}", ttl_runs,
            origin, region, vec_dims);
    }

    for z in 0..vec_dims[2] {
        for y in 0..vec_dims[1] {
            for x in 0..vec_dims[0] {
                let idx = (z * vec_dims[1] * vec_dims[0]) + 
                    (y * vec_dims[0]) + x;

                // Print:
                if print && z < slices_to_print {
                    if within_region([x, y, z], origin, region) {
                        if vec[idx] == in_region_val {
                            printc!(lime: "[{:02}]", vec[idx]);
                        } else {
                            printc!(red_bold: "[{:02}]", vec[idx]);
                        }
                    } else {
                        if vec[idx] == out_region_val {
                            printc!(dark_grey: "[{:02}]", vec[idx]); 
                        } else {
                            printc!(yellow: "[{:02}]", vec[idx]);
                        }
                    }
                }

                // Verify:
                if within_region([x, y, z], origin, region) {
                    assert!(vec[idx] == in_region_val, "vec[{}] should be '{}' but is '{}'", 
                        idx, in_region_val, vec[idx]);
                } else {
                    assert!(vec[idx] == out_region_val, "vec[{}] should be '{}' but is '{}'", 
                        idx, out_region_val, vec[idx]);
                }
            }
            if print && z < slices_to_print { print!("\n"); }
        }
        if print && z < slices_to_print { print!("\n"); }
    }
    if print { print!("\n"); }
}



#[test]
fn test_buffer_ops_rect() {
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
    let mut vec = vec![0.0f32; proque.dims().to_len().unwrap()];
    let buf = unsafe { Buffer::new_unchecked(
        flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR,
        proque.dims().to_len().unwrap(), Some(&vec), proque.queue()) };

    let kernel_add = proque.create_kernel("add")
        .arg_buf(&buf)
        .arg_scl(ADDEND);


    //========================================================================
    //========================================================================
    //============================ Warm Up Run ===============================
    //========================================================================
    //========================================================================
    // Make sure that pro_que's dims are correct:
    let dims = proque.dims().to_size().unwrap();
    assert_eq!(DIMS, dims);

    // Verify buffer and vector lengths:
    let len = proque.dims().to_len().unwrap();
    assert_eq!(buf.len(), len);
    assert_eq!(vec.len(), len);

    // KERNEL RUN #1 -- make sure everything's working normally:
    kernel_add.enqueue();
    let mut ttl_runs = 1i32;

    // READ AND VERIFY #1 (LINEAR):
    buf.read(0, &mut vec).unwrap();

    for idx in 0..proque.dims().to_len().unwrap() {
        // DEBUG:
        // print!("[{:02}]", vec[i]);
        // if i % 20 == 19 { print!("\n"); }
        assert!(vec[idx] == ADDEND * ttl_runs as f32, 
            "vec[{}]: {}", idx, vec[idx]);
    }

    print!("\n");

    // Warm up the verify function:
    verify_vec_rect([0, 0, 0], dims, ADDEND * ttl_runs as f32,
        ADDEND * (ttl_runs - 1) as f32, dims, &vec, ttl_runs, false);

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
        let (read_region, vec_origin) = gen_region_origin(&dims);

        //====================================================================
        //=============== `core::enqueue_read_buffer_rect()` =================
        //====================================================================
        // Reset vec:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf, true, 
            [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
            &mut vec, None::<&core::EventList>, None).unwrap(); }

        // Run kernel:
        kernel_add.enqueue();
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let old_val = ADDEND * (ttl_runs - 1) as f32;

        // Read from the random region into our vec:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf, true, 
            buf_origin, vec_origin, read_region.clone(), row_pitch, slc_pitch, 
            row_pitch, slc_pitch, &mut vec, None::<&core::EventList>, None).unwrap(); }

        // Verify:
        verify_vec_rect(vec_origin, read_region, cur_val, old_val, dims, &vec, ttl_runs, false);

        //====================================================================
        //================== `Buffer::cmd().read().rect()` ===================
        //====================================================================
        // Reset vec:
        buf.cmd().read(&mut vec).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
            row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();

        // Run kernel:
        kernel_add.enqueue();
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let old_val = ADDEND * (ttl_runs - 1) as f32;

        // Read from the random region into our vec:
        buf.cmd().read(&mut vec).rect(buf_origin, vec_origin, read_region.clone(), row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();

        // Verify:
        verify_vec_rect(vec_origin, read_region, cur_val, old_val, dims, &vec, ttl_runs, false);
    }

    //========================================================================
    //========================================================================
    //=============================== Write ==================================
    //========================================================================
    //========================================================================
    // Prepare a kernel which will write a single value to the entire buffer
    // and which can be updated on each run (to act as a 'reset').
    let mut kernel_eq = proque.create_kernel("eq")
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
        let (read_region, buf_origin) = gen_region_origin(&dims);

        //====================================================================
        //=============== `core::enqueue_write_buffer_rect()` ================
        //====================================================================
        // Set up values. Device buffer will now be init'd one step behind host vec.
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let nxt_val = ADDEND * (ttl_runs + 1) as f32;
        kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enqueue();

        // Write `next_val` to all of `vec`. This will be our 'in-region' value:
        for ele in vec.iter_mut() { *ele = nxt_val }
        
        // Write to the random region:
        core::enqueue_write_buffer_rect(proque.queue(), &buf, false, 
            buf_origin, vec_origin, read_region.clone(), row_pitch, slc_pitch, 
            row_pitch, slc_pitch, &vec, None::<&core::EventList>, None).unwrap();
        // Read the entire buffer back into the vector:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf, true, 
            [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
            &mut vec, None::<&core::EventList>, None).unwrap(); }
        // Verify that our random region was in fact written correctly:
        verify_vec_rect(buf_origin, read_region, nxt_val, cur_val, dims, &vec, ttl_runs, true);

        //====================================================================
        //================= `Buffer::cmd().write().rect()` ===================
        //====================================================================
        // Set up values. Device buffer will now be init'd one step behind host vec.
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let nxt_val = ADDEND * (ttl_runs + 1) as f32;
        kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enqueue();

        // Write `next_val` to all of `vec`. This will be our 'in-region' value:
        for ele in vec.iter_mut() { *ele = nxt_val }

        // Write to the random region:
        buf.cmd().write(&mut vec).rect(buf_origin, vec_origin, read_region.clone(), row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).block(false).enq().unwrap();
        // Read the entire buffer back into the vector:
        buf.cmd().read(&mut vec).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
            row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();
        // Verify that our random region was in fact written correctly:
        verify_vec_rect(buf_origin, read_region, nxt_val, cur_val, dims, &vec, ttl_runs, true);
    }

    //========================================================================
    //========================================================================
    //================================ Copy ==================================
    //========================================================================
    //========================================================================
    // Source Buffer:
    let mut vec_src = vec![0.0f32; proque.dims().to_len().unwrap()];
    let buf_src = unsafe { Buffer::new_unchecked(
        flags::MEM_READ_ONLY | flags::MEM_HOST_WRITE_ONLY | flags::MEM_COPY_HOST_PTR,
        proque.dims().to_len().unwrap(), Some(&vec_src), proque.queue()) };

    // Destination Buffer:
    let mut vec_dst = vec![0.0f32; proque.dims().to_len().unwrap()];
    let buf_dst = unsafe { Buffer::new_unchecked(
        flags::MEM_WRITE_ONLY | flags::MEM_HOST_READ_ONLY | flags::MEM_COPY_HOST_PTR,
        proque.dims().to_len().unwrap(), Some(&vec_dst), proque.queue()) };

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
        let (read_region, dst_origin) = gen_region_origin(&dims);

        //====================================================================
        //=============== `core::enqueue_copy_buffer_rect()` ================
        //====================================================================
        // Set up values. Src buffer will be one step ahead of dst buffer.
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let nxt_val = ADDEND * (ttl_runs + 1) as f32;

        // Reset destination buffer to current val:
        kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enqueue();

        // Set all of `vec_src` to equal the 'next' value. This will be our
        // 'in-region' value and will be written to the device before copying.
        for ele in vec_src.iter_mut() { *ele = nxt_val }

        // Write the source vec to the source buf:
        core::enqueue_write_buffer_rect(proque.queue(), &buf_src, true, [0, 0, 0], [0, 0, 0], 
            dims, row_pitch, slc_pitch, row_pitch, slc_pitch, &vec_src,
            None::<&core::EventList>, None).unwrap();
        
        // Copy from the source buffer to the random region on the destination buffer:
        core::enqueue_copy_buffer_rect::<f32, _>(proque.queue(), &buf_src, &buf_dst,
            src_origin, dst_origin, read_region.clone(), row_pitch, slc_pitch, 
            row_pitch, slc_pitch, None::<&core::EventList>, None).unwrap();
        // Read the entire destination buffer into the destination vec:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf_dst, true, 
            [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
            &mut vec_dst, None::<&core::EventList>, None).unwrap(); }
        // Verify that our random region was in fact written correctly:
        verify_vec_rect(dst_origin, read_region, nxt_val, cur_val, dims, &vec_dst, ttl_runs, true);

        //====================================================================
        //================= `Buffer::cmd().copy().rect()` ===================
        //====================================================================
        // Set up values. Src buffer will be one step ahead of dst buffer.
        ttl_runs += 1;
        let cur_val = ADDEND * ttl_runs as f32;
        let nxt_val = ADDEND * (ttl_runs + 1) as f32;

        // Reset destination buffer to current val:
        kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enqueue();

        // Set all of `vec_src` to equal the 'next' value. This will be our
        // 'in-region' value and will be written to the device before copying.
        for ele in vec_src.iter_mut() { *ele = nxt_val }

        // Write the source vec to the source buf:
        buf_src.cmd().write(&vec_src).rect([0, 0, 0], [0, 0, 0], dims, row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();
        
        // Copy from the source buffer to the random region on the destination buffer:
        buf_src.cmd().copy(&buf_dst, 0, 0).rect(src_origin, dst_origin, read_region.clone(), row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();
        // Read the entire destination buffer into the destination vec:
        buf_dst.cmd().read(&mut vec_dst).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
            row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();
        // Verify that our random region was in fact written correctly:
        verify_vec_rect(dst_origin, read_region, nxt_val, cur_val, dims, &vec_dst, ttl_runs, true);
    }


    printlnc!(lime: "{} total test runs complete.\n", ttl_runs);
}