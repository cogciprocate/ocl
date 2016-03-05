//! Tests various image operations.
//!
//! Runs both the core function and the 'standard' method call for each.

use core;
use flags;
use core::OclNum;
use standard::{ProQue, Image, Sampler};
use enums::{AddressingMode, FilterMode, ImageChannelOrder, ImageChannelDataType, MemObjectType};
use tests;

const ADDEND: [i32; 4] = [1; 4];
const DIMS: [usize; 3] = [12, 12, 12];
const TEST_ITERS: i32 = 220;
const PRINT_ITERS_MAX: i32 = 3;
const PRINT_SLICES_MAX: usize = 12;

#[test]
fn test_image_ops() {
    let src = r#"
        __kernel void add(
                    sampler_t sampler_host,
                    __private int4 addend,
                    read_only image3d_t img_src,
                    write_only image3d_t img_dst)
        {
            int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
            int4 pixel = read_imagei(img_src, sampler_host, coord);
            pixel += addend;
            write_imagei(img_dst, coord, pixel);
        }
    "#;

    let proque = ProQue::builder()
        .src(src)
        .dims(DIMS)
        .build().unwrap();   

    let sampler = Sampler::new(proque.context(), true, AddressingMode::None, FilterMode::Nearest).unwrap();


    // Source and destination images and a vec to shuffle data:
    let mut vec = vec![0i32; proque.dims().to_len().unwrap()];
    let img_src = Image::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnsignedInt32)
        .image_type(MemObjectType::Image3d)
        .dims(proque.dims())
        .flags(flags::MEM_READ_ONLY | flags::MEM_HOST_WRITE_ONLY | flags::MEM_COPY_HOST_PTR)
        .build_with_data(proque.queue(), &vec).unwrap();
    let img_dst = Image::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnsignedInt32)
        .image_type(MemObjectType::Image3d)
        .dims(proque.dims())
        .flags(flags::MEM_WRITE_ONLY | flags::MEM_HOST_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .build_with_data(proque.queue(), &vec).unwrap();

    let kernel_add = proque.create_kernel("add")
        .arg_smp(&sampler)
        .arg_vec(&ADDEND)
        .arg_img(&img_src)
        .arg_img(&img_dst);

    //========================================================================
    //========================================================================
    //============================ Warm Up Run ===============================
    //========================================================================
    //========================================================================
    // Make sure that pro_que's dims are correct:
    let dims = proque.dims().to_size().unwrap();
    assert_eq!(DIMS, dims);

    // Verify image and vector lengths:
    let len = proque.dims().to_len().unwrap();
    assert_eq!(img_src.dims().to_len().unwrap(), len);
    assert_eq!(img_dst.dims().to_len().unwrap(), len);
    assert_eq!(vec.len(), len);

    println!("\n\nAAAA\n ");

    // KERNEL RUN #1 -- make sure everything's working normally:
    kernel_add.enqueue();
    let mut ttl_runs = 1i32;

    println!("\n\nBBBB\n ");

    // READ AND VERIFY #1 (LINEAR):
    img_dst.read(&mut vec).unwrap();

    println!("\n\nCCCC\n ");

    for idx in 0..proque.dims().to_len().unwrap() {
        // DEBUG:
        // print!("[{:02}]", vec[i]);
        // if i % 20 == 19 { print!("\n"); }
        assert!(vec[idx] == ADDEND[0] * ttl_runs, "vec[{}]: {}", idx, vec[idx]);
    }

    print!("\n");

    // Warm up the verify function:
    tests::verify_vec_rect([0, 0, 0], dims, ADDEND[0] * ttl_runs,
        ADDEND[0] * (ttl_runs - 1), dims, &vec, ttl_runs, false);

    // //========================================================================
    // //========================================================================
    // //=============================== Read ===================================
    // //========================================================================
    // //========================================================================
    // // Image origin doesn't matter since it's all the same value host side:
    // let img_origin = [0, 0, 0];
    // // Lengths of the two non-major dimensions.
    // let row_pitch = dims[0];
    // let slc_pitch = dims[0] * dims[1];

    // for _ in 0..TEST_ITERS {
    //     // Generate a random size region and origin point:
    //     let (read_region, vec_origin) = tests::gen_region_origin(&dims);

    //     //====================================================================
    //     //=============== `core::enqueue_read_image_rect()` =================
    //     //====================================================================
    //     // Reset vec:
    //     unsafe { core::enqueue_read_image_rect(proque.queue(), &img, true, 
    //         [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
    //         &mut vec, None::<&core::EventList>, None).unwrap(); }

    //     // Run kernel:
    //     kernel_add.enqueue();
    //     ttl_runs += 1;
    //     let cur_val = ADDEND * ttl_runs as f32;
    //     let old_val = ADDEND * (ttl_runs - 1) as f32;

    //     // Read from the random region into our vec:
    //     unsafe { core::enqueue_read_image_rect(proque.queue(), &img, true, 
    //         img_origin, vec_origin, read_region.clone(), row_pitch, slc_pitch, 
    //         row_pitch, slc_pitch, &mut vec, None::<&core::EventList>, None).unwrap(); }

    //     // Verify:
    //     tests::verify_vec_rect(vec_origin, read_region, cur_val, old_val, dims, &vec, ttl_runs, false);














        // //====================================================================
        // //================== `Image::cmd().read().rect()` ===================
        // //====================================================================
        // // Reset vec:
        // img.cmd().read(&mut vec).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
        //     row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();

        // // Run kernel:
        // kernel_add.enqueue();
        // ttl_runs += 1;
        // let cur_val = ADDEND * ttl_runs as f32;
        // let old_val = ADDEND * (ttl_runs - 1) as f32;

        // // Read from the random region into our vec:
        // img.cmd().read(&mut vec).rect(img_origin, vec_origin, read_region.clone(), row_pitch,
        //     slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();

        // // Verify:
        // tests::verify_vec_rect(vec_origin, read_region, cur_val, old_val, dims, &vec, ttl_runs, false);
    // }










    // //========================================================================
    // //========================================================================
    // //=============================== Write ==================================
    // //========================================================================
    // //========================================================================
    // // Prepare a kernel which will write a single value to the entire image
    // // and which can be updated on each run (to act as a 'reset').
    // let mut kernel_eq = proque.create_kernel("eq")
    // .arg_img_named("img", Some(&img))
    // .arg_scl_named("val", Some(0.0f32));

    // // Vector origin doesn't matter for this:
    // let vec_origin = [0, 0, 0];
    // // Lengths of the two non-major dimensions.
    // let row_pitch = dims[0];
    // let slc_pitch = dims[0] * dims[1];

    // // Reset kernel runs count:
    // ttl_runs = 0;

    // for _ in 0..TEST_ITERS {
    //     // Generate a random size region and origin point. For the write test
    //     // it's the img origin we care about, not the vec.
    //     let (read_region, img_origin) = tests::gen_region_origin(&dims);

    //     //====================================================================
    //     //=============== `core::enqueue_write_image_rect()` ================
    //     //====================================================================
    //     // Set up values. Device image will now be init'd one step behind host vec.
    //     ttl_runs += 1;
    //     let cur_val = ADDEND * ttl_runs as f32;
    //     let nxt_val = ADDEND * (ttl_runs + 1) as f32;
    //     kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enqueue();

    //     // Write `next_val` to all of `vec`. This will be our 'in-region' value:
    //     for ele in vec.iter_mut() { *ele = nxt_val }
        
    //     // Write to the random region:
    //     core::enqueue_write_image_rect(proque.queue(), &img, false, 
    //         img_origin, vec_origin, read_region.clone(), row_pitch, slc_pitch, 
    //         row_pitch, slc_pitch, &vec, None::<&core::EventList>, None).unwrap();
    //     // Read the entire image back into the vector:
    //     unsafe { core::enqueue_read_image_rect(proque.queue(), &img, true, 
    //         [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
    //         &mut vec, None::<&core::EventList>, None).unwrap(); }
    //     // Verify that our random region was in fact written correctly:
    //     tests::verify_vec_rect(img_origin, read_region, nxt_val, cur_val, dims, &vec, ttl_runs, true);

    //     //====================================================================
    //     //================= `Image::cmd().write().rect()` ===================
    //     //====================================================================
    //     // Set up values. Device image will now be init'd one step behind host vec.
    //     ttl_runs += 1;
    //     let cur_val = ADDEND * ttl_runs as f32;
    //     let nxt_val = ADDEND * (ttl_runs + 1) as f32;
    //     kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enqueue();

    //     // Write `next_val` to all of `vec`. This will be our 'in-region' value:
    //     for ele in vec.iter_mut() { *ele = nxt_val }

    //     // Write to the random region:
    //     img.cmd().write(&mut vec).rect(img_origin, vec_origin, read_region.clone(), row_pitch,
    //         slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).block(false).enq().unwrap();
    //     // Read the entire image back into the vector:
    //     img.cmd().read(&mut vec).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
    //         row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();
    //     // Verify that our random region was in fact written correctly:
    //     tests::verify_vec_rect(img_origin, read_region, nxt_val, cur_val, dims, &vec, ttl_runs, true);
    // }

    // //========================================================================
    // //========================================================================
    // //================================ Copy ==================================
    // //========================================================================
    // //========================================================================
    // // Source Image:
    // let mut vec_src = vec![0.0f32; proque.dims().to_len().unwrap()];
    // let img_src = unsafe { Image::new_unchecked(
    //     flags::MEM_READ_ONLY | flags::MEM_HOST_WRITE_ONLY | flags::MEM_COPY_HOST_PTR,
    //     proque.dims().to_len().unwrap(), Some(&vec_src), proque.queue()) };

    // // Destination Image:
    // let mut vec_dst = vec![0.0f32; proque.dims().to_len().unwrap()];
    // let img_dst = unsafe { Image::new_unchecked(
    //     flags::MEM_WRITE_ONLY | flags::MEM_HOST_READ_ONLY | flags::MEM_COPY_HOST_PTR,
    //     proque.dims().to_len().unwrap(), Some(&vec_dst), proque.queue()) };

    // // Source origin doesn't matter for this:
    // let src_origin = [0, 0, 0];
    // // Lengths of the two non-major dimensions.
    // let row_pitch = dims[0];
    // let slc_pitch = dims[0] * dims[1];

    // // Set our 'eq' kernel's image to our dst image for reset purposes:
    // kernel_eq.set_arg_img_named("img", Some(&img_dst)).unwrap();

    // // Reset kernel runs count:
    // ttl_runs = 0;

    // for _ in 0..TEST_ITERS {
    //     // Generate a random size region and origin point. For the copy test
    //     // it's the dst origin we care about, not the src. Each vector+image
    //     // combo now holds the same value.
    //     let (read_region, dst_origin) = tests::gen_region_origin(&dims);

    //     //====================================================================
    //     //=============== `core::enqueue_copy_image_rect()` ================
    //     //====================================================================
    //     // Set up values. Src image will be one step ahead of dst image.
    //     ttl_runs += 1;
    //     let cur_val = ADDEND * ttl_runs as f32;
    //     let nxt_val = ADDEND * (ttl_runs + 1) as f32;

    //     // Reset destination image to current val:
    //     kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enqueue();

    //     // Set all of `vec_src` to equal the 'next' value. This will be our
    //     // 'in-region' value and will be written to the device before copying.
    //     for ele in vec_src.iter_mut() { *ele = nxt_val }

    //     // Write the source vec to the source img:
    //     core::enqueue_write_image_rect(proque.queue(), &img_src, true, [0, 0, 0], [0, 0, 0], 
    //         dims, row_pitch, slc_pitch, row_pitch, slc_pitch, &vec_src,
    //         None::<&core::EventList>, None).unwrap();
        
    //     // Copy from the source image to the random region on the destination image:
    //     core::enqueue_copy_image_rect::<f32, _>(proque.queue(), &img_src, &img_dst,
    //         src_origin, dst_origin, read_region.clone(), row_pitch, slc_pitch, 
    //         row_pitch, slc_pitch, None::<&core::EventList>, None).unwrap();
    //     // Read the entire destination image into the destination vec:
    //     unsafe { core::enqueue_read_image_rect(proque.queue(), &img_dst, true, 
    //         [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
    //         &mut vec_dst, None::<&core::EventList>, None).unwrap(); }
    //     // Verify that our random region was in fact written correctly:
    //     tests::verify_vec_rect(dst_origin, read_region, nxt_val, cur_val, dims, &vec_dst, ttl_runs, true);

    //     //====================================================================
    //     //================= `Image::cmd().copy().rect()` ===================
    //     //====================================================================
    //     // Set up values. Src image will be one step ahead of dst image.
    //     ttl_runs += 1;
    //     let cur_val = ADDEND * ttl_runs as f32;
    //     let nxt_val = ADDEND * (ttl_runs + 1) as f32;

    //     // Reset destination image to current val:
    //     kernel_eq.set_arg_scl_named("val", cur_val).unwrap().enqueue();

    //     // Set all of `vec_src` to equal the 'next' value. This will be our
    //     // 'in-region' value and will be written to the device before copying.
    //     for ele in vec_src.iter_mut() { *ele = nxt_val }

    //     // Write the source vec to the source img:
    //     img_src.cmd().write(&vec_src).rect([0, 0, 0], [0, 0, 0], dims, row_pitch,
    //         slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();
        
    //     // Copy from the source image to the random region on the destination image:
    //     img_src.cmd().copy(&img_dst, 0, 0).rect(src_origin, dst_origin, read_region.clone(), row_pitch,
    //         slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).enq().unwrap();
    //     // Read the entire destination image into the destination vec:
    //     img_dst.cmd().read(&mut vec_dst).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
    //         row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();
    //     // Verify that our random region was in fact written correctly:
    //     tests::verify_vec_rect(dst_origin, read_region, nxt_val, cur_val, dims, &vec_dst, ttl_runs, true);
    // }


    // printlnc!(lime: "{} total test runs complete.\n", ttl_runs);
}