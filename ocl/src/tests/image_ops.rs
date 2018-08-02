//! Tests various image operations.
//!
//! * TODO: Test fill and copy to buffer.
//!
//! Runs both the core function and the 'standard' method call for each.

use core::{self, DeviceInfo, DeviceInfoResult};
use flags;
use standard::{ProQue, Image, Sampler, Platform, Device};
use enums::{AddressingMode, FilterMode, ImageChannelOrder, ImageChannelDataType, MemObjectType};
use prm::{Int4};
use tests;

// const ADDEND: [i32; 4] = [1; 4];
const DIMS: [usize; 3] = [64, 128, 4];
const TEST_ITERS: i32 = 4;

#[test]
fn image_ops() {
    #[allow(non_snake_case)]
    let ADDEND: Int4 = Int4::new(1, 1, 1, 1);

    let src = r#"
        #pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

        __kernel void add(
                    sampler_t sampler_host,
                    __private int4 addend,
                    __read_only image3d_t img_src,
                    __write_only image3d_t img_dst)
        {
            int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
            int4 pixel = read_imagei(img_src, sampler_host, coord);
            pixel += addend;
            write_imagei(img_dst, coord, pixel);
        }

        __kernel void fill(
                    sampler_t sampler_host,
                    __private int4 pixel,
                    __write_only image3d_t img)
        {
            int4 coord = (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);
            write_imagei(img, coord, pixel);
        }
    "#;

    let platform = Platform::default();
    let device = Device::first(platform).unwrap();

    // Ensure `cl_khr_3d_image_writes` is available.
    match device.info(DeviceInfo::Extensions).unwrap() {
        DeviceInfoResult::Extensions(exts) => {
            println!("exts: \n {:?}", exts);
            if !exts.contains("cl_khr_3d_image_writes") {
                println!("Skipping 'tests::image_ops': The 'cl_khr_3d_image_writes' is not available.");
                return;
            }
        },
        _ => unreachable!(),
    }

    let proque = ProQue::builder()
        .platform(platform)
        .device(device)
        .src(src)
        .dims(DIMS)
        .build().unwrap();

    let sampler = Sampler::new(proque.context(), false, AddressingMode::None, FilterMode::Nearest).unwrap();

    let mut vec = vec![0i32; proque.dims().to_len() * 4];

    // Source and destination images and a vec to shuffle data:
    let img_src = Image::<i32>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::SignedInt32)
        .image_type(MemObjectType::Image3d)
        .dims(proque.dims())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(&vec)
        .queue(proque.queue().clone())
        .build().unwrap();
    let img_dst = Image::<i32>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::SignedInt32)
        .image_type(MemObjectType::Image3d)
        .dims(proque.dims())
        .flags(flags::MEM_WRITE_ONLY | flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(&vec)
        .queue(proque.queue().clone())
        .build().unwrap();

    let kernel_add = proque.kernel_builder("add")
        .arg_sampler(&sampler)
        .arg(ADDEND)
        .arg(&img_src)
        .arg(&img_dst)
        .build().unwrap();

    let kernel_fill_src = proque.kernel_builder("fill")
        .arg_sampler(&sampler)
        .arg_named::<Int4, _>("pixel", &Int4::splat(0))
        .arg(&img_src)
        .build().unwrap();

    //========================================================================
    //========================================================================
    //============================ Warm Up Run ===============================
    //========================================================================
    //========================================================================
    // Make sure that pro_que's dims are correct:
    let dims = proque.dims().to_lens().unwrap();
    assert_eq!(DIMS, dims);
    assert_eq!(DIMS, kernel_add.default_global_work_size().to_lens().unwrap());

    // Verify image and vector lengths:
    let len = proque.dims().to_len();
    assert_eq!(img_src.dims().to_len(), len);
    assert_eq!(img_dst.dims().to_len(), len);

    let pixel_element_len = img_src.pixel_element_len();
    assert_eq!(vec.len(), len * pixel_element_len);

    // KERNEL RUN #1 -- make sure everything's working normally:
    unsafe { kernel_add.enq().unwrap(); }
    let mut ttl_runs = 1i32;

    // READ AND VERIFY #1 (LINEAR):
    img_dst.read(&mut vec).enq().unwrap();

    // Verify that the `verify_vec_rect` function isn't letting something slip:
    for idx in 0..vec.len() {
        assert!(vec[idx] == ADDEND[0] * ttl_runs, "vec[{}]: {}", idx, vec[idx]);
    }

    print!("\n");

    // Warm up the verify function:
    tests::verify_vec_rect([0, 0, 0], dims, ADDEND[0] * ttl_runs,
        ADDEND[0] * (ttl_runs - 1), dims, pixel_element_len, &vec, ttl_runs, true).unwrap();

    //========================================================================
    //========================================================================
    //======================= Read / Write / Copy ============================
    //========================================================================
    //========================================================================

    for _ in 0..TEST_ITERS {
        let (region, origin) = (dims, [0, 0, 0]);

        //====================================================================
        //=================== `core::enqueue_..._image()` ====================
        //====================================================================

        // Write to src:
        unsafe {
            core::enqueue_write_image(proque.queue(), &img_src, true,
                origin, region, 0, 0,
                &vec, None::<core::Event>, None::<&mut core::Event>).unwrap();
        }

        // Add from src to dst:
        unsafe { kernel_add.enq().expect("[FIXME]: HANDLE ME!"); }
        ttl_runs += 1;
        let (cur_val, old_val) = (ADDEND[0] * ttl_runs, ADDEND[0] * (ttl_runs - 1));

        // Read into vec:
        unsafe { core::enqueue_read_image(proque.queue(), &img_dst, true,
            origin, region, 0, 0,
            &mut vec, None::<core::Event>, None::<&mut core::Event>).unwrap(); }

        // Just to make sure read is complete:
        proque.queue().finish().unwrap();

        // Verify:
        tests::verify_vec_rect(origin, region, cur_val, old_val,
            dims, pixel_element_len, &vec, ttl_runs, true).unwrap();


        // Run kernel:
        ttl_runs += 1;
        let (cur_val, old_val) = (ADDEND[0] * ttl_runs, ADDEND[0] * (ttl_runs - 1));
        let cur_pixel = Int4::new(cur_val, cur_val, cur_val, cur_val);
        unsafe {
            kernel_fill_src.set_arg("pixel", cur_pixel).unwrap();
            kernel_fill_src.enq().expect("[FIXME]: HANDLE ME!");
        }

        core::enqueue_copy_image::<_, _>(proque.queue(), &img_src, &img_dst,
            origin, origin, region, None::<core::Event>, None::<&mut core::Event>).unwrap();

        // Read into vec:
        unsafe { core::enqueue_read_image(proque.queue(), &img_dst, true,
            origin, region, 0, 0,
            &mut vec, None::<core::Event>, None::<&mut core::Event>).unwrap(); }

        // Just to make sure read is complete:
        proque.queue().finish().unwrap();

        // Verify:
        tests::verify_vec_rect(origin, region, cur_val, old_val,
            dims, pixel_element_len, &vec, ttl_runs, true).unwrap();

        //====================================================================
        //========================= `Image::cmd()...` ========================
        //====================================================================
        // Write to src:
        img_src.cmd().write(&vec).enq().unwrap();

        // Add from src to dst:
        unsafe {
            kernel_add.enq().expect("[FIXME]: HANDLE ME!");
        }
        ttl_runs += 1;
        let (cur_val, old_val) = (ADDEND[0] * ttl_runs, ADDEND[0] * (ttl_runs - 1));

        // // Read into vec:
        img_dst.cmd().read(&mut vec).enq().unwrap();

        // Verify:
        tests::verify_vec_rect(origin, region, cur_val, old_val,
            dims, pixel_element_len, &vec, ttl_runs, true).unwrap();


        // Run kernel:
        ttl_runs += 1;
        let (cur_val, old_val) = (ADDEND[0] * ttl_runs, ADDEND[0] * (ttl_runs - 1));
        let cur_pixel = Int4::new(cur_val, cur_val, cur_val, cur_val);
        unsafe {
            kernel_fill_src.set_arg("pixel", cur_pixel).unwrap();
            kernel_fill_src.enq().expect("[FIXME]: HANDLE ME!");
        }

        img_src.cmd().copy(&img_dst, origin).enq().unwrap();

        // Read into vec:
        img_dst.cmd().read(&mut vec).enq().unwrap();

        // Verify:
        tests::verify_vec_rect(origin, region, cur_val, old_val,
            dims, pixel_element_len, &vec, ttl_runs, true).unwrap();

    }

    println!("{} total test runs complete.\n", ttl_runs);
}
