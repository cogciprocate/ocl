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
            out_region_val: T, vec_dims: [usize; 3], vec: &[T], kernel_runs: i32) 
{
    let print = kernel_runs <= PRINT_ITERS_MAX;
    let slices_to_print = PRINT_SLICES_MAX;

    if print {
        println!("Verifying run: '{}', origin: {:?}, region: {:?}, vec_dims: {:?}", kernel_runs,
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

    let kernel = proque.create_kernel("add")
        .arg_buf(&buf)
        .arg_scl(ADDEND);


    //========================================================================
    //============================ Warm Up Run ===============================
    //========================================================================
    // Make sure that pro_que's dims are correct:
    let dims = proque.dims().to_size().unwrap();
    assert_eq!(DIMS, dims);

    // Verify buffer and vector lengths:
    let len = proque.dims().to_len().unwrap();
    assert_eq!(buf.len(), len);
    assert_eq!(vec.len(), len);

    // KERNEL RUN #1 -- make sure everything's working normally:
    kernel.enqueue();
    let mut kernel_runs = 1i32;

    // READ AND VERIFY #1 (LINEAR):
    buf.read(0, &mut vec).unwrap();

    for idx in 0..proque.dims().to_len().unwrap() {
        // DEBUG:
        // print!("[{:02}]", vec[i]);
        // if i % 20 == 19 { print!("\n"); }
        assert!(vec[idx] == ADDEND * kernel_runs as f32, 
            "vec[{}]: {}", idx, vec[idx]);
    }

    print!("\n");

    // Warm up the verify function:
    verify_vec_rect([0, 0, 0], dims, ADDEND * kernel_runs as f32,
        ADDEND * (kernel_runs - 1) as f32, dims, &vec, kernel_runs);

    //========================================================================
    //=============================== Read ===================================
    //========================================================================
    for _ in 0..TEST_ITERS {
        // Generate a random size region and origin point:
        let (read_region, vec_origin) = gen_region_origin(&dims);
        // Buffer origin doesn't matter since it's all the same value host side:
        let buf_origin = [0, 0, 0];
        // Lengths of the two non-major dimensions.
        let row_pitch = dims[0];
        let slc_pitch = dims[0] * dims[1];

        //====================================================================
        //=============== `core::enqueue_read_buffer_rect()` =================
        //====================================================================
        // RUN KERNEL:
        kernel.enqueue();
        kernel_runs += 1;
        let cur_val = ADDEND * kernel_runs as f32;
        let old_val = ADDEND * (kernel_runs - 1) as f32;

        // READ RANDOM REGION AND VERIFY:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf, true, 
            buf_origin, vec_origin, read_region.clone(), row_pitch, slc_pitch, 
            row_pitch, slc_pitch, &mut vec, None::<&core::EventList>, None).unwrap(); }
        verify_vec_rect(vec_origin, read_region, cur_val, old_val, dims, &vec, kernel_runs);

        // RESET AND VERIFY:
        unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf, true, 
            [0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch, row_pitch, slc_pitch,
            &mut vec, None::<&core::EventList>, None).unwrap(); }
        verify_vec_rect(vec_origin, read_region, cur_val, cur_val, dims, &vec, kernel_runs);

        //====================================================================
        //======================== `Buffer::cmd()` ===========================
        //====================================================================
        // RUN KERNEL:
        kernel.enqueue();
        kernel_runs += 1;
        let cur_val = ADDEND * kernel_runs as f32;
        let old_val = ADDEND * (kernel_runs - 1) as f32;

        // READ RANDOM REGION AND VERIFY:
        buf.cmd().read(&mut vec).rect(buf_origin, vec_origin, read_region.clone(), row_pitch,
            slc_pitch, row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();
        verify_vec_rect(vec_origin, read_region, cur_val, old_val, dims, &vec, kernel_runs);

        // RESET AND VERIFY
        buf.cmd().read(&mut vec).rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
            row_pitch, slc_pitch).queue(proque.queue()).block(true).enq().unwrap();
        verify_vec_rect(vec_origin, read_region, cur_val, cur_val, dims, &vec, kernel_runs);
    }


    // for _ in 0..TEST_ITERS {
    //     let (read_region, vec_origin) = gen_region_origin(&dims);
    //     let buf_origin = [0, 0, 0];

    //     kernel.enqueue();
    //     kernel_runs += 1;

    //     let row_pitch = dims[0];
    //     let slc_pitch = dims[0] * dims[1];

    //     // READ RANDOM REGION AND VERIFY
    //     buf.cmd().read(&mut vec)
    //         .rect(buf_origin, vec_origin, read_region.clone(), row_pitch, slc_pitch,
    //             row_pitch, slc_pitch)
    //         .queue(proque.queue()).block(true).enq().unwrap();

    //     verify_vec_rect(vec_origin, read_region, cur_val,
    //         old_val, dims, &vec, kernel_runs);

    //     buf.cmd().read(&mut vec)
    //         .rect([0, 0, 0], [0, 0, 0], dims, row_pitch, slc_pitch,
    //             row_pitch, slc_pitch)
    //         .queue(proque.queue()).block(true).enq().unwrap();

    //     verify_vec_rect(vec_origin, read_region, cur_val,
    //         cur_val, dims, &vec, kernel_runs);
    // }


    printlnc!(lime: "{} total test runs complete.\n", kernel_runs);
    
    // panic!("SUCCESS!");

    // [DST_BUFFER]:
    // let mut dst_vec = vec![0.0f32; proque.dims().to_len().unwrap()];
    // let dst_buf = unsafe { Buffer::new_unchecked(
    //     flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR,
    //     proque.dims().to_len().unwrap(), Some(&vec_1), proque.queue()) };

}

    // pub fn enqueue_copy_buffer_rect<L: AsRef<EventList>>(
    //  command_queue: &CommandQueue, buf: &Mem, buf_1: &Mem, src_origin: [usize; 3],
    //  dst_origin: [usize; 3], region: [usize; 3], src_row_pitch: usize, src_slc_pitch: usize, 
    //  dst_row_pitch: usize, dst_slc_pitch: usize, wait_list: Option<&L>, new_event: Option<&mut ClEventPtrNew>,
    //  ) -> OclResult<()>


