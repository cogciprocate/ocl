
const ADDEND: f32 = 10.0;
const DIMS: [usize; 3] = [8, 8, 8];

#[test]
fn test_buffer_ops_rect() {
    use core;
    use flags;
    use standard::{ProQue, Buffer};

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
    let mut vec_buf_0 = vec![0.0f32; proque.dims().to_len()];
    let buf_0 = unsafe { Buffer::new_unchecked(
        flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR,
        proque.dims().to_len(), Some(&vec_buf_0), proque.queue()) };

    // DST_BUFFER:
    // let mut vec_buf_1 = vec![0.0f32; proque.dims().to_len()];
    // let buf_1 = unsafe { Buffer::new_unchecked(
    //     flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR,
    //     proque.dims().to_len(), Some(&vec_buf_1), proque.queue()) };

    let kernel = proque.create_kernel("add")
        .arg_buf(&buf_0)
        .arg_scl(ADDEND);

    // Kernel run #1:
    kernel.enqueue();
    let mut kernel_runs = 1;

    buf_0.read(0, &mut vec_buf_0).unwrap();

    for idx in 0..proque.dims().to_len() {
        // print!("[{:02}]", vec_buf_0[i]);
        // if i % 20 == 19 { print!("\n"); }
        assert!(vec_buf_0[idx] == ADDEND * kernel_runs as f32, 
            "vec[{}]: {}", idx, vec_buf_0[idx]);
    }    

    // Kernel run #2:
    kernel.enqueue();
    kernel_runs += 1;

    let buf_0_origin = [0, 0, 0];
    let vec_buf_0_origin = [2, 3, 4];
    let region_size = proque.dims().to_size();
    let read_region_size = [region_size[0] / 4, region_size[1] / 4, region_size[2] / 4];
    // let read_region_size = region_size.clone();

    println!("region_size: {:?}", region_size);
    println!("vec_buf_0.len(): {}", vec_buf_0.len());

    println!("buf_0 info: {}", buf_0);

    unsafe { core::enqueue_read_buffer_rect(proque.queue(), &buf_0, true, 
        buf_0_origin, vec_buf_0_origin, read_region_size.clone(), 
        region_size[0], region_size[0] * region_size[1], 
        region_size[0], region_size[0] * region_size[1], 
        &mut vec_buf_0, None::<&core::EventList>, None).unwrap(); }

    let slices_to_print = 8;

    for z in 0..region_size[2] {
        for y in 0..region_size[1] {
            for x in 0..region_size[0] {
                // let idx = (x * region_size[1] * region_size[2]) + 
                //     (y * region_size[2]) + z;
                let idx = (z * region_size[1] * region_size[0]) + 
                    (y * region_size[0]) + x;
                if z < slices_to_print {
                    if vec_buf_0[idx] == ADDEND * kernel_runs as f32 {
                        printc!(lime: "[{:02}]", vec_buf_0[idx]);
                    } else {
                        printc!(dark_grey: "[{:02}]", vec_buf_0[idx]);   
                    }
                } else {
                    
                }

                // if x >= src_vec_origin[0] && x < (src_vec_origin[0] + read_region_size[0]) {
                //     let idx = x * y * z;
                //     print!("[{}]", vec_buf_0[idx]);
                // }
            }
            if z < slices_to_print { print!("\n"); }
        }
        if z < slices_to_print { print!("\n"); }
    }

    // for i in 0..dims[0] {
    //     assert_eq!(vec_buf_0[i], ADDEND);

    //     if i >= copy_range.0 && i < copy_range.1 {
    //         assert_eq!(vec_buf_1[i], ADDEND);
    //     } else {
    //         assert!(vec_buf_1[i] == 0.0, "dst_vec: {}, idx: {}", vec_buf_1[i], i);
    //     }
    // }
    panic!("SUCCESS!");
}


    // pub fn enqueue_copy_buffer_rect<L: AsRef<EventList>>(
    //  command_queue: &CommandQueue, buf_0: &Mem, buf_1: &Mem, src_origin: [usize; 3],
    //  dst_origin: [usize; 3], region: [usize; 3], src_row_pitch: usize, src_slc_pitch: usize, 
    //  dst_row_pitch: usize, dst_slc_pitch: usize, wait_list: Option<&L>, new_event: Option<&mut ClEventPtrNew>,
    //  ) -> OclResult<()>




    // // Copy src to dst:
    // let copy_range = (IDX, proque.dims()[0] - 100);
    // buf_0.cmd().copy(&buf_1, copy_range.0, copy_range.1 - copy_range.0).enq().unwrap();

    // // Read both buffers from device.
    // buf_0.fill_vec();
    // buf_1.fill_vec();

    // for i in 0..proque.dims()[0] {
    //     assert_eq!(buf_0[i], ADDEND);

    //     if i >= copy_range.0 && i < copy_range.1 {
    //         assert_eq!(buf_1[i], ADDEND);
    //     } else {
    //         assert!(buf_1[i] == 0.0, "dst_buf: {}, idx: {}", buf_1[i], i);
    //     }
    // }