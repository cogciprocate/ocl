extern crate ocl;
extern crate ocl_core;

const BUFFER_DIMENSIONS: usize = 2 << 20;
const PLATFORM_ID: usize = 0;
const DEVICE_ID: usize = 0;


fn scalar_map() {
    let kernel_src = r#"
        __kernel void add (__global float* in, float scalar) {
            in[get_global_id(0)] += scalar;
        }
        "#;

    let plt = ocl::Platform::list()[PLATFORM_ID];
    let dev = ocl::Device::list_all(&plt).unwrap()[DEVICE_ID];
    let context = ocl::Context::builder()
        .platform(plt)
        .devices(dev)
        .build()
        .unwrap();
    let queue = ocl::Queue::new(&context, dev, None).unwrap();
    let program = ocl::Program::builder()
        .src(kernel_src)
        .devices(dev)
        .build(&context)
        .unwrap();

    // Creation of buffer using ocl API will result in filling of the buffer as well
    let in_buff = ocl::Buffer::new::<_, _, &mut ocl::Event>(queue.clone(),
                                   Some(ocl::core::MEM_ALLOC_HOST_PTR),
                                   BUFFER_DIMENSIONS,
                                   None::<&[f32]>,
                                   None)
        .expect("Creating buffer failed");

    unsafe {
        let mut buff_datum = ocl_core::enqueue_map_buffer::<f32, _, _, _>(&queue,
                                                             in_buff.core(),
                                                             true,
                                                             ocl_core::MAP_WRITE,
                                                             0,
                                                             BUFFER_DIMENSIONS,
                                                             None::<ocl::Event>,
                                                             None::<&mut ocl::Event>)
            .expect("Mapping memory object failed");
        // Wait until mapping is finished
        queue.finish().unwrap();

        let datum: Vec<f32> = vec![10_f32; BUFFER_DIMENSIONS];
        let mut datum_slice = buff_datum.as_slice_mut(datum.len());
        datum_slice.copy_from_slice(&datum);
        ocl_core::enqueue_unmap_mem_object(&queue, in_buff.core(), &mut buff_datum, None::<ocl::Event>, None::<&mut ocl::Event>)
            .expect("Unmap of memory object failed");
        // Wait until unmapping is finished
        queue.finish().unwrap();
        // Don't deallocate vector, it'll lead to double free of the buffer pointed by buff_datum
    }
    let mut check_datum: Vec<f32> = vec![0_f32; BUFFER_DIMENSIONS];
    in_buff.read(&mut check_datum)
        .enq()
        .expect("Reading from in_buff failed");
    for &ele in check_datum.iter() {
        assert_eq!(ele, 10_f32);
    }

    let kern = ocl::Kernel::new(String::from("add"), &program, queue.clone())
        .expect("Kernel creation failed")
        .gws(BUFFER_DIMENSIONS)
        .arg_buf(&in_buff)
        .arg_scl(5_f32);

    kern.cmd().enq().unwrap();

    let mut read_datum: Vec<f32> = vec![0_f32; BUFFER_DIMENSIONS];
    in_buff.read(&mut read_datum)
        .enq()
        .expect("Reading from in_buff after kernel exec failed");
    for &ele in read_datum.iter() {
        assert_eq!(ele, 15_f32);
    }
}

fn vector_map() {
    let kernel_src = r#"
        __kernel void add (__global float16* in, float scalar) {
            float16 invalue = in[get_global_id(0)];
            /* Use only first value */
            invalue.s0 += scalar;
            in[get_global_id(0)] = invalue;
        }
        "#;
    let plt = ocl::Platform::list()[PLATFORM_ID];
    let dev = ocl::Device::list_all(&plt).unwrap()[DEVICE_ID];
    let context = ocl::Context::builder()
        .platform(plt)
        .devices(dev)
        .build()
        .unwrap();
    let queue = ocl::Queue::new(&context, dev, None).unwrap();
    let program = ocl::Program::builder()
        .src(kernel_src)
        .devices(dev)
        .build(&context)
        .unwrap();
    let in_buff = ocl::Buffer::new::<_, _, &mut ocl::Event>(queue.clone(),
                                   Some(ocl::core::MEM_ALLOC_HOST_PTR),
                                   BUFFER_DIMENSIONS,
                                   None::<&[ocl::aliases::ClFloat16]>,
                                   None)
        .expect("Creating buffer failed");

    unsafe {
        let mut event = ocl::EventList::new();
        let mut buff_datum =
            ocl_core::enqueue_map_buffer::<ocl::aliases::ClFloat16, _, _, _>(&queue,
                                                                    in_buff.core(),
                                                                    true,
                                                                    ocl_core::MAP_WRITE,
                                                                    0,
                                                                    BUFFER_DIMENSIONS,
                                                                    None::<ocl::Event>,
                                                                    Some(&mut event))
                .expect("Mapping memory object failed");
        queue.finish().unwrap();

        let mut value: ocl::aliases::ClFloat16 = Default::default();
        // Use only first value
        value.0 = 10_f32;
        let datum: Vec<ocl::aliases::ClFloat16> = vec![value; BUFFER_DIMENSIONS];
        let mut datum_slice = buff_datum.as_slice_mut(datum.len());
        datum_slice.copy_from_slice(&datum);
        ocl_core::enqueue_unmap_mem_object(&queue, in_buff.core(), &mut buff_datum, None::<ocl::Event>, None::<&mut ocl::Event>)
            .expect("Unmap of memory object failed");
        queue.finish().unwrap();
    }
    let mut check_datum: Vec<ocl::aliases::ClFloat16> = vec![Default::default(); BUFFER_DIMENSIONS];

    in_buff.read(&mut check_datum)
        .enq()
        .expect("Reading from in_buff failed");

    for &ele in check_datum.iter() {
        assert_eq!(ele.0, 10_f32);
    }

    let kern = ocl::Kernel::new(String::from("add"), &program, queue.clone())
        .expect("Kernel creation failed")
        .gws(BUFFER_DIMENSIONS)
        .arg_buf(&in_buff)
        .arg_scl(5_f32);

    kern.cmd().enq().unwrap();

    let mut read_datum: Vec<ocl::aliases::ClFloat16> = vec![Default::default(); BUFFER_DIMENSIONS];
    in_buff.read(&mut read_datum)
        .enq()
        .expect("Reading from in_buff after kernel exec failed");

    for &ele in read_datum.iter() {
        assert_eq!(ele.0, 15_f32);
    }
}

fn main() {
    scalar_map();
    vector_map();
}
