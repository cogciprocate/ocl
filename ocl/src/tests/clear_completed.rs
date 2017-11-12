use standard::{ProQue, EventList};

#[test]
fn clear_completed() {
    let src = r#"
        __kernel void add(__global float* buffer, float addend) {
            buffer[get_global_id(0)] += addend;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims([1 << 10])
        .build().unwrap();

    let buffer = pro_que.create_buffer::<f32>().unwrap();

    let kernel = pro_que.create_kernel("add").unwrap()
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    let mut event_list = EventList::new();

    let mut vec = vec![0.0f32; buffer.len()];

    for _ in 0..2048 {
        unsafe { kernel.cmd().enew(&mut event_list).enq().unwrap(); }

        buffer.read(&mut vec).enq().unwrap();

        event_list.clear_completed().unwrap();
    }
}

