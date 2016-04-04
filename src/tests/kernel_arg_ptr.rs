//! This currently should fail and does not...

use std::thread;
use std::time::Duration;
use standard::{ProQue, Kernel};

static SRC: &'static str = r#"
    __kernel void add(__global float* buffer, float addend) {
        buffer[get_global_id(0)] += addend;
    }
"#;

#[test]
fn kernel_arg_ptr_out_of_scope() {
    let pro_que = ProQue::builder()
        .src(SRC)
        .dims([1024])
        .build().unwrap();    

	let kernel = make_a_kernel(&pro_que);

	for i in 0..50 {
		make_a_kernel(&pro_que);
	}

	thread::sleep(Duration::from_millis(2000));

	for _ in 0..5 {
		kernel.enq().unwrap();
	}
}

fn make_a_kernel(pro_que: &ProQue) -> Kernel {
	let throwaway_vec = vec![0usize; 2 << 20];
	let buffer1 = Box::new(pro_que.create_buffer::<f32>().unwrap());
	let throwaway_vec = vec![0usize; 2 << 20];
	let buffer2 = Box::new(pro_que.create_buffer::<f32>().unwrap());
	let throwaway_vec = vec![0usize; 2 << 20];
	let buffer3 = Box::new(pro_que.create_buffer::<f32>().unwrap());
	let throwaway_vec = vec![0usize; 2 << 20];
	let buffer4 = Box::new(pro_que.create_buffer::<f32>().unwrap());

	let mut kernel = pro_que.create_kernel("add").unwrap()
        // .arg_buf(&buffer)
        .arg_buf_named::<f32>("buf", None)
        .arg_scl(10.0f32);

    kernel.set_arg_buf_named("buf", Some(&buffer3));

    kernel.set_arg_buf_named("buf", Some(&buffer2));

    kernel
}