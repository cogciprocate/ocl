use std::thread;
use std::time::Duration;
use standard::{ProQue, Kernel, Buffer};

static SRC: &'static str = r#"
    __kernel void add(__global float* buffer, float addend) {
        buffer[get_global_id(0)] += addend;
    }
"#;

/// Create a vector on the heap, then a buffer right after it. Assign the
/// buffer as a kernel argument. Let them both fall out of scope. Run kernel.
/// If a copy of the pointer to the buffer is not made by the kernel, it will
/// be deallocated and when the kernel tries to access it via the pointer it
/// has internally stored within its argument, it may cause a segfault or
/// error.
///
#[test]
fn kernel_arg_ptr_out_of_scope() {
    let pro_que = ProQue::builder()
        .src(SRC)
        .dims([1024])
        .build().unwrap();

    let mut kernel = pro_que.kernel_builder("add")
        .arg_named("buf", None::<&Buffer<f32>>)
        .arg(10.0f32)
        .build().unwrap();

    set_arg(&mut kernel, &pro_que);
    let throwaway_vec = vec![99usize; 1 << 24];
    thread::sleep(Duration::from_millis(100));

    for _ in 0..5 {
        unsafe { kernel.enq().unwrap(); }
    }

    assert!(throwaway_vec[1 << 15] == 99);
}

fn set_arg(kernel: &mut Kernel, pro_que: &ProQue) {
    let throwaway_vec = vec![99usize; 1 << 22];
    let buffer = pro_que.create_buffer::<f32>().unwrap();
    assert!(throwaway_vec[1 << 15] == 99);

    kernel.set_arg("buf", Some(&buffer)).unwrap();
}
