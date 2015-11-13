use libc;
// use std::ptr;
use super::cl_h;

use super::{ Context, BuildConfig, Envoy, SimpleDims, ProQue, EventList };

const SCL: f32 = 5.0;
const DSS: usize = 100;
#[test]
fn test_async_events() {
	// Create a context & program/queue: 
	let mut ocl_pq = ProQue::new(&Context::new(None, None).unwrap(), None);

	// Build program:
	ocl_pq.build(BuildConfig::new().kern_file("cl/kernel_file.cl")).unwrap();

	// Set up data set size and work dimensions:
	let data_set_size = DSS;
	let envoy_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result envoys (our data containers):
	// let source_envoy = Envoy::shuffled(&envoy_dims, 0f32, 20f32, &ocl_pq);
	let mut result_envoy = Envoy::new(&envoy_dims, 0f32, &ocl_pq.queue());

	// Our scalar:
	let scalar = SCL;

	// Create kernel:
	let kernel = ocl_pq.create_kernel("add_scalar".to_string(), envoy_dims.work_size())
		.arg_scl(scalar)
		.arg_env(&mut result_envoy)
	;

	// Create event list:
	let mut kernel_event = EventList::new();

	//#############################################################################################

	// let fn_verify = | |

	// Repeat the test. First iteration 
	for itr in 1..11 {
		println!("Enqueuing kernel [{}]...", itr);
		kernel.enqueue(None, Some(&mut kernel_event));

		let mut read_event = EventList::new();

		unsafe {
			// result_envoy.read(Some(&read_event), None);

			println!("Enqueuing read buffer [{}]...", itr);
			super::enqueue_read_buffer(ocl_pq.queue().obj(), result_envoy.buffer_obj(), false, 
					result_envoy.vec_mut(), 0, None, Some(&mut read_event));

			let mut buncha_stuff = BunchaStuff {
				env: &result_envoy as *const Envoy<f32>, 
				dss: data_set_size as usize, 
				scl: scalar as f32, 
				itr: itr as usize,
			};

			println!("Setting callback (verify_result, buncha_stuff) [{}]...", itr);
			read_event.set_callback(verify_result, &mut buncha_stuff as *mut _ as *mut libc::c_void);
		}

		// println!("Waiting for read_event...");
		// super::wait_for_event(read_event.events()[0]);
		
		// ocl_pq.queue().finish();

		// read_event.wait();
		//

		// for idx in 0..data_set_size {
		// 	assert_eq!(result_envoy[idx], (coeff as f32) * scalar);
		// }

		println!("Releasing read_event [{}]...", itr);
		read_event.release();
	}

	ocl_pq.queue().finish();

	// panic!();
}

// #[repr(C)]
struct BunchaStuff {
	env: *const Envoy<f32>, 
	dss: usize,
	scl: f32, 
	itr: usize,
}


// #[allow(dead_code)]
extern /*"C"*/ fn verify_result(event: cl_h::cl_event, status: cl_h::cl_int, user_data: *mut libc::c_void) {
	let buncha_stuff = user_data as *const BunchaStuff;

	unsafe {
		let result_envoy: *const Envoy<f32> = (*buncha_stuff).env as *const Envoy<f32>;
		let data_set_size: usize = (*buncha_stuff).dss;
		let scalar: f32 = (*buncha_stuff).scl;
		let itr: usize = (*buncha_stuff).itr;
		

	    println!("Event: `{:?}` has completed with status: `{}`, data_set_size: '{}`, scalar: {}, itr: `{}`.", 
	    	event, status, data_set_size, scalar, itr);

		// for idx in 0..data_set_size {
		// 	assert_eq!((*result_envoy)[idx], (itr as f32) * scalar);
		// }
    }
}

#[allow(dead_code)]
extern fn callback_test(event: cl_h::cl_event, status: cl_h::cl_int, user_data: *mut libc::c_void) {
    println!("Event: `{:?}` has completed with status: `{}` and data: `{:?}`", 
    	event, status, user_data);
}

#[allow(dead_code)]
extern fn shithead_test(event: cl_h::cl_event, status: cl_h::cl_int, user_data: *mut libc::c_void) {
    println!("Event: `{:?}` has completed with status: `{}` and data: `{:?}`. And you're a shithead.", 
    	event, status, user_data);
}


// #[cfg(target_os = "linux")]
// #[link(name = "OpenCL")]
// extern fn verify_callback() {
// 	for idx in 0..data_set_size {
// 		assert_eq!(result_envoy[idx], (i as f32) * scalar);
// 	}
// }


// #[cfg(target_os = "linux")]
// #[link(name = "OpenCL")]
// extern fn callback_test(event: cl_h::cl_event, status: cl_h::cl_int, user_data: *mut libc::c_void) {
//     println!("Event: `{:?}` has completed with status: `{}` and data: `{:?}`", 
//     	event, status, user_data);
// }




#[test]
fn test_basics() {
	// Create a context with the default platform and device types:
	let ocl_cxt = Context::new(None, None).unwrap();

	// Create a program/queue with the default device: 
	let mut ocl_pq = ProQue::new(&ocl_cxt, None);

	// Create build configuration:
	let build_config = BuildConfig::new().kern_file("cl/kernel_file.cl");

	// Build with our configuration and check for errors:
	ocl_pq.build(build_config).expect("ocl program build");

	// Set up our data set size and work dimensions:
	let data_set_size = 9000;
	let envoy_dims = SimpleDims::OneDim(data_set_size);

	// Create a source envoy (array) with randomized values and an empty result envoy:
	let source_envoy = Envoy::scrambled(&envoy_dims, 0f32, 20.0, &ocl_pq.queue());
	let mut result_envoy = Envoy::new(&envoy_dims, 0f32, &ocl_pq.queue());

	// Our coefficient:
	let coeff = 50.0;

	// Create kernel:
	let kernel = ocl_pq.create_kernel("multiply_by_scalar".to_string(), envoy_dims.work_size())
		.arg_env(&source_envoy)
		.arg_scl(coeff)
		.arg_env(&mut result_envoy)
	;

	// Enqueue kernel with no events:
	kernel.enqueue(None, None);

	// Read results:
	result_envoy.read_wait();

	// Check results:
	for idx in 0..data_set_size {
		assert_eq!(result_envoy[idx], source_envoy[idx] * coeff);

		// if idx < 20 { 
		// 	println!("source_envoy[idx]: {}, coeff: {}, result_envoy[idx]: {}",
		// 	source_envoy[idx], coeff, result_envoy[idx]); 
		// }
	}
}
