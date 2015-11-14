use libc::{ c_void };
// use std::ptr;
use cl_h::{ cl_event, cl_int };

use super::{ Context, BuildConfig, Envoy, SimpleDims, ProQue, EventList };


struct TestStuff {
	env: *const Envoy<f32>, 
	dss: usize,
	scl: f32, 
	itr: usize,
}

#[allow(unused_variables, dead_code)]
extern fn verify_result(event: cl_event, status: cl_int, user_data: *mut c_void) {
	let buncha_stuff = user_data as *const TestStuff;

	unsafe {
		let result_envoy: *const Envoy<f32> = (*buncha_stuff).env as *const Envoy<f32>;
		let data_set_size: usize = (*buncha_stuff).dss;
		let scalar: f32 = (*buncha_stuff).scl;
		let itr: usize = (*buncha_stuff).itr;
		
	    // println!("Event: `{:?}` has completed with status: `{}`, data_set_size: '{}`, \
	    // 	 scalar: {}, itr: `{}`.", event, status, data_set_size, scalar, itr);

		for idx in 0..data_set_size {
			assert_eq!((*result_envoy)[idx], ((itr + 1) as f32) * scalar);

			// if idx < 20 {
			// 	print!("[{}]", (*result_envoy)[idx]);
			// }
		}
		// print!("\n\n");
    }
}


#[test]
fn test_async_events() {
	// Create a context & program/queue: 
	let mut ocl_pq = ProQue::new(&Context::new(None, None).unwrap(), None);

	// Build program:
	ocl_pq.build(BuildConfig::new().kern_file("cl/kernel_file.cl")).unwrap();

	// Set up data set size and work dimensions:
	let data_set_size = 900000;
	let envoy_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result envoys (our data containers):
	// let source_envoy = Envoy::shuffled(&envoy_dims, 0f32, 20f32, &ocl_pq);
	let mut result_envoy = Envoy::new(&envoy_dims, 0f32, &ocl_pq.queue());

	// Our scalar:
	let scalar = 5.0f32;

	// Create kernel:
	let kernel = ocl_pq.create_kernel("add_scalar".to_string(), envoy_dims.work_size())
		.arg_scl(scalar)
		.arg_env(&mut result_envoy)
	;

	// Create event list:
	let mut kernel_event = EventList::new();	

	//#############################################################################################

	let iters = 20;

	// Create storage for per-event data:
	let mut buncha_stuffs = Vec::<TestStuff>::with_capacity(iters);

	// Repeat the test. First iteration 
	for i in 0..iters {
		println!("Enqueuing kernel [i:{}]...", i);
		kernel.enqueue(None, Some(&mut kernel_event));

		let mut read_event = EventList::new();
		
		println!("Enqueuing read buffer [i:{}]...", i);
		result_envoy.read(None, Some(&mut read_event));

		// unsafe {			
		// 	super::enqueue_read_buffer(ocl_pq.queue().obj(), result_envoy.buffer_obj(), false, 
		// 			result_envoy.vec_mut(), 0, None, Some(&mut read_event));
		// }

		buncha_stuffs.push(TestStuff {
			env: &result_envoy as *const Envoy<f32>, 
			dss: data_set_size, 
			scl: scalar, 
			itr: i,
		});

		let last_idx = buncha_stuffs.len() - 1;		

		unsafe {
			println!("Setting callback (verify_result, buncha_stuff[{}]) [i:{}]...", last_idx, i);
			read_event.set_callback(verify_result, 
				&mut buncha_stuffs[last_idx] as *mut _ as *mut c_void);
		}

		println!("Releasing read_event [i:{}]...", i);
		read_event.release();
	}

	// Wait for all queued tasks to finish so that verify_result() will be called:
	ocl_pq.queue().finish();
}



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
	let data_set_size = 900000;
	let envoy_dims = SimpleDims::OneDim(data_set_size);

	// Create a source envoy (array) with randomized values and an empty result envoy:
	let source_envoy = Envoy::scrambled(&envoy_dims, 0.0f32, 200.0, &ocl_pq.queue());
	let mut result_envoy = Envoy::new(&envoy_dims, 0.0f32, &ocl_pq.queue());

	// Our coefficient:
	let coeff = 432.1;

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

		if idx < 20 { 
			println!("source_envoy[idx]: {}, coeff: {}, result_envoy[idx]: {}",
			source_envoy[idx], coeff, result_envoy[idx]); 
		}
	}
}
