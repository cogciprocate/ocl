use super::{ Context, BuildConfig, Envoy, SimpleDims, ProQue, EventList };


#[test]
fn test_async_events() {
	// Create a context & program/queue: 
	let mut ocl_pq = ProQue::new(&Context::new(None, None).unwrap(), None);

	// Build program:
	ocl_pq.build(BuildConfig::new().kern_file("cl/kernel_file.cl")).unwrap();

	// Set up data set size and work dimensions:
	let data_set_size = 100;
	let envoy_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result envoys (our data containers):
	// let source_envoy = Envoy::shuffled(&envoy_dims, 0f32, 20f32, &ocl_pq);
	let mut result_envoy = Envoy::new(&envoy_dims, 0f32, &ocl_pq.queue());

	// Our scalar:
	let scalar = 5f32;

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
	for i in 1..20 {
		kernel.enqueue(None, Some(&mut kernel_event));

		let mut read_event = EventList::new();

		unsafe {
			// Altering `wait` bool will affect success if no wait list is used.
			// Toggle passing an event list or not:
			if false {
				super::enqueue_read_buffer(ocl_pq.queue().obj(), result_envoy.buffer_obj(), false, 
					result_envoy.vec_mut(), 0, None, None);
			} else {
				super::enqueue_read_buffer(ocl_pq.queue().obj(), result_envoy.buffer_obj(), false, 
					result_envoy.vec_mut(), 0, None, Some(&mut read_event));
			}
		}

		// read_event.wait();
		read_event.set_callback();
		read_event.wait();

		// result_envoy.read(Some(&kernel_event), None);

		for idx in 0..data_set_size {
			assert_eq!(result_envoy[idx], (i as f32) * scalar);
		}
	}

	panic!();
}

// fn verify_callback() {
// 	for idx in 0..data_set_size {
// 		assert_eq!(result_envoy[idx], (i as f32) * scalar);
// 	}
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

		if idx < 20 { 
			println!("source_envoy[idx]: {}, coeff: {}, result_envoy[idx]: {}",
			source_envoy[idx], coeff, result_envoy[idx]); 
		}
	}
}
