use libc::{ c_void };
// use std::ptr;
use cl_h::{ cl_event, cl_int };

use super::{ Context, BuildConfig, Envoy, SimpleDims, ProQue, EventList };

const PRINT_DEBUG: bool = false;

struct TestEventsStuff {
	seed_env: *const Envoy<u32>, 
	res_env: *const Envoy<u32>, 
	data_set_size: usize,
	addend: u32, 
	itr: usize,
}

// Callback for `test_events()`.
extern fn _test_events_verify_result(event: cl_event, status: cl_int, user_data: *mut c_void) {
	let buncha_stuff = user_data as *const TestEventsStuff;

	unsafe {
		let seed_envoy: *const Envoy<u32> = (*buncha_stuff).seed_env as *const Envoy<u32>;
		let result_envoy: *const Envoy<u32> = (*buncha_stuff).res_env as *const Envoy<u32>;
		let data_set_size: usize = (*buncha_stuff).data_set_size;
		let addend: u32 = (*buncha_stuff).addend;
		let itr: usize = (*buncha_stuff).itr;
		
		if PRINT_DEBUG { println!("Event: `{:?}` has completed with status: `{}`, data_set_size: '{}`, \
		    	 addend: {}, itr: `{}`.", event, status, data_set_size, addend, itr); }

		for idx in 0..data_set_size {
			assert_eq!((*result_envoy)[idx], ((*seed_envoy)[idx] + ((itr + 1) as u32) * addend));

			if PRINT_DEBUG && (idx < 20) {
				print!("[{}]", (*result_envoy)[idx]);
			}
		}

		if PRINT_DEBUG { print!("\n\n"); }
    }
}


#[test]
fn test_events() {
	// Create a context & program/queue: 
	let mut ocl_pq = ProQue::new(&Context::new(None, None).unwrap(), None);

	// Build program:
	ocl_pq.build(BuildConfig::new().kern_file("cl/kernel_file.cl")).unwrap();

	// Set up data set size and work dimensions:
	let data_set_size = 900000;
	let our_test_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result envoys (our data containers):
	let seed_envoy = Envoy::scrambled(&our_test_dims, 0u32, 500u32, &ocl_pq.queue());
	let mut result_envoy = Envoy::new(&our_test_dims, 0u32, &ocl_pq.queue());

	// Our addend:
	let addend = 10u32;

	// Create kernel with the source initially set to our seed values.
	let mut kernel = ocl_pq.create_kernel("add_scalar", our_test_dims.work_size())
		.arg_env_named("src", Some(&seed_envoy))
		.arg_scl(addend)
		.arg_env(&mut result_envoy)
	;

	// Create event list:
	let mut kernel_event = EventList::new();	

	//#############################################################################################

	// Define how many iterations we wish to run:
	let iters = 20;

	// Create storage for per-event data:
	let mut buncha_stuffs = Vec::<TestEventsStuff>::with_capacity(iters);

	// Run our test:
	for itr in 0..iters {
		// Store information for use by the result callback function into a vector
		// which will persist until all of the commands have completed (as long as
		// we are sure to allow the queue to finish before returning).
		buncha_stuffs.push(TestEventsStuff {
			seed_env: &seed_envoy as *const Envoy<u32>,
			res_env: &result_envoy as *const Envoy<u32>, 
			data_set_size: data_set_size, 
			addend: addend, 
			itr: itr,
		});

		// Change the source envoy to the result after seed values have been copied.
		// Yes, this is far from optimal...
		// Should just copy the values in the first place but oh well.
		if itr != 0 {
			kernel.set_arg_env_named("src", &result_envoy);
		}

		if PRINT_DEBUG { println!("Enqueuing kernel [itr:{}]...", itr); }
		kernel.enqueue(None, Some(&mut kernel_event));

		let mut read_event = EventList::new();
		
		if PRINT_DEBUG { println!("Enqueuing read buffer [itr:{}]...", itr); }
		result_envoy.read(None, Some(&mut read_event));
	

		let last_idx = buncha_stuffs.len() - 1;		

		unsafe {
			if PRINT_DEBUG { println!("Setting callback (verify_result, buncha_stuff[{}]) [i:{}]...", 
				last_idx, itr); }
			read_event.set_callback(_test_events_verify_result, 
				// &mut buncha_stuffs[last_idx] as *mut _ as *mut c_void);
				&mut buncha_stuffs[last_idx]);
		}

		if PRINT_DEBUG { println!("Releasing read_event [i:{}]...", itr); }
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
	let our_test_dims = SimpleDims::OneDim(data_set_size);

	// Create a source envoy (array) with randomized values and an empty result envoy:
	let source_envoy = Envoy::scrambled(&our_test_dims, 0.0f32, 200.0, &ocl_pq.queue());
	let mut result_envoy = Envoy::new(&our_test_dims, 0.0f32, &ocl_pq.queue());

	// Our coefficient:
	let coeff = 5432.1;

	// Create kernel:
	let kernel = ocl_pq.create_kernel("multiply_by_scalar", our_test_dims.work_size())
		.arg_env(&source_envoy)
		.arg_scl(coeff)
		.arg_env(&mut result_envoy)
	;

	// Enqueue kernel depending on and creating no events:
	kernel.enqueue(None, None);

	// Read results:
	result_envoy.read_wait();

	// Check results and print the first 20:
	for idx in 0..data_set_size {
		assert_eq!(result_envoy[idx], source_envoy[idx] * coeff);

		if PRINT_DEBUG && (idx < 20) { 
			println!("source_envoy[idx]: {}, coeff: {}, result_envoy[idx]: {}",
			source_envoy[idx], coeff, result_envoy[idx]); 
		}
	}
}
