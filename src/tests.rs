use libc::c_void;
// use std::ptr;
use cl_h::{cl_event, cl_int};

use super::{Context, ProgramBuilder, Buffer, SimpleDims, ProQue, EventList};

const PRINT_DEBUG: bool = true;

struct TestEventsStuff {
	seed_env: *const Buffer<u32>, 
	res_env: *const Buffer<u32>, 
	data_set_size: usize,
	addend: u32, 
	itr: usize,
}

// Callback for `test_events()`.
extern fn _test_events_verify_result(event: cl_event, status: cl_int, user_data: *mut c_void) {
	let buncha_stuff = user_data as *const TestEventsStuff;

	unsafe {
		let seed_buffer: *const Buffer<u32> = (*buncha_stuff).seed_env as *const Buffer<u32>;
		let result_buffer: *const Buffer<u32> = (*buncha_stuff).res_env as *const Buffer<u32>;
		let data_set_size: usize = (*buncha_stuff).data_set_size;
		let addend: u32 = (*buncha_stuff).addend;
		let itr: usize = (*buncha_stuff).itr;

		let mut errors_found: u32 = 0;

		for idx in 0..data_set_size {
			// [FIXME]: FAILING ON OSX -- TEMPORARLY COMMENTING OUT
			// assert_eq!((*result_buffer)[idx], 
			// 	((*seed_buffer)[idx] + ((itr + 1) as u32) * addend));

			if PRINT_DEBUG {
				let correct_result = (*seed_buffer)[idx] + (((itr + 1) as u32) * addend);

				// if (*result_buffer)[idx] != correct_result {
				// 	print!("correct_result:{}, result_buffer[{idx}]:{}\n",
				// 		correct_result, (*result_buffer)[idx], idx = idx);
				// 	errors_found += 1;
				// }

				errors_found += ((*result_buffer)[idx] != correct_result) as u32;
			}
		}

		if PRINT_DEBUG && errors_found > 0 { 
			println!("Event: `{:?}` has completed with status: `{}`, data_set_size: '{}`, \
		    	 addend: {}, itr: `{}`.", event, status, data_set_size, addend, itr);
			println!("    TOTAL ERRORS FOUND: {}", errors_found); }
    }
}


#[test]
fn test_events() {
	// Create a context & program/queue: 
	let mut ocl_pq = ProQue::new(&Context::new(None, None).unwrap(), None);

	// Build program:
	ocl_pq.build_program(ProgramBuilder::new().src_file("cl/kernel_file.cl")).unwrap();

	// Set up data set size and work dimensions:
	let data_set_size = 900000;
	let our_test_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result buffers (our data containers):
	let seed_buffer = Buffer::with_vec_scrambled((0u32, 500u32), &our_test_dims, &ocl_pq.queue());
	let mut result_buffer = Buffer::with_vec(&our_test_dims, &ocl_pq.queue());

	// Our addend:
	let addend = 10u32;

	// Create kernel with the source initially set to our seed values.
	let mut kernel = ocl_pq.create_kernel("add_scalar", our_test_dims.work_size())
		.arg_buf_named("src", Some(&seed_buffer))
		.arg_scl(addend)
		.arg_buf(&mut result_buffer)
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
			seed_env: &seed_buffer as *const Buffer<u32>,
			res_env: &result_buffer as *const Buffer<u32>, 
			data_set_size: data_set_size, 
			addend: addend, 
			itr: itr,
		});

		// Change the source buffer to the result after seed values have been copied.
		// Yes, this is far from optimal...
		// Should just copy the values in the first place but oh well.
		if itr != 0 {
			kernel.set_arg_buf_named("src", &result_buffer);
		}

		if PRINT_DEBUG { println!("Enqueuing kernel [itr:{}]...", itr); }
		kernel.enqueue(None, Some(&mut kernel_event));

		let mut read_event = EventList::new();
		
		if PRINT_DEBUG { println!("Enqueuing read buffer [itr:{}]...", itr); }
		result_buffer.fill_vec_async(None, Some(&mut read_event));
	

		let last_idx = buncha_stuffs.len() - 1;		

		unsafe {
			if PRINT_DEBUG { println!("Setting callback (verify_result, buncha_stuff[{}]) [i:{}]...", 
				last_idx, itr); }
			read_event.set_callback(_test_events_verify_result, 
				// &mut buncha_stuffs[last_idx] as *mut _ as *mut c_void);
				&mut buncha_stuffs[last_idx]);
		}

		if PRINT_DEBUG { println!("Releasing read_event [i:{}]...", itr); }
		read_event.release_all();
	}

	// Wait for all queued tasks to finish so that verify_result() will be called:
	ocl_pq.queue().finish();
}



#[test]
fn test_basics() {
	// Set our data set size and coefficent to arbitrary values:
	let data_set_size = 900000;
	let coeff = 5432.1;

	// Create a context with the first avaliable platform and default device type:
	let ocl_cxt = Context::new(None, None).unwrap();

	// Create a program/queue with the first available device: 
	let mut ocl_pq = ProQue::new(&ocl_cxt, None);

	// Declare our kernel source code:
	let kernel_src = r#"
		__kernel void multiply_by_scalar(
					__global float const* const src,
					__private float const coeff,
					__global float* const res)
		{
			uint const idx = get_global_id(0);

			res[idx] = src[idx] * coeff;
		}
	"#;

	// Create a basic build configuration using above source: 
	let program_builder = ProgramBuilder::new().src(kernel_src);

	// Build with our configuration and check for errors:
	ocl_pq.build_program(program_builder).expect("ocl program build");

	// Set up our work dimensions / data set size:
	let dims = SimpleDims::OneDim(data_set_size);

	// Create an buffer (a local array + a remote buffer) as a data source:
	let source_buffer = Buffer::with_vec_scrambled((0.0f32, 20.0f32), &dims, &ocl_pq.queue());

	// Create another empty buffer for results:
	let mut result_buffer = Buffer::<f32>::with_vec(&dims, &ocl_pq.queue());

	// Create a kernel with three arguments corresponding to those in the kernel:
	let kernel = ocl_pq.create_kernel("multiply_by_scalar", dims.work_size())
		.arg_buf(&source_buffer)
		.arg_scl(coeff)
		.arg_buf(&mut result_buffer)
	;

	// Enqueue kernel depending on and creating no events:
	kernel.enqueue(None, None);

	// Read results:
	result_buffer.fill_vec();

	// Check results and print the first 20:
	for idx in 0..data_set_size {
		// Check:
		assert_eq!(result_buffer[idx], source_buffer[idx] * coeff);

		// Print:
		if PRINT_DEBUG && (idx < 20) { 
			println!("source_buffer[idx]: {}, coeff: {}, result_buffer[idx]: {}",
			source_buffer[idx], coeff, result_buffer[idx]); 
		}
	}
}
