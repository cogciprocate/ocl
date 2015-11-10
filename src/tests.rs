use super::{ Context, BuildOptions, Envoy, SimpleDims, ProQueue, EventList };


#[test]
fn test_async_events() {
	// Create a context & program/queue: 
	let mut ocl_pq = ProQueue::new(&Context::new(None, None).unwrap(), None);

	// Build program:
	ocl_pq.build(BuildOptions::new("").kern_file("cl/kernel_file.cl".to_string())).unwrap();

	// Set up data set size and work dimensions:
	let data_set_size = 100;
	let env_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result envoys (our data containers):
	let env_source = Envoy::shuffled(&env_dims, 0f32, 20f32, &ocl_pq);
	let mut env_result = Envoy::new(&env_dims, 0f32, &ocl_pq);

	// Our coefficient:
	let coeff = 5f32;

	// Create kernel:
	let kernel = ocl_pq.create_kernel("multiply_by_scalar".to_string(), env_dims.work_size())
		.arg_env(&env_source)
		.arg_scl(coeff)
		.arg_env(&mut env_result)
	;

	// Create event list:
	let mut kernel_event = EventList::new();

	// Enqueue kernel:
	kernel.enqueue(None, Some(&mut kernel_event));

	// Read results:
	env_result.read();

	// Check results:
	for idx in 0..data_set_size {
		assert_eq!(env_result[idx], env_source[idx] * coeff);
	}
}


#[test]
fn test_basics() {
	// Create a context:
	let ocl_cxt = Context::new(None, None).unwrap();

	// Create a program/queue: 
	let mut ocl_pq = ProQueue::new(&ocl_cxt, None);

	// Create build options passing optional command line switches and other options:
	let build_options = BuildOptions::new("-cl-unsafe-math-optimizations")
		.kern_file("cl/kernel_file.cl".to_string());

	// Build:
	ocl_pq.build(build_options).unwrap();

	// Set up our data set size and work dimensions:
	let data_set_size = 100;
	let env_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result envoys (our data containers):
	let env_source = Envoy::shuffled(&env_dims, 0f32, 20f32, &ocl_pq);
	let mut env_result = Envoy::new(&env_dims, 0f32, &ocl_pq);

	// Our coefficient:
	let coeff = 5f32;

	// Create our kernel:
	let kernel = ocl_pq.create_kernel("multiply_by_scalar".to_string(), env_dims.work_size())
		.arg_env(&env_source)
		.arg_scl(coeff)
		.arg_env(&mut env_result)
	;

	// Enqueue kernel:
	kernel.enqueue(None, None);

	// Read results:
	env_result.read();

	// Check results:
	for idx in 0..data_set_size {
		assert_eq!(env_result[idx], env_source[idx] * coeff);
	}
}
