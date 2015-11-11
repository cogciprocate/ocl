use super::{ Context, BuildOptions, Envoy, SimpleDims, ProQueue, EventList };


#[test]
fn test_async_events() {
	// Create a context & program/queue: 
	let mut ocl_pq = ProQueue::new(&Context::new(None, None).unwrap(), None);

	// Build program:
	ocl_pq.build(BuildOptions::new("").kern_file("cl/kernel_file.cl".to_string())).unwrap();

	// Set up data set size and work dimensions:
	let data_set_size = 100;
	let envoy_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result envoys (our data containers):
	// let source_envoy = Envoy::shuffled(&envoy_dims, 0f32, 20f32, &ocl_pq);
	let mut result_envoy = Envoy::new(&envoy_dims, 0f32, &ocl_pq);

	// Our scalar:
	let scalar = 1f32;

	// Create kernel:
	let kernel = ocl_pq.create_kernel("add_scalar".to_string(), envoy_dims.work_size())
		.arg_scl(scalar)
		.arg_env(&mut result_envoy)
	;

	// Create event list:
	let mut kernel_event = EventList::new();

	//#############################################################################################

	// Repeat the test. First iteration 
	for i in 1..20 {
		kernel.enqueue(None, Some(&mut kernel_event));
		result_envoy.read_wait();

		for idx in 0..data_set_size {
			assert_eq!(result_envoy[idx], i as f32);
		}
	}
}




#[test]
fn test_basics() {
	// Create a context with the default platform and device types:
	let ocl_cxt = Context::new(None, None).unwrap();

	// Create a program/queue with the default device: 
	let mut ocl_pq = ProQueue::new(&ocl_cxt, None);

	// Create build options passing optional command line switches and other options:
	let build_options = BuildOptions::new("-cl-unsafe-math-optimizations")
		.kern_file("cl/kernel_file.cl".to_string());

	// Build:
	ocl_pq.build(build_options).unwrap();

	// Set up our data set size and work dimensions:
	let data_set_size = 100;
	let envoy_dims = SimpleDims::OneDim(data_set_size);

	// Create source and result envoys (our data containers):
	let source_envoy = Envoy::shuffled(&envoy_dims, 0f32, 20f32, &ocl_pq);
	let mut result_envoy = Envoy::new(&envoy_dims, 0f32, &ocl_pq);

	// Our coefficient:
	let coeff = 5f32;

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
	}
}
