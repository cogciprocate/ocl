

#[test]
fn test_basics() {
	use super::{ Context, BuildOptions, Envoy, SimpleDims, ProQueue };

	// Create a context:
	let ocl_cxt = Context::new(None, None).unwrap();

	// Create a Program/Queue: 
	let mut ocl_pq = ProQueue::new(&ocl_cxt, None);

	// Create build options passing optional command line switches and other options:
	let build_options = BuildOptions::new("-cl-unsafe-math-optimizations")
		.kern_file("cl/kernel_file.cl".to_string());

	// Build:
	ocl_pq.build(build_options).unwrap();

	// Set up our work dimensions:
	let data_set_size = 100;
	let env_dims = SimpleDims::OneDim(data_set_size);

	// Create source and destination Envoys (our data containers):
	let src_env = Envoy::shuffled(&env_dims, 0f32, 20f32, &ocl_pq);
	let mut dst_env = Envoy::new(&env_dims, 0f32, &ocl_pq);

	// Our coefficient:
	let coeff = 5f32;

	// Create our kernel:
	let kernel = ocl_pq.new_kernel("multiply_by_scalar".to_string(), env_dims.work_size())
		.arg_env(&src_env)
		.arg_scl(coeff)
		.arg_env(&mut dst_env)
	;

	// Enqueue kernel:
	kernel.enqueue();

	// Read results:
	dst_env.read();

	// Check results:
	for idx in 0..data_set_size {
		assert_eq!(dst_env[idx], src_env[idx] * coeff);
	}
}
