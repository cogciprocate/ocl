#Ocl

OpenCL interfaces for Rust. Makes easy to use many of the most useful features of OpenCL such as creating kernels and data containers. Advanced features are supported via direct rust wrapper function calls. All interfaces and wrappers are virtually zero-cost and will have runtime performance on a par with standard C++ libraries.

Interfaces are highly unstable for a few more weeks. Probably won't eat your laundry but will break constantly until 0.1 which should be ready before December 2015.

##Installation

Ensure that OpenCL is installed for your preferred platform. Remember that Intel and AMD both have OpenCL libraries for your CPU if you're having trouble getting your GPU to work. Make sure that `clinfo` or some other diagnostic command will run. You may want to check that `/usr/lib/libOpenCL.so.1` exists. Go ahead and link `/usr/lib/libOpenCL.so -> libOpenCL.so.1` just in case it's not already.

Add

```
[dependencies]
ocl = "0.1"
```

or (to be cutting edge)

```
[dependencies.ocl]
git = "https://github.com/cogciprocate/ocl_rust.git"
```

to your project's `Cargo.toml` then, of course

```
extern crate ocl;
```

to your crate main file (`main.rs` or `lib.rs`).


##Usage
```
use ocl::{ Context, BuildOptions, Envoy, SimpleDims, ProQueue };

fn main() {
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

	// Create a kernel:
	let kernel = ocl_pq.new_kernel("multiply_by_scalar".to_string(), env_dims.work_size())
		.arg_env(&src_env)
		.arg_scl(coeff)
		.arg_env(&mut dst_env)
	;

	// Enqueue our kernel:
	kernel.enqueue();

	// Read results from device:
	dst_env.read();

	// Check results:
	for idx in 0..data_set_size {
		assert_eq!(dst_env[idx], src_env[idx] * coeff);
	}
}
```

`kernel_file.cl` contents:
```
__kernel void multiply_by_scalar(
			__global float const* const src,
			__private float const coeff,
			__global float* const dst)
{
	uint const idx = get_global_id(0);

	dst[idx] = src[idx] * coeff;
}

```


More details and documentation to come.
