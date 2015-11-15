# ocl

OpenCL interfaces for Rust. Makes easy to use the most common features of OpenCL. All interfaces are virtually zero-cost and will have runtime performance on a par with standard C++ libraries.

Interfaces are still unstable. Probably won't eat your laundry but some of the interfaces may change (in hopefully obvious ways).


##Goals

To provide a simple and intuitive way to interact with OpenCL devices with:
- The full functionality of the OpenCL C ABI
- An absolute minimum of boilerplate
- Zero performance overhead (or as close as possible)


##Platforms

Tested only on Linux. Please [provide feedback](https://github.com/cogciprocate/ocl_rust/issues) about failures and successes on other platforms.


##Installation

Ensure that an OpenCL library is installed for your preferred platform. Remember that Intel and AMD both have OpenCL libraries for your CPU if you're having trouble getting your GPU to work. Make sure that `clinfo` or some other diagnostic command will run. 

*If/when troubleshooting your preferred OpenCL drivers: check that `/usr/lib/libOpenCL.so.1` exists. Go ahead and link `/usr/lib/libOpenCL.so -> libOpenCL.so.1` just in case it's not already.*


Add:

```
[dependencies]
ocl = "0.1"
```

or (to live dangerously):

```
[dependencies.ocl]
git = "https://github.com/cogciprocate/ocl_rust.git"
```

to your project's `Cargo.toml` then, of course:

```
extern crate ocl;
```

to your crate main file (`main.rs` or `lib.rs`).


##Usage
```
use ocl::{ Context, ProQueue, BuildOptions, SimpleDims, Envoy };

fn main() {
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

		if idx < 20 { 
			println!("source_envoy[idx]: {}, coeff: {}, result_envoy[idx]: {}",
			source_envoy[idx], coeff, result_envoy[idx]); 
		}
	}
}
```

`.\cl\kernel_file.cl` contents:

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

##Help

Please ask questions and provide any positive or negative feedback by opening an [issue](https://github.com/cogciprocate/ocl_rust/issues).


##Upcoming
Event queuing/waiting was temporarily removed for polishing and will be coming back for 0.2.0. *[Update] Currently in on master branch. Cargo crate update coming soon.*

Lots more details and documentation to come.
