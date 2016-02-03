# ocl

[![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl)


**[Documentation](http://doc.cogciprocate.com/ocl/)**

OpenCL interfaces for Rust. Makes easy to use the most common features of OpenCL. All interfaces are virtually zero-cost and perform on a par with any C++ libraries.

Interfaces are still unstable. Won't eat your laundry but some of the conventions may change (in hopefully obvious ways).


##Goals

To provide a simple and intuitive way to interact with OpenCL devices with:
- The full functionality of the OpenCL C ABI 
   - *This is a work in progress. Please [file an issue](https://github.com/cogciprocate/ocl_rust/issues) about any functionality you might want so it can be prioritized.*
- An absolute minimum of boilerplate
- As close as possible to zero performance overhead


##Platforms

Tested so far only on Linux. Please [provide feedback](https://github.com/cogciprocate/ocl_rust/issues) about failures and successes on other platforms. *Note: Probably working in OS X, need confirmation.*


##Installation

Ensure that an OpenCL library is installed for your preferred platform and  that `clinfo` or some other diagnostic command will run.

Add:

```
[dependencies]
ocl = "0.4"
```

to your project's `Cargo.toml`.


##Example 
From 'examples/basics.rs':

```
use ocl::{Context, ProQue, BuildConfig, SimpleDims, Buffer};
extern crate ocl;

const RESULTS_TO_PRINT: usize = 20;

fn main() {
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
	let build_config = BuildConfig::new().kern_embed(kernel_src);

	// Build with our configuration and check for errors:
	ocl_pq.build(build_config).expect("ocl program build");

	// Set up our work dimensions / data set size:
	let dims = SimpleDims::OneDim(data_set_size);

	// Create a 'Buffer' (a local vector + a remote buffer) as a data source:
	let source_buffer: Buffer<f32> = 
		Buffer::with_vec_scrambled(0.0f32, 20.0f32, &dims, &ocl_pq.queue());

	// Create another empty buffer for results:
	let mut result_buffer: Buffer<f32> = Buffer::with_vec(&dims, &ocl_pq.queue());

	// Create a kernel with three arguments corresponding to those in the kernel:
	let kernel = ocl_pq.create_kernel("multiply_by_scalar", dims.work_size())
		.arg_env(&source_buffer)
		.arg_scl(coeff)
		.arg_env(&mut result_buffer)
	;

	// Enqueue kernel depending on and creating no events:
	kernel.enqueue(None, None);

	// Read results from the device into the buffer's vector:
	result_buffer.fill_vec_wait();

	// Check results and print the first 20:
	for idx in 0..data_set_size {
		// Check:
		assert_eq!(result_buffer[idx], source_buffer[idx] * coeff);

		// Print:
		if idx < RESULTS_TO_PRINT { 
			println!("source_buffer[idx]: {}, coeff: {}, result_buffer[idx]: {}",
			source_buffer[idx], coeff, result_buffer[idx]); 
		}
	}
}

```

###Recent Changes

'Envoy' has undergone a major redesign: [Issue #4](https://github.com/cogciprocate/ocl/issues/4) and has been renamed to 'Buffer'.

###Upcoming Changes

At the top of the list are cleaning up and consolidating error handling [Issue #8](https://github.com/cogciprocate/ocl/issues/8) and finishing [documentation](http://doc.cogciprocate.com/ocl/) (now about 60% complete).

##Help

*If troubleshooting your OpenCL drivers:* check that `/usr/lib/libOpenCL.so.1` exists. Go ahead and link `/usr/lib/libOpenCL.so -> libOpenCL.so.1` just in case it's not already (AMD drivers sometimes don't create this link).  Intel and AMD also have OpenCL libraries for your CPU if you're having trouble getting your GPU to work (intel: [windows](http://registrationcenter.intel.com/irc_nas/5198/opencl_runtime_15.1_x64_setup.msi), [linux](http://registrationcenter.intel.com/irc_nas/5193/opencl_runtime_15.1_x64_5.0.0.57.tgz)). 

Please ask questions and provide feedback by opening an [issue](https://github.com/cogciprocate/ocl_rust/issues).

