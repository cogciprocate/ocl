# ocl

[![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl)

OpenCL interfaces for Rust. Makes easy to use the most common features of OpenCL. All interfaces are virtually zero-cost and perform on a par with any C++ libraries.

Interfaces are still unstable. Won't eat your laundry but some of the conventions may change (in hopefully obvious ways).


##Goals

To provide a simple and intuitive way to interact with OpenCL devices with:
- The full functionality of the OpenCL C ABI
- An absolute minimum of boilerplate
- Zero performance overhead (or as close as absolutely possible)


##Platforms

Tested so far only on Linux. Please [provide feedback](https://github.com/cogciprocate/ocl_rust/issues) about failures and successes on other platforms.


##Installation

Ensure that an OpenCL library is installed for your preferred platform and  that `clinfo` or some other diagnostic command will run.

Add:

```
[dependencies]
ocl = "0.2"
```

to your project's `Cargo.toml`.


##Example

Create `{your_project_dir}\cl\kernel_file.cl` with the following contents:

```
__kernel void multiply_by_scalar(
			__global float const* const src,
			__private float const coeff,
			__global float* const res)
{
	uint const idx = get_global_id(0);

	res[idx] = src[idx] * coeff;
}

```

`main.rs`:

```
use ocl::{ Context, ProQue, BuildConfig, SimpleDims, Envoy };
extern crate ocl;

const PRINT_DEBUG: bool = true;

fn main() {
	// Set our data set size and coefficent to arbitrary values:
	let data_set_size = 900000;
	let coeff = 5432.1;

	// Create a context with the default platform and device type (GPU):
	// * Use: `Context::new(None, Some(ocl::CL_DEVICE_TYPE_CPU))` for CPU.
	let ocl_cxt = Context::new(None, None).unwrap();

	// Create a program/queue with the first available device: 
	let mut ocl_pq = ProQue::new(&ocl_cxt, None);

	// Create a basic build configuration:
	let build_config = BuildConfig::new().kern_file("cl/kernel_file.cl");

	// Build with our configuration and check for errors:
	ocl_pq.build(build_config).expect("ocl program build");

	// Set up our work dimensions / data set size:
	let our_test_dims = SimpleDims::OneDim(data_set_size);

	// Create an envoy (an array + an OpenCL buffer) as a data source:
	let source_envoy = Envoy::scrambled(&our_test_dims, 0.0f32, 200.0, &ocl_pq.queue());

	// Create another empty one for results:
	let mut result_envoy = Envoy::new(&our_test_dims, 0.0f32, &ocl_pq.queue());

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
		// Check:
		assert_eq!(result_envoy[idx], source_envoy[idx] * coeff);

		// Print:
		if PRINT_DEBUG && (idx < 20) { 
			println!("source_envoy[idx]: {}, coeff: {}, result_envoy[idx]: {}",
			source_envoy[idx], coeff, result_envoy[idx]); 
		}
	}
}
```

##Upcoming Changes

Envoy will be receiving a major rework and a possible rename. [Issue #4](https://github.com/cogciprocate/ocl/issues/4).

##Help

*If troubleshooting your OpenCL drivers:* check that `/usr/lib/libOpenCL.so.1` exists. Go ahead and link `/usr/lib/libOpenCL.so -> libOpenCL.so.1` just in case it's not already (AMD drivers sometimes don't create this link).  Intel and AMD also have OpenCL libraries for your CPU if you're having trouble getting your GPU to work (intel: [windows](http://registrationcenter.intel.com/irc_nas/5198/opencl_runtime_15.1_x64_setup.msi), [linux](http://registrationcenter.intel.com/irc_nas/5193/opencl_runtime_15.1_x64_5.0.0.57.tgz)). 

Please ask questions and provide feedback by opening an [issue](https://github.com/cogciprocate/ocl_rust/issues).



Lots more details and documentation to come.
