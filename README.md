# ocl

[![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl)
** [Documentation](http://doc.cogciprocate.com/ocl/) | [Recent Changes](https://github.com/cogciprocate/ocl/blob/master/RELEASES.md) **

OpenCL interfaces for Rust. Makes easy to use the most common features of OpenCL. All interfaces are virtually zero-cost and perform on a par with any C++ libraries.

Interfaces are still unstable. Won't eat your laundry but some of the conventions may change (in hopefully obvious ways).


## Goals

To provide a simple and intuitive way to interact with OpenCL devices with:
- The full functionality of the OpenCL C ABI 
   - *This is a work in progress. Please [file an issue](https://github.com/cogciprocate/ocl_rust/issues) about any functionality you might want so it can be prioritized.*
- An absolute minimum of boilerplate
- As close as possible to zero performance overhead


## Platforms

Tested so far only on Linux. Please [provide feedback](https://github.com/cogciprocate/ocl_rust/issues) about failures and successes on other platforms. *Note: Probably working in OS X, need confirmation.*


## Installation

Ensure that an OpenCL library is installed for your preferred platform and  that `clinfo` or some other diagnostic command will run.

Add:

```
[dependencies]
ocl = "0.5"
```

to your project's `Cargo.toml`.


## Example 
From 'examples/basics.rs':

```
extern crate ocl;

// Number of results to print out:
const RESULTS_TO_PRINT: usize = 20;

// Our arbitrary data set size and coefficent:
const DATA_SET_SIZE: usize = 900000;
const COEFF: f32 = 5432.1;

// Our kernel source code:
static KERNEL_SRC: &'static str = r#"
	__kernel void multiply_by_scalar(
				__global float const* const src,
				__private float const coeff,
				__global float* const res)
	{
		uint const idx = get_global_id(0);

		res[idx] = src[idx] * coeff;
	}
"#;


fn main() {
	use ocl::{ProQue, SimpleDims, Buffer};

	// Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
	let ocl_pq = ProQue::builder().src(KERNEL_SRC).build().expect("ProQue build");

	// Set up our work dimensions / data set size:
	let dims = SimpleDims::OneDim(DATA_SET_SIZE);

	// Create a 'Buffer' (a device buffer + a local vector) as a data source
	// and initialize it with random floats between 0.0 and 20.0:
	let source_buffer: Buffer<f32> = 
		Buffer::with_vec_scrambled((0.0, 20.0), &dims, &ocl_pq.queue());

	// Create another empty buffer for results:
	let mut result_buffer: Buffer<f32> = Buffer::with_vec(&dims, &ocl_pq.queue());

	// Create a kernel with three arguments corresponding to those in the kernel:
	let kernel = ocl_pq.create_kernel("multiply_by_scalar", dims.work_size())
		.arg_env(&source_buffer)
		.arg_scl(COEFF)
		.arg_env(&mut result_buffer);

	// Enqueue kernel depending on and creating no events:
	kernel.enqueue(None, None);

	// Read results from the device into result_buffer's local vector:
	result_buffer.fill_vec_wait();

	// Check results and print the first 20:
	for idx in 0..DATA_SET_SIZE {
		assert_eq!(result_buffer[idx], source_buffer[idx] * COEFF);

		if idx < RESULTS_TO_PRINT { 
			println!("source[{idx}]: {}, \tcoeff: {}, \tresult[{idx}]: {}",
			source_buffer[idx], COEFF, result_buffer[idx], idx = idx); 
		}
	}
}
```
### Recent Changes

See the **[RELEASES](https://github.com/cogciprocate/ocl/blob/master/RELEASES.md)** log.

### Upcoming Changes

At the top of the list are cleaning up and consolidating error handling [Issue #8](https://github.com/cogciprocate/ocl/issues/8) and finishing [documentation](http://doc.cogciprocate.com/ocl/) (now about 60% complete).

## Help

*If troubleshooting your OpenCL drivers:* check that `/usr/lib/libOpenCL.so.1` exists. Go ahead and link `/usr/lib/libOpenCL.so -> libOpenCL.so.1` just in case it's not already (AMD drivers sometimes don't create this link).  Intel and AMD also have OpenCL libraries for your CPU if you're having trouble getting your GPU to work (intel: [windows](http://registrationcenter.intel.com/irc_nas/5198/opencl_runtime_15.1_x64_setup.msi), [linux](http://registrationcenter.intel.com/irc_nas/5193/opencl_runtime_15.1_x64_5.0.0.57.tgz)). 

Please ask questions and provide feedback by opening an [issue](https://github.com/cogciprocate/ocl_rust/issues).

