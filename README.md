#Ocl

OpenCL interfaces for Rust. Makes easy to use many of the most useful features of OpenCL such as creating kernels and data containers. All advanced features are supported via direct rust wrapper function calls. All interfaces and wrappers are zero-cost and will have runtime performance on a par with standard C++ libraries.

##Installation

Ensure that OpenCL is installed for your preferred platform. Remember that Intel and AMD both have OpenCL libraries for your CPU if you're having trouble getting your GPU to work. Make sure that `clinfo` or some other diagnostic command will run. You may want to check that `/usr/lib/libOpenCL.so.1` exists. Go ahead and link `/usr/lib/libOpenCL.so -> libOpenCL.so.1` just in case if it's not already.

Add

```
[dependencies]
ocl = "0.1"
```

or

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
use ocl::{ BuildOptions, Envoy };

fn main() {
	// Create a context with no specified platform or devices:
	// (defaults to GPUs from the first available platform)
	let ocl_cxt = ocl::Context::new(None, None);

	// Create a Program/Queue with no specified device 
	// (defaults to first device found):
	let ocl_pq = ocl::ProQueue::new(&ocl_cxt, None);

	// Create build options passing optional command line switches and other options:
	let build_options = BuildOptions::new("-cl-fast-relaxed-math")
		.kern_file("my_kernel_file.cl".to_string(); [FIXME]: Explain and describe paths

	// Build:
	ocl_pq.build(build_options).unwrap();

	// Create source and destination Envoys (our data containers):
	let src_env = Envoy::new( [FIXME]: Incomplete...
	let dst_env = Envoy::new( [FIXME]: Explain dimensionality and Envoy length, etc.

	// Create a Kernel:
	let kernel = ocl_pq.new_kernel("multiply_by".to_string(), WorkSize::OneDim())
		.arg_env(&src_env)
		.arg_env(&dst_env)
		.arg_scl(5)
	;

	// Populate source Envoy:


	// Enqueue kernel:
	kernel.enqueue();

	// Get a reference to the Envoy's internal Vec with '.vec()' or '.vec_mut()':


	// Check results etc.


}
```

More details, examples, and documentation coming soon!
