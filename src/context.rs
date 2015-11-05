
// use formatting::MT;
use super::{ cl_h, cl_platform_id, cl_device_id, cl_device_type, cl_context, DEFAULT_PLATFORM };

/// An OpenCL context for a particular platform and set of device types.
///
/// Wraps a 'cl_context' such as that returned by 'clCreateContext'.
pub struct Context {
	platform_opt: Option<cl_platform_id>,
	devices: Vec<cl_device_id>,
	context: cl_context,
}

impl Context {
	/// Constructs a new `Context` within a specified platform and set of device types.
	/// 
	/// The desired platform may be specified by passing a valid index from a list 
	/// obtainable from the ocl::get_platform_ids() function and wrapping it with a 
	/// Some (ex. `Some(2)`). Pass `None` to use the first platform avaliable (0). 
	/// 
	/// The device types mask may be specified using a union of any or all of the 
	/// following flags:
	/// 	-`CL_DEVICE_TYPE_DEFAULT`: The default OpenCL device in the system.
	/// 	-`CL_DEVICE_TYPE_CPU`: An OpenCL device that is the host processor. 
	/// 	The host processor runs the OpenCL implementations and is a single or 
	/// 	multi-core CPU.
	/// 	-`CL_DEVICE_TYPE_GPU`: An OpenCL device that is a GPU. By this we mean 
	/// 	that the device can also be used to accelerate a 3D API such as OpenGL or 
	///		DirectX.
	///		-`CL_DEVICE_TYPE_ACCELERATOR`: Dedicated OpenCL accelerators (for example 
	///		the IBM CELL Blade). These devices communicate with the host processor 
	///		using a peripheral interconnect such as PCIe.
	///		-`CL_DEVICE_TYPE_ALL`: A union of all flags.
	///
	/// Passing `None` will use the flag: `CL_DEVICE_TYPE_GPU`.
	///
	/// # Examples
	///	
	/// ```
	///	use ocl::Context;
	///
	///	fn main() {
	///		// Create a context with the first platform and all GPUs in the system.
	///		let ocl_context = Context::new(None, None);
	///		
	///		// Do fun stuff...
	/// }
	/// ```
	///
	///
	/// ```
	/// use ocl::{ self, Context };
	/// 
	/// fn main() {
	/// 	let platform_ids = ocl::get_platform_ids();
	/// 	assert!(platform_ids.len() >= 2);
	/// 
	/// 	let device_types = CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU;
	///
	/// 	// Create a context using the 2nd platform and both CPU and GPU devices.
	/// 	let ocl_context = Context::new(Some(1), Some(device_types));
	/// 	
	/// 	// ...
	/// }
	/// ```	
	///
	/// # Panics
	/// 	-`get_device_ids()` (work in progress)
	///
	/// # Failures
	/// 	-No platforms.
	/// 	-Invalid platform index.
	/// 	-No devices.
	///
	/// # TODO:
	/// 	-Add a more in-depth constructor which accepts an arbitrary list of 
	///		devices (or sub-devices), a list of cl_context_properties, and no platform.
	///
	/// # Maybe Someday TODO:
	///		-Handle context callbacks.
	///
	pub fn new(platform_idx_opt: Option<usize>, device_types_opt: Option<cl_device_type>) 
			-> Result<Context, &'static str>
	{
		let platforms = super::get_platform_ids();
		if platforms.len() == 0 { return Err("\nNo OpenCL platforms found!\n"); }

		let platform = match platform_idx_opt {
			Some(pf_idx) => {
				match platforms.get(pf_idx) {
					Some(&pf) => pf,
					None => return Err("Invalid OpenCL platform index specified. \
						Use 'get_platform_ids()' for a list."),
				}				
			},

			None => platforms[DEFAULT_PLATFORM],
		};
		
		let devices: Vec<cl_device_id> = super::get_device_ids(platform, device_types_opt);
		if devices.len() == 0 { return Err("\nNo OpenCL devices found!\n"); }

		// println!("{}OCL::NEW(): device list: {:?}", MT, devices);

		let context: cl_context = super::create_context(&devices);

		Ok(Context {
			platform_opt: Some(platform),
			devices: devices,
			context:  context,
		})
	}

	/// Releases the current context.
	pub fn release_components(&mut self) {		
    	unsafe {
			cl_h::clReleaseContext(self.context);
		}
	}

	/// Returns the current context as a `*mut libc::c_void`.
	pub fn context(&self) -> cl_context {
		self.context
	}

	/// Returns a list of `*mut libc::c_void` corresponding to devices valid for use in this context.
	pub fn devices(&self) -> &Vec<cl_device_id> {
		&self.devices
	}

	/// Returns the platform our context pertains to.
	pub fn platform(&self) -> Option<cl_platform_id> {
		self.platform_opt
	}

	/// Returns a valid device regardless of whether or not the index passed is valid by performing a modulo operation on it.
	pub fn valid_device(&self, selected_idx: usize) -> cl_device_id {
		let valid_idx = selected_idx % self.devices.len();
		self.devices[valid_idx]
	}
}

