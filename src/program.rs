//! An OpenCL program, sometimes referred to as a build.

use std::ffi::CString;

use wrapper;
use cl_h::{self, cl_program, cl_context, cl_device_id};
use super::{BuildConfig, Context};

/// An OpenCL program, sometimes referred to as a build.
///
/// To use with multiple devices, create manually with `::from_parts()`.
#[derive(Clone)]
pub struct Program {
	obj: cl_program,
	context_obj: cl_context,
	device_ids: Vec<cl_device_id>,
}

// [FIXME] TODO: ERROR HANDLING
impl Program {
	/// Returns a new program.
	pub fn new(build_config: BuildConfig, context: &Context, device_idx: Option<usize>
			) -> Result<Program, String> 
	{
		let device_id = context.resolve_device_id(device_idx);

		Program::from_parts(
			try!(build_config.kernel_strings().map_err(|e| e.to_string())), 
			try!(build_config.compiler_options().map_err(|e| e.to_string())), 
			context.obj(), 
			&vec![device_id])
	}

	/// Returns a new program built from pre-created build components and device
	/// list.
	// SOMEDAY TODO: Keep track of line number range for each kernel string and print 
	// out during build failure.
	pub fn from_parts(
				kernel_strings: Vec<CString>, 
				cmplr_opts: CString, 
				context_obj: cl_context, 
				device_ids: &Vec<cl_device_id>,
			) -> Result<Program, String> 
	{
		// let kern_c_str = try!(parse_kernel_files(&build_config));

		let obj = try!(wrapper::new_program(kernel_strings, cmplr_opts, 
			context_obj, device_ids).map_err(|e| e.to_string()));

		// [FIXME] TEMPORARY UNWRAP:
		// [FIXME] IS THIS A DUPLICATE CALL -- YES?
		// Temporarily disabling (maybe permanent).
		// wrapper::program_build_info(obj, device_ids).unwrap();

		Ok(Program {
			obj: obj,
			context_obj: context_obj,
			device_ids: device_ids.clone(),
		})
	}

	/// Returns the associated OpenCL program object.
	pub fn obj(&self) -> cl_program {
		self.obj
	}

	/// Decrements the associated OpenCL program object's reference count.
	// Note: Do not move this to a Drop impl in case this Program has been cloned.
	pub fn release(&mut self) {
		unsafe { 
			cl_h::clReleaseProgram(self.obj);
		}
	}
}
