//! An OpenCL command queue.
// use std::mem;
// use std::ptr;

use cl_h::{self, cl_command_queue, cl_context, cl_device_id};
use super::Context;

/// An OpenCL command queue.
// TODO: Implement a constructor which accepts a cl_device_id.
#[derive(Clone)]
pub struct Queue {
	obj: cl_command_queue,
	context_obj: cl_context,
	device_id: cl_device_id,
}

impl Queue {
	/// Returns a new Queue on the device specified by `device_idx`. 
	///
	/// 'device_idx` refers to a index in the list of devices generated when creating
	/// `context`. For a list of these devices, call `context.device_ids()`. If 
	/// `device_idx` is out of range, it will automatically 'wrap around' via a 
	/// modulo operation and therefore is valid up to the limit of `usize`. See
	/// the documentation for `Context` for more information.
	/// 
	pub fn new(context: &Context, device_idx: Option<usize>) -> Queue {
		let device_id = context.resolve_device_id(device_idx);
		let obj: cl_command_queue = super::create_command_queue(context.obj(), device_id); 

		Queue {
			obj: obj,
			context_obj: context.obj(),
			device_id: device_id,			
		}
	}	

	/// Blocks until all commands in this queue have completed.
	pub fn finish(&self) {
		super::finish(self.obj);
	}

	/// Returns the OpenCL command queue object associated with this queue.
	pub fn obj(&self) -> cl_command_queue {
		self.obj
	}

	/// Returns the OpenCL context object associated with this queue.
	pub fn context_obj(&self) -> cl_context {
		self.context_obj
	}

	/// Returns the OpenCL device id associated with this queue.
	///
	/// Not to be confused with the zero-indexed `device_idx` passed to `::new()`
	/// when creating this queue.
	pub fn device_id(&self) -> cl_device_id {
		self.device_id
	}

	/// Decrements the reference counter of the associated OpenCL command queue object.
	// Note: Do not move this to a Drop impl in case this Queue has been cloned.
	pub fn release(&mut self) {
		unsafe {
			cl_h::clReleaseCommandQueue(self.obj);
		}
	}
}
