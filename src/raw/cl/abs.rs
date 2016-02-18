//! Abstract data type wrappers.
//!
//! ### Reference: (from [SDK](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/abstractDataTypes.html))
//! 
//! The following table describes abstract data types supported by OpenCL:
//! Type	Description	API Type
//! _cl_platform_id *	The ID for a platform.	cl_platform_id
//! _cl_device_id *	The ID for a device.	cl_device_id
//! _cl_context *	A context.	cl_context
//! _cl_command_queue *	A command queue.	cl_command_queue
//! _cl_mem *	A memory object.	cl_mem
//! _cl_program *	A program.	cl_program
//! _cl_kernel *	A kernel.	cl_kernel
//! _cl_event *	An event. Also see event_t.	cl_event
//! _cl_sampler *	A sampler. Also see sampler_t.	cl_sampler
//! 
//! [FIXME]: Bring back `Send` and `Sync` to those types deemed worthy 
//! (Everything but kernel? How should we roll?)

use libc;
use cl_h::{cl_platform_id, cl_device_id,  cl_context, cl_command_queue, cl_mem, cl_program, 
	cl_kernel, cl_event, cl_sampler};


/// cl_platform_id
#[derive(Clone, Copy, Debug)]
pub struct PlatformIdRaw(cl_platform_id);

impl PlatformIdRaw {
	pub unsafe fn new(ptr: cl_platform_id) -> PlatformIdRaw {
		PlatformIdRaw(ptr)
	}

	pub fn null() -> PlatformIdRaw {
		PlatformIdRaw(0 as *mut libc::c_void)
	}

	pub fn as_ptr(&self) -> cl_platform_id {
		self.0
	}
}

unsafe impl Sync for PlatformIdRaw {}
unsafe impl Send for PlatformIdRaw {}


/// cl_device_id
#[derive(Clone, Copy, Debug)]
pub struct DeviceIdRaw(cl_device_id);

impl DeviceIdRaw {
	pub unsafe fn new(ptr: cl_device_id) -> DeviceIdRaw {
		DeviceIdRaw(ptr)
	}

	pub fn null() -> DeviceIdRaw {
		DeviceIdRaw(0 as *mut libc::c_void)
	}

	pub fn as_ptr(&self) -> cl_device_id {
		self.0
	}
}

unsafe impl Sync for DeviceIdRaw {}
unsafe impl Send for DeviceIdRaw {}


/// cl_context
#[derive(Clone, Copy, Debug)]
pub struct ContextRaw(cl_context);

impl ContextRaw {
	pub unsafe fn new(ptr: cl_context) -> ContextRaw {
		ContextRaw(ptr)
	}

	pub fn as_ptr(&self) -> cl_context {
		self.0
	}
}

unsafe impl Sync for ContextRaw {}
unsafe impl Send for ContextRaw {}


/// cl_command_queue
#[derive(Clone, Copy, Debug)]
pub struct CommandQueueRaw(cl_command_queue);

impl CommandQueueRaw {
	pub unsafe fn new(ptr: cl_command_queue) -> CommandQueueRaw {
		CommandQueueRaw(ptr)
	}

	pub fn as_ptr(&self) -> cl_command_queue {
		self.0
	}
}

unsafe impl Sync for CommandQueueRaw {}
unsafe impl Send for CommandQueueRaw {}


/// cl_mem
#[derive(Clone, Copy, Debug)]
pub struct MemRaw(cl_mem);

impl MemRaw {
	pub unsafe fn new(ptr: cl_mem) -> MemRaw {
		MemRaw(ptr)
	}

	pub fn null() -> MemRaw {
		MemRaw(0 as *mut libc::c_void)
	}

	pub fn as_ptr(&self) -> cl_mem {
		self.0
	}
}

unsafe impl Sync for MemRaw {}
unsafe impl Send for MemRaw {}


/// cl_program
#[derive(Clone, Copy, Debug)]
pub struct ProgramRaw(cl_program);

impl ProgramRaw {
	pub unsafe fn new(ptr: cl_program) -> ProgramRaw {
		ProgramRaw(ptr)
	}

	pub fn as_ptr(&self) -> cl_program {
		self.0
	}
}

unsafe impl Sync for ProgramRaw {}
unsafe impl Send for ProgramRaw {}


/// cl_kernel
///
/// ### Thread Safety
///
/// Not thread safe: do not implement `Send` or `Sync`.
#[derive(Clone, Copy, Debug)]
pub struct KernelRaw(cl_kernel);

impl KernelRaw {
	pub unsafe fn new(ptr: cl_kernel) -> KernelRaw {
		KernelRaw(ptr)
	}

	pub fn as_ptr(&self) -> cl_kernel {
		self.0
	}
}


/// cl_event
#[derive(Clone, Copy, Debug)]
pub struct EventRaw(cl_event);

impl EventRaw {
	pub unsafe fn new(ptr: cl_event) -> EventRaw {
		EventRaw(ptr)
	}

	pub fn null() -> EventRaw {
		EventRaw(0 as *mut libc::c_void)
	}

	pub fn as_ptr(&self) -> cl_event {
		self.0
	}
}

unsafe impl Sync for EventRaw {}
unsafe impl Send for EventRaw {}


/// cl_sampler
#[derive(Clone, Copy, Debug)]
pub struct SamplerRaw(cl_sampler);

impl SamplerRaw {
	pub unsafe fn new(ptr: cl_sampler) -> SamplerRaw {
		SamplerRaw(ptr)
	}

	pub fn as_ptr(&self) -> cl_sampler {
		self.0
	}
}

unsafe impl Sync for SamplerRaw {}
unsafe impl Send for SamplerRaw {}
