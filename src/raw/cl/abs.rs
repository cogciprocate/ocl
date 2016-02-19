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

use std::mem;
use std::fmt::Debug;
use std::marker::PhantomData;
use libc;
use cl_h::{cl_platform_id, cl_device_id,  cl_context, cl_command_queue, cl_mem, cl_program, 
	cl_kernel, cl_event, cl_sampler};
use raw::{self, OclNum, /*EventPtr, EventListPtr*/};


const EL_INIT_CAPACITY: usize = 64;
const EL_CLEAR_SIZE: usize = 48;
const EL_CLEAR_INTERVAL: isize = 32;

// Clear the list automatically. Usefulness and performance impact of this is
// currently under evaluation.
const EL_AUTO_CLEAR: bool = false;


/// cl_platform_id
#[repr(C)]
#[derive(Clone, Debug)]
pub struct PlatformIdRaw(cl_platform_id);

impl PlatformIdRaw {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_platform_id) -> PlatformIdRaw {
		PlatformIdRaw(ptr)
	}

	pub unsafe fn null() -> PlatformIdRaw {
		PlatformIdRaw(0 as *mut libc::c_void)
	}

	/// Returns a pointer, do not store it.
	pub unsafe fn as_ptr(&self) -> cl_platform_id {
		self.0
	}
}

unsafe impl Sync for PlatformIdRaw {}
unsafe impl Send for PlatformIdRaw {}



/// cl_device_id
#[repr(C)]
#[derive(Clone, Debug)]
pub struct DeviceIdRaw(cl_device_id);

impl DeviceIdRaw {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_device_id) -> DeviceIdRaw {
		DeviceIdRaw(ptr)
	}

	pub unsafe fn null() -> DeviceIdRaw {
		DeviceIdRaw(0 as *mut libc::c_void)
	}

	/// Returns a pointer, do not store it.
	pub unsafe fn as_ptr(&self) -> cl_device_id {
		self.0
	}
}

unsafe impl Sync for DeviceIdRaw {}
unsafe impl Send for DeviceIdRaw {}



/// cl_context
#[derive(Debug)]
pub struct ContextRaw(cl_context);

impl ContextRaw {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_context) -> ContextRaw {
		ContextRaw(ptr)
	}

	/// Returns a pointer, do not store it.
	pub unsafe fn as_ptr(&self) -> cl_context {
		self.0
	}
}

impl Clone for ContextRaw {
	fn clone(&self) -> ContextRaw {
		unsafe { raw::retain_context(self).unwrap(); }
		ContextRaw(self.0)
	}
}

impl Drop for ContextRaw {
	fn drop(&mut self) {
		unsafe { raw::release_context(self).unwrap(); }
	}
}

unsafe impl Sync for ContextRaw {}
unsafe impl Send for ContextRaw {}



/// cl_command_queue
#[derive(Debug)]
pub struct CommandQueueRaw(cl_command_queue);

impl CommandQueueRaw {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_command_queue) -> CommandQueueRaw {
		CommandQueueRaw(ptr)
	}

	/// Returns a pointer, do not store it.
	pub unsafe fn as_ptr(&self) -> cl_command_queue {
		self.0
	}
}

impl Clone for CommandQueueRaw {
	fn clone(&self) -> CommandQueueRaw {
		unsafe { raw::retain_command_queue(self).unwrap(); }
		CommandQueueRaw(self.0)
	}
}

impl Drop for CommandQueueRaw {
	fn drop(&mut self) {
		unsafe { raw::release_command_queue(self).unwrap(); }
	}
}

unsafe impl Sync for CommandQueueRaw {}
unsafe impl Send for CommandQueueRaw {}



/// cl_mem
#[derive(Debug)]
pub struct MemRaw<T: OclNum>(cl_mem, PhantomData<T>);

impl<T: OclNum> MemRaw<T> {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_mem) -> MemRaw<T> {
		MemRaw(ptr, PhantomData)
	}

	// pub unsafe fn null() -> MemRaw<T> {
	// 	MemRaw(0 as *mut libc::c_void, PhantomData)
	// }

	/// Returns a pointer, do not store it.
	pub unsafe fn as_ptr(&self) -> cl_mem {
		self.0
	}
}

impl<T: OclNum> Clone for MemRaw<T> {
	fn clone(&self) -> MemRaw<T> {
		unsafe { raw::retain_mem_object(self).unwrap(); }
		MemRaw(self.0, PhantomData)
	}
}

impl<T: OclNum> Drop for MemRaw<T> {
	fn drop(&mut self) {
		unsafe { raw::release_mem_object(self).unwrap(); }
	}
}

unsafe impl<T: OclNum> Sync for MemRaw<T> {}
unsafe impl<T: OclNum> Send for MemRaw<T> {}



/// cl_program
#[derive(Debug)]
pub struct ProgramRaw(cl_program);

impl ProgramRaw {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_program) -> ProgramRaw {
		ProgramRaw(ptr)
	}

	/// Returns a pointer, do not store it.
	pub unsafe fn as_ptr(&self) -> cl_program {
		self.0
	}
}

impl Clone for ProgramRaw {
	fn clone(&self) -> ProgramRaw {
		unsafe { raw::retain_program(self).unwrap(); }
		ProgramRaw(self.0)
	}
}

impl Drop for ProgramRaw {
	fn drop(&mut self) {
		unsafe { raw::release_program(self).unwrap(); }
	}
}

unsafe impl Sync for ProgramRaw {}
unsafe impl Send for ProgramRaw {}

// impl Drop for Program {
//     fn drop(&mut self) {
//         // println!("DROPPING PROGRAM");
//         unsafe { raw::release_program(self.obj_raw).unwrap(); }
//     }
// }


/// cl_kernel
///
/// ### Thread Safety
///
/// Not thread safe: do not implement `Send` or `Sync`.
#[derive(Debug)]
pub struct KernelRaw(cl_kernel);

impl KernelRaw {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_kernel) -> KernelRaw {
		KernelRaw(ptr)
	}

	/// Returns a pointer, do not store it.
	pub unsafe fn as_ptr(&self) -> cl_kernel {
		self.0
	}
}

impl Clone for KernelRaw {
	fn clone(&self) -> KernelRaw {
		unsafe { raw::retain_kernel(self).unwrap(); }
		KernelRaw(self.0)
	}
}

impl Drop for KernelRaw {
	fn drop(&mut self) {
		unsafe { raw::release_kernel(self).unwrap(); }
	}
}



/// Reference counted list of `cl_event` pointers.
#[derive(Debug)]
pub struct EventListRaw {
	events: Vec<cl_event>,
	// Kinda experimental:
	clear_counter: isize, 
}

impl EventListRaw {
	/// Returns a new, empty, `EventListRaw`.
    pub fn new() -> EventListRaw {
        EventListRaw { 
            events: Vec::with_capacity(EL_INIT_CAPACITY),
            clear_counter: EL_CLEAR_INTERVAL,
        }
    }

    /// Pushes a new event onto the list.
    ///
    /// Technically, copies of `event`s contained pointer (a `cl_event`) then 
    /// `mem::forget`s it. This seems preferrable to incrementing the reference
    /// count (with `raw::retain_event`) then letting `event` drop which just decrements it right back.
    pub unsafe fn push(&mut self, event: EventRaw) {
        self.events.push((*event.as_ptr()));
        mem::forget(event);
        self.decr_counter();
    }

    /// Appends a new null element to the end of the list and returns a mutable reference to that element.
    pub fn allot(&mut self) -> &mut EventRaw {
        unsafe { self.events.push(EventRaw::null()); }
        self.events.last_mut().unwrap()
    }

    /// Counts down the auto-list-clear counter.
    fn decr_counter(&mut self) {
    	if EL_AUTO_CLEAR {
    		self.clear_counter -= 1

    		if self.clear_counter <= 0 && self.events.len() > EL_CLEAR_SIZE {
                // self.clear_completed();
                unimplemented!()
            }
		}
	}
}

// impl Clone for EventListRaw {
// 	fn clone(&self) -> EventRaw {
// 		unsafe { raw::retain_event(self).unwrap(); }
// 		EventRaw(self.0)
// 	}
// }

impl Drop for EventListRaw {
	fn drop(&mut self) {
		for event in self.events.iter() {
			unsafe { raw::release_event(event).unwrap(); }
		}
	}
}

// unsafe impl EventListPtr for EventRaw {}
unsafe impl Sync for EventListRaw {}
unsafe impl Send for EventListRaw {}



/// cl_event
#[derive(Debug)]
pub struct EventRaw(cl_event);

impl EventRaw {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_event) -> EventRaw {
		EventRaw(ptr)
	}

	/// For passage directly to an 'event creation' function (such as enqueue).
	pub unsafe fn null() -> EventRaw {
		EventRaw(0 as cl_event)
	}

	// /// Returns a pointer, do not store it unless you will manage its
	// /// associated reference count carefully (as does `EventListRaw`).
	// pub unsafe fn as_ptr(&self) -> cl_event {
	// 	self.0
	// }

	/// Returns an immutable reference to a pointer, do not deref and store it unless 
	/// you will manage its associated reference count carefully.
	pub unsafe fn as_ptr_ref(&self) -> &cl_event {
		&self.0
	}

	/// Returns a mutable reference to a pointer, do not deref then modify or store it 
	/// unless you will manage its associated reference count carefully.
	pub unsafe fn as_ptr_mut(&mut self) -> &mut cl_event {
		&mut self.0
	}
}

impl Clone for EventRaw {
	fn clone(&self) -> EventRaw {
		unsafe { raw::retain_event(self).unwrap(); }
		EventRaw(self.0)
	}
}

impl Drop for EventRaw {
	fn drop(&mut self) {
		unsafe { raw::release_event(self).unwrap(); }
	}
}

// unsafe impl EventPtr for EventRaw {}
unsafe impl Sync for EventRaw {}
unsafe impl Send for EventRaw {}



/// cl_sampler
#[derive(Debug)]
pub struct SamplerRaw(cl_sampler);

impl SamplerRaw {
	/// Only call this when passing a newly created pointer directly from 
	/// `clCreate...`. Do not use this to clone or copy.
	pub unsafe fn from_fresh_ptr(ptr: cl_sampler) -> SamplerRaw {
		SamplerRaw(ptr)
	}

	/// Returns a pointer, do not store it.
	pub unsafe fn as_ptr(&self) -> cl_sampler {
		self.0
	}
}

impl Clone for SamplerRaw {
	fn clone(&self) -> SamplerRaw {
		unsafe { raw::retain_sampler(self).unwrap(); }
		SamplerRaw(self.0)
	}
}

impl Drop for SamplerRaw {
	fn drop(&mut self) {
		unsafe { raw::release_sampler(self).unwrap(); }
	}
}

unsafe impl Sync for SamplerRaw {}
unsafe impl Send for SamplerRaw {}
