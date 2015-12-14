use std::ptr;
use libc::{ c_void };

use cl_h::{ self, cl_event, cl_int };

/// A list of OpenCL events which contain status information about the command that
/// created them. Used to coordinate the activity of multiple commands.
// [FIXME] TODO: impl Index.
#[derive(Debug)]
pub struct EventList {
	events: Vec<cl_event>,
}

impl EventList {
	/// Returns a new, empty, `EventList`.
	pub fn new() -> EventList {
		EventList { events: Vec::with_capacity(16) }
	}

	/// Merges the copied contents of this list and another into a new list and returns it.
	pub fn union(&self, other_list: &EventList) -> EventList {
		let mut new_list = EventList { events: Vec::with_capacity(other_list.events().len() 
			+ self.events.len() + 8) };
		new_list.events.extend(self.events().iter().cloned());
		new_list.events.extend(other_list.events().iter().cloned());

		new_list
	}

	/// Appends a new null element to the end of the list and returns a mutable slice
	/// containing only that element.
	pub fn allot(&mut self) -> &mut [cl_event] {
		self.events.push(ptr::null_mut());
		let len = self.events.len();
		&mut self.events[(len - 1)..len]
	}

	/// Returns an immutable slice to the events list.
	pub fn events(&self) -> &[cl_event] {
		&self.events[..]
	}

	/// Returns a const pointer to the list, useful for passing directly to the c ffi.
	pub fn as_ptr(&self) -> *const cl_event {
		self.events().as_ptr()
	}

	/// Waits for all events in list to complete.
	pub fn wait(&self) {
		if self.events.len() > 0 {
			super::wait_for_events(self);
		}
	}

	/// Sets a callback function, `callback_receiver`, to trigger upon completion of
	/// the *last event* added to the event list with an optional reference to user 
	/// data.
	///
	/// #Safety
	/// `user_data` must be guaranteed to still exist if and when `callback_receiver` 
	/// is ever called.
	pub unsafe fn set_callback<T>(&self, 
				callback_receiver: extern fn (cl_event, cl_int, *mut c_void),
				// user_data: *mut c_void,
				user_data: &mut T,
			)
	{
		if self.events.len() > 0 {
			super::set_event_callback(
				self.events[self.events.len() - 1], 
				cl_h::CL_COMPLETE, 
				callback_receiver,
				user_data as *mut _ as *mut c_void,
			)
		}
	}

	// pub fn set_callback_envoy(&self, 
	// 			callback_receiver: extern fn (cl_event, cl_int, *mut libc::c_void),
	// 			user_data: *mut libc::c_void,
	// 		)
	// {
	// 	if self.events.len() > 0 {
	// 		super::set_event_callback(
	// 			self.events[self.events.len() - 1], 
	// 			CL_COMPLETE, 
	// 			callback_receiver,
	// 			user_data,
	// 		)
	// 	}
	// }

	/// Returns the number of events in the list.
	pub fn count(&self) -> u32 {
		self.events.len() as u32
	}

	/// Clears this list regardless of whether or not its events have completed.
	pub fn clear(&mut self) {
		self.events.clear();
	}

	/// Clears all completed events from the list.
	pub fn clear_completed(&mut self) {
		let mut ce_idxs: Vec<usize> = Vec::with_capacity(8);

		let mut idx = 0;
		for event in self.events.iter() {
			if super::get_event_status(*event) == cl_h::CL_COMPLETE {
				ce_idxs.push(idx)
			}

			idx += 1;
		}

		for idx in ce_idxs.into_iter().rev() {
			self.events.remove(idx);
		}
	}

	/// Releases all events in the list by decrementing their reference counts by one
	/// and empties the list.
	///
	/// Events will continue to exist until their creating commands have completed 
	/// and no other commands are waiting for them to complete.
	pub fn release_all(&mut self) {
		for &mut event in &mut self.events {
	    	let err = unsafe {
				cl_h::clReleaseEvent(event)
			};

			super::must_succeed("clReleaseEvent", err);
		}

		self.clear();
	}
}
