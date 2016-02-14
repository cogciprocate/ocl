//! Event list for coordinating enqueued tasks.
use std::ptr;
use libc::c_void;

use raw::{self, EventRaw};
use cl_h::{self, cl_event};

/// A list of OpenCL events which contain status information about the command that
/// created them. Used to coordinate the activity of multiple commands.
// [FIXME] TODO: impl Index.
#[derive(Debug)]
pub struct EventList {
    events: Vec<cl_event>,
}

impl EventList {
    /// Returns a new, empty, `EventList`.
    #[inline]
    pub fn new() -> EventList {
        EventList { events: Vec::with_capacity(16) }
    }

    /// Merges the copied contents of this list and another into a new list and returns it.
    #[inline]
    pub fn union(&self, other_list: &EventList) -> EventList {
        let mut new_list = EventList { events: Vec::with_capacity(other_list.events().len() 
            + self.events.len() + 8) };
        new_list.events.extend(self.events().iter().cloned());
        new_list.events.extend(other_list.events().iter().cloned());

        new_list
    }

    /// Appends a new null element to the end of the list and returns a mutable slice
    /// containing only that element.
    #[inline]
    pub fn allot(&mut self) -> &mut cl_event {
        self.events.push(ptr::null_mut());
        self.events.last_mut().unwrap()
    }

    // #[inline]
    pub fn last(&self) -> Option<&cl_event> {
        self.events.last()
    }


    // #[inline]
    pub fn last_mut(&mut self) -> Option<&mut cl_event> {
        self.events.last_mut()
    }

    /// Returns an immutable slice to the events list.
    #[inline]
    pub fn events(&self) -> &[cl_event] {
        &self.events[..]
    }

    /// Returns a mutable slice to the events list.
    #[inline]
    pub fn events_mut(&mut self) -> &mut [cl_event] {
        &mut self.events[..]
    }

    /// Returns a const pointer to the list, useful for passing directly to the c ffi.
    #[inline]
    pub fn as_ptr(&self) -> *const cl_event {
        self.events().as_ptr()
    }

    /// Returns a mut pointer to the list, useful for passing directly to the c ffi.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut cl_event {
        self.events_mut().as_mut_ptr()
    }

    /// Waits for all events in list to complete.
    pub fn wait(&self) {
        if self.events.len() > 0 {
            raw::wait_for_events(self.count(), self.events());
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
                callback_receiver: extern fn (cl_event, i32, *mut c_void),
                // user_data: *mut c_void,
                user_data: &mut T,
            )
    {
        if self.events.len() > 0 {
            raw::set_event_callback(
                self.events[self.events.len() - 1], 
                cl_h::CL_COMPLETE, 
                callback_receiver,
                user_data as *mut _ as *mut c_void,
            )
        }
    }

    // pub fn set_callback_buffer(&self, 
    //          callback_receiver: extern fn (cl_event, i32, *mut libc::c_void),
    //          user_data: *mut libc::c_void,
    //      )
    // {
    //  if self.events.len() > 0 {
    //      raw::set_event_callback(
    //          self.events[self.events.len() - 1], 
    //          CL_COMPLETE, 
    //          callback_receiver,
    //          user_data,
    //      )
    //  }
    // }

    /// Returns the number of events in the list.
    #[inline]
    pub fn count(&self) -> u32 {
        self.events.len() as u32
    }

    /// Clears this list regardless of whether or not its events have completed.
    #[inline]
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Clears each completed event from the list.
    pub fn clear_completed(&mut self) {
        let mut ce_idxs: Vec<usize> = Vec::with_capacity(8);

        let mut idx = 0;
        for event in self.events.iter() {
            if raw::get_event_status(*event) == cl_h::CL_COMPLETE {
                ce_idxs.push(idx)
            }

            idx += 1;
        }

        for idx in ce_idxs.into_iter().rev() {
            self.events.remove(idx);
            
            // let ev = self.events.remove(idx);
            // Release?
            // raw::release_event(ev);
        }
    }

    /// Releases all events in the list by decrementing their reference counts by one
    /// and empties the list.
    ///
    /// Events will continue to exist until their creating commands have completed 
    /// and no other commands are waiting for them to complete.
    pub fn release_all(&mut self) {
        for &mut event in &mut self.events {
            raw::release_event(event);
        }

        self.clear();
    }
}
