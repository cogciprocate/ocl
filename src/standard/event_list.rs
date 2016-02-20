//! Event list for coordinating enqueued commands.

// use std::ptr;
use libc::c_void;

use error::{Result as OclResult, Error as OclError};
use standard::Event;
use core::{self, EventCallbackFn, EventList as EventListCore, CommandExecutionStatus};
// use cl_h::{self, cl_event};


/// A list of events for coordinating enqueued commands.
///
/// Events contain status information about the command that
/// created them. Used to coordinate the activity of multiple commands with
/// more fine-grained control than the queue alone.
///
/// For access to individual events use `get_clone` and `last_clone` then
/// either store or discard the result.
///
// [FIXME] TODO: impl Index.
#[derive(Debug, Clone)]
pub struct EventList {
    event_list_core: EventListCore,
}

impl EventList {
    /// Returns a new, empty, `EventList`.
    pub fn new() -> EventList {
        EventList { 
            event_list_core: EventListCore::new(),
        }
    }

    // pub fn push(&mut self, event: Event) {
    //     self.events.push(event);
    // }

    // /// Appends a new null element to the end of the list and returns...
    // /// [FIXME]: Update
    // pub fn allot(&mut self) -> &mut Event {
    //     unsafe { self.events.push(Event::null()); }
    //     self.events.last_mut().unwrap()
    // }

    /// Returns a new copy of an event by index.
    pub fn get_clone(&self, index: usize) -> Option<Event> {
        match self.event_list_core.get_clone(index) {
            Some(ev_res) => {
                match ev_res {
                    Ok(ev) => Some(Event::from_event_core(ev)),
                    Err(_) => None,
                }
            },
            None => None,
        }
    }

    /// Returns a copy of the last event in the list.
    pub fn last_clone(&self) -> Option<Event> {
        match self.event_list_core.last_clone() {
            Some(ev_res) => {
                match ev_res {
                    Ok(ev) => Some(Event::from_event_core(ev)),
                    Err(_) => None,
                }
            },
            None => None,
        }
    }

    // Returns a reference to the underlying `core` event list.
    pub fn core_as_ref(&self) -> &EventListCore {
        &self.event_list_core
    }

    // Returns a mutableb reference to the underlying `core` event list.
    pub fn core_as_mut(&mut self) -> &mut EventListCore {
        &mut self.event_list_core
    }

    // /// Returns a mutable slice to the events list.
    // #[inline]
    // pub fn events_mut(&mut self) -> &mut [Event] {
    //     &mut self.events[..]
    // }

    // pub fn event_list_core(&self) -> Vec<EventCore> {
    //     self.events().iter().map(|ref event| event.as_core()).collect()
    // }

    // /// Returns a const pointer to the list, useful for passing directly to the c ffi.
    // #[inline]
    // pub fn core_as_ref(&self) -> *const Event {
    //     self.events().as_ptr()
    // }

    // /// Returns a mut pointer to the list, useful for passing directly to the c ffi.
    // #[inline]
    // pub fn as_mut_ptr(&mut self) -> *mut Event {
    //     self.events.as_mut_ptr()
    // }

    /// Waits for all events in list to complete.
    pub fn wait(&self) {
        if self.event_list_core.len() > 0 {
            // let event_list_core = self.event_list_core();
            core::wait_for_events(self.event_list_core.count(), &self.event_list_core);
        }
    }

    // /// Merges the copied contents of this list and another into a new list and returns it.
    // pub fn union(&self, other_list: &EventList) -> EventList {
    //     let new_cap = other_list.events().len() + self.events.len() + EXTRA_CAPACITY;

    //     let mut new_list = EventList {
    //         events: Vec::with_capacity(new_cap),
    //         clear_counter: 0,
    //     };

    //     new_list.events.extend(self.events().iter().cloned());
    //     new_list.events.extend(other_list.events().iter().cloned());

    //     if AUTO_CLEAR {
    //         new_list.clear_completed();
    //     }

    //     new_list
    // }

    /// Sets a callback function, `callback_receiver`, to trigger upon completion of
    /// the *last event* added to the event list with an optional reference to user 
    /// data.
    ///
    /// # Safety
    ///
    /// `user_data` must be guaranteed to still exist if and when `callback_receiver` 
    /// is ever called.
    ///
    /// TODO: Create a safer type wrapper for `callback_receiver`
    pub unsafe fn set_callback<T>(&self, 
                // callback_receiver: extern fn (cl_event, i32, *mut c_void),
                callback_receiver: EventCallbackFn,
                // user_data: *mut c_void,
                user_data: &mut T,
                ) -> OclResult<()>
    {
        // let event_list_core = self.event_list_core();
        // match self.event_list_core.last_clone() {
        //     Ok(last_event) => {
        //         core::set_event_callback(&last_event, CommandExecutionStatus::Complete,
        //             callback_receiver, user_data as *mut _ as *mut c_void)
        //     },
        //     Err(err) => Err(err),
        // }

        let event_core = try!(try!(self.event_list_core.last_clone().ok_or(
            OclError::new("ocl::EventList::set_callback: This event list is empty."))));

        core::set_event_callback(&event_core, CommandExecutionStatus::Complete,
                    callback_receiver, user_data as *mut _ as *mut c_void)
    }

    pub fn clear_completed(&mut self) -> OclResult<()> {
        self.event_list_core.clear_completed()
    }

    // pub fn set_callback_buffer(&self, 
    //          callback_receiver: extern fn (Event, i32, *mut libc::c_void),
    //          user_data: *mut libc::c_void,
    //      )
    // {
    //  if self.events.len() > 0 {
    //      core::set_event_callback(
    //          self.events[self.events.len() - 1], 
    //          CL_COMPLETE, 
    //          callback_receiver,
    //          user_data,
    //      )
    //  }
    // }

    /// Returns the number of events in the list.
    pub fn len(&self) -> usize {
        self.event_list_core.len()
    }

    // /// Clears this list regardless of whether or not its events have completed.
    // #[inline]
    // pub fn clear(&mut self) {
    //     self.events.clear();

    //     if AUTO_CLEAR {
    //         self.clear_counter = 0;
    //     }
    // }

    // /// Releases all events in the list by decrementing their reference counts by one
    // /// then empties the list.
    // ///
    // /// Events will continue to exist until their creating commands have completed 
    // /// and no other commands are waiting for them to complete.
    // pub unsafe fn release_all(&mut self) {
    //     for event in &mut self.events {
    //         core::release_event(event.as_core()).unwrap();
    //     }

    //     self.clear();
    // }
}

// impl Drop for EventList {
//     fn drop(&mut self) {
//         // println!("DROPPING EVENT LIST");
//         unsafe { self.release_all(); }
//     }
// }
