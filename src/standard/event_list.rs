//! Event list for coordinating enqueued commands.

use std::ops::{Deref, DerefMut};
use libc::c_void;

use cl_h;
use error::{Result as OclResult, Error as OclError};
use standard::Event;
use core::{self, EventCallbackFn, EventList as EventListCore, CommandExecutionStatus, ClEventPtrNew};
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
                    Ok(ev) => unsafe { Some(Event::from_core(ev)) },
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
                    Ok(ev) => unsafe { Some(Event::from_core(ev)) },
                    Err(_) => None,
                }
            },
            None => None,
        }
    }

    /// Sets a callback function, `callback_receiver`, to trigger upon completion of
    /// the *last event* added to the event list with an optional reference to user 
    /// data.
    ///
    /// # Safety
    ///
    /// `user_data` must be guaranteed to still exist if and when `callback_receiver` 
    /// is ever called.
    ///
    /// TODO: Create a safer type wrapper for `callback_receiver`.
    /// TODO: Move this method to `Event`.
    pub unsafe fn set_callback<T>(&self, 
                callback_receiver: Option<EventCallbackFn>,
                user_data: &mut T,
                ) -> OclResult<()>
    {
        let event_core = try!(try!(self.event_list_core.last_clone().ok_or(
            OclError::new("ocl::EventList::set_callback: This event list is empty."))));

        core::set_event_callback(&event_core, CommandExecutionStatus::Complete,
                    callback_receiver, user_data as *mut _ as *mut c_void)
    }

    // pub fn clear_completed(&mut self) -> OclResult<()> {
    //     self.event_list_core.clear_completed()
    // }

    /// Returns the number of events in the list.
    pub fn len(&self) -> usize {
        self.event_list_core.len()
    }

    // Returns a reference to the underlying `core` event list.
    pub fn core_as_ref(&self) -> &EventListCore {
        &self.event_list_core
    }

    // Returns a mutable reference to the underlying `core` event list.
    pub fn core_as_mut(&mut self) -> &mut EventListCore {
        &mut self.event_list_core
    }

    /// Waits for all events in list to complete.
    pub fn wait(&self) -> OclResult<()> {
        if self.event_list_core.len() > 0 {
            core::wait_for_events(self.event_list_core.count(), &self.event_list_core)
        } else {
            Ok(())
        }
    }
}

impl AsRef<EventListCore> for EventList {
    fn as_ref(&self) -> &EventListCore {
        &self.event_list_core
    }
}

impl Deref for EventList {
    type Target = EventListCore;

    fn deref(&self) -> &EventListCore {
        &self.event_list_core
    }
}

impl DerefMut for EventList {
    fn deref_mut(&mut self) -> &mut EventListCore {
        &mut self.event_list_core
    }
}

unsafe impl ClEventPtrNew for EventList {
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_h::cl_event> {
        Ok(self.event_list_core.allot())
    }
}
