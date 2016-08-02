//! An `OpenCL` event.

use std;
use std::ops::{Deref, DerefMut};
use std::convert::Into;
use libc::c_void;
use cl_h;
use core::error::{Error as OclError, Result as OclResult};
use core::{self, Event as EventCore, EventInfo, EventInfoResult, ProfilingInfo, ProfilingInfoResult,
    ClEventPtrNew, ClWaitList, EventList as EventListCore, CommandExecutionStatus, EventCallbackFn};

/// An event representing a command or user created event.
#[derive(Clone, Debug)]
pub struct Event(Option<EventCore>);

impl Event {
    /// Creates a new, empty event which must be filled by a newly initiated
    /// command, becoming associated with it.
    pub fn empty() -> Event {
        Event(None)
    }

    /// Creates a new `Event` from a `EventCore`.
    ///
    /// ## Safety
    ///
    /// Not meant to be called directly.
    pub unsafe fn from_core(event_core: EventCore) -> Event {
        Event(Some(event_core))
    }

    /// Waits for all events in list to complete before returning.
    ///
    /// Similar in function to `Queue::finish()`.
    ///
    pub fn wait(&self) -> OclResult<()> {
        assert!(!self.is_empty(), "ocl::Event::wait(): {}", self.err_empty());
        core::wait_for_event(self.0.as_ref().unwrap())
    }

    /// Returns info about the event.
    pub fn info(&self, info_kind: EventInfo) -> EventInfoResult {
        match self.0 {
            Some(ref core) => {
                // match core::get_event_info(core, info_kind) {
                //     Ok(pi) => pi,
                //     Err(err) => EventInfoResult::Error(Box::new(err)),
                // }
                core::get_event_info(core, info_kind)
            },
            None => EventInfoResult::Error(Box::new(self.err_empty())),
        }
    }

    /// Returns info about the event.
    pub fn profiling_info(&self, info_kind: ProfilingInfo) -> ProfilingInfoResult {
        match self.0 {
            Some(ref core) => {
                // match core::get_event_profiling_info(core, info_kind) {
                //     Ok(pi) => pi,
                //     Err(err) => ProfilingInfoResult::Error(Box::new(err)),
                // }
                core::get_event_profiling_info(core, info_kind)
            },
            None => ProfilingInfoResult::Error(Box::new(self.err_empty())),
        }
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    pub fn core_as_ref(&self) -> Option<&EventCore> {
        self.0.as_ref()
    }

    /// Returns a mutable reference to the core pointer wrapper usable by
    /// functions in the `core` module.
    pub fn core_as_mut(&mut self) -> Option<&mut EventCore> {
        self.0.as_mut()
    }

    /// Returns true if this event is 'empty' and has not yet been associated
    /// with a command.
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    fn err_empty(&self) -> OclError {
        OclError::new("This `ocl::Event` is empty and cannot be used until \
            filled by a command.")
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Event")
            .field("CommandQueue", &self.info(EventInfo::CommandQueue))
            .field("CommandType", &self.info(EventInfo::CommandType))
            .field("ReferenceCount", &self.info(EventInfo::ReferenceCount))
            .field("CommandExecutionStatus", &self.info(EventInfo::CommandExecutionStatus))
            .field("Context", &self.info(EventInfo::Context))
            .finish()
    }
}

impl Into<String> for Event {
    fn into(self) -> String {
        format!("{}", self)
    }
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl AsRef<EventCore> for Event {
    fn as_ref(&self) -> &EventCore {
        self.0.as_ref().ok_or(self.err_empty()).expect("ocl::Event::as_ref()")
    }
}

impl Deref for Event {
    type Target = EventCore;

    fn deref(&self) -> &EventCore {
        self.0.as_ref().ok_or(self.err_empty()).expect("ocl::Event::deref()")
    }
}

impl DerefMut for Event {
    fn deref_mut(&mut self) -> &mut EventCore {
        assert!(!self.is_empty(), "ocl::Event::deref_mut(): {}", self.err_empty());
        self.0.as_mut().unwrap()
    }
}

unsafe impl ClEventPtrNew for Event {
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_h::cl_event> {
        if !self.is_empty() {
            return OclError::err("ocl::Event: Attempting to use a non-empty event as a new event
                is not allowed. Please create a new, empty, event with ocl::Event::empty().");
        }

        unsafe {
            self.0 = Some(EventCore::null());
            Ok(self.0.as_mut().unwrap().as_ptr_mut())
        }
    }
}

unsafe impl ClWaitList for Event {
    unsafe fn as_ptr_ptr(&self) -> *const cl_h::cl_event {
        // self.0.as_ref().ok_or(self.err_empty()).expect("ocl::Event::as_ref()").as_ptr_ptr()
        match self.0 {
            Some(ref ec) => ec.as_ptr_ptr(),
            None => 0 as *const cl_h::cl_event,
        }
    }

    fn count(&self) -> u32 {
        match self.0 {
            Some(ref ec) => ec.count(),
            None => 0,
        }
    }
}



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

    /// Returns if there is no events.
    pub fn is_empty(&self) -> bool {
        self.event_list_core.len() == 0
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
        if self.event_list_core.is_empty() == false {
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

unsafe impl ClWaitList for EventList {
    unsafe fn as_ptr_ptr(&self) -> *const cl_h::cl_event {
        self.event_list_core.as_ptr_ptr()
    }

    fn count(&self) -> u32 {
        self.event_list_core.count()
    }
}
