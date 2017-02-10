//! An `OpenCL` event.

use std;
use std::ops::{Deref, DerefMut};
use std::convert::Into;
use libc::c_void;
use futures::{Future, Poll, Async};
use ffi::cl_event;
use core::error::{Error as OclError, Result as OclResult};
use core::{self, Event as EventCore, NullEvent as NullEventCore, UserEvent as UserEventCore,
    EventInfo, EventInfoResult, ProfilingInfo, ProfilingInfoResult, ClNullEventPtr, ClWaitListPtr,
    EventList as EventListCore, CommandExecutionStatus, EventCallbackFn};


/// An event representing a command or user created event.
///
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

    /// Returns true if this event is complete, false if it is not complete or
    /// if this event is not yet associated with a command.
    ///
    /// This is the fastest possible way to determine the event completion
    /// status.
    ///
    #[inline]
    pub fn is_complete(&self) -> OclResult<bool> {
        match self.0 {
            Some(ref core) => core.is_complete(),
            None => Ok(false),
        }
    }

    /// Waits for all events in list to complete before returning.
    ///
    /// Similar in function to `Queue::finish()`.
    ///
    pub fn wait(&self) -> OclResult<()> {
        assert!(!self.is_empty(), "ocl::Event::wait(): {}", self.err_empty());
        core::wait_for_event(self.0.as_ref().unwrap())
    }

    /// Returns true if this event is 'empty' and has not yet been associated
    /// with a command.
    ///
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
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
    ///
    #[deprecated(since="0.13.0", note="Use `::core` instead.")]
    #[inline]
    pub fn core_as_ref(&self) -> Option<&EventCore> {
        self.0.as_ref()
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    ///
    #[inline]
    pub fn core(&self) -> Option<&EventCore> {
        self.0.as_ref()
    }

    /// Returns a mutable reference to the core pointer wrapper usable by
    /// functions in the `core` module.
    ///
    #[deprecated(since="0.13.0", note="Use `::core_mut` instead.")]
    #[inline]
    pub fn core_as_mut(&mut self) -> Option<&mut EventCore> {
        self.0.as_mut()
    }

    /// Returns a mutable reference to the core pointer wrapper usable by
    /// functions in the `core` module.
    ///
    #[inline]
    pub fn core_mut(&mut self) -> Option<&mut EventCore> {
        self.0.as_mut()
    }

    fn err_empty(&self) -> OclError {
        OclError::string("This `ocl::Event` is empty and cannot be used until \
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

    fn _ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
        if !self.is_empty() {
            return OclError::err_string("ocl::Event: Attempting to use a non-empty event as a new event
                is not allowed. Please create a new, empty, event with ocl::Event::empty().");
        }

        unsafe {
            self.0 = Some(EventCore::from_raw(0 as *mut c_void));
            Ok(self.0.as_mut().unwrap().as_ptr_mut())
        }
    }

    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        match self.0 {
            Some(ref ec) => ec.as_ptr_ptr(),
            None => 0 as *const cl_event,
        }
    }

    fn _count(&self) -> u32 {
        match self.0 {
            Some(ref ec) => ec.count(),
            None => 0,
        }
    }
}

impl From<NullEventCore> for Event {
    #[inline]
    fn from(nev: NullEventCore) -> Event {
        match nev.validate() {
            Ok(nev) => Event(Some(nev)),
            Err(_) => Event(None),
        }
    }
}

impl From<EventCore> for Event {
    #[inline]
    fn from(ev: EventCore) -> Event {
        Event(Some(ev))
    }
}

impl From<UserEventCore> for Event {
    #[inline]
    fn from(uev: UserEventCore) -> Event {
        if uev.is_valid() {
            Event(Some(uev.into()))
        } else {
            Event(None)
        }
    }
}

// impl Into<EventCore> for Event {
//     fn into(self) -> EventCore {
//         match self.0 {
//             Some(evc) => evc,
//             None => unsafe { EventCore::null() },
//         }
//     }
// }

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

unsafe impl ClNullEventPtr for Event {
    #[inline]
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
        self._ptr_mut_ptr_new()
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut Event {
    #[inline]
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
        self._ptr_mut_ptr_new()
    }
}

unsafe impl ClWaitListPtr for Event {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }

    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}

unsafe impl<'a> ClWaitListPtr for  &'a Event {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }

    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}

impl Future for Event {
    type Item = ();
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.is_complete().map(|_| Async::Ready(()))
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

    /// Adds an event to the list.
    pub fn push(&mut self, event: Event) {
        match event.0 {
            Some(ecore) => self.event_list_core.push(ecore),
            None => panic!("EventList::push: Unable to push null (empty) event."),
        }
    }

    /// Removes the last event from the list and returns it.
    pub fn pop(&mut self) -> Option<Event> {
        self.event_list_core.pop().map(|ev| unsafe { Event::from_core(ev) })
    }

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
                user_data: *mut T,
                ) -> OclResult<()>
    {
        let event_core = try!(try!(self.event_list_core.last_clone().ok_or(
            OclError::string("ocl::EventList::set_callback: This event list is empty."))));

        core::set_event_callback(&event_core, CommandExecutionStatus::Complete,
                    callback_receiver, user_data as *mut _ as *mut c_void)
    }

    /// Clears all events from the list whether or not they have completed.
    pub fn clear(&mut self) -> OclResult<()> {
        self.event_list_core.clear()
    }

    /// Clears events which have completed.
    pub fn clear_completed(&mut self) -> OclResult<()> {
        self.event_list_core.clear_completed()
    }

    /// Returns the number of events in the list.
    pub fn len(&self) -> usize {
        self.event_list_core.len()
    }

    /// Returns if there is no events.
    pub fn is_empty(&self) -> bool {
        self.event_list_core.len() == 0
    }

    /// Returns a reference to the underlying `core` event list.
    ///
    #[deprecated(since="0.13.0", note="Use `::core` instead.")]
    #[inline]
    pub fn core_as_ref(&self) -> &EventListCore {
        &self.event_list_core
    }

    /// Returns a reference to the underlying `core` event list.
    ///
    #[inline]
    pub fn core(&self) -> &EventListCore {
        &self.event_list_core
    }

    /// Returns a mutable reference to the underlying `core` event list.
    ///
    #[deprecated(since="0.13.0", note="Use `::core_mut` instead.")]
    #[inline]
    pub fn core_as_mut(&mut self) -> &mut EventListCore {
        &mut self.event_list_core
    }

    /// Returns a mutable reference to the underlying `core` event list.
    ///
    #[inline]
    pub fn core_mut(&mut self) -> &mut EventListCore {
        &mut self.event_list_core
    }

    /// Waits for all events in list to complete.
    pub fn wait(&self) -> OclResult<()> {
        if !self.event_list_core.is_empty() {
            core::wait_for_events(self.event_list_core.count(), &self.event_list_core)
        } else {
            Ok(())
        }
    }

    #[inline]
    fn _ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
        Ok(self.event_list_core.allot())
    }

    #[inline]
    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        self.event_list_core.as_ptr_ptr()
    }

    #[inline]
    fn _count(&self) -> u32 {
        self.event_list_core.count()
    }
}

impl Into<EventListCore> for EventList {
    #[inline]
    fn into(self) ->  EventListCore {
        self.event_list_core
    }
}

impl AsRef<EventListCore> for EventList {
    #[inline]
    fn as_ref(&self) -> &EventListCore {
        &self.event_list_core
    }
}

impl Deref for EventList {
    type Target = EventListCore;

    #[inline]
    fn deref(&self) -> &EventListCore {
        &self.event_list_core
    }
}

impl DerefMut for EventList {
    #[inline]
    fn deref_mut(&mut self) -> &mut EventListCore {
        &mut self.event_list_core
    }
}

unsafe impl ClNullEventPtr for EventList {
    #[inline]
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
        self._ptr_mut_ptr_new()
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut EventList {
    #[inline]
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
        self._ptr_mut_ptr_new()
    }
}

unsafe impl ClWaitListPtr for EventList {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }

    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}

unsafe impl<'a> ClWaitListPtr for &'a mut EventList {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }

    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}
