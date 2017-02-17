//! An OpenCL event.

use std;
use std::ops::{Deref, DerefMut};
// use std::convert::Into;
use libc::c_void;
use futures::{task, Future, Poll, Async};
use ffi::cl_event;
use core::error::{Error as OclError, Result as OclResult};
use core::{self, Event as EventCore, EventInfo, EventInfoResult, ProfilingInfo,
    ProfilingInfoResult, ClNullEventPtr, ClWaitListPtr, ClEventRef, Context,
    CommandQueue as CommandQueueCore};
use standard::_unpark_task;

/// An event representing a command or user created event.
///
#[derive(Clone, Debug)]
// pub struct Event(Option<EventCore>);
// pub struct Event(EventCoreKind);
// pub enum Event {
//     Event(EventCore),
//     UserEvent(UserEventCore),
//     Empty,
// }
#[repr(C)]
pub struct Event(EventCore);

impl Event {
    /// Creates a new, empty (null) event which must be filled by a command,
    /// associating the event with it.
    pub fn empty() -> Event {
        // Event(None)
        Event(EventCore::null())
    }

    /// Creates a new, empty event which must be filled by a newly initiated
    /// command, associating the event with it.
    pub fn user(context: &Context) -> OclResult<Event> {
        EventCore::user(context).map(Event)
    }

    // /// Creates a new `Event` from a `EventCore`.
    // ///
    // /// ## Safety
    // ///
    // /// Not meant to be called directly.
    // pub unsafe fn from_core_unchecked(event_core: EventCore) -> Event {
    //     Event::Event(event_core)
    // }

    // /// Returns true if this event is complete, false if it is not complete or
    // /// if this event is not yet associated with a command.
    // ///
    // /// This is the fastest possible way to determine the event completion
    // /// status.
    // ///
    // #[inline]
    // pub fn is_complete(&self) -> OclResult<bool> {
    //     // match *self {
    //     //     Some(ref core) => core.is_complete(),
    //     //     None => Ok(false),
    //     // }

    //     // match *self {
    //     //     Event::Event(ref core) => core.is_complete(),
    //     //     Event::UserEvent(ref core) => core.is_complete(),
    //     //     Event::Empty => Err(().into()),
    //     // }

    //     self.0.is_complete()
    // }

    // /// Waits on the host thread for commands identified by this event object
    // /// to complete.
    // ///
    // pub fn wait_for(&self) -> OclResult<()> {
    //     // assert!(!self.is_empty(), "ocl::Event::wait(): {}", self.err_empty());
    //     // core::wait_for_event(self.0.as_ref().unwrap())
    //     match *self {
    //         Event::Event(ref core) => core.wait_for(),
    //         Event::UserEvent(ref core) => core.wait_for(),
    //         Event::Empty => Err(format!("ocl::Event::wait(): {}", self.err_empty()).into()),
    //     }
    // }

    /// Returns true if this event is 'empty' and has not yet been associated
    /// with a command.
    ///
    #[inline]
    pub fn is_empty(&self) -> bool {
        // match *self {
        //     Event::Event(_) => false,
        //     Event::UserEvent(_) => false,
        //     Event::Empty => true,
        // }
        self.0.is_null()
    }

    /// Returns info about the event.
    pub fn info(&self, info_kind: EventInfo) -> EventInfoResult {
        // match *self {
        //     Some(ref core) => {
        //         // match core::get_event_info(core, info_kind) {
        //         //     Ok(pi) => pi,
        //         //     Err(err) => EventInfoResult::Error(Box::new(err)),
        //         // }
        //         core::get_event_info(core, info_kind)
        //     },
        //     None => EventInfoResult::Error(Box::new(self.err_empty())),
        // }
        // match *self {
        //     Event::Event(ref core) => core::get_event_info(core, info_kind),
        //     Event::UserEvent(ref core) => core::get_event_info(core, info_kind),
        //     Event::Empty => EventInfoResult::Error(Box::new(self.err_empty())),
        // }
        core::get_event_info(&self.0, info_kind)
    }

    /// Returns info about the event.
    pub fn profiling_info(&self, info_kind: ProfilingInfo) -> ProfilingInfoResult {
        // match *self {
        //     Some(ref core) => {
        //         // match core::get_event_profiling_info(core, info_kind) {
        //         //     Ok(pi) => pi,
        //         //     Err(err) => ProfilingInfoResult::Error(Box::new(err)),
        //         // }
        //         core::get_event_profiling_info(core, info_kind)
        //     },
        //     None => ProfilingInfoResult::Error(Box::new(self.err_empty())),
        // }
        // match *self {
        //     Event::Event(ref core) => core::get_event_profiling_info(core, info_kind),
        //     Event::UserEvent(ref core) => core::get_event_profiling_info(core, info_kind),
        //     Event::Empty => ProfilingInfoResult::Error(Box::new(self.err_empty())),
        // }
        core::get_event_profiling_info(&self.0, info_kind)
    }

    /// Returns this event's associated command queue.
    pub fn queue_core(&self) -> OclResult<CommandQueueCore> {
        match self.info(EventInfo::CommandQueue) {
            EventInfoResult::CommandQueue(queue_core) => Ok(queue_core),
            EventInfoResult::Error(err) => Err(*err),
            _ => unreachable!(),
        }
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    ///
    #[inline]
    pub fn core(&self) -> &EventCore {
        // self.0.as_ref()
        // match *self {
        //     Core::Event(ref core) => core::wait_for_event(core),
        //     Core::UserEvent(ref core) => core::wait_for_event(core),
        //     Core::Null => Err(format!("ocl::Event::wait(): {}", self.err_empty()).into()),
        // }
        // match *self {
        //     Event::Event(ref core) => EventVariantRef::Event(core),
        //     Event::UserEvent(ref core) => EventVariantRef::UserEvent(core),
        //     Event::Empty => EventVariantRef::Null,
        // }
        &self.0
    }

    // /// Returns a mutable reference to the core pointer wrapper usable by
    // /// functions in the `core` module.
    // ///
    // #[inline]
    // pub fn core_mut(&mut self) -> EventVariantMut {
    //     // self.0.as_mut()
    //     match *self {
    //         Event::Event(ref mut core) => EventVariantMut::Event(core),
    //         Event::UserEvent(ref mut core) => EventVariantMut::UserEvent(core),
    //         Event::Empty => EventVariantMut::Null,
    //     }
    // }

    // /// Sets a callback function, `callback_receiver`, to trigger upon
    // /// completion of this event passing an optional reference to user data.
    // ///
    // /// # Safety
    // ///
    // /// `user_data` must be guaranteed to still exist if and when `callback_receiver`
    // /// is called.
    // ///
    // pub unsafe fn set_callback(&self, receiver: Option<EventCallbackFn>,
    //         user_data_ptr: *mut c_void) -> OclResult<()>
    // {
    //     match *self {
    //         Event::Event(ref core) => core.set_callback(receiver, user_data_ptr),
    //         Event::UserEvent(ref core) => core.set_callback(receiver, user_data_ptr),
    //         Event::Empty => Err("ocl::Event::set_callback: This event is uninitialized (null).".into()),
    //     }
    // }

    // pub fn as_ptr(&self) -> cl_event {
    //     match *self {
    //         Event::Event(ref core) => core.as_ptr(),
    //         Event::UserEvent(ref core) => core.as_ptr(),
    //         Event::Empty => 0 as cl_event,
    //     }
    // }

    // fn err_empty(&self) -> OclError {
    //     OclError::string("This `ocl::Event` is empty and cannot be used until \
    //         filled by a command.")
    // }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Event")
            .field("CommandQueue", &self.info(EventInfo::CommandQueue))
            .field("CommandType", &self.info(EventInfo::CommandType))
            .field("ReferenceCount", &self.info(EventInfo::ReferenceCount))
            .field("CommandExecutionStatus", &self.info(EventInfo::CommandExecutionStatus))
            .field("Context", &self.info(EventInfo::Context))
            .finish()
    }

    // fn _alloc_new(&mut self) -> *mut cl_event {
    //     // assert!(self.is_empty(), "ocl::Event: Attempting to use a non-empty event as a new event
    //     //     is not allowed. Please create a new, empty, event with ocl::Event::empty().");
    //     if self.is_empty() {
    //         // unsafe {
    //         //     *self = Event::Event(EventCore::null());

    //         //     if let Event::Event(ref mut ev) = *self {
    //         //         ev.as_ptr_mut()
    //         //     } else {
    //         //         unreachable!();
    //         //     }
    //         // }
    //         self.0 = EventCore::null();
    //         self.0.as_ptr_mut()
    //     } else {
    //         panic!("ocl::Event: Attempting to use a non-empty event as a new event
    //             is not allowed. Please create a new, empty, event with ocl::Event::empty().");
    //     }
    // }

    // unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
    //     // match *self {
    //     //     Some(ref ec) => ec.as_ptr_ptr(),
    //     //     None => 0 as *const cl_event,
    //     // }
    //     // match *self {
    //     //     Event::Event(ref core) => core.as_ptr_ptr(),
    //     //     Event::UserEvent(ref core) => core.as_ptr_ptr(),
    //     //     Event::Empty => panic!("<ocl::Event as ClWaitListPtr>::as_ptr_ptr: Event is null"),
    //     // }
    //     self.0.as_ptr_ptr()
    // }

    #[inline]
    fn _count(&self) -> u32 {
        // match *self {
        //     Event::Event(ref core) => core.count(),
        //     Event::UserEvent(ref core) => core.count(),
        //     Event::Empty => panic!("<ocl::Event as ClWaitListPtr>::as_ptr_ptr: Event is null"),
        // }
        if self.0.is_null() { 0 } else { 1 }
    }
}

impl Deref for Event {
    type Target = EventCore;

    fn deref(&self) -> &EventCore {
        &self.0
    }
}

impl DerefMut for Event {
    fn deref_mut(&mut self) -> &mut EventCore {
        &mut self.0
    }
}

impl AsRef<EventCore> for Event {
    fn as_ref(&self) -> &EventCore {
        &self.0
    }
}

impl From<EventCore> for Event {
    #[inline]
    fn from(ev: EventCore) -> Event {
        if ev.is_valid() {
            Event(ev)
        } else {
            panic!("ocl::Event::from::<EventCore>: Invalid event.");
        }
    }
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl<'e> ClEventRef<'e> for Event {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
    //     match *self {
    //         Event::Event(ref core) => core.as_ptr_ref(),
    //         Event::UserEvent(ref core) => core.as_ptr_ref(),
    //         Event::Empty => panic!("<ocl::Event as ClWaitListPtr>::as_ptr_ptr: Event is null"),
    //     }
        self.0.as_ptr_ref()
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut Event {
    #[inline] fn alloc_new(&mut self) -> *mut cl_event { (&mut self.0).alloc_new() }
}

unsafe impl ClWaitListPtr for Event {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self.0.as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self._count() }
}

unsafe impl<'a> ClWaitListPtr for  &'a Event {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self.0.as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self._count() }
}

/// Non-blocking, proper implementation.
#[cfg(not(feature = "disable_event_callbacks"))]
impl Future for Event {
    type Item = ();
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.is_complete() {
            Ok(true) => {
                Ok(Async::Ready(()))
            }
            Ok(false) => {
                let task_box = Box::new(task::park());
                let task_ptr = Box::into_raw(task_box) as *mut _ as *mut c_void;
                unsafe { self.0.set_callback(Some(_unpark_task), task_ptr)?; };
                Ok(Async::NotReady)
            },
            Err(err) => Err(err),
        }
    }
}

/// Blocking implementation (yuk).
#[cfg(feature = "disable_event_callbacks")]
impl Future for Event {
    type Item = ();
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.wait_for()?;
        Ok(Async::Ready(()))
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
    // event_list_core: EventListCore,
    events: Vec<Event>,
}

impl EventList {
    /// Returns a new, empty, `EventList`.
    pub fn new() -> EventList {

        EventList {
            events: Vec::new(),
        }
    }

    /// Returns a new, empty, `EventList` with an initial capacity of `cap`.
    pub fn with_capacity(cap: usize) -> EventList {
        EventList {
            events: Vec::with_capacity(cap),
        }
    }

    /// Adds an event to the list.
    pub fn push(&mut self, event: Event) {
        self.events.push(event)
    }

    /// Removes the last event from the list and returns it.
    pub fn pop(&mut self) -> Option<Event> {
        self.events.pop()
    }

    // /// Returns a new copy of an event by index.
    // pub fn get(&self, index: usize) -> Option<&Event> {
    //     self.events.get(index)
    // }

    // /// Returns a copy of the last event in the list.
    // pub fn last(&self) -> Option<&Event> {
    //     self.events.last()
    // }

    /// Clears all events from the list whether or not they have completed.
    ///
    /// Forwards any errors related to releasing events.
    ///
    #[inline]
    pub fn clear(&mut self) {
        self.events.clear()
    }

    /// Clears events which have completed.
    //
    // [TODO]: Reimplement optimized version using `util::vec_remove_rebuild`
    // (from old `EventListCore`).
    pub fn clear_completed(&mut self) -> OclResult<()> {
        let mut events = Vec::with_capacity(self.events.len());

        std::mem::swap(&mut events, &mut self.events);

        for event in events {
            if !event.is_complete()? {
                self.events.push(event);
            }
        }

        Ok(())
    }

    // /// Returns the number of events in the list.
    // pub fn len(&self) -> usize {
    //     self.events.len()
    // }

    // /// Returns if there are no events.
    // pub fn is_empty(&self) -> bool {
    //     self.events.len() == 0
    // }

    /// Waits on the host thread for all events in list to complete.
    pub fn wait_for(&self) -> OclResult<()> {
        for event in self.events.iter() {
            event.wait_for()?;
        }

        Ok(())
    }

    #[inline]
    fn _alloc_new(&mut self) -> *mut cl_event {
        self.events.push(Event::empty());
        self.events.last_mut().unwrap() as *mut _ as *mut cl_event
    }

    #[inline]
    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        match self.events.first() {
            Some(ev) => ev as *const _ as *const cl_event,
            None => 0 as *const cl_event,
        }
    }

    #[inline]
    fn _count(&self) -> u32 {
        self.events.len() as u32
    }
}

impl Deref for EventList {
    type Target = [Event];

    fn deref(&self) -> &[Event] {
        self.events.as_slice()
    }
}

impl DerefMut for EventList {
    fn deref_mut(&mut self) -> &mut [Event] {
        self.events.as_mut_slice()
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut EventList {
    #[inline] fn alloc_new(&mut self) -> *mut cl_event { self._alloc_new() }
}

unsafe impl ClWaitListPtr for EventList {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self._count() }
}

unsafe impl<'a> ClWaitListPtr for &'a EventList {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self._count() }
}
