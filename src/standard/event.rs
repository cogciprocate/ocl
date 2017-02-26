//! An OpenCL event.

use std;
use std::ops::{Deref, DerefMut};
use futures::{task, Future, Poll, Async};
use ffi::cl_event;
use core::error::{Error as OclError, Result as OclResult};
use core::{self, Event as EventCore, EventInfo, EventInfoResult, ProfilingInfo,
    ProfilingInfoResult, ClNullEventPtr, ClWaitListPtr, ClEventRef, Context,
    CommandQueue as CommandQueueCore};
use standard::{_unpark_task, box_raw_void};

/// An event representing a command or user created event.
///
#[derive(Clone, Debug)]
#[repr(C)]
#[must_use = "futures do nothing unless polled"]
pub struct Event(EventCore);

impl Event {
    /// Creates a new, empty (null) event which must be filled by a command,
    /// associating the event with it.
    pub fn empty() -> Event {
        Event(EventCore::null())
    }

    /// Creates a new, empty event which must be filled by a newly initiated
    /// command, associating the event with it.
    pub fn user(context: &Context) -> OclResult<Event> {
        EventCore::user(context).map(Event)
    }

    /// Returns true if this event is 'empty' and has not yet been associated
    /// with a command.
    ///
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_null()
    }

    /// Returns info about the event.
    pub fn info(&self, info_kind: EventInfo) -> EventInfoResult {
        core::get_event_info(&self.0, info_kind)
    }

    /// Returns info about the event.
    pub fn profiling_info(&self, info_kind: ProfilingInfo) -> ProfilingInfoResult {
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
        &self.0
    }

    /// Consumes the `Event`, returning the wrapped `cl_event` pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Event` using
    /// [`Event::from_raw`][from_raw].
    ///
    /// [from_raw]: struct.Event.html#method.from_raw
    ///
    #[inline]
    pub fn into_raw(self) -> cl_event {
        self.0.into_raw()
    }

    /// Constructs an `Event` from a raw `cl_event` pointer.
    ///
    /// The raw pointer must have been previously returned by a call to a
    /// [`Event::into_raw`][into_raw].
    ///
    /// [into_raw]: struct.Event.html#method.into_raw
    #[inline]
    pub unsafe fn from_raw(ptr: cl_event) -> Event {
        EventCore::from_raw(ptr).into()
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

    #[inline]
    fn _count(&self) -> u32 {
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

impl Into<EventCore> for Event {
    fn into(self) -> EventCore {
        self.0
    }
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl<'e> ClEventRef<'e> for Event {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
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
///
/// [NOTE]: There is currently no check to ensure that only one callback is
/// created (is this ok?).
///
#[cfg(feature = "event_callbacks")]
impl Future for Event {
    type Item = ();
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.is_complete() {
            Ok(true) => {
                Ok(Async::Ready(()))
            }
            Ok(false) => {
                let task_ptr = box_raw_void(task::park());
                unsafe { self.0.set_callback(_unpark_task, task_ptr)?; };
                Ok(Async::NotReady)
            },
            Err(err) => Err(err),
        }
    }
}

/// Blocking implementation (yuk).
#[cfg(not(feature = "event_callbacks"))]
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
