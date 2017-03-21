//! An OpenCL event.

use std;
use std::borrow::Borrow;
use std::ops::{Deref, DerefMut};
#[cfg(feature = "event_callbacks")]
use futures::task;
use futures::{Future, Poll, Async};
use ffi::cl_event;
use core::{self, Event as EventCore, EventInfo, EventInfoResult, ProfilingInfo,
    ProfilingInfoResult, ClNullEventPtr, ClWaitListPtr, ClEventPtrRef,
    CommandQueue as CommandQueueCore, ClContextPtr};
use core::error::{Error as OclError, Result as OclResult};
#[cfg(feature = "event_callbacks")]
use standard::{_unpark_task, box_raw_void};
use standard::Queue;

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
    pub fn user<C: ClContextPtr>(context: C) -> OclResult<Event> {
        EventCore::user(context).map(Event)
    }

    /// Returns true if this event is 'empty' and has not yet been associated
    /// with a command.
    ///
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_null()
    }

    /// Sets a callback function to trigger upon completion of this event
    /// which will unpark the current task.
    ///
    /// To be used within the context of a futures task.
    ///
    /// ## Panics
    ///
    /// This function will panic if a task is not currently being executed.
    /// That is, this method can be dangerous to call outside of an
    /// implementation of poll.
    #[cfg(feature = "event_callbacks")]
    pub fn set_unpark_callback(&self) -> OclResult<()> {
        let task_ptr = box_raw_void(task::park());
        unsafe { self.set_callback(_unpark_task, task_ptr) }
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

impl From<Event> for EventCore {
    #[inline]
    fn from(ev: Event) -> EventCore {
        if ev.is_valid() {
            ev.0
        } else {
            panic!("ocl::EventCore::from::<Event>: Invalid event.");
        }
    }
}


impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

unsafe impl<'e> ClEventPtrRef<'e> for Event {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        self.0.as_ptr_ref()
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut Event {
    #[inline] fn alloc_new(&mut self) -> *mut cl_event { (&mut self.0).alloc_new() }

    #[inline] unsafe fn clone_from<E: AsRef<EventCore>>(&mut self, ev: E) {
        self.0.clone_from(ev.as_ref())
    }
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
/// * [NOTE]: There is currently no check to ensure that only one callback is
///   created (is this ok?).
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
                self.set_unpark_callback()?;
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
// * [FIXME] TODO: impl Index.
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

    /// Pushes an `Option<Event>` to the list if it is `Some(...)`.
    pub fn push_some<E: Into<Event>>(&mut self, event: Option<E>) {
        if let Some(e) = event { self.push(e.into()) }
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
    // * TODO: Reimplement optimized version using
    //   `util::vec_remove_rebuild` (from old `EventListCore`).
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

    /// Enqueue a marker event representing the completion of each and every
    /// event in this list.
    pub fn enqueue_marker(&self, queue: &Queue) -> OclResult<Event> {
        if self.events.is_empty() { return Err("EventList::enqueue_marker: List empty.".into()); }
        // let mut marker = Event::empty();
        queue.enqueue_marker(Some(self))
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

    #[inline] unsafe fn clone_from<E: AsRef<EventCore>>(&mut self, ev: E) {
        assert!(ev.as_ref().is_valid());
        *self._alloc_new() = ev.as_ref().clone().into_raw()
    }
}

unsafe impl ClWaitListPtr for EventList {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self._count() }
}

unsafe impl<'a> ClWaitListPtr for &'a EventList {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self._count() }
}


// pub trait EventOption {
//     type Event;
// }

// impl BorrowEvent for Event {
//     type Event = Event;
// }

/// A stack allocated array of `cl_event` pointers with a maximum length of 8.
///
/// Do not store this list beyond the enqueue call(s) with which this list is
/// first used. Use `EventList` for a persistent list.
///
//
// * [NOTE]: Consider implementing using
//   `https://github.com/servo/rust-smallvec` instead.
//
#[derive(Debug, Clone)]
pub struct RawEventArray {
    list: [cl_event; 8],
    count: u32,
}

impl RawEventArray {
    pub unsafe fn null() -> RawEventArray {
        RawEventArray {
            list: [0 as cl_event; 8],
            count: 0,
        }
    }

    #[inline]
    pub fn push<E>(&mut self, e: E) where E: AsRef<EventCore> {
        if (self.count as usize) < self.list.len() {
            self.list[(self.count as usize)] = unsafe { *e.as_ref().as_ptr_ref() };
            self.count += 1;
        } else {
            panic!("RawEventArray::push: List is full.");
        }
    }

    #[inline]
    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        match self.list.first() {
            Some(ev) => ev as *const _ as *const cl_event,
            None => 0 as *const cl_event,
        }
    }

    /// Enqueue a marker event representing the completion of each and every
    /// event in this list.
    pub fn to_marker(&self, queue: &Queue) -> OclResult<Option<Event>> {
        if self.list.is_empty() { return Ok(None); }
        queue.enqueue_marker(Some(self)).map(|marker_event| Some(marker_event))
    }

    /// Enqueue a marker event representing the completion of each and every
    /// event in this list.
    pub fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        self.to_marker(queue)
    }
}

impl<'e, E> From<&'e [E]> for RawEventArray where E: Borrow<Event> {
    fn from(events: &[E]) -> RawEventArray {
        let mut list = unsafe { RawEventArray::null() };

        for e in events {
            list.push(e.borrow());
        }

        list
    }
}

impl<'e, E> From<&'e [Option<E>]> for RawEventArray where E: Borrow<Event> {
    fn from(events: &[Option<E>]) -> RawEventArray {
        let mut list = unsafe { RawEventArray::null() };

        for event in events {
            if let Some(ref e) = *event {
                list.push(e.borrow());
            }
        }

        list
    }
}

impl<'e, 'f, E> From<&'e [&'f Option<E>]> for RawEventArray where 'e: 'f, E: Borrow<Event> {
    fn from(events: &[&Option<E>]) -> RawEventArray {
        let mut list = unsafe { RawEventArray::null() };

        for &event in events {
            if let Some(ref e) = *event {
                list.push(e.borrow());
            }
        }

        list
    }
}

macro_rules! impl_raw_list_from_arrays {
    ($( $len:expr ),*) => ($(
        impl<'e, E> From<[E; $len]> for RawEventArray where E: Borrow<Event> {
            fn from(events: [E; $len]) -> RawEventArray {
                let mut list = unsafe { RawEventArray::null() };

                for e in &events {
                    list.push(e.borrow());
                }

                list
            }
        }

        impl<'e, E> From<[Option<E>; $len]> for RawEventArray where E: Borrow<Event> {
            fn from(events: [Option<E>; $len]) -> RawEventArray {
                let mut list = unsafe { RawEventArray::null() };

                for event in &events {
                    if let Some(ref e) = *event {
                        list.push(e.borrow());
                    }
                }

                list
            }
        }

        impl<'e, 'f, E> From<[&'f Option<E>; $len]> for RawEventArray where 'e: 'f, E: Borrow<Event> {
            fn from(events: [&'f Option<E>; $len]) -> RawEventArray {
                let mut list = unsafe { RawEventArray::null() };

                for &event in &events {
                    if let Some(ref e) = *event {
                        list.push(e.borrow());
                    }
                }

                list
            }
        }
    )*);
}

impl_raw_list_from_arrays!(1, 2, 3, 4, 5, 6, 7, 8);

unsafe impl ClWaitListPtr for RawEventArray {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self.count }
}

unsafe impl<'a> ClWaitListPtr for &'a RawEventArray {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self.count }
}


/// Conversion to a 'marker' event.
pub trait IntoMarker {
    fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>>;
}

// impl<'e> IntoMarker for &'e [Event] {
//     fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
//         RawEventArray::from(self).into_marker (queue)
//     }
// }

impl<'s, 'e> IntoMarker for &'s [&'e Event] where 'e: 's {
    fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        RawEventArray::from(self).into_marker(queue)
    }
}

impl<'s, 'e> IntoMarker for &'s [Option<&'e Event>] where 'e: 's {
    fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        RawEventArray::from(self).into_marker(queue)
    }
}

impl<'s, 'o, 'e> IntoMarker for &'s [&'o Option<&'e Event>] where 'e: 's + 'o, 'o: 's {
    fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        RawEventArray::from(self).into_marker(queue)
    }
}

macro_rules! impl_marker_arrays {
    ($( $len:expr ),*) => ($(
        // impl<'e> IntoMarker for [Event; $len] {
        //     fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        //         RawEventArray::from(self).into_marker (queue)
        //     }
        // }

        impl<'s, 'e> IntoMarker for [&'e Event; $len] where 'e: 's {
            fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
                RawEventArray::from(self).into_marker(queue)
            }
        }

        impl<'s, 'e> IntoMarker for [Option<&'e Event>; $len] where 'e: 's {
            fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
                RawEventArray::from(self).into_marker(queue)
            }
        }

        impl<'s, 'o, 'e> IntoMarker for [&'o Option<&'e Event>; $len] where 'e: 's + 'o, 'o: 's {
            fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
                RawEventArray::from(self).into_marker(queue)
            }
        }
    )*);
}

impl_marker_arrays!(1, 2, 3, 4, 5, 6, 7, 8);


/// Conversion to a stack allocated array of `cl_event` pointers.
pub trait IntoRawEventArray {
    fn into_raw_list(self) -> RawEventArray;
}

// impl<'e> IntoRawEventArray  for &'e [Event] {
//     fn into_raw_list(self) -> RawEventArray {
//         RawEventArray::from(self)
//     }
// }

impl<'s, 'e> IntoRawEventArray  for &'s [&'e Event] where 'e: 's {
    fn into_raw_list(self) -> RawEventArray {
        RawEventArray::from(self)
    }
}

impl<'s, 'e> IntoRawEventArray  for &'s [Option<&'e Event>] where 'e: 's {
    fn into_raw_list(self) -> RawEventArray {
        RawEventArray::from(self)
    }
}

impl<'s, 'o, 'e> IntoRawEventArray  for &'s [&'o Option<&'e Event>] where 'e: 's + 'o, 'o: 's {
    fn into_raw_list(self) -> RawEventArray {
        RawEventArray::from(self)
    }
}


macro_rules! impl_raw_list_arrays {
    ($( $len:expr ),*) => ($(
        // impl<'e> IntoRawEventArray  for [Event; $len] {
        //     fn into_raw_list(self) -> RawEventArray {
        //         RawEventArray::from(self)
        //     }
        // }

        impl<'s, 'e> IntoRawEventArray  for [&'e Event; $len] where 'e: 's {
            fn into_raw_list(self) -> RawEventArray {
                RawEventArray::from(self)
            }
        }

        impl<'s, 'e> IntoRawEventArray  for [Option<&'e Event>; $len] where 'e: 's {
            fn into_raw_list(self) -> RawEventArray {
                RawEventArray::from(self)
            }
        }

        impl<'s, 'o, 'e> IntoRawEventArray  for [&'o Option<&'e Event>; $len] where 'e: 's + 'o, 'o: 's {
            fn into_raw_list(self) -> RawEventArray {
                RawEventArray::from(self)
            }
        }
    )*);
}

impl_raw_list_arrays!(1, 2, 3, 4, 5, 6, 7, 8);