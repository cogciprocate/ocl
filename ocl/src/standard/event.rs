//! An OpenCL event.
//!
//
// ### Notes
//
// * `EventArray` (a stack allocated event list with a maximum length of 8,
//   akin to `RawEventArray`) is incomplete (TODO: Complete it).
// * It's not yet clear whether or not to keep EventArray and EventList
//   separate or to combine them into a smart-list that might be either one
//   depending on the circumstances.
// * It would be nice to have a "master" list type but it doesn't look like
//   that's particularly feasable (although ClWaitListPtrEnum basically serves
//   that role as long as lifetimes aren't an issue).
//

extern crate nodrop;

use self::nodrop::NoDrop;
use crate::core::{
    self, ClContextPtr, ClEventPtrRef, ClNullEventPtr, ClWaitListPtr,
    CommandQueue as CommandQueueCore, Event as EventCore, EventInfo, EventInfoResult,
    ProfilingInfo, ProfilingInfoResult,
};
use crate::error::{Error as OclError, Result as OclResult};
use crate::ffi::cl_event;
use crate::standard::{ClWaitListPtrEnum, Queue};
#[cfg(not(feature = "async_block"))]
use crate::standard::{_unpark_task, box_raw_void};
#[cfg(not(feature = "async_block"))]
use futures::task;
use futures::{Async, Future, Poll};
use std::borrow::Borrow;
use std::cell::Ref;
use std::ops::{Deref, DerefMut};
use std::{fmt, mem, ptr};

const PRINT_DEBUG: bool = false;

/// An event representing a command or user created event.
///
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
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
        EventCore::user(context).map(Event).map_err(OclError::from)
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
    #[cfg(not(feature = "async_block"))]
    pub fn set_unpark_callback(&self) -> OclResult<()> {
        let task_ptr = box_raw_void(task::current());
        unsafe {
            self.set_callback(_unpark_task, task_ptr)
                .map_err(OclError::from)
        }
    }

    /// Registers a user event to have its status set to complete
    /// (`CommandExecutionStatus::Complete`) immediately upon completion of
    /// this event.
    ///
    /// ## Deadlocks
    ///
    /// Due to the nature of OpenCL queue implementations, care must be taken
    /// when using this function. OpenCL queue deadlocks may occur.
    ///
    /// OpenCL queues generally use one thread per queue for the purposes of
    /// callbacks, etc. As a rule of thumb, ensure that any OpenCL commands
    /// preceding the causation/source event (`self`) are in a separate queue
    /// from any commands with the dependent/target event (`user_event`).
    ///
    /// ## Safety
    ///
    /// The caller must ensure that `user_event` was created with
    /// `Event::user()` and that it's status is
    /// `CommandExecutionStatus::Submitted` (the default upon creation).
    ///
    #[cfg(not(feature = "async_block"))]
    pub unsafe fn register_event_relay(&self, user_event: Event) -> OclResult<()> {
        let unmap_event_ptr = user_event.into_raw();
        self.set_callback(core::_complete_user_event, unmap_event_ptr)
            .map_err(OclError::from)
    }

    /// Returns info about the event.
    pub fn info(&self, info_kind: EventInfo) -> OclResult<EventInfoResult> {
        core::get_event_info(&self.0, info_kind).map_err(OclError::from)
    }

    /// Returns info about the event.
    pub fn profiling_info(&self, info_kind: ProfilingInfo) -> OclResult<ProfilingInfoResult> {
        core::get_event_profiling_info(&self.0, info_kind).map_err(OclError::from)
    }

    /// Returns this event's associated command queue.
    pub fn queue_core(&self) -> OclResult<CommandQueueCore> {
        match self.info(EventInfo::CommandQueue)? {
            EventInfoResult::CommandQueue(queue_core) => Ok(queue_core),
            _ => unreachable!(),
        }
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    ///
    #[inline]
    pub fn as_core(&self) -> &EventCore {
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

    fn fmt_info(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Event")
            .field("CommandQueue", &self.info(EventInfo::CommandQueue))
            .field("CommandType", &self.info(EventInfo::CommandType))
            .field("ReferenceCount", &self.info(EventInfo::ReferenceCount))
            .field(
                "CommandExecutionStatus",
                &self.info(EventInfo::CommandExecutionStatus),
            )
            .field("Context", &self.info(EventInfo::Context))
            .finish()
    }

    #[inline]
    fn _count(&self) -> u32 {
        if self.0.is_null() {
            0
        } else {
            1
        }
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

impl Default for Event {
    fn default() -> Event {
        Event::empty()
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

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_info(f)
    }
}

unsafe impl<'e> ClEventPtrRef<'e> for Event {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        self.0.as_ptr_ref()
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut Event {
    #[inline]
    fn alloc_new(&mut self) -> *mut cl_event {
        (&mut self.0).alloc_new()
    }

    #[inline]
    unsafe fn clone_from<E: AsRef<EventCore>>(&mut self, ev: E) {
        self.0.clone_from(ev.as_ref())
    }
}

unsafe impl ClWaitListPtr for Event {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self.0.as_ptr_ptr()
    }
    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}

unsafe impl<'a> ClWaitListPtr for &'a Event {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self.0.as_ptr_ptr()
    }
    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}

impl Future for Event {
    type Item = ();
    type Error = OclError;

    // Non-blocking, proper implementation.
    //
    // * NOTE: There is currently no check to ensure that only one callback is
    //   created (is this ok?).
    //   - TODO: Look into possible effects of unparking a task multiple times.
    //
    #[cfg(not(feature = "async_block"))]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        debug_assert!(self.0.is_valid());

        match self.is_complete() {
            Ok(true) => Ok(Async::Ready(())),
            Ok(false) => {
                self.set_unpark_callback()?;
                Ok(Async::NotReady)
            }
            Err(err) => Err(OclError::from(err)),
        }
    }

    // Blocking implementation (yuk).
    #[cfg(feature = "async_block")]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        debug_assert!(self.0.is_valid());
        self.wait_for()?;
        Ok(Async::Ready(()))
    }
}

/// Returns an empty, initialized (zeroed) event array.
fn empty_event_array() -> NoDrop<[Event; 8]> {
    NoDrop::new(Default::default())
}

/// Swaps an event with a new empty (null) event and returns it.
fn take(event: &mut Event) -> Event {
    let mut eold = Event::empty();
    mem::swap(event, &mut eold);
    eold
}

/// Polls events for `EventArray` and `EventList`
fn poll_events(events: &[Event]) -> Poll<(), OclError> {
    if PRINT_DEBUG {
        println!(
            "####### EventList/Array::poll: Polling Event list (thread: '{}')",
            ::std::thread::current().name().unwrap_or("<unnamed>")
        );
    }

    for event in events.iter() {
        if cfg!(feature = "async_block") {
            if PRINT_DEBUG {
                println!(
                    "####### EventList/Array::poll: waiting for for event: {:?} \
                (thread: '{}')",
                    event,
                    ::std::thread::current().name().unwrap_or("<unnamed>")
                );
            }
            event.wait_for()?;
        } else {
            if !event.is_complete()? {
                #[cfg(not(feature = "async_block"))]
                event.set_unpark_callback()?;
                if PRINT_DEBUG {
                    println!(
                        "####### EventList/Array::poll: callback set for event: {:?} \
                    (thread: '{}')",
                        event,
                        ::std::thread::current().name().unwrap_or("<unnamed>")
                    );
                }
                return Ok(Async::NotReady);
            } else {
                if PRINT_DEBUG {
                    println!(
                        "####### EventList/Array::poll: event complete: {:?} \
                    (thread: '{}')",
                        event,
                        ::std::thread::current().name().unwrap_or("<unnamed>")
                    );
                }
            }
        }
    }

    // let res = future::join_all(self.events.clone()).poll().map(|res| res.map(|_| ()) );
    if PRINT_DEBUG {
        println!(
            "####### EventList/Array::poll: All events complete (thread: '{}')",
            ::std::thread::current().name().unwrap_or("<unnamed>")
        );
    }

    Ok(Async::Ready(()))
    // res
}

/// A list of events for coordinating enqueued commands.
///
/// Events contain status information about the command that
/// created them. Used to coordinate the activity of multiple commands with
/// more fine-grained control than the queue alone.
///
/// For access to individual events use `get_clone` or `last_clone`.
///
//
// * [FIXME] TODO: impl Index.
// #[derive(Debug)]
//
// * [NOTE]: Consider replacing with
//   `https://github.com/servo/rust-smallvec` instead.
//
pub struct EventArray {
    array: NoDrop<[Event; 8]>,
    len: usize,
}

impl EventArray {
    /// Returns a new, empty, `EventArray`.
    pub fn new() -> EventArray {
        EventArray {
            array: empty_event_array(),
            len: 0,
        }
    }

    /// Pushes a new event into the list.
    pub fn push<E: Into<Event>>(&mut self, event: E) -> Result<(), Event> {
        if (self.len) < self.array.len() {
            let event = event.into();
            debug_assert!(self.array[self.len].is_empty());
            self.array[self.len] = event;
            self.len += 1;
            Ok(())
        } else {
            Err(event.into())
        }
    }

    /// Removes the last event from the list and returns it.
    pub fn pop(&mut self) -> Option<Event> {
        if self.len > 0 {
            self.len -= 1;
            Some(take(&mut self.array[self.len]))
        } else {
            None
        }
    }

    /// Removes an event from the list and returns it, swapping the last element into its place.
    pub fn swap_remove(&mut self, idx: usize) -> Event {
        assert!(idx < self.len);
        let old = take(&mut self.array[idx]);
        let src_ptr = &mut self.array[self.len - 1] as *mut Event;
        let dst_ptr = &mut self.array[idx] as *mut Event;
        unsafe {
            ptr::swap(src_ptr, dst_ptr);
        }
        self.len -= 1;
        old
    }

    /// Removes an event from the list and returns it, shifting elements after it to the left.
    ///
    /// [MAY DEPRICATE]: Prefer `::swap_remove`, this function is really unnecessary.
    pub fn remove(&mut self, idx: usize) -> Event {
        assert!(idx < self.len);
        let old = take(&mut self.array[idx]);

        // Shift everything after `idx` to the left:
        unsafe {
            let ptr = self.array.as_mut_ptr().add(idx);
            ptr::copy(ptr.offset(1), ptr, self.len - idx - 1);
        }
        self.len -= 1;
        old
    }

    /// Clears all events from the list whether or not they have completed.
    ///
    /// Forwards any errors related to releasing events.
    ///
    #[inline]
    pub fn clear(&mut self) {
        for ev in &mut self.array[..self.len] {
            let _ = take(ev);
        }
        self.len = 0;
    }

    /// Clears events which have already completed.
    pub fn clear_completed(&mut self) -> OclResult<()> {
        let mut new_len = 0;

        for idx in 0..self.len {
            if self.array[idx].is_complete()? {
                let _ = take(&mut self.array[idx]);
            } else {
                let dst_ptr = &mut self.array[new_len] as *mut Event;
                unsafe {
                    ptr::swap(&mut self.array[idx], dst_ptr);
                }
                new_len += 1;
            }
        }

        self.len = new_len;
        Ok(())
    }

    /// Blocks the host thread until all events in this list are complete.
    pub fn wait_for(&self) -> OclResult<()> {
        for ev in &self.array[..self.len] {
            ev.wait_for()?;
        }
        Ok(())
    }

    /// Enqueue a marker event representing the completion of each and every
    /// event in this list.
    pub fn enqueue_marker(&self, queue: &Queue) -> OclResult<Event> {
        if self.array.is_empty() {
            return Err("EventArray::enqueue_marker: List empty.".into());
        }
        queue.enqueue_marker(Some(self)).map_err(OclError::from)
    }

    /// The number of events in this list.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns a slice of events in the list.
    #[inline]
    pub fn as_slice(&self) -> &[Event] {
        &self.array[..self.len]
    }

    /// Returns a mutable slice of events in the list.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Event] {
        &mut self.array[..self.len]
    }

    #[inline]
    fn _alloc_new(&mut self) -> Result<*mut cl_event, &'static str> {
        match self.push(Event::empty()) {
            Ok(_) => Ok((&mut self.array[self.len - 1]) as *mut _ as *mut cl_event),
            Err(_) => Err(
                "EventArray::_alloc_new: Error attempting to add a new event \
                to the event list: this list is full",
            ),
        }
    }

    #[inline]
    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        if self.len > 0 {
            &self.array[0] as *const _ as *const cl_event
        } else {
            ptr::null()
        }
    }

    #[inline]
    fn _count(&self) -> u32 {
        self.len as u32
    }
}

// Due to a fix to coherence rules
// (https://github.com/rust-lang/rust/pull/46192) we must manually implement
// this for each event type.
macro_rules! from_event_into_event_array(
    ($e:ty) => (
        impl<'a> From<$e> for EventArray {
            fn from(event: $e) -> EventArray {
                let mut array = empty_event_array();
                array[0] = event.into();

                EventArray {
                    array: array,
                    len: 1,
                }
            }
        }
    )
);
from_event_into_event_array!(EventCore);
from_event_into_event_array!(Event);

impl<'a, E> From<&'a E> for EventArray
where
    E: Into<Event> + Clone,
{
    #[inline]
    fn from(event: &E) -> EventArray {
        Self::from(event.clone().into())
    }
}

impl<'a, E> From<&'a [E]> for EventArray
where
    E: Into<Event> + Clone,
{
    fn from(events: &[E]) -> EventArray {
        let mut array = empty_event_array();

        for (idx, event) in events.iter().enumerate() {
            array[idx] = event.clone().into();
        }

        EventArray {
            array,
            len: events.len(),
        }
    }
}

impl Deref for EventArray {
    type Target = [Event];

    #[inline]
    fn deref(&self) -> &[Event] {
        self.as_slice()
    }
}

impl DerefMut for EventArray {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Event] {
        self.as_mut_slice()
    }
}

impl Clone for EventArray {
    fn clone(&self) -> EventArray {
        let mut new_a = empty_event_array();
        for i in 0..self.len {
            new_a[i] = self.array[i].clone();
        }
        EventArray {
            array: new_a,
            len: self.len,
        }
    }
}

impl Drop for EventArray {
    fn drop(&mut self) {
        // ptr::drop_in_place(self.as_mut_slice());
        self.clear();

        for idx in 0..self.array.len() {
            debug_assert!(self.array[idx].is_empty());
        }
    }
}

impl fmt::Debug for EventArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "EventArray {{ array: {:?}, len: {} }}",
            &self.array[..],
            self.len
        )
    }
}

impl Future for EventArray {
    type Item = ();
    type Error = OclError;

    /// Polls each event from this list.
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        poll_events(self.as_slice())
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut EventArray {
    #[inline]
    fn alloc_new(&mut self) -> *mut cl_event {
        self._alloc_new()
            .expect("<EventArray as ClNullEventPtr>::alloc_new")
    }

    #[inline]
    unsafe fn clone_from<E: AsRef<EventCore>>(&mut self, ev: E) {
        assert!(ev.as_ref().is_valid());
        *self
            ._alloc_new()
            .expect("<EventArray as ClNullEventPtr>::clone_from") = ev.as_ref().clone().into_raw()
    }
}

unsafe impl ClWaitListPtr for EventArray {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }
    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}

unsafe impl<'a> ClWaitListPtr for &'a EventArray {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }
    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}

/// The guts of an EventList.
#[derive(Debug, Clone)]
enum Inner {
    Array(EventArray),
    Vec(Vec<Event>),
}

/// A list of events for coordinating enqueued commands.
///
/// Events contain status information about the command that
/// created them. Used to coordinate the activity of multiple commands with
/// more fine-grained control than the queue alone.
///
/// For access to individual events use `get_clone` or `last_clone`.
///
/// `EventList` is a dynamically allocated list. It will be (internally) stack
/// allocated (as an `[Event; 8]`) until it reaches a length of 9 at which
/// time it will become heap-allocated (a `Vec<Event>`).
///
/// Converting back from heap to stack allocation is currently not
/// implemented.
///
// * [FIXME] TODO: impl Index.
//
// * [NOTE]: Consider implementing using
//   `https://github.com/servo/rust-smallvec` instead.
//
#[derive(Debug, Clone)]
pub struct EventList {
    inner: Inner,
}

impl EventList {
    /// Returns a new, empty, stack-allocated `EventList`.
    #[inline]
    pub fn new() -> EventList {
        EventList {
            inner: Inner::Array(EventArray::new()),
        }
    }

    /// Returns a new, empty, EventList` with an initial capacity of `cap`.
    ///
    /// If `cap` is greater than 8, the event list will be heap-allocated.
    #[inline]
    pub fn with_capacity(cap: usize) -> EventList {
        if cap <= 8 {
            EventList {
                inner: Inner::Array(EventArray::new()),
            }
        } else {
            EventList {
                inner: Inner::Vec(Vec::with_capacity(cap)),
            }
        }
    }

    /// Converts the contained list from a stack allocated to a heap allocated
    /// array [or vice-versa].
    ///
    /// FIXME: Implement conversion from Vec -> array.
    fn convert(&mut self) {
        let new_inner: Inner;

        match self.inner {
            Inner::Array(ref a) => {
                let vec = a.as_slice().to_owned();
                new_inner = Inner::Vec(vec);
            }
            Inner::Vec(ref _v) => unimplemented!(),
        }

        self.inner = new_inner;
    }

    /// Adds an event to the list.
    //
    // TODO: Clean this up. There must be a simpler way to do this.
    #[inline]
    pub fn push<E: Into<Event>>(&mut self, event: E) {
        let mut event: Option<Event> = Some(event.into());

        match self.inner {
            Inner::Array(ref mut a) => {
                if let Some(ev) = event {
                    match a.push(ev) {
                        Ok(_) => return,
                        Err(ev) => event = Some(ev),
                    }
                }
            }
            Inner::Vec(ref mut v) => {
                if let Some(ev) = event {
                    v.push(ev);
                }
                return;
            }
        }

        if let Some(ev) = event {
            self.convert();
            self.push(ev);
        }
    }

    /// Removes the last event from the list and returns it.
    #[inline]
    pub fn pop(&mut self) -> Option<Event> {
        match self.inner {
            Inner::Array(ref mut a) => a.pop(),
            Inner::Vec(ref mut v) => v.pop(),
        }
    }

    /// Clears all events from the list whether or not they have completed.
    ///
    /// Forwards any errors related to releasing events.
    ///
    #[inline]
    pub fn clear(&mut self) {
        match self.inner {
            Inner::Array(ref mut a) => a.clear(),
            Inner::Vec(ref mut v) => v.clear(),
        }
    }

    /// Clears events which have completed.
    pub fn clear_completed(&mut self) -> OclResult<()> {
        match self.inner {
            Inner::Array(ref mut a) => a.clear_completed(),
            Inner::Vec(ref mut v) => {
                // * TODO: Reimplement optimized version using
                //   `util::vec_remove_rebuild` (from old `EventListCore`).
                let mut events = Vec::with_capacity(v.capacity());
                mem::swap(&mut events, v);
                for event in events {
                    if !event.is_complete()? {
                        v.push(event);
                    }
                }
                Ok(())
            }
        }
    }

    /// Blocks the host thread until all events in this list are complete.
    pub fn wait_for(&self) -> OclResult<()> {
        match self.inner {
            Inner::Array(ref a) => a.wait_for(),
            Inner::Vec(ref v) => {
                for event in v.iter() {
                    event.wait_for()?;
                }
                Ok(())
            }
        }
    }

    /// Enqueue a marker event representing the completion of each and every
    /// event in this list.
    ///
    /// ### Platform Compatibility
    ///
    /// Some device/platform combinations (particularly older Intel CPUs on
    /// Intel platform drivers) may have intermittent issues waiting on
    /// markers when multiple threads are in use. This is rare and can be
    /// circumvented by using AMD platform drivers instead. Please file an
    /// issue immediately if you run into problems on your platform so that we
    /// may make note of it here in the documentation.
    ///
    pub fn enqueue_marker(&self, queue: &Queue) -> OclResult<Event> {
        match self.inner {
            Inner::Array(ref a) => a.enqueue_marker(queue).map_err(OclError::from),
            Inner::Vec(ref v) => {
                if v.is_empty() {
                    return Err("EventList::enqueue_marker: List empty.".into());
                }
                queue.enqueue_marker(Some(self)).map_err(OclError::from)
            }
        }
    }

    /// Returns a slice of the contained events.
    #[inline]
    pub fn as_slice(&self) -> &[Event] {
        match self.inner {
            Inner::Array(ref a) => a.as_slice(),
            Inner::Vec(ref v) => v.as_slice(),
        }
    }

    /// Returns a mutable slice of the contained events.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [Event] {
        match self.inner {
            Inner::Array(ref mut a) => a.as_mut_slice(),
            Inner::Vec(ref mut v) => v.as_mut_slice(),
        }
    }

    #[inline]
    fn _alloc_new(&mut self) -> *mut cl_event {
        match self.inner {
            Inner::Array(ref mut a) => match a._alloc_new() {
                Ok(ptr) => return ptr,
                Err(_) => (),
            },
            Inner::Vec(ref mut v) => {
                v.push(Event::empty());
                return v.last_mut().unwrap() as *mut _ as *mut cl_event;
            }
        }

        self.convert();
        self._alloc_new()
    }

    #[inline]
    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        match self.inner {
            Inner::Array(ref a) => a._as_ptr_ptr(),
            Inner::Vec(ref v) => match v.first() {
                Some(ev) => ev as *const _ as *const cl_event,
                None => ptr::null(),
            },
        }
    }

    #[inline]
    fn _count(&self) -> u32 {
        match self.inner {
            Inner::Array(ref a) => a._count(),
            Inner::Vec(ref v) => v.len() as u32,
        }
    }
}

// Due to a fix to coherence rules
// (https://github.com/rust-lang/rust/pull/46192) we must manually implement
// this for each event type.
macro_rules! from_event_into_event_list(
    ($e:ty) => (
        impl<'a> From<$e> for EventList {
            #[inline]
            fn from(event: $e) -> EventList {
                EventList { inner: Inner::Array(EventArray::from(event)) }
            }
        }
    )
);
from_event_into_event_list!(EventCore);
from_event_into_event_list!(Event);

impl<'a, E> From<&'a E> for EventList
where
    E: Into<Event> + Clone,
{
    #[inline]
    fn from(event: &E) -> EventList {
        EventList {
            inner: Inner::Array(EventArray::from(event)),
        }
    }
}

impl<'a> From<Vec<Event>> for EventList {
    #[inline]
    fn from(events: Vec<Event>) -> EventList {
        EventList {
            inner: Inner::Vec(events),
        }
    }
}

// Due to a fix to coherence rules
// (https://github.com/rust-lang/rust/pull/46192) we must manually implement
// this for each event type.
macro_rules! from_event_option_ref_into_event_list(
    ($e:ty) => (
        impl<'a> From<&'a Option<$e>> for EventList {
            fn from(event: &Option<$e>) -> EventList {
                let mut el = EventList::new();
                if let Some(ref e) = *event { el.push::<Event>(e.clone().into()) }
                el
            }
        }
    )
);
from_event_option_ref_into_event_list!(EventCore);
from_event_option_ref_into_event_list!(Event);

impl<'a, 'b, E> From<Option<&'b E>> for EventList
where
    'b: 'a,
    E: Into<Event> + Clone,
{
    fn from(event: Option<&E>) -> EventList {
        let mut el = EventList::new();
        if let Some(e) = event {
            el.push(e.clone().into())
        }
        el
    }
}

impl<'a, 'b, E> From<&'a Option<&'b E>> for EventList
where
    'b: 'a,
    E: Into<Event> + Clone,
{
    fn from(event: &Option<&E>) -> EventList {
        let mut el = EventList::new();
        if let Some(e) = *event {
            el.push(e.clone().into())
        }
        el
    }
}

impl<'a, E> From<&'a [E]> for EventList
where
    E: Into<Event> + Clone,
{
    fn from(events: &[E]) -> EventList {
        if events.len() <= 8 {
            EventList {
                inner: Inner::Array(EventArray::from(events)),
            }
        } else {
            EventList {
                inner: Inner::Vec(events.iter().map(|e| e.clone().into()).collect()),
            }
        }
    }
}

// Due to a fix to coherence rules
// (https://github.com/rust-lang/rust/pull/46192) we must manually implement
// this for each event type.
macro_rules! from_event_option_into_event_list(
    ($e:ty) => (
        impl<'a> From<&'a [Option<$e>]> for EventList {
            fn from(events: &[Option<$e>]) -> EventList {
                let mut el = EventList::with_capacity(events.len());
                for event in events {
                    if let Some(ref e) = *event { el.push::<Event>(e.clone().into()) }
                }
                el
            }
        }
    )
);
from_event_option_into_event_list!(EventCore);
from_event_option_into_event_list!(Event);

impl<'a, 'b, E> From<&'a [Option<&'b E>]> for EventList
where
    'b: 'a,
    E: Into<Event> + Clone,
{
    fn from(events: &[Option<&E>]) -> EventList {
        let mut el = EventList::with_capacity(events.len());
        for event in events {
            if let Some(e) = *event {
                el.push(e.clone().into())
            }
        }
        el
    }
}

// Due to a fix to coherence rules
// (https://github.com/rust-lang/rust/pull/46192) we must manually implement
// this for each event type.
macro_rules! from_event_option_slice_into_event_list(
    ($e:ty) => (
        impl<'a, 'b> From<&'a [&'b Option<$e>]> for EventList where 'b: 'a {
            fn from(events: &[&Option<$e>]) -> EventList {
                let mut el = EventList::with_capacity(events.len());
                for event in events {
                    if let Some(ref e) = **event { el.push::<Event>(e.clone().into()) }
                }
                el
            }
        }
    )
);
from_event_option_slice_into_event_list!(EventCore);
from_event_option_slice_into_event_list!(Event);

impl<'a, 'b, 'c, E> From<&'a [&'b Option<&'c E>]> for EventList
where
    'c: 'b,
    'b: 'a,
    E: Into<Event> + Clone,
{
    fn from(events: &[&Option<&E>]) -> EventList {
        let mut el = EventList::with_capacity(events.len());

        for event in events {
            if let Some(e) = **event {
                el.push(e.clone().into())
            }
        }
        el
    }
}

// Due to a fix to coherence rules
// (https://github.com/rust-lang/rust/pull/46192) we must manually implement
// this for each event type.
macro_rules! from_event_option_array_into_event_list(
    ($e:ty, $len:expr) => (
        impl<'e> From<[Option<$e>; $len]> for EventList {
            fn from(events: [Option<$e>; $len]) -> EventList {
                let mut el = EventList::with_capacity(events.len());
                for idx in 0..events.len() {
                    let event_opt = unsafe { ptr::read(events.get_unchecked(idx)) };
                    if let Some(event) = event_opt { el.push::<Event>(event.into()); }
                }
                mem::forget(events);
                el
            }
        }
    )
);

// Due to a fix to coherence rules
// (https://github.com/rust-lang/rust/pull/46192) we must manually implement
// this for each event type.
macro_rules! from_event_option_ref_array_into_event_list(
    ($e:ty, $len:expr) => (
        impl<'e, 'f> From<[&'f Option<$e>; $len]> for EventList where 'e: 'f {
            fn from(events: [&'f Option<$e>; $len]) -> EventList {
                let mut el = EventList::with_capacity(events.len());
                for event_opt in &events {
                    if let Some(event) = *event_opt {
                        el.push(event.clone());
                    }
                }
                el
            }
        }
    )
);

macro_rules! impl_event_list_from_arrays {
    ($( $len:expr ),*) => ($(
        impl<'e, E> From<[E; $len]> for EventList where E: Into<Event> {
            fn from(events: [E; $len]) -> EventList {
                let mut el = EventList::with_capacity(events.len());
                for idx in 0..events.len() {
                    let event = unsafe { ptr::read(events.get_unchecked(idx)) };
                    el.push(event.into());
                }
                // Ownership has been unsafely transfered to the new event
                // list without modifying the event reference count. Not
                // forgetting the source array would cause a double drop.
                mem::forget(events);
                el
            }
        }

        from_event_option_array_into_event_list!(EventCore, $len);
        from_event_option_array_into_event_list!(Event, $len);

        impl<'e, E> From<[Option<&'e E>; $len]> for EventList where E: Into<Event> + Clone {
            fn from(events: [Option<&E>; $len]) -> EventList {
                let mut el = EventList::with_capacity(events.len());
                for event_opt in &events {
                    if let Some(event) = event_opt.cloned() {
                        el.push(event);
                    }
                }
                el
            }
        }

        from_event_option_ref_array_into_event_list!(EventCore, $len);
        from_event_option_ref_array_into_event_list!(Event, $len);

        impl<'e, 'f, E> From<[&'f Option<&'e E>; $len]> for EventList where 'e: 'f, E: Into<Event> + Clone {
            fn from(events: [&'f Option<&'e E>; $len]) -> EventList {
                let mut el = EventList::with_capacity(events.len());
                for event_opt in &events {
                    if let Some(event) = event_opt.cloned() {
                        el.push(event);
                    }
                }
                el
            }
        }
    )*);
}

impl_event_list_from_arrays!(1, 2, 3, 4, 5, 6, 7, 8);

impl<'a> From<&'a [cl_event]> for EventList {
    fn from(raw_events: &[cl_event]) -> EventList {
        let mut event_list = EventList::new();
        for &ptr in raw_events {
            let event_core = unsafe {
                EventCore::from_raw_copied_ptr(ptr)
                    .expect("EventList::from: Error converting from raw 'cl_event'")
            };
            event_list.push(event_core);
        }
        event_list
    }
}

impl<'a> From<EventArray> for EventList {
    #[inline]
    fn from(events: EventArray) -> EventList {
        EventList {
            inner: Inner::Array(events),
        }
    }
}

impl<'a> From<RawEventArray> for EventList {
    #[inline]
    fn from(raw_events: RawEventArray) -> EventList {
        EventList::from(raw_events.as_slice())
    }
}

impl<'a> From<Box<dyn ClWaitListPtr>> for EventList {
    fn from(trait_obj: Box<dyn ClWaitListPtr>) -> EventList {
        let raw_slice = unsafe {
            ::std::slice::from_raw_parts(trait_obj.as_ptr_ptr(), trait_obj.count() as usize)
        };

        Self::from(raw_slice)
    }
}

impl<'a> From<&'a Box<dyn ClWaitListPtr>> for EventList {
    fn from(trait_obj: &Box<dyn ClWaitListPtr>) -> EventList {
        let raw_slice = unsafe {
            ::std::slice::from_raw_parts(trait_obj.as_ptr_ptr(), trait_obj.count() as usize)
        };

        Self::from(raw_slice)
    }
}

impl<'a> From<Ref<'a, dyn ClWaitListPtr>> for EventList {
    fn from(trait_obj: Ref<'a, dyn ClWaitListPtr>) -> EventList {
        let raw_slice = unsafe {
            ::std::slice::from_raw_parts(trait_obj.as_ptr_ptr(), trait_obj.count() as usize)
        };

        Self::from(raw_slice)
    }
}

impl<'a> From<ClWaitListPtrEnum<'a>> for EventList {
    /// Returns an `EventList` containing owned copies of each element in
    /// this `ClWaitListPtrEnum`.
    fn from(wlpe: ClWaitListPtrEnum<'a>) -> EventList {
        match wlpe {
            ClWaitListPtrEnum::Null => EventList::with_capacity(0),
            ClWaitListPtrEnum::RawEventArray(e) => e.as_slice().into(),
            ClWaitListPtrEnum::EventCoreOwned(e) => EventList::from(vec![e.into()]),
            ClWaitListPtrEnum::EventOwned(e) => EventList::from(vec![e]),
            ClWaitListPtrEnum::EventCore(e) => EventList::from(vec![e.clone().into()]),
            ClWaitListPtrEnum::Event(e) => EventList::from(vec![e.clone().into()]),
            ClWaitListPtrEnum::EventList(e) => e.clone(),
            ClWaitListPtrEnum::EventSlice(e) => EventList::from(e),
            ClWaitListPtrEnum::EventPtrSlice(e) => EventList::from(e),
            ClWaitListPtrEnum::RefEventList(e) => (*e).clone(),
            ClWaitListPtrEnum::RefTraitObj(e) => e.into(),
            ClWaitListPtrEnum::BoxTraitObj(e) => e.into(),
        }
    }
}

impl Deref for EventList {
    type Target = [Event];

    #[inline]
    fn deref(&self) -> &[Event] {
        self.as_slice()
    }
}

impl DerefMut for EventList {
    #[inline]
    fn deref_mut(&mut self) -> &mut [Event] {
        self.as_mut_slice()
    }
}

impl IntoIterator for EventList {
    type Item = Event;
    type IntoIter = ::std::vec::IntoIter<Event>;

    // * TODO: Currently converts a contained array to a vec. Will need
    //   something better eventually (perhaps wait for impl Trait to
    //   stabilize).
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        match self.inner {
            Inner::Array(a) => {
                let mut el = EventList {
                    inner: Inner::Array(a),
                };
                el.convert();
                el.into_iter()
            }
            Inner::Vec(v) => v.into_iter(),
        }
    }
}

impl Future for EventList {
    type Item = ();
    type Error = OclError;

    /// Polls each event from this list.
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        poll_events(self.as_slice())
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut EventList {
    #[inline]
    fn alloc_new(&mut self) -> *mut cl_event {
        self._alloc_new()
    }

    #[inline]
    unsafe fn clone_from<E: AsRef<EventCore>>(&mut self, ev: E) {
        assert!(ev.as_ref().is_valid());
        *self._alloc_new() = ev.as_ref().clone().into_raw()
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

unsafe impl<'a> ClWaitListPtr for &'a EventList {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }
    #[inline]
    fn count(&self) -> u32 {
        self._count()
    }
}

/// A stack allocated array of `cl_event` pointers with a maximum length of 8.
///
/// Do not store this list beyond the enqueue call(s) with which this list is
/// first used. Use `EventList` or `EventArray` for a persistent list.
///
//
// * [NOTE]: Consider implementing using
//   `https://github.com/servo/rust-smallvec` instead.
//
#[derive(Debug, Clone)]
pub struct RawEventArray {
    list: [cl_event; 8],
    len: usize,
}

impl RawEventArray {
    pub unsafe fn null() -> RawEventArray {
        RawEventArray {
            list: [0 as cl_event; 8],
            len: 0,
        }
    }

    #[inline]
    pub fn push<E>(&mut self, e: E)
    where
        E: AsRef<EventCore>,
    {
        if (self.len) < self.list.len() {
            self.list[(self.len)] = unsafe { *(e.as_ref().as_ptr_ref()) };
            self.len += 1;
        } else {
            panic!("RawEventArray::push: List is full.");
        }
    }

    #[inline]
    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        if self.len > 0 {
            &self.list as *const _ as *const cl_event
        } else {
            ptr::null()
        }
    }

    /// Enqueue a marker event representing the completion of each and every
    /// event in this list.
    pub fn to_marker(&self, queue: &Queue) -> OclResult<Option<Event>> {
        if self.list.is_empty() {
            return Ok(None);
        }
        queue
            .enqueue_marker(Some(self))
            .map(Some)
            .map_err(OclError::from)
    }

    /// Enqueue a marker event representing the completion of each and every
    /// event in this list.
    pub fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        self.to_marker(queue)
    }

    /// Returns a slice of events in the list.
    pub fn as_slice(&self) -> &[cl_event] {
        &self.list[..self.len]
    }

    /// Returns a mutable slice of events in the list.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [cl_event] {
        &mut self.list[..self.len]
    }

    /// The number of events in this list.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'e, E> From<&'e [E]> for RawEventArray
where
    E: Borrow<Event>,
{
    fn from(events: &[E]) -> RawEventArray {
        let mut list = unsafe { RawEventArray::null() };
        for e in events {
            list.push(e.borrow());
        }
        list
    }
}

impl<'e, E> From<&'e [Option<E>]> for RawEventArray
where
    E: Borrow<Event>,
{
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

impl<'e, 'f, E> From<&'e [&'f Option<E>]> for RawEventArray
where
    'e: 'f,
    E: Borrow<Event>,
{
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
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }
    #[inline]
    fn count(&self) -> u32 {
        self.len as u32
    }
}

unsafe impl<'a> ClWaitListPtr for &'a RawEventArray {
    #[inline]
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self._as_ptr_ptr()
    }
    #[inline]
    fn count(&self) -> u32 {
        self.len as u32
    }
}

/// Conversion to a 'marker' event.
pub trait IntoMarker {
    fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>>;
}

impl<'s, 'e> IntoMarker for &'s [&'e Event]
where
    'e: 's,
{
    fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        RawEventArray::from(self).into_marker(queue)
    }
}

impl<'s, 'e> IntoMarker for &'s [Option<&'e Event>]
where
    'e: 's,
{
    fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        RawEventArray::from(self).into_marker(queue)
    }
}

impl<'s, 'o, 'e> IntoMarker for &'s [&'o Option<&'e Event>]
where
    'e: 's + 'o,
    'o: 's,
{
    fn into_marker(self, queue: &Queue) -> OclResult<Option<Event>> {
        RawEventArray::from(self).into_marker(queue)
    }
}

macro_rules! impl_marker_arrays {
    ($( $len:expr ),*) => ($(
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
    fn into_raw_array(self) -> RawEventArray;
}

impl<'s, 'e> IntoRawEventArray for &'s [&'e Event]
where
    'e: 's,
{
    fn into_raw_array(self) -> RawEventArray {
        RawEventArray::from(self)
    }
}

impl<'s, 'e> IntoRawEventArray for &'s [Option<&'e Event>]
where
    'e: 's,
{
    fn into_raw_array(self) -> RawEventArray {
        RawEventArray::from(self)
    }
}

impl<'s, 'o, 'e> IntoRawEventArray for &'s [&'o Option<&'e Event>]
where
    'e: 's + 'o,
    'o: 's,
{
    fn into_raw_array(self) -> RawEventArray {
        RawEventArray::from(self)
    }
}

macro_rules! impl_raw_list_arrays {
    ($( $len:expr ),*) => ($(
        impl<'s, 'e> IntoRawEventArray  for [&'e Event; $len] where 'e: 's {
            fn into_raw_array(self) -> RawEventArray {
                RawEventArray::from(self)
            }
        }

        impl<'s, 'e> IntoRawEventArray  for [Option<&'e Event>; $len] where 'e: 's {
            fn into_raw_array(self) -> RawEventArray {
                RawEventArray::from(self)
            }
        }

        impl<'s, 'o, 'e> IntoRawEventArray  for [&'o Option<&'e Event>; $len] where 'e: 's + 'o, 'o: 's {
            fn into_raw_array(self) -> RawEventArray {
                RawEventArray::from(self)
            }
        }
    )*);
}

impl_raw_list_arrays!(1, 2, 3, 4, 5, 6, 7, 8);
