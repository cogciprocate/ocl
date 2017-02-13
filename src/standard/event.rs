//! An `OpenCL` event.

use std;
// use std::ops::{Deref, DerefMut};
use std::convert::Into;
use libc::c_void;
use futures::{Future, Poll, /*Async*/};
use ffi::cl_event;
use core::error::{Error as OclError, Result as OclResult};
use core::{self, Event as EventCore, UserEvent as UserEventCore,
    EventInfo, EventInfoResult, ProfilingInfo, ProfilingInfoResult, ClNullEventPtr, ClWaitListPtr,
    EventList as EventListCore, CommandExecutionStatus, EventCallbackFn, EventVariant, EventVariantRef,
    EventVariantMut, ClEventRef};


// #[derive(Clone, Debug)]
// pub enum EventCoreKind {
//     Event(EventCore),
//     UserEvent(UserEventCore),
//     Null,
// }

// #[derive(Clone, Debug)]
// pub enum EventCoreKindRef<'a> {
//     Event(&'a EventCore),
//     UserEvent(&'a UserEventCore),
//     Null,
// }

// #[derive(Debug)]
// pub enum EventCoreKindMut<'a> {
//     Event(&'a mut EventCore),
//     UserEvent(&'a mut UserEventCore),
//     Null,
// }


/// An event representing a command or user created event.
///
#[derive(Clone, Debug)]
// pub struct Event(Option<EventCore>);
// pub struct Event(EventCoreKind);
pub enum Event {
    Event(EventCore),
    UserEvent(UserEventCore),
    Empty,
}

impl Event {
    /// Creates a new, empty event which must be filled by a newly initiated
    /// command, associating the event with it.
    pub fn empty() -> Event {
        // Event(None)
        Event::Empty
    }

    /// Creates a new `Event` from a `EventCore`.
    ///
    /// ## Safety
    ///
    /// Not meant to be called directly.
    pub unsafe fn from_core_unchecked(event_core: EventCore) -> Event {
        // Event(Some(event_core))
        Event::Event(event_core)
    }

    /// Returns true if this event is complete, false if it is not complete or
    /// if this event is not yet associated with a command.
    ///
    /// This is the fastest possible way to determine the event completion
    /// status.
    ///
    #[inline]
    pub fn is_complete(&self) -> OclResult<bool> {
        // match *self {
        //     Some(ref core) => core.is_complete(),
        //     None => Ok(false),
        // }

        match *self {
            Event::Event(ref core) => core.is_complete(),
            Event::UserEvent(ref core) => core.is_complete(),
            Event::Empty => Err(().into()),
        }
    }

    /// Waits for event to complete (blocks) before returning.
    ///
    /// Similar in function to `Queue::finish()`.
    ///
    pub fn wait(&self) -> OclResult<()> {
        // assert!(!self.is_empty(), "ocl::Event::wait(): {}", self.err_empty());
        // core::wait_for_event(self.0.as_ref().unwrap())
        match *self {
            Event::Event(ref core) => core::wait_for_event(core),
            Event::UserEvent(ref core) => core::wait_for_event(core),
            Event::Empty => Err(format!("ocl::Event::wait(): {}", self.err_empty()).into()),
        }
    }

    /// Returns true if this event is 'empty' and has not yet been associated
    /// with a command.
    ///
    #[inline]
    pub fn is_empty(&self) -> bool {
        match *self {
            Event::Event(_) => false,
            Event::UserEvent(_) => false,
            Event::Empty => true,
        }
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
        match *self {
            Event::Event(ref core) => core::get_event_info(core, info_kind),
            Event::UserEvent(ref core) => core::get_event_info(core, info_kind),
            Event::Empty => EventInfoResult::Error(Box::new(self.err_empty())),
        }
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
        match *self {
            Event::Event(ref core) => core::get_event_profiling_info(core, info_kind),
            Event::UserEvent(ref core) => core::get_event_profiling_info(core, info_kind),
            Event::Empty => ProfilingInfoResult::Error(Box::new(self.err_empty())),
        }
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    ///
    #[inline]
    pub fn core(&self) -> EventVariantRef {
        // self.0.as_ref()
        // match *self {
        //     Core::Event(ref core) => core::wait_for_event(core),
        //     Core::UserEvent(ref core) => core::wait_for_event(core),
        //     Core::Null => Err(format!("ocl::Event::wait(): {}", self.err_empty()).into()),
        // }
        match *self {
            Event::Event(ref core) => EventVariantRef::Event(core),
            Event::UserEvent(ref core) => EventVariantRef::UserEvent(core),
            Event::Empty => EventVariantRef::Null,
        }
    }

    /// Returns a mutable reference to the core pointer wrapper usable by
    /// functions in the `core` module.
    ///
    #[inline]
    pub fn core_mut(&mut self) -> EventVariantMut {
        // self.0.as_mut()
        match *self {
            Event::Event(ref mut core) => EventVariantMut::Event(core),
            Event::UserEvent(ref mut core) => EventVariantMut::UserEvent(core),
            Event::Empty => EventVariantMut::Null,
        }
    }

    /// Sets a callback function, `callback_receiver`, to trigger upon
    /// completion of this event passing an optional reference to user data.
    ///
    /// # Safety
    ///
    /// `user_data` must be guaranteed to still exist if and when `callback_receiver`
    /// is called.
    ///
    pub unsafe fn set_callback(&self, receiver: Option<EventCallbackFn>,
            user_data_ptr: *mut c_void) -> OclResult<()>
    {
        match *self {
            Event::Event(ref core) => core.set_callback(receiver, user_data_ptr),
            Event::UserEvent(ref core) => core.set_callback(receiver, user_data_ptr),
            Event::Empty => Err("ocl::Event::set_callback: This event is uninitialized (null).".into()),
        }
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

    fn _alloc_new(&mut self) -> *mut cl_event {
        // assert!(self.is_empty(), "ocl::Event: Attempting to use a non-empty event as a new event
        //     is not allowed. Please create a new, empty, event with ocl::Event::empty().");
        if self.is_empty() {
            unsafe {
                *self = Event::Event(EventCore::from_raw_create_ptr(0 as *mut c_void));

                if let Event::Event(ref mut ev) = *self {
                    ev.as_ptr_mut()
                } else {
                    unreachable!();
                }
            }
        } else {
            panic!("ocl::Event: Attempting to use a non-empty event as a new event
                is not allowed. Please create a new, empty, event with ocl::Event::empty().");
        }
    }

    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        // match *self {
        //     Some(ref ec) => ec.as_ptr_ptr(),
        //     None => 0 as *const cl_event,
        // }
        match *self {
            Event::Event(ref core) => core.as_ptr_ptr(),
            Event::UserEvent(ref core) => core.as_ptr_ptr(),
            Event::Empty => panic!("<ocl::Event as ClWaitListPtr>::as_ptr_ptr: Event is null"),
        }
    }

    fn _count(&self) -> u32 {
        match *self {
            Event::Event(ref core) => core.count(),
            Event::UserEvent(ref core) => core.count(),
            Event::Empty => panic!("<ocl::Event as ClWaitListPtr>::as_ptr_ptr: Event is null"),
        }
    }
}

// impl From<NullEventCore> for Event {
//     #[inline]
//     fn from(nev: NullEventCore) -> Event {
//         match nev.validate() {
//             Ok(nev) => Event(Event::Event(nev)),
//             Err(err) => panic!("ocl::Event::from::<NullEventCore>: {}", err),
//         }
//     }
// }

impl From<EventCore> for Event {
    #[inline]
    fn from(ev: EventCore) -> Event {
        if ev.is_valid() {
            Event::Event(ev)
        } else {
            panic!("ocl::Event::from::<EventCore>: Invalid event.");
        }
    }
}

impl From<UserEventCore> for Event {
    #[inline]
    fn from(uev: UserEventCore) -> Event {
        if uev.is_valid() {
            Event::UserEvent(uev)
        } else {
            panic!("ocl::Event::from::<UserEventCore>: Invalid event.");
        }
    }
}

impl From<EventVariant> for Event {
    #[inline]
    fn from(variant: EventVariant) -> Event {
        match variant {
            EventVariant::Null => panic!("ocl::Event::from::<EventVariant>: Invalid event variant (`Null`)."),
            EventVariant::Event(ev) => {
                if ev.is_valid() {
                    Event::Event(ev)
                } else {
                    panic!("ocl::Event::from::<UserEventCore>: Invalid (likely null) event.");
                }
            }
            EventVariant::UserEvent(uev) => {
                if uev.is_valid() {
                    Event::UserEvent(uev)
                } else {
                    panic!("ocl::Event::from::<UserEventCore>: Invalid (likely null) user event.");
                }
            }
        }
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

impl<'e> ClEventRef<'e> for Event {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        match *self {
            Event::Event(ref core) => core.as_ptr_ref(),
            Event::UserEvent(ref core) => core.as_ptr_ref(),
            Event::Empty => panic!("<ocl::Event as ClWaitListPtr>::as_ptr_ptr: Event is null"),
        }
    }
}

// impl AsRef<EventCore> for Event {
//     fn as_ref(&self) -> &EventCore {
//         self.0.as_ref().ok_or(self.err_empty()).expect("ocl::Event::as_ref()")
//     }
// }

// impl Deref for Event {
//     type Target = EventCore;

//     fn deref(&self) -> &EventCore {
//         // self.0.as_ref().ok_or(self.err_empty()).expect("ocl::Event::deref()")

//         match *self {
//             Event::Event(ref core) => core,
//             Event::UserEvent(_) => panic!("ocl::Event::deref::<EventCore>: Event is a user event."),
//             Event::Empty => panic!("ocl::Event::deref::<EventCore>: Event is null."),
//         }
//     }
// }

// impl DerefMut for Event {
//     fn deref_mut(&mut self) -> &mut EventCore {
//         assert!(!self.is_empty(), "ocl::Event::deref_mut(): {}", self.err_empty());
//         self.0.as_mut().unwrap()
//     }
// }

unsafe impl<'a> ClNullEventPtr for &'a mut Event {
    #[inline] fn alloc_new(self) -> *mut cl_event { (*self)._alloc_new() }
}

unsafe impl ClWaitListPtr for Event {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self._count() }
}

unsafe impl<'a> ClWaitListPtr for  &'a Event {
    #[inline] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline] fn count(&self) -> u32 { self._count() }
}

impl Future for Event {
    type Item = ();
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        unimplemented!();

        // // Do this right:
        // self.is_complete().map(|_| Async::Ready(()))
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
        match event {
            // Some(ecore) => self.event_list_core.push(core),
            // None => panic!("EventList::push: Unable to push null (empty) event."),

            Event::Event(core) =>  self.event_list_core.push_event(core),
            Event::UserEvent(core) => self.event_list_core.push_user_event(core),
            Event::Empty => panic!("EventList::push: Unable to push null (empty) event."),
        }

    }

    /// Removes the last event from the list and returns it.
    pub fn pop(&mut self) -> Option<Event> {
        self.event_list_core.pop().map(|ev| Event::from(ev))
    }

    /// Returns a new copy of an event by index.
    pub fn get_clone(&self, index: usize) -> Option<Event> {
        match self.event_list_core.get_clone(index) {
            Some(ev_res) => {
                match ev_res {
                    Ok(ev) => unsafe { Some(Event::from_core_unchecked(ev)) },
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
                    Ok(ev) => unsafe { Some(Event::from_core_unchecked(ev)) },
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
    pub unsafe fn last_set_callback<T>(&self,
                callback_receiver: Option<EventCallbackFn>,
                user_data: *mut T,
                ) -> OclResult<()>
    {
        let event_core = self.event_list_core.last_clone().ok_or(
            OclError::string("ocl::EventList::set_callback: This event list is empty."))??;

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
    fn _alloc_new(&mut self) -> *mut cl_event {
        self.event_list_core.allot()
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

// impl Deref for EventList {
//     type Target = EventListCore;

//     #[inline]
//     fn deref(&self) -> &EventListCore {
//         &self.event_list_core
//     }
// }

// impl DerefMut for EventList {
//     #[inline]
//     fn deref_mut(&mut self) -> &mut EventListCore {
//         &mut self.event_list_core
//     }
// }

// unsafe impl ClNullEventPtr for EventList {
//     #[inline]
//     fn alloc_new(&mut self) -> *mut cl_event {
//         self._alloc_new()
//     }
// }

unsafe impl<'a> ClNullEventPtr for &'a mut EventList {
    #[inline]
    fn alloc_new(self) -> *mut cl_event {
        self._alloc_new()
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
