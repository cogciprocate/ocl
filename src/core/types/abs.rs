//! Abstract data type wrappers.
//!
//! ### Reference
//!
//! The following table describes abstract data types supported by OpenCL
//! (from [SDK]):
//!
//! * cl_platform_id: The ID for a platform.
//! * cl_device_id: The ID for a device.
//! * cl_context: A context.
//! * cl_command_queue: A command queue.
//! * cl_mem: A memory object.
//! * cl_program: A program.
//! * cl_kernel: A kernel.
//! * cl_event: An event.
//! * cl_sampler: A sampler.
//!
//! The following new derived wrappers are also included in this module:
//!
//! * [cl_event]: A list of events.
//!
//!
//! ### Who cares. Why bother?
//!
//! These types ensure as best they can that stored pointers to any of the
//! above objects will be valid until that pointer is dropped by the Rust
//! runtime (which obviously is not a 100% guarantee).
//!
//! What this means is that you can share, clone, store, and throw away these
//! types, and any types that contain them, among multiple threads, for as
//! long as you'd like, with an insignificant amount of overhead, without
//! having to worry about the dangers of dereferencing those types later on.
//! As good as the OpenCL library generally is about this, it fails in many
//! cases to provide complete protection against segfaults due to
//! dereferencing old pointers particularly on certain *ahem* platforms.
//!
//!
//!
//!
//!
//! [SDK]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/abstractDataTypes.html

use std::mem;
use std::ptr;
use std::fmt::Debug;
use std::marker::Sized;
use libc;
use cl_h::{cl_platform_id, cl_device_id,  cl_context, cl_command_queue, cl_mem, cl_program,
    cl_kernel, cl_event, cl_sampler};
use core::{self, CommandExecutionStatus};
use error::{Result as OclResult, Error as OclError};
use util;

//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// TODO: Evaluate optimal parameters:
const EL_INIT_CAPACITY: usize = 64;
const EL_CLEAR_MAX_LEN: usize = 48;
const EL_CLEAR_INTERVAL: i32 = 32;
const EL_CLEAR_AUTO: bool = true;

const DEBUG_PRINT: bool = false;

//=============================================================================
//================================== TRAITS ===================================
//=============================================================================

/// Types with a mutable pointer to a new, null raw event pointer.
pub unsafe trait ClEventPtrNew: Debug {
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event>;
}

/// Types with a reference to a raw event pointer.
///
/// Using references just to be extra paranoid about copying raw pointers.
pub trait ClEventRef<'e> {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event;
}

/// Types with a reference to a raw event array and an associated element
/// count.
///
/// Using references just to be extra paranoid about copying raw pointers.
pub unsafe trait ClWaitList: Debug {
    unsafe fn as_ptr_ptr(&self) -> *const cl_event;
    fn count (&self) -> u32;
}

impl<'e, L> ClEventRef<'e> for &'e L where L: ClEventRef<'e> {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        (*self).as_ptr_ref()
    }
}


/// Types with a reference to a raw platform_id pointer.
pub unsafe trait ClPlatformIdPtr: Sized {
    unsafe fn as_ptr(&self) -> cl_platform_id {
        debug_assert!(mem::size_of_val(self) == mem::size_of::<PlatformId>());
        // mem::transmute_copy()
        let core = self as *const Self as *const _ as *const PlatformId;
        (*core).as_ptr()
    }
}

/// Types with a reference to a raw device_id pointer.
pub unsafe trait ClDeviceIdPtr: Sized {
    unsafe fn as_ptr(&self) -> cl_device_id {
        debug_assert!(mem::size_of_val(self) == mem::size_of::<DeviceId>());
        // mem::transmute_copy(self)
        let core = self as *const Self as *const _ as *const DeviceId;
        (*core).as_ptr()
    }
}

//=============================================================================
//=================================== TYPES ===================================
//=============================================================================

/// Wrapper used by `EventList` to send event pointers to core functions
/// cheaply.
pub struct EventRefWrapper<'e>(&'e cl_event, u32);

impl<'e> ClEventRef<'e> for EventRefWrapper<'e> {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        self.0
    }
}




/// cl_platform_id
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PlatformId(cl_platform_id);

impl PlatformId {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_platform_id) -> PlatformId {
        PlatformId(ptr)
    }

    pub unsafe fn null() -> PlatformId {
        PlatformId(0 as *mut libc::c_void)
    }

    /// Returns a pointer.
    pub unsafe fn as_ptr(&self) -> cl_platform_id {
        self.0
    }
}

unsafe impl ClPlatformIdPtr for PlatformId {}
unsafe impl Sync for PlatformId {}
unsafe impl Send for PlatformId {}

impl PartialEq<PlatformId> for PlatformId {
    fn eq(&self, other: &PlatformId) -> bool {
        self.0 == other.0
    }
}



/// cl_device_id
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeviceId(cl_device_id);

impl DeviceId {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_device_id) -> DeviceId {
        DeviceId(ptr)
    }

    pub unsafe fn null() -> DeviceId {
        DeviceId(0 as *mut libc::c_void)
    }

    /// Returns a pointer.
    pub unsafe fn as_ptr(&self) -> cl_device_id {
        self.0
    }
}

unsafe impl ClDeviceIdPtr for DeviceId {}
unsafe impl Sync for DeviceId {}
unsafe impl Send for DeviceId {}

impl PartialEq<DeviceId> for DeviceId {
    fn eq(&self, other: &DeviceId) -> bool {
        self.0 == other.0
    }
}



/// cl_context
#[derive(Debug)]
pub struct Context(cl_context);

impl Context {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_context) -> Context {
        Context(ptr)
    }

    /// Only call this when passing a copied pointer such as from an
    /// `clGet*****Info` function.
    pub unsafe fn from_copied_ptr(ptr: cl_command_queue) -> Context {
        let copy = Context(ptr);
        core::retain_context(&copy).unwrap();
        copy
    }

    /// Returns a pointer, do not store it.
    pub unsafe fn as_ptr(&self) -> cl_context {
        self.0
    }
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}

impl Clone for Context {
    fn clone(&self) -> Context {
        unsafe { core::retain_context(self).unwrap(); }
        Context(self.0)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { core::release_context(self).ok(); }
    }
}

impl PartialEq<Context> for Context {
    fn eq(&self, other: &Context) -> bool {
        self.0 == other.0
    }
}


/// cl_command_queue
#[derive(Debug)]
pub struct CommandQueue(cl_command_queue);

impl CommandQueue {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_command_queue) -> CommandQueue {
        CommandQueue(ptr)
    }

    /// Only call this when passing a copied pointer such as from an
    /// `clGet*****Info` function.
    pub unsafe fn from_copied_ptr(ptr: cl_command_queue) -> CommandQueue {
        let copy = CommandQueue(ptr);
        core::retain_command_queue(&copy).unwrap();
        copy
    }

    /// Returns a pointer, do not store it.
    pub unsafe fn as_ptr(&self) -> cl_command_queue {
        self.0
    }
}

impl Clone for CommandQueue {
    fn clone(&self) -> CommandQueue {
        unsafe { core::retain_command_queue(self).unwrap(); }
        CommandQueue(self.0)
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        unsafe { core::release_command_queue(self).ok(); }
    }
}

impl AsRef<CommandQueue> for CommandQueue {
    fn as_ref(&self) -> &CommandQueue {
        self
    }
}

unsafe impl Sync for CommandQueue {}
unsafe impl Send for CommandQueue {}



/// cl_mem
#[derive(Debug)]
pub struct Mem(cl_mem);

impl Mem {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_mem) -> Mem {
        Mem(ptr)
    }

    // pub unsafe fn null() -> Mem {
    //  Mem(0 as *mut libc::c_void, PhantomData)
    // }

    /// Returns a pointer, do not store it.
    pub unsafe fn as_ptr(&self) -> cl_mem {
        self.0
    }
}

impl Clone for Mem {
    fn clone(&self) -> Mem {
        unsafe { core::retain_mem_object(self).unwrap(); }
        Mem(self.0)
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        unsafe { core::release_mem_object(self).ok(); }
    }
}

unsafe impl Sync for Mem {}
unsafe impl Send for Mem {}



/// cl_program
#[derive(Debug)]
pub struct Program(cl_program);

impl Program {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_program) -> Program {
        Program(ptr)
    }

    /// Returns a pointer, do not store it.
    pub unsafe fn as_ptr(&self) -> cl_program {
        self.0
    }
}

impl Clone for Program {
    fn clone(&self) -> Program {
        unsafe { core::retain_program(self).unwrap(); }
        Program(self.0)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { core::release_program(self).ok(); }
    }
}

unsafe impl Sync for Program {}
unsafe impl Send for Program {}

// impl Drop for Program {
//     fn drop(&mut self) {
//         // println!("DROPPING PROGRAM");
//         unsafe { core::release_program(self.obj_core).unwrap(); }
//     }
// }


/// cl_kernel
///
/// ### Thread Safety
///
/// Not thread safe: do not implement `Send` or `Sync`.
///
/// It's possible to do with some work but it's not worth the bother, just
/// make another identical kernel in the other thread and call it good.
///
///
#[derive(Debug)]
pub struct Kernel(cl_kernel);

impl Kernel {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_kernel) -> Kernel {
        Kernel(ptr)
    }

    /// Returns a pointer, do not store it.
    pub unsafe fn as_ptr(&self) -> cl_kernel {
        self.0
    }
}

impl Clone for Kernel {
    fn clone(&self) -> Kernel {
        unsafe { core::retain_kernel(self).unwrap(); }
        Kernel(self.0)
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { core::release_kernel(self).ok(); }
    }
}



/// cl_event
#[derive(Debug)]
pub struct Event(cl_event);

impl Event {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_event) -> Event {
        Event(ptr)
    }

    /// Only use when cloning from a pre-existing and valid `cl_event`.
    pub unsafe fn from_cloned_ptr(ptr: cl_event) -> OclResult<Event> {
        let new_core = Event(ptr);

        if new_core.is_valid() {
            try!(core::retain_event(&new_core));
            Ok(new_core)
        } else {
            OclError::err("core::Event::from_cloned_ptr: Invalid pointer `ptr`.")
        }
    }

    /// For passage directly to an 'event creation' function (such as enqueue).
    pub unsafe fn null() -> Event {
        Event(0 as cl_event)
    }

    // /// Returns a pointer, do not store it unless you will manage its
    // /// associated reference count carefully (as does `EventList`).
    // pub unsafe fn as_ptr(&self) -> cl_event {
    //  self.0
    // }

    /// Returns an immutable reference to a pointer, do not deref and store it unless
    /// you will manage its associated reference count carefully.
    pub unsafe fn as_ptr_ref(&self) -> &cl_event {
        &self.0
    }

    /// Returns a mutable reference to a pointer, do not deref then modify or store it
    /// unless you will manage its associated reference count carefully.
    pub unsafe fn as_ptr_mut(&mut self) -> &mut cl_event {
        &mut self.0
    }

    /// [FIXME]: ADD VALIDITY CHECK BY CALLING '_INFO' OR SOMETHING:
    /// NULL CHECK IS NOT ENOUGH
    pub fn is_valid(&self) -> bool {
        !self.0.is_null()
    }
}

unsafe impl ClEventPtrNew for Event {
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
        if self.0.is_null() {
            Ok(&mut self.0)
        } else {
            unsafe { try!(core::release_event(self)); }
            Ok(&mut self.0)
        }
    }
}

impl<'e> ClEventRef<'e> for Event {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        &self.0
    }
}

unsafe impl ClWaitList for Event {
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        if self.0.is_null() { 0 as *const cl_event } else { &self.0 as *const cl_event }
    }

    fn count(&self) -> u32 {
        if self.0.is_null() { 0 } else { 1 }
    }
}

impl Clone for Event {
    fn clone(&self) -> Event {
        unsafe { core::retain_event(self).expect("core::Event::clone"); }
        Event(self.0)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.is_valid() {
            unsafe { core::release_event(self).ok(); }
        }
    }
}

// unsafe impl EventPtr for Event {}
unsafe impl Sync for Event {}
unsafe impl Send for Event {}



/// List of `cl_event`s.
#[derive(Debug)]
pub struct EventList {
    event_ptrs: Vec<cl_event>,
    clear_max_len: usize,
    clear_counter_max: i32,
    clear_auto: bool,
    clear_counter: i32,
}

impl EventList {
    /// Returns a new, empty, `EventList`.
    pub fn new() -> EventList {
        EventList {
            event_ptrs: Vec::with_capacity(EL_INIT_CAPACITY),
            clear_max_len: EL_CLEAR_MAX_LEN,
            clear_counter_max: EL_CLEAR_INTERVAL,
            clear_auto: EL_CLEAR_AUTO,
            clear_counter: 0,
        }
    }

    /// Pushes a new event onto the list.
    ///
    /// Technically, copies `event`'s contained pointer (a `cl_event`) then
    /// `mem::forget`s it. This seems preferrable to incrementing the reference
    /// count (with `core::retain_event`) then letting `event` drop which just decrements it right back.
    pub fn push(&mut self, event: Event) {
        assert!(event.is_valid());

        unsafe {
            self.event_ptrs.push((*event.as_ptr_ref()));
            mem::forget(event);
        }
        self.decr_counter();
    }

    /// Appends a new null element to the end of the list and returns a reference to it.
    pub fn allot(&mut self) -> &mut cl_event {
        self.event_ptrs.push(0 as cl_event);
        self.event_ptrs.last_mut().unwrap()
    }

    pub fn len(&self) -> usize {
        self.event_ptrs.len()
    }

    pub fn count(&self) -> u32 {
        self.event_ptrs.len() as u32
    }

    /// Returns an immutable reference to a pointer, do not deref and store it unless
    /// you will manage its associated reference count carefully.
    pub unsafe fn as_ptr_ref(&self) -> &cl_event {
        self.event_ptrs.first().expect("ocl::core::EventList::as_ptr_ref(): \
            Attempted to take a reference to the first element of an empty list.")
    }

    /// Clones an event by index.
    pub fn get_clone(&self, index: usize) -> Option<OclResult<Event>> {
        self.event_ptrs.get(index).map(|ptr| unsafe { Event::from_cloned_ptr(*ptr) } )
    }

    /// Clones the last event.
    pub fn last_clone(&self) -> Option<OclResult<Event>> {
        self.event_ptrs.last().map(|ptr| unsafe { Event::from_cloned_ptr(*ptr) } )
    }

    /// Clears each completed event from the list.
    ///
    /// TODO: TEST THIS
    pub fn clear_completed(&mut self) -> OclResult<()> {
        if self.len() < 16 { return Ok(()) }

        let mut cmpltd_events: Vec<usize> = Vec::with_capacity(EL_CLEAR_MAX_LEN);
        let mut idx = 0;

        for event_ptr in self.event_ptrs.iter() {
            let status = try!(core::get_event_status(&EventRefWrapper(event_ptr, 1)));

            if status == CommandExecutionStatus::Complete {
                cmpltd_events.push(idx)
            }

            idx += 1;
        }

        // Release completed events:
        for &idx in cmpltd_events.iter() {
            unsafe {
                try!(core::release_event(&EventRefWrapper(&self.event_ptrs[idx], 1)));
            }
        }

        try!(util::vec_remove_rebuild(&mut self.event_ptrs, &cmpltd_events[..], 2));

        if EL_CLEAR_AUTO {
            self.clear_counter = EL_CLEAR_INTERVAL;
        }

        Ok(())
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

    /// Counts down the auto-list-clear counter.
    fn decr_counter(&mut self) {
        if EL_CLEAR_AUTO {
            self.clear_counter -= 1;

            if self.clear_counter <= 0 && self.event_ptrs.len() > EL_CLEAR_MAX_LEN {
                // self.clear_completed();
                unimplemented!();
            }
        }
    }
}

unsafe impl ClEventPtrNew for EventList {
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
        Ok(self.allot())
    }
}

unsafe impl ClWaitList for EventList {
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        match self.event_ptrs.first() {
            Some(ele) => ele as *const cl_event,
            None => ptr::null(),
        }
    }

    fn count(&self) -> u32 {
        self.event_ptrs.len() as u32
    }
}

impl Clone for EventList {
    /// Clones this list in a thread safe manner.
    fn clone(&self) -> EventList {
        for event_ptr in self.event_ptrs.iter() {
            if !(*event_ptr).is_null() {
                unsafe { core::retain_event(&EventRefWrapper(event_ptr, 1))
                    .expect("core::EventList::clone") }
            }
        }

        EventList {
            event_ptrs: self.event_ptrs.clone(),
            clear_max_len: self.clear_max_len,
            clear_counter_max: self.clear_counter_max,
            clear_auto: self.clear_auto,
            clear_counter: self.clear_counter,
        }
    }
}

impl Drop for EventList {
    fn drop(&mut self) {
        if DEBUG_PRINT { print!("Dropping events... "); }
        for event_ptr in self.event_ptrs.iter() {
            unsafe { core::release_event(&EventRefWrapper(event_ptr, 1)).ok(); }
            if DEBUG_PRINT { print!("{{.}}"); }
        }
        if DEBUG_PRINT { print!("\n"); }
    }
}

impl AsRef<EventList> for EventList {
    fn as_ref(&self) -> &EventList {
        self
    }
}

unsafe impl Sync for EventList {}
unsafe impl Send for EventList {}



/// cl_sampler
#[derive(Debug)]
pub struct Sampler(cl_sampler);

impl Sampler {
    /// Only call this when passing a newly created pointer directly from
    /// `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_fresh_ptr(ptr: cl_sampler) -> Sampler {
        Sampler(ptr)
    }

    /// Returns a pointer, do not store it.
    pub unsafe fn as_ptr(&self) -> cl_sampler {
        self.0
    }
}

impl Clone for Sampler {
    fn clone(&self) -> Sampler {
        unsafe { core::retain_sampler(self).unwrap(); }
        Sampler(self.0)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { core::release_sampler(self).ok(); }
    }
}

unsafe impl Sync for Sampler {}
unsafe impl Send for Sampler {}
