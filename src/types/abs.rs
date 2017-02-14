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
//! * cl_events: A list of events.
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
//! [SDK]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/abstractDataTypes.html

use std::mem;
use std::ptr;
use std::slice;
use std::fmt::Debug;
use std::marker::Sized;
use std::ops::Deref;
// use std::borrow::Borrow;
use libc::c_void;
use ffi::{/*self,*/ cl_platform_id, cl_device_id,  cl_context, cl_command_queue, cl_mem, cl_program,
    cl_kernel, cl_event, cl_sampler};
use ::{CommandExecutionStatus, OpenclVersion, PlatformInfo, DeviceInfo, DeviceInfoResult,
    ContextInfo, ContextInfoResult, CommandQueueInfo, CommandQueueInfoResult, ProgramInfo,
    ProgramInfoResult, KernelInfo, KernelInfoResult, Status, EventCallbackFn, OclPrm};
use error::{Result as OclResult, Error as OclError};
use functions;
use util;

//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// TODO: Evaluate optimal parameters:
const EL_INIT_CAPACITY: usize = 64;
const EL_CLEAR_MAX_LEN: usize = 48;
const EL_CLEAR_INTERVAL: i32 = 32;
const EL_CLEAR_AUTO: bool = false;

// const DEBUG_PRINT: bool = false;

//=============================================================================
//================================== TRAITS ===================================
//=============================================================================

/// `AsRef` with a type being carried along for convenience.
pub trait AsMem<T>
        where T: OclPrm
{
    fn as_mem(&self) -> &Mem;
}

impl<'a, T, M> AsMem<T> for &'a M where T: OclPrm, M: AsMem<T> {
    fn as_mem(&self) -> &Mem {
        (*self).as_mem()
    }
}

/// Types which can be passed as the primary (`ptr`) argument value to
/// `::enqueue_read_buffer`, `::enqueue_write_buffer`,
/// `::enqueue_read_buffer_rect`, `::enqueue_write_buffer_rect`,
/// `::enqueue_read_image`, or `::enqueue_write_image`.
///
/// These may be device or host side memory buffers.
///
/// Types returned from `::enqueue_map_...` and all of their derivatives as
/// well as types created with `::create_buffer` and `::create_image` all
/// implement this trait.
///
pub unsafe trait MemCmdRw {}

/// Types which can be passed to any and all `::enqueue_...` functions as the
/// primary (`ptr`) argument and can also be passed as kernel `cl_mem` arguments.
///
/// These are strictly device side memory buffers.
///
/// Types created with `::create_buffer` and `::create_image` implement this
/// trait.
pub unsafe trait MemCmdAll {}


/// Types with a fixed set of associated devices and an associated platform.
pub trait ClVersions {
    fn device_versions(&self) -> OclResult<Vec<OpenclVersion>>;
    fn platform_version(&self) -> OclResult<OpenclVersion>;

    fn verify_device_versions(&self, required_version: [u16; 2]) -> OclResult<()> {
        functions::verify_versions(&try!(self.device_versions()), required_version)
    }

    fn verify_platform_version(&self, required_version: [u16; 2]) -> OclResult<()> {
        let ver = [try!(self.platform_version())];
        functions::verify_versions(&ver, required_version)
    }
}


/// Types with a reference to a raw event pointer.
///
pub trait ClEventRef<'e> {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event;
}

impl<'e, L> ClEventRef<'e> for &'e L where L: ClEventRef<'e> {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        (*self).as_ptr_ref()
    }
}



/// Types with a mutable pointer to a new, null raw event pointer.
///
pub unsafe trait ClNullEventPtr: Debug {
    fn alloc_new(&mut self) -> *mut cl_event;
}

unsafe impl ClNullEventPtr for () {
    fn alloc_new(&mut self) -> *mut cl_event {
        ptr::null_mut() as *mut _ as *mut cl_event
    }
}



/// Types with a reference to a raw event array and an associated element
/// count.
///
/// [TODO]: Create an enum to be used with this trait.
///
pub unsafe trait ClWaitListPtr: Debug {
    /// Returns a pointer to the first pointer in this list.
    unsafe fn as_ptr_ptr(&self) -> *const cl_event;
    /// Returns the number of items in this wait list.
    fn count (&self) -> u32;
}

unsafe impl<'a> ClWaitListPtr for &'a [cl_event] {
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self.as_ptr()
    }

    fn count (&self) -> u32 {
        self.len() as u32
    }
}

unsafe impl<'a> ClWaitListPtr for () {
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        ptr::null() as *const _ as *const cl_event
    }

    fn count (&self) -> u32 {
        0 as u32
    }
}


/// Types with a reference to a raw platform_id pointer.
pub unsafe trait ClPlatformIdPtr: Sized + Debug {
    unsafe fn as_ptr(&self) -> cl_platform_id {
        debug_assert!(mem::size_of_val(self) == mem::size_of::<PlatformId>());
        // mem::transmute_copy()
        let core = self as *const Self as *const _ as *const PlatformId;
        (*core).as_ptr()
    }
}

unsafe impl ClPlatformIdPtr for () {
    unsafe fn as_ptr(&self) -> cl_platform_id {
        ptr::null_mut() as *mut _ as cl_platform_id
    }
}


/// Types with a reference to a raw device_id pointer.
pub unsafe trait ClDeviceIdPtr: Sized + Debug {
    unsafe fn as_ptr(&self) -> cl_device_id {
        debug_assert!(mem::size_of_val(self) == mem::size_of::<DeviceId>());
        // mem::transmute_copy(self)
        let core = self as *const Self as *const _ as *const DeviceId;
        (*core).as_ptr()
    }
}

unsafe impl ClDeviceIdPtr for () {
    unsafe fn as_ptr(&self) -> cl_device_id {
        ptr::null_mut() as *mut _ as cl_device_id
    }
}

//=============================================================================
//=================================== TYPES ===================================
//=============================================================================

/// Wrapper used by `EventList` to send event pointers to core functions
/// cheaply.
pub struct EventRefWrapper(cl_event);

impl EventRefWrapper {
    pub unsafe fn new(ptr: cl_event) -> EventRefWrapper {
        EventRefWrapper(ptr)
    }
}

impl<'e> ClEventRef<'e> for EventRefWrapper {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        &self.0
    }
}


/// cl_platform_id
#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, Eq)]
pub struct PlatformId(cl_platform_id);

impl PlatformId {
    /// Creates a new `PlatformId` wrapper from a raw pointer.
    pub unsafe fn from_raw(ptr: cl_platform_id) -> PlatformId {
        // assert!(!ptr.is_null(), "Null pointer passed.");
        PlatformId(ptr)
    }

    /// Returns an invalid `PlatformId` used for initializing data structures
    /// meant to be filled with valid ones.
    pub unsafe fn null() -> PlatformId {
        PlatformId(0 as *mut c_void)
    }

    /// Returns a pointer.
    pub fn as_ptr(&self) -> cl_platform_id {
        self.0
    }

    /// Returns the queried and parsed OpenCL version for this platform.
    pub fn version(&self) -> OclResult<OpenclVersion> {
        if !self.0.is_null() {
            functions::get_platform_info(self, PlatformInfo::Version).as_opencl_version()
        } else {
            OclError::err_string("PlatformId::version(): This platform_id is invalid.")
        }
    }
}

unsafe impl ClPlatformIdPtr for PlatformId {}
unsafe impl<'a> ClPlatformIdPtr for &'a PlatformId {}
unsafe impl Sync for PlatformId {}
unsafe impl Send for PlatformId {}

impl PartialEq<PlatformId> for PlatformId {
    fn eq(&self, other: &PlatformId) -> bool {
        self.0 == other.0
    }
}

impl ClVersions for PlatformId {
    fn device_versions(&self) -> OclResult<Vec<OpenclVersion>> {
        let devices = try!(functions::get_device_ids(self, Some(::DEVICE_TYPE_ALL), None));
        functions::device_versions(&devices)
    }

    // [FIXME]: TEMPORARY
    fn platform_version(&self) -> OclResult<OpenclVersion> {
        self.version()
    }
}



/// cl_device_id
#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, Eq)]
pub struct DeviceId(cl_device_id);

impl DeviceId {
    /// Creates a new `DeviceId` wrapper from a raw pointer.
    pub unsafe fn from_raw(ptr: cl_device_id) -> DeviceId {
        // assert!(!ptr.is_null(), "Null pointer passed.");
        DeviceId(ptr)
    }

    /// Returns an invalid `DeviceId` used for initializing data structures
    /// meant to be filled with valid ones.
    pub unsafe fn null() -> DeviceId {
        DeviceId(0 as *mut c_void)
    }

    /// Returns a pointer.
    pub fn as_ptr(&self) -> cl_device_id {
        self.0
    }

    /// Returns the queried and parsed OpenCL version for this device.
    pub fn version(&self) -> OclResult<OpenclVersion> {
        if !self.0.is_null() {
            functions::get_device_info(self, DeviceInfo::Version).as_opencl_version()
        } else {
            OclError::err_string("DeviceId::device_versions(): This device_id is invalid.")
        }
    }
}

unsafe impl ClDeviceIdPtr for DeviceId {}
unsafe impl<'a> ClDeviceIdPtr for &'a DeviceId {}
unsafe impl Sync for DeviceId {}
unsafe impl Send for DeviceId {}

impl PartialEq<DeviceId> for DeviceId {
    fn eq(&self, other: &DeviceId) -> bool {
        self.0 == other.0
    }
}

impl ClVersions for DeviceId {
    fn device_versions(&self) -> OclResult<Vec<OpenclVersion>> {
        self.version().map(|dv| vec![dv])
    }

    fn platform_version(&self) -> OclResult<OpenclVersion> {
        let platform = match functions::get_device_info(self, DeviceInfo::Platform) {
            DeviceInfoResult::Platform(p) => p,
            DeviceInfoResult::Error(e) => return Err(OclError::from(*e)),
            _ => unreachable!(),
        };

        functions::get_platform_info(&platform, PlatformInfo::Version).as_opencl_version()
    }
}



/// cl_context
#[derive(Debug)]
pub struct Context(cl_context);

impl Context {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_context) -> Context {
        // assert!(!ptr.is_null(), "Null pointer passed.");
        Context(ptr)
    }

    /// Only call this when passing a copied pointer such as from an
    /// `clGet*****Info` function.
    pub unsafe fn from_raw_copied_ptr(ptr: cl_context) -> Context {
        assert!(!ptr.is_null(), "Null pointer passed.");
        let copy = Context(ptr);
        functions::retain_context(&copy).unwrap();
        copy
    }

    /// Returns a pointer, do not store it.
    pub fn as_ptr(&self) -> cl_context {
        self.0
    }

    /// Returns the devices associated with this context.
    pub fn devices(&self) -> OclResult<Vec<DeviceId>> {
        match functions::get_context_info(self, ContextInfo::Devices) {
            ContextInfoResult::Devices(ds) => Ok(ds),
            ContextInfoResult::Error(e) => return Err(OclError::from(*e)),
            _ => unreachable!(),
        }
    }
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}

impl Clone for Context {
    fn clone(&self) -> Context {
        unsafe { functions::retain_context(self).unwrap(); }
        Context(self.0)
    }
}

impl Drop for Context {
    /// Panics in the event of an error of type `Error::Status` except when
    /// the status code is `CL_INVALID_CONTEXT` (which is ignored).
    ///
    /// This is done because certain platforms error with `CL_INVALID_CONTEXT`
    /// for unknown reasons and as far as we know can be safely ignored.
    ///
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = functions::release_context(self) {
                if let OclError::Status { ref status, .. } = e {
                    if status == &Status::CL_INVALID_CONTEXT {
                        return;
                    }
                }
                panic!("{:?}", e);
            }
        }
    }
}

impl PartialEq<Context> for Context {
    fn eq(&self, other: &Context) -> bool {
        self.0 == other.0
    }
}

impl ClVersions for Context {
    fn device_versions(&self) -> OclResult<Vec<OpenclVersion>> {
        let devices = try!(self.devices());
        functions::device_versions(&devices)
    }

    fn platform_version(&self) -> OclResult<OpenclVersion> {
        let devices = try!(self.devices());
        devices[0].platform_version()
    }
}


/// cl_command_queue
#[derive(Debug)]
pub struct CommandQueue(cl_command_queue);

impl CommandQueue {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_command_queue) -> CommandQueue {
        // assert!(!ptr.is_null(), "Null pointer passed.");
        CommandQueue(ptr)
    }

    /// Only call this when passing a copied pointer such as from an
    /// `clGet*****Info` function.
    pub unsafe fn from_raw_copied_ptr(ptr: cl_command_queue) -> CommandQueue {
        assert!(!ptr.is_null(), "Null pointer passed.");
        let copy = CommandQueue(ptr);
        functions::retain_command_queue(&copy).unwrap();
        copy
    }

    /// Returns a pointer, do not store it.
    pub fn as_ptr(&self) -> cl_command_queue {
        self.0
    }

    /// Returns the `DeviceId` associated with this command queue.
    pub fn device(&self) -> OclResult<DeviceId> {
        match functions::get_command_queue_info(self, CommandQueueInfo::Device) {
            CommandQueueInfoResult::Device(d) => Ok(d),
            CommandQueueInfoResult::Error(e) => Err(OclError::from(*e)),
            _ => unreachable!(),
        }
    }
}

impl Clone for CommandQueue {
    fn clone(&self) -> CommandQueue {
        unsafe { functions::retain_command_queue(self).unwrap(); }
        CommandQueue(self.0)
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        unsafe { functions::release_command_queue(self).unwrap(); }
    }
}

impl AsRef<CommandQueue> for CommandQueue {
    fn as_ref(&self) -> &CommandQueue {
        self
    }
}

unsafe impl Sync for CommandQueue {}
unsafe impl Send for CommandQueue {}

impl ClVersions for CommandQueue{
    fn device_versions(&self) -> OclResult<Vec<OpenclVersion>> {
        let device = try!(self.device());
        device.version().map(|dv| vec![dv])
    }

    fn platform_version(&self) -> OclResult<OpenclVersion> {
        try!(self.device()).platform_version()
    }
}



/// cl_mem
#[derive(Debug)]
pub struct Mem(cl_mem);

impl Mem {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_mem) -> Mem {
        // Don't bother checking this, sometimes null pointers get passed during an error.
        // assert!(!ptr.is_null(), "Null pointer passed.");
        Mem(ptr)
    }

	/// Only call this when passing a copied pointer such as from an
	/// `clGet*****Info` function.
	pub unsafe fn from_raw_copied_ptr(ptr: cl_mem) -> Mem {
        assert!(!ptr.is_null(), "Null pointer passed.");
		let copy = Mem(ptr);
		functions::retain_mem_object(&copy).unwrap();
		copy
	}

    /// Returns a pointer, do not store it.
    #[inline(always)]
    pub fn as_ptr(&self) -> cl_mem {
        self.0
    }
}

impl Clone for Mem {
    fn clone(&self) -> Mem {
        unsafe { functions::retain_mem_object(self).unwrap(); }
        Mem(self.0)
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        unsafe { functions::release_mem_object(self).unwrap(); }
    }
}

impl<T> AsMem<T> for Mem where T: OclPrm {
    #[inline(always)]
    fn as_mem(&self) -> &Mem {
        self
    }
}


unsafe impl<'a> MemCmdRw for Mem {}
unsafe impl<'a> MemCmdRw for &'a Mem {}
unsafe impl<'a> MemCmdRw for &'a mut Mem {}
unsafe impl<'a> MemCmdAll for Mem {}
unsafe impl<'a> MemCmdAll for &'a Mem {}
unsafe impl<'a> MemCmdAll for &'a mut Mem {}
unsafe impl Sync for Mem {}
unsafe impl Send for Mem {}


pub struct MappedMem<T: OclPrm>(*mut T);

impl<T: OclPrm> MappedMem<T> {
    #[inline(always)]
    pub unsafe fn from_raw(ptr: *mut T) -> MappedMem<T> {
        assert!(!ptr.is_null(), "MappedMem::from_raw: Null pointer passed.");
        MappedMem(ptr)
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *mut T {
        self.0
    }

    #[inline(always)]
    pub fn as_void_ptr(&self) -> *mut c_void {
        self.0 as *mut _ as *mut c_void
    }

    #[inline(always)]
    pub unsafe fn as_slice<'a>(&self, len: usize) -> &'a [T] {
        slice::from_raw_parts(self.0, len)
    }

    #[inline(always)]
    pub unsafe fn as_slice_mut<'a>(&mut self, len: usize) -> &'a mut [T] {
        slice::from_raw_parts_mut(self.0, len)
    }
}

impl<T> AsMem<T> for MappedMem<T> where T: OclPrm {
    #[inline(always)]
    fn as_mem(&self) -> &Mem {
        unsafe { &*(self as *const _ as *const Mem) }
    }
}

unsafe impl<T: OclPrm> MemCmdRw for MappedMem<T> {}
unsafe impl<'a, T: OclPrm> MemCmdRw for &'a MappedMem<T> {}
unsafe impl<'a, T: OclPrm> MemCmdRw for &'a mut MappedMem<T> {}
unsafe impl<T: OclPrm> Send for MappedMem<T> {}
unsafe impl<T: OclPrm> Sync for MappedMem<T> {}



/// cl_program
#[derive(Debug)]
pub struct Program(cl_program);

impl Program {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_program) -> Program {
        assert!(!ptr.is_null(), "Null pointer passed.");
        Program(ptr)
    }

	/// Only call this when passing a copied pointer such as from an
	/// `clGet*****Info` function.
	pub unsafe fn from_raw_copied_ptr(ptr: cl_program) -> Program {
        assert!(!ptr.is_null(), "Null pointer passed.");
		let copy = Program(ptr);
		functions::retain_program(&copy).unwrap();
		copy
	}

	/// Returns a pointer, do not store it.
    #[inline(always)]
	pub fn as_ptr(&self) -> cl_program {
		self.0
	}

    /// Returns the devices associated with this program.
    pub fn devices(&self) -> OclResult<Vec<DeviceId>> {
        match functions::get_program_info(self, ProgramInfo::Devices) {
            ProgramInfoResult::Devices(d) => Ok(d),
            ProgramInfoResult::Error(e) => Err(OclError::from(*e)),
            _ => unreachable!(),
        }
    }
}

impl Clone for Program {
    fn clone(&self) -> Program {
        unsafe { functions::retain_program(self).unwrap(); }
        Program(self.0)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { functions::release_program(self).unwrap(); }
    }
}

unsafe impl Sync for Program {}
unsafe impl Send for Program {}

impl ClVersions for Program {
    fn device_versions(&self) -> OclResult<Vec<OpenclVersion>> {
        let devices = try!(self.devices());
        functions::device_versions(&devices)
    }

    fn platform_version(&self) -> OclResult<OpenclVersion> {
        let devices = try!(self.devices());
        devices[0].platform_version()
    }
}


/// cl_kernel
///
/// ### Thread Safety
///
/// Currently not thread safe: does not implement `Send` or `Sync`. It's
/// probably possible to implement one or both with some work but it's
/// potentially problematic on certain (all?) platforms due to issues while
/// setting arguments. If you need to transfer a kernel you're better off
/// creating another one in the other thread or using some other mechanism
/// such as channels to manipulate kernels in other threads. This issue will
/// be revisited in the future (please provide input by filing an issue if you
/// have any thoughts on the matter).
///
/// [UPDATE]: Enabling `Send` for a while to test.
///
///
#[derive(Debug)]
pub struct Kernel(cl_kernel);

impl Kernel {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_kernel) -> Kernel {
        assert!(!ptr.is_null(), "Null pointer passed.");
        Kernel(ptr)
    }

    /// Returns a pointer, do not store it.
    #[inline(always)]
    pub fn as_ptr(&self) -> cl_kernel {
        self.0
    }

    /// Returns the program associated with this kernel.
    pub fn program(&self) -> OclResult<Program> {
        match functions::get_kernel_info(self, KernelInfo::Program) {
            KernelInfoResult::Program(d) => Ok(d),
            KernelInfoResult::Error(e) => Err(OclError::from(*e)),
            _ => unreachable!(),
        }
    }
}

impl Clone for Kernel {
    fn clone(&self) -> Kernel {
        unsafe { functions::retain_kernel(self).unwrap(); }
        Kernel(self.0)
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { functions::release_kernel(self).unwrap(); }
    }
}

impl ClVersions for Kernel {
    fn device_versions(&self) -> OclResult<Vec<OpenclVersion>> {
        let devices = try!(try!(self.program()).devices());
        functions::device_versions(&devices)
    }

    fn platform_version(&self) -> OclResult<OpenclVersion> {
        let devices = try!(try!(self.program()).devices());
        devices[0].platform_version()
    }
}

unsafe impl Send for Kernel {}



// /// cl_event
// #[derive(Clone, Debug)]
// pub struct NullEvent(cl_event);

// impl NullEvent {
//     /// For passage directly to an 'event creation' function (such as enqueue...).
//     #[inline]
//     pub fn new() -> NullEvent {
//         NullEvent(0 as cl_event)
//     }

//     pub fn validate(&mut self) -> OclResult<Event> {
//         if self.is_valid() {
//             let event = Ok(Event(self.0));
//             mem::forget(self);
//             event
//         } else {
//             Err("NullEvent::validate: Event is not valid.".into())
//         }
//     }

//     /// Returns a mutable reference to a pointer, do not deref then modify or store it
//     /// unless you will manage its associated reference count carefully.
//     ///
//     #[inline]
//     pub unsafe fn as_ptr_mut(&mut self) -> &mut cl_event {
//         &mut self.0
//     }

//     /// [FIXME]: ADD VALIDITY CHECK BY CALLING '_INFO' OR SOMETHING:
//     /// NULL CHECK IS NOT ENOUGH
//     ///
//     /// This still leads to crazy segfaults when non-event pointers (random
//     /// whatever addresses) are passed. Need better check.
//     ///
//     #[inline]
//     pub fn is_valid(&self) -> bool {
//         !self.0.is_null()
//     }

//     fn _alloc_new(&mut self) -> *mut cl_event {
//         assert!(self.0.is_null(), "NullEvent (new event) has been used twice.");
//         &mut self.0
//     }
// }

// impl<'e> ClEventRef<'e> for NullEvent {
//     #[inline(always)] unsafe fn as_ptr_ref(&'e self) -> &'e cl_event { &self.0 }
// }

// unsafe impl<'a> ClNullEventPtr for &'a mut NullEvent {
//     #[inline(always)] fn alloc_new(self) -> *mut cl_event { self._alloc_new() }

//     #[inline(always)] fn validate(self) -> OclResult<Event> {
//         let valid_event = self.validate();
//         self.0 = 0 as cl_event;
//         valid_event
//     }
// }

// unsafe impl Sync for NullEvent {}
// unsafe impl Send for NullEvent {}



/// cl_event
#[derive(Debug)]
pub struct Event(cl_event);

impl Event {
    /// For passage directly to an 'event creation' function (such as enqueue...).
    #[inline]
    pub fn null() -> Event {
        Event(0 as cl_event)
    }

    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    #[inline]
    pub unsafe fn from_raw_create_ptr(ptr: cl_event) -> Event {
        assert!(!ptr.is_null(), "ocl_core::Event::from_raw_create_ptr: Null pointer passed.");
        Event(ptr)
    }

    /// Only use when cloning or copying from a pre-existing and valid
    /// `cl_event`.
    ///
    #[inline]
    pub unsafe fn from_raw_copied_ptr(ptr: cl_event) -> OclResult<Event> {
        assert!(!ptr.is_null(), "ocl_core::Event::from_raw_copied_ptr: Null pointer passed.");
        let new_core = Event(ptr);
        functions::retain_event(&new_core)?;
        Ok(new_core)
    }

    /// Queries the command status associated with this event and returns true
    /// if it is complete, false if incomplete or upon error.
    ///
    /// This is the fastest possible way to determine event status.
    ///
    #[inline]
    pub fn is_complete(&self) -> OclResult<bool> {
        // assert!(!self.0.is_null(), "ocl_core::Event::is_complete: Event is null.");
        if self.0.is_null() { return Err("ocl_core::Event::is_complete: Event is null.".into()) }
        functions::event_is_complete(self)
    }

    /// Sets a callback function, `callback_receiver`, to trigger upon
    /// completion of this event list with an optional pointer to user data.
    ///
    /// # Safety
    ///
    /// `user_data` must be guaranteed to still exist if and when `callback_receiver`
    /// is ever called.
    ///
    /// TODO: Create a safer type wrapper for `callback_receiver` (using an
    /// `Arc`?, etc.) within `ocl`.
    ///
    pub unsafe fn set_callback(&self,
            callback_receiver_opt: Option<EventCallbackFn>,
            user_data_ptr: *mut c_void,
            ) -> OclResult<()>
    {
        if self.is_valid() {
            let callback_receiver = match callback_receiver_opt {
                Some(cbr) => Some(cbr),
                None => Some(::_dummy_event_callback as EventCallbackFn),
            };

            ::set_event_callback(
                self,
                CommandExecutionStatus::Complete,
                callback_receiver,
                user_data_ptr as *mut _ as *mut c_void,
            )

        } else {
            Err("ocl_core::Event::set_callback: This event is null. Cannot set callback until \
                internal event pointer is actually created by a `clCreate...` function.".into())
        }
    }

    /// Returns a pointer, do not store it unless you will manage its
    /// associated reference count carefully (as does `EventList`).
    pub fn as_ptr(&self) -> cl_event {
        self.0
    }

    /// Returns an immutable reference to a pointer, do not deref and store it unless
    /// you will manage its associated reference count carefully.
    ///
    #[inline]
    pub unsafe fn as_ptr_ref(&self) -> &cl_event {
        &self.0
    }

    /// Returns a mutable reference to a pointer, do not deref then modify or store it
    /// unless you will manage its associated reference count carefully.
    ///
    #[inline]
    pub unsafe fn as_ptr_mut(&mut self) -> &mut cl_event {
        &mut self.0
    }

    /// [FIXME]: ADD VALIDITY CHECK BY CALLING '_INFO' OR SOMETHING:
    /// NULL CHECK IS NOT ENOUGH
    ///
    /// This still leads to crazy segfaults when non-event pointers (random
    /// whatever addresses) are passed. Need better check.
    ///
    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.0.is_null()
    }

    /// Consumes the `Event`, returning the wrapped `cl_event` pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Event` using
    /// [`Event::from_raw`][from_raw].
    ///
    /// [from_raw]: struct.Event.html#method.from_raw
    ///
    pub fn into_raw(self) -> cl_event {
        let ptr = self.0;
        mem::forget(self);
        ptr
    }

    /// Constructs an `Event` from a raw `cl_event` pointer.
    ///
    /// The raw pointer must have been previously returned by a call to a
    /// [`Event::into_raw`][into_raw].
    ///
    /// [into_raw]: struct.Event.html#method.into_raw
    #[inline]
    pub unsafe fn from_raw(ptr: cl_event) -> Event {
        assert!(!ptr.is_null(), "Null pointer passed.");
        Event(ptr)
    }

    /// Ensures this contains a null event and returns a mutable pointer to it.
    fn _alloc_new(&mut self) -> *mut cl_event {
        assert!(self.0.is_null(), "ocl_core::Event::alloc_new: An 'Event' cannot be \
            used as target for event creation (as a new event) more than once.");
        &mut self.0
    }

    /// Returns a pointer pointer expected when used as a wait list.
    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        if self.0.is_null() { 0 as *const cl_event } else { &self.0 as *const cl_event }
    }

    /// Returns a count expected when used as a wait list.
    fn _count(&self) -> u32 {
        if self.0.is_null() { 0 } else { 1 }
    }
}

// impl From<NullEvent> for Event {
//     fn from(null_event: NullEvent) -> Event {
//         assert!(null_event.is_valid(), "Event::from::<NullEvent>: Invalid NullEvent.");
//         Event(null_event.0)
//     }
// }

// impl From<UserEvent> for Event {
//     fn from(uev: UserEvent) -> Event {
//         uev.into()
//     }
// }

unsafe impl<'a> ClNullEventPtr for &'a mut Event {
    #[inline(always)] fn alloc_new(&mut self) -> *mut cl_event { self._alloc_new() }
}

impl<'e> ClEventRef<'e> for Event {
    #[inline(always)] unsafe fn as_ptr_ref(&'e self) -> &'e cl_event { &self.0 }
}

unsafe impl ClWaitListPtr for Event {
    #[inline(always)] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline(always)] fn count(&self) -> u32 { self._count() }
}

unsafe impl<'a> ClWaitListPtr for &'a Event {
    #[inline(always)] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline(always)] fn count(&self) -> u32 { self._count() }
}

impl Clone for Event {
    fn clone(&self) -> Event {
        unsafe { functions::retain_event(self).expect("core::Event::clone"); }
        Event(self.0)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // Ignore errors here, some platforms just suck.
            unsafe { functions::release_event(self).ok(); }
        }
    }
}

// unsafe impl EventPtr for Event {}
unsafe impl Sync for Event {}
unsafe impl Send for Event {}



#[derive(Debug)]
pub struct UserEvent(cl_event);

impl UserEvent {
    #[inline]
    pub fn new(context: &Context) -> OclResult<UserEvent> {
        functions::create_user_event(context)
    }

    // /// Only call this when passing **the original** newly created pointer
    // /// directly from `clCreate...`. Do not use this to clone or copy.
    // #[inline]
    // pub unsafe fn from_raw_create_ptr(ptr: cl_event) -> UserEvent {
    //     UserEvent(ptr)
    // }

    /// Only use when cloning or copying from a pre-existing and valid
    /// `cl_event`.
    ///
    #[inline]
    pub unsafe fn from_raw_copied_ptr(ptr: cl_event) -> OclResult<UserEvent> {
        assert!(!ptr.is_null(), "Null pointer passed.");
        let new_core = UserEvent(ptr);
        try!(functions::retain_event(&new_core));
        Ok(new_core)
    }

    /// Sets the status for this user created event. Setting status to
    /// completion will cause commands waiting upon this event to execute
    ///
    /// Valid options are (for OpenCL versions 1.1 - 2.1):
    ///
    /// `CommandExecutionStatus::Complete`
    /// `CommandExecutionStatus::Running`
    /// `CommandExecutionStatus::Submitted`
    /// `CommandExecutionStatus::Queued`
    ///
    /// To the best of the author's knowledge, the only variant that matters
    /// is `::Complete`. Everything else is functionally equivalent and is
    /// useful only for debugging or profiling purposes (this may change).
    ///
    #[inline]
    pub fn set_status(&self, status: CommandExecutionStatus) -> OclResult<()> {
        println!("UserEvent::set_status: Setting user event status for event: {:?}", self);
        functions::set_user_event_status(self, status)
    }

    /// Sets this user created event to `CommandExecutionStatus::Complete`.
    #[inline]
    pub fn set_complete(&self) -> OclResult<()> {
        self.set_status(CommandExecutionStatus::Complete)
    }

    /// Queries the command status associated with this event and returns true
    /// if it is complete, false if incomplete or upon error.
    ///
    /// This is the fastest possible way to determine event status.
    ///
    #[inline]
    pub fn is_complete(&self) -> OclResult<bool> {
        functions::event_is_complete(self)
    }

    /// Sets a callback function, `callback_receiver`, to trigger upon
    /// completion of this event list with an optional pointer to user data.
    ///
    /// # Safety
    ///
    /// `user_data` must be guaranteed to still exist if and when `callback_receiver`
    /// is ever called.
    ///
    /// TODO: Create a safer type wrapper for `callback_receiver` (using an
    /// `Arc`?, etc.) within `ocl`.
    ///
    pub unsafe fn set_callback(&self,
            callback_receiver_opt: Option<EventCallbackFn>,
            user_data_ptr: *mut c_void,
            ) -> OclResult<()>
    {
        if self.is_valid() {
            let callback_receiver = match callback_receiver_opt {
                Some(cbr) => Some(cbr),
                None => Some(::_dummy_event_callback as EventCallbackFn),
            };

            ::set_event_callback(
                self,
                CommandExecutionStatus::Complete,
                callback_receiver,
                user_data_ptr as *mut _ as *mut c_void,
            )

        } else {
            Err("ocl_core::Event::set_callback: This event is null. Cannot set callback until \
                internal event pointer is actually created by a `clCreate...` function.".into())
        }
    }

    /// Returns a pointer, do not store it unless you will manage its
    /// associated reference count carefully (as does `EventList`).
    pub fn as_ptr(&self) -> cl_event {
        self.0
    }

    /// Returns an immutable reference to a pointer, do not deref and store it unless
    /// you will manage its associated reference count carefully.
    #[inline]
    pub unsafe fn as_ptr_ref(&self) -> &cl_event {
        &self.0
    }

    /// Returns a mutable reference to a pointer, do not deref then modify or store it
    /// unless you will manage its associated reference count carefully.
    #[inline]
    pub unsafe fn as_ptr_mut(&mut self) -> &mut cl_event {
        &mut self.0
    }

    /// [FIXME]: ADD VALIDITY CHECK BY CALLING '_INFO' OR SOMETHING:
    /// NULL CHECK IS NOT ENOUGH
    ///
    /// This still leads to crazy segfaults when non-event pointers (random
    /// whatever addresses) are passed. Need better check.
    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.0.is_null()
    }

    /// Consumes the `UserEvent`, returning the wrapped `cl_event` pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `UserEvent` using
    /// [`UserEvent::from_raw`][from_raw].
    ///
    /// [from_raw]: struct.UserEvent.html#method.from_raw
    ///
    pub fn into_raw(self) -> cl_event {
        let ptr = self.0;
        mem::forget(self);
        ptr
    }

    /// Constructs an `UserEvent` from a raw `cl_event` pointer.
    ///
    /// The raw pointer must have been previously returned by a call to a
    /// [`UserEvent::into_raw`][into_raw].
    ///
    /// [into_raw]: struct.UserEvent.html#method.into_raw
    pub unsafe fn from_raw(ptr: cl_event) -> UserEvent {
        assert!(!ptr.is_null(), "Null pointer passed.");
        UserEvent(ptr)
    }

    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        if self.0.is_null() { 0 as *const cl_event } else { &self.0 as *const cl_event }
    }

    fn _count(&self) -> u32 {
        if self.0.is_null() { 0 } else { 1 }
    }
}

impl<'e> ClEventRef<'e> for UserEvent {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        &self.0
    }
}

unsafe impl ClWaitListPtr for UserEvent {
    #[inline(always)] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline(always)] fn count(&self) -> u32 { self._count() }
}

unsafe impl<'a> ClWaitListPtr for &'a UserEvent {
    #[inline(always)] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline(always)] fn count(&self) -> u32 { self._count() }
}

impl Deref for UserEvent {
    type Target = Event;

    fn deref(&self) -> &Event {
        unsafe { &*(&self as *const _ as *const Event) }
    }
}

impl Clone for UserEvent {
    fn clone(&self) -> UserEvent {
        unsafe { functions::retain_event(self).expect("core::Event::clone"); }
        UserEvent(self.0)
    }
}

impl Drop for UserEvent {
    fn drop(&mut self) {
        debug_assert!(!self.0.is_null());
        // Ignore errors here, some platforms just suck.
        unsafe { functions::release_event(self).ok(); }
    }
}

// unsafe impl EventPtr for Event {}
unsafe impl Sync for UserEvent {}
unsafe impl Send for UserEvent {}



#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum EventKind {
    // Null,
    Event,
    UserEvent,
}


// #[derive(Clone, Debug)]
pub enum EventVariant {
    Null,
    Event(Event),
    UserEvent(UserEvent),
}


#[derive(Clone, Debug)]
pub enum EventVariantRef<'a> {
    Null,
    Event(&'a Event),
    UserEvent(&'a UserEvent),
}


#[derive(Debug)]
pub enum EventVariantMut<'a> {
    Null,
    Event(&'a mut Event),
    UserEvent(&'a mut UserEvent),
}




/// List of `cl_event`s.
#[derive(Debug)]
pub struct EventList {
    event_ptrs: Vec<cl_event>,
    event_kinds: Vec<EventKind>,
    clear_max_len: usize,
    clear_counter_max: i32,
    clear_counter: i32,
    clear_auto: bool,
}

impl EventList {
    /// Returns a new, empty, `EventList`.
    pub fn new() -> EventList {
        EventList::with_capacity(EL_INIT_CAPACITY)
    }

    /// Returns a new, empty, `EventList` with an initial capacity of `cap`.
    pub fn with_capacity(cap: usize) -> EventList {
        EventList {
            event_ptrs: Vec::with_capacity(cap),
            event_kinds: Vec::with_capacity(cap),
            clear_max_len: EL_CLEAR_MAX_LEN,
            clear_counter_max: EL_CLEAR_INTERVAL,
            clear_auto: EL_CLEAR_AUTO,
            clear_counter: 0,
        }
    }

    /// Pushes a new event onto the list.
    //
    // Technically, copies `event`'s contained pointer (a `cl_event`) then
    // `mem::forget`s it. This seems preferrable to incrementing the reference
    // count (with `functions::retain_event`) then letting `event` drop which just decrements it right back.
    //
    pub fn push_event(&mut self, event: Event) {
        // assert!(event.is_valid(), "Cannot push an empty (null) 'Event' into an 'EventList'.");
        debug_assert!(self.event_ptrs.len() == self.event_kinds.len());
        self.event_kinds.push(EventKind::Event);

        unsafe {
            self.event_ptrs.push(*event.as_ptr_ref());
            mem::forget(event);
        }

        self.decr_counter();
    }

    /// Pushes a new event onto the list.
    //
    // Technically, copies `event`'s contained pointer (a `cl_event`) then
    // `mem::forget`s it. This seems preferrable to incrementing the reference
    // count (with `functions::retain_event`) then letting `event` drop which just decrements it right back.
    //
    pub fn push_user_event(&mut self, user_event: UserEvent) {
        // assert!(event.is_valid(), "Cannot push an empty (null) 'Event' into an 'EventList'.");
        debug_assert!(self.event_ptrs.len() == self.event_kinds.len());
        self.event_kinds.push(EventKind::UserEvent);

        unsafe {
            self.event_ptrs.push(*user_event.as_ptr_ref());
            mem::forget(user_event);
        }

        self.decr_counter();
    }

    /// Removes the last event from the list and returns it.
    //
    // Does not increment reference count as it will not have been decremented
    // when added to list.
    //
    pub fn pop(&mut self) -> Option<EventVariant> {
        // self.event_ptrs.pop().map(|ptr| unsafe { Event::from_raw_copied_ptr(ptr) } )
        // self.event_ptrs.pop().map(|ptr| Event(ptr))
        debug_assert!(self.event_ptrs.len() == self.event_kinds.len());

        let kind = self.event_kinds.pop();
        let ptr = self.event_ptrs.pop();

        match kind.and_then(|kind| ptr.and_then(|ptr| Some((kind, ptr)))) {
            Some((kind, ptr)) => {
                match kind {
                    // EventKind::Null => Some(EventVariant::Null),
                    EventKind::Event => Some(EventVariant::Event(Event(ptr))),
                    EventKind::UserEvent => Some(EventVariant::UserEvent(UserEvent(ptr))),
                }
            },
            None => None
        }
    }

    /// Appends a new null element to the end of the list and returns a reference to it.
    pub fn allot(&mut self) -> &mut cl_event {
        debug_assert!(self.event_ptrs.len() == self.event_kinds.len());
        self.event_kinds.push(EventKind::Event);
        self.event_ptrs.push(0 as cl_event);
        self.event_ptrs.last_mut().unwrap()
    }

    /// Returns an immutable reference to a pointer, do not deref and store it unless
    /// you will manage its associated reference count carefully.
    pub unsafe fn as_ptr_ref(&self) -> &cl_event {
        self.event_ptrs.first().expect("ocl::core::EventList::as_ptr_ref(): \
            Attempted to take a reference to the first element of an empty list.")
    }

    /// Clones an event by index.
    pub fn get_event_cloned(&self, index: usize) -> Option<OclResult<Event>> {
        match self.event_kinds.get(index) {
            Some(kind) => {
                if let EventKind::UserEvent = *kind {
                    return Some(Err("EventList::event_cloned: Cannot clone a `UserEvent` with this \
                        method. Use `::get_user_event_cloned` instead.".into()))
                }
            },
            None => return None,
        }

        self.event_ptrs.get(index).map(|ptr| unsafe { Event::from_raw_copied_ptr(*ptr) } )
    }

    /// Clones the last event.
    pub fn last_event_cloned(&self) -> Option<OclResult<Event>> {
        match self.event_kinds.last() {
            Some(kind) => {
                if let EventKind::UserEvent = *kind {
                    return Some(Err("EventList::event_cloned: Cannot clone a `UserEvent` with this \
                        method. Use `::last_user_event_cloned` instead.".into()))
                }
            },
            None => return None,
        }

        self.event_ptrs.last().map(|ptr| unsafe { Event::from_raw_copied_ptr(*ptr) } )
    }

    pub fn get_user_event_cloned(&self) -> Option<OclResult<UserEvent>> {
        unimplemented!()
    }

    pub fn last_user_event_cloned(&self) -> Option<OclResult<UserEvent>> {
        unimplemented!()
    }

    /// Clears the list.
    pub fn clear(&mut self) -> OclResult<()> {
        for &ptr in self.event_ptrs.iter() {
            unsafe { functions::release_event(&EventRefWrapper(ptr))?; }
        }

        self.clear_counter = EL_CLEAR_INTERVAL;
        self.event_ptrs.clear();
        self.event_kinds.clear();
        Ok(())
    }

    /// Clears each completed event from the list.
    ///
    /// TODO: TEST THIS
    pub fn clear_completed(&mut self) -> OclResult<()> {
        if self.len() < 16 { return Ok(()) }

        let mut cmpltd_events: Vec<usize> = Vec::with_capacity(EL_CLEAR_MAX_LEN);

        for (idx, &event_ptr) in self.event_ptrs.iter().enumerate() {
            let status = try!(functions::get_event_status(&EventRefWrapper(event_ptr)));

            if status == CommandExecutionStatus::Complete {
                cmpltd_events.push(idx)
            }
        }

        // Release completed events:
        for &idx in &cmpltd_events {
            unsafe {
                try!(functions::release_event(&EventRefWrapper(self.event_ptrs[idx])));
            }
        }

        try!(util::vec_remove_rebuild(&mut self.event_ptrs, &cmpltd_events[..], 2));
        try!(util::vec_remove_rebuild(&mut self.event_kinds, &cmpltd_events[..], 2));

        debug_assert!(self.event_ptrs.len() == self.event_kinds.len());

        self.clear_counter = EL_CLEAR_INTERVAL;

        Ok(())
    }


    // /// Merges thecontents of this list and another into a new list and returns it.
    // /// Make these merge without dropping the damn pointers
    // pub fn union(self, other_list: EventList) -> EventList {
    //     let new_cap = other_list.event_ptrs.len() + self.event_ptrs.len() + 8;
    //     let mut new_list = EventList::with_capacity(new_cap);

    //     new_list.event_ptrs.extend(self.event_ptrs.iter().cloned());
    //     new_list.event_kinds.extend(self.event_kinds.iter().cloned());
    //     new_list.event_ptrs.extend(other_list.event_ptrs.iter().cloned());
    //     new_list.event_kinds.extend(other_list.event_kinds.iter().cloned());

    //     new_list
    // }

    /// Counts down the auto-list-clear counter.
    fn decr_counter(&mut self) {
        if EL_CLEAR_AUTO {
            self.clear_counter -= 1;

            if self.clear_counter <= 0 && self.event_ptrs.len() > EL_CLEAR_MAX_LEN {
                self.clear_completed().unwrap();
            }
        }
    }

    #[inline(always)] pub fn len(&self) -> usize { self.event_ptrs.len() }
    #[inline(always)] pub fn is_empty(&self) -> bool { self.len() == 0 }
    #[inline(always)] pub fn count(&self) -> u32 { self.event_ptrs.len() as u32 }

    unsafe fn _as_ptr_ptr(&self) -> *const cl_event {
        match self.event_ptrs.first() {
            Some(ele) => ele as *const cl_event,
            None => ptr::null(),
        }
    }

    #[inline(always)]
    fn _count(&self) -> u32 {
        self.event_ptrs.len() as u32
    }
}

unsafe impl<'a> ClNullEventPtr for &'a mut EventList {
    #[inline(always)] fn alloc_new(&mut self) -> *mut cl_event { self.allot() }
}

unsafe impl ClWaitListPtr for EventList {
    #[inline(always)] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline(always)] fn count(&self) -> u32 { self.count() }
}

unsafe impl<'a> ClWaitListPtr for &'a EventList {
    #[inline(always)] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline(always)] fn count(&self) -> u32 { self._count() }
}

impl Clone for EventList {
    /// Clones this list in a thread safe manner.
    fn clone(&self) -> EventList {
        for &event_ptr in &self.event_ptrs {
            if !event_ptr.is_null() {
                unsafe { functions::retain_event(&EventRefWrapper(event_ptr))
                    .expect("core::EventList::clone") }
            }
        }

        EventList {
            event_ptrs: self.event_ptrs.clone(),
            event_kinds: self.event_kinds.clone(),
            clear_max_len: self.clear_max_len,
            clear_counter_max: self.clear_counter_max,
            clear_auto: self.clear_auto,
            clear_counter: self.clear_counter,
        }
    }
}

impl Drop for EventList {
    /// Re-creates the appropriate event wrapper for each pointer in the list
    /// and drops it.
    fn drop(&mut self) {
        for (&event_ptr, event_kind) in self.event_ptrs.iter().zip(self.event_kinds.iter()) {
            // unsafe { functions::release_event(&EventRefWrapper(event_ptr)).unwrap(); }
            match *event_kind {
                // EventKind::Null => (),
                EventKind::Event => { mem::drop(Event(event_ptr)) },
                EventKind::UserEvent => { mem::drop(UserEvent(event_ptr)) },
            }
        }
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
    pub unsafe fn from_raw_create_ptr(ptr: cl_sampler) -> Sampler {
        assert!(!ptr.is_null(), "Null pointer passed.");
        Sampler(ptr)
    }

    /// Returns a pointer, do not store it.
    pub unsafe fn as_ptr(&self) -> cl_sampler {
        self.0
    }
}

impl Clone for Sampler {
    fn clone(&self) -> Sampler {
        unsafe { functions::retain_sampler(self).unwrap(); }
        Sampler(self.0)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { functions::release_sampler(self).unwrap(); }
    }
}

unsafe impl Sync for Sampler {}
unsafe impl Send for Sampler {}
