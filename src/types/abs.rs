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

// #![doc(html_root_url="https://docs.rs/ocl-core/0.3/")]

use std::mem;
use std::ptr;
use std::slice;
use std::cell::Ref;
use std::fmt::Debug;
use libc::c_void;
use ffi::{cl_platform_id, cl_device_id,  cl_context, cl_command_queue, cl_mem, cl_program,
    cl_kernel, cl_event, cl_sampler};
use ::{CommandExecutionStatus, OpenclVersion, PlatformInfo, DeviceInfo, DeviceInfoResult,
    ContextInfo, ContextInfoResult, CommandQueueInfo, CommandQueueInfoResult, ProgramInfo,
    ProgramInfoResult, KernelInfo, KernelInfoResult, Status, EventCallbackFn, OclPrm};
use error::{Result as OclResult, Error as OclError};
use functions;

//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

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
pub trait ClEventPtrRef<'e> {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event;
}

impl<'e, L> ClEventPtrRef<'e> for &'e L where L: ClEventPtrRef<'e> {
    unsafe fn as_ptr_ref(&'e self) -> &'e cl_event {
        (*self).as_ptr_ref()
    }
}



/// Types with a mutable pointer to a new, null raw event pointer.
///
pub unsafe trait ClNullEventPtr: Debug {
    fn alloc_new(&mut self) -> *mut cl_event;
    unsafe fn alloc_from<E: AsRef<Event>>(&mut self, ev: E);
}

unsafe impl ClNullEventPtr for () {
    fn alloc_new(&mut self) -> *mut cl_event {
        panic!("Void events may not be used.");
    }

    unsafe fn alloc_from<E: AsRef<Event>>(&mut self, _: E) {
        panic!("Void events may not be used.");
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

unsafe impl<'a, W> ClWaitListPtr for Ref<'a, W> where W: ClWaitListPtr {
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        (*(*self)).as_ptr_ptr()
    }

    fn count (&self) -> u32 {
        0 as u32
    }
}

unsafe impl<'a> ClWaitListPtr for &'a [cl_event] {
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self.as_ptr()
    }

    fn count (&self) -> u32 {
        self.len() as u32
    }
}

unsafe impl<'a> ClWaitListPtr for &'a [Event] {
    unsafe fn as_ptr_ptr(&self) -> *const cl_event {
        self.as_ptr() as *const _ as *const cl_event
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
// pub unsafe trait ClPlatformIdPtr: Sized + Debug {
pub unsafe trait ClPlatformIdPtr: Debug {
    // unsafe fn as_ptr(&self) -> cl_platform_id {
    //     debug_assert!(mem::size_of_val(self) == mem::size_of::<PlatformId>());
    //     // mem::transmute_copy()
    //     let core = self as *const Self as *const _ as *const PlatformId;
    //     (*core).as_ptr()
    // }
    fn as_ptr(&self) -> cl_platform_id;
}

unsafe impl<'a, P> ClPlatformIdPtr for &'a P where P: ClPlatformIdPtr {
    fn as_ptr(&self) -> cl_platform_id {
        (*self).as_ptr()
    }
}

unsafe impl ClPlatformIdPtr for () {
    fn as_ptr(&self) -> cl_platform_id {
        ptr::null_mut() as *mut _ as cl_platform_id
    }
}


/// Types with a reference to a raw device_id pointer.
// pub unsafe trait ClDeviceIdPtr: Sized + Debug {
pub unsafe trait ClDeviceIdPtr: Debug {
    // unsafe fn as_ptr(&self) -> cl_device_id {
    //     debug_assert!(mem::size_of_val(self) == mem::size_of::<DeviceId>());
    //     // mem::transmute_copy(self)
    //     let core = self as *const Self as *const _ as *const DeviceId;
    //     (*core).as_ptr()
    // }
    fn as_ptr(&self) -> cl_device_id;
}


unsafe impl<'a, P> ClDeviceIdPtr for &'a P where P: ClDeviceIdPtr {
    fn as_ptr(&self) -> cl_device_id {
        (*self).as_ptr()
    }
}

unsafe impl ClDeviceIdPtr for () {
    fn as_ptr(&self) -> cl_device_id {
        ptr::null_mut() as *mut _ as cl_device_id
    }
}

//=============================================================================
//=================================== TYPES ===================================
//=============================================================================

/// Wrapper used by `EventList` to send event pointers to core functions
/// cheaply.
#[repr(C)]
pub struct EventRefWrapper(cl_event);

impl EventRefWrapper {
    pub unsafe fn new(ptr: cl_event) -> EventRefWrapper {
        EventRefWrapper(ptr)
    }
}

impl<'e> ClEventPtrRef<'e> for EventRefWrapper {
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

unsafe impl ClPlatformIdPtr for PlatformId {
    fn as_ptr(&self) -> cl_platform_id {
        self.0
    }
}

// unsafe impl<'a> ClPlatformIdPtr for &'a PlatformId {
//     fn as_ptr(&self) -> cl_platform_id {
//         (*self).0
//     }
// }

// unsafe impl<'a> ClPlatformIdPtr for &'a PlatformId {}
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
        assert!(!ptr.is_null(), "Null pointer passed.");
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

// unsafe impl ClDeviceIdPtr for DeviceId {}
// unsafe impl<'a> ClDeviceIdPtr for &'a DeviceId {}

unsafe impl ClDeviceIdPtr for DeviceId {
    fn as_ptr(&self) -> cl_device_id {
        self.0
    }
}

// unsafe impl<'a> ClDeviceIdPtr for &'a DeviceId {
//     fn as_ptr(&self) -> cl_device_id {
//         (*self).0
//     }
// }

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
#[repr(C)]
#[derive(Debug)]
pub struct Context(cl_context);

impl Context {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_context) -> Context {
        assert!(!ptr.is_null(), "Null pointer passed.");
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

    /// Returns the platform associated with this context, if any.
    ///
    /// Errors upon the usual OpenCL errors.
    ///
    /// Returns `None` if the context properties do not specify a platform.
    pub fn platform(&self) -> OclResult<Option<PlatformId>> {
        functions::get_context_platform(self)
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
#[repr(C)]
#[derive(Debug)]
pub struct CommandQueue(cl_command_queue);

impl CommandQueue {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_command_queue) -> CommandQueue {
        assert!(!ptr.is_null(), "Null pointer passed.");
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

    /// Returns the `Context` associated with this command queue.
    pub fn context(&self) -> OclResult<Context> {
        match functions::get_command_queue_info(self, CommandQueueInfo::Context) {
            CommandQueueInfoResult::Context(c) => Ok(c),
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
#[repr(C)]
#[derive(Debug)]
pub struct Mem(cl_mem);

impl Mem {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_mem) -> Mem {
        assert!(!ptr.is_null(), "Null pointer passed.");
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

/// A pointer to a region of mapped (pinned) memory.
//
// [NOTE]: Do not derive/impl `Clone` or `Sync`. Will not be thread safe
// without a mutex.
//
#[repr(C)]
#[derive(Debug)]
pub struct MemMap<T>(*mut T);

impl<T: OclPrm> MemMap<T> {
    #[inline(always)]
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw(ptr: *mut T) -> MemMap<T> {
        assert!(!ptr.is_null(), "MemMap::from_raw: Null pointer passed.");
        MemMap(ptr)
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.0
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
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

impl<T> AsMem<T> for MemMap<T> where T: OclPrm {
    #[inline(always)]
    fn as_mem(&self) -> &Mem {
        unsafe { &*(self as *const _ as *const Mem) }
    }
}

unsafe impl<T: OclPrm> MemCmdRw for MemMap<T> {}
unsafe impl<'a, T: OclPrm> MemCmdRw for &'a MemMap<T> {}
unsafe impl<'a, T: OclPrm> MemCmdRw for &'a mut MemMap<T> {}
unsafe impl<T: OclPrm> Send for MemMap<T> {}
// unsafe impl<T: OclPrm> Sync for MemMap<T> {}



/// cl_program
#[repr(C)]
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
#[repr(C)]
#[derive(Debug)]
pub struct Kernel(cl_kernel);

impl Kernel {
    /// Only call this when passing **the original** newly created pointer
    /// directly from `clCreate...`. Do not use this to clone or copy.
    pub unsafe fn from_raw_create_ptr(ptr: cl_kernel) -> Kernel {
        assert!(!ptr.is_null(), "Null pointer passed.");
        Kernel(ptr)
    }

    /// Only call this when passing a copied pointer such as from an
    /// `clGet*****Info` function.
    ///
    // [TODO]: Evaluate usefulness.
    pub unsafe fn from_raw_copied_ptr(ptr: cl_kernel) -> Kernel {
        assert!(!ptr.is_null(), "Null pointer passed.");
        let copy = Kernel(ptr);
        functions::retain_kernel(&copy).unwrap();
        copy
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

    pub fn devices(&self) -> OclResult<Vec<DeviceId>> {
        self.program().and_then(|p| p.devices())
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


/// cl_event
#[repr(C)]
#[derive(Debug)]
pub struct Event(cl_event);

impl Event {
    /// For passage directly to an 'event creation' function (such as enqueue...).
    #[inline]
    pub fn null() -> Event {
        Event(0 as cl_event)
    }

    /// Creates and returns a new 'user' event.
    ///
    /// User events are events which are meant to have their completion status
    /// set from the host side (that means you).
    ///
    #[inline]
    pub fn user(context: &Context) -> OclResult<Event> {
        functions::create_user_event(context)
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
    #[inline]
    pub unsafe fn from_raw_copied_ptr(ptr: cl_event) -> OclResult<Event> {
        assert!(!ptr.is_null(), "ocl_core::Event::from_raw_copied_ptr: Null pointer passed.");
        let copy = Event(ptr);
        functions::retain_event(&copy)?;
        Ok(copy)
    }

    /// Sets the status for this event. Setting status to completion will
    /// cause commands waiting upon this event to execute.
    ///
    ///  Will return an error if this event is not a 'user' event (created
    /// with `::user()`).
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
    /// `::set_complete` is probably what you want.
    ///
    #[inline]
    pub fn set_status(&self, status: CommandExecutionStatus) -> OclResult<()> {
        functions::set_user_event_status(self, status)
    }

    /// Sets this user created event to `CommandExecutionStatus::Complete`.
    ///
    /// Will return an error if this event is not a 'user' event (created
    /// with `::user()`).
    ///
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

    /// Causes the command queue to wait until this event is complete before returning.
    #[inline]
    pub fn wait_for(&self) -> OclResult <()> {
        ::wait_for_event(self)
    }

    /// Returns whether or not this event is associated with a command or is a
    /// user event.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
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

    /// Sets a callback function, `callback_receiver`, to trigger upon
    /// completion of this event list with an optional pointer to user data.
    ///
    /// The callback function must have a signature matching:
    /// `extern "C" fn (ffi::cl_event, i32, *mut libc::c_void)`
    ///
    /// # Safety
    ///
    /// `user_data` must be guaranteed to still exist if and when `callback_receiver`
    /// is ever called.
    ///
    /// TODO: Create a safer type wrapper for `callback_receiver` (using an
    /// `Arc`?, etc.) within `ocl`.
    ///
    //
    // [NOTE]: Making callback_receiver optional is pointless. There is no way
    // to unset a previously set callback.
    pub unsafe fn set_callback(&self,
            callback_receiver: EventCallbackFn,
            user_data_ptr: *mut c_void,
            ) -> OclResult<()>
    {
        if self.is_valid() {
            ::set_event_callback(self, CommandExecutionStatus::Complete,
                Some(callback_receiver), user_data_ptr as *mut _ as *mut c_void)
        } else {
            Err("ocl_core::Event::set_callback: This event is null. Cannot set callback until \
                internal event pointer is actually created by a `clCreate...` function.".into())
        }
    }

    /// Returns an immutable reference to a pointer, do not deref and store it unless
    /// you will manage its associated reference count carefully.
    ///
    ///
    /// ### Warning
    ///
    /// DO NOT store this pointer.
    ///
    /// DO NOT send this pointer across threads unless you are incrementing
    /// the reference count before sending and decrementing after sending.
    ///
    /// Use `::into_raw` for these purposes. Thank you.
    ///
    #[inline]
    pub unsafe fn as_ptr_ref(&self) -> &cl_event {
        &self.0
    }

    /// Returns a mutable reference to a pointer, do not deref then modify or store it
    /// unless you will manage its associated reference count carefully.
    ///
    ///
    /// ### Warning
    ///
    /// DO NOT store this pointer.
    ///
    /// DO NOT send this pointer across threads unless you are incrementing
    /// the reference count before sending and decrementing after sending.
    ///
    /// Use `::into_raw` for these purposes. Thank you.
    ///
    #[inline]
    pub unsafe fn as_ptr_mut(&mut self) -> &mut cl_event {
        &mut self.0
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

unsafe impl<'a> ClNullEventPtr for &'a mut Event {
    #[inline(always)] fn alloc_new(&mut self) -> *mut cl_event { self._alloc_new() }

    #[inline(always)] unsafe fn alloc_from<E: AsRef<Event>>(&mut self, ev: E) {
        let ptr = ev.as_ref().clone().into_raw();
        assert!(!ptr.is_null());
        self.0 = ptr;
        // functions::retain_event(*self).expect("core::Event::alloc_from");
    }
}

unsafe impl ClWaitListPtr for Event {
    #[inline(always)] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline(always)] fn count(&self) -> u32 { self._count() }
}

unsafe impl<'a> ClWaitListPtr for &'a Event {
    #[inline(always)] unsafe fn as_ptr_ptr(&self) -> *const cl_event { self._as_ptr_ptr() }
    #[inline(always)] fn count(&self) -> u32 { self._count() }
}

impl<'e> ClEventPtrRef<'e> for Event {
    #[inline(always)] unsafe fn as_ptr_ref(&'e self) -> &'e cl_event { &self.0 }
}

impl Clone for Event {
    fn clone(&self) -> Event {
        assert!(!self.0.is_null(), "ocl_core::Event::clone: \
            Cannot clone a null (empty) event.");
        unsafe { functions::retain_event(self).expect("core::Event::clone"); }
        Event(self.0)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // Ignore errors here? Some platforms just suck.
            unsafe { functions::release_event(self).unwrap(); }
        }
    }
}

// unsafe impl EventPtr for Event {}
unsafe impl Sync for Event {}
unsafe impl Send for Event {}


/// cl_sampler
#[repr(C)]
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
