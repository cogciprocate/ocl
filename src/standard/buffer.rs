//! Interfaces to a buffer.

use std;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use core::{self, OclPrm, Mem as MemCore, CommandQueue as CommandQueueCore, MemFlags, 
    MemInfo, MemInfoResult, ClEventPtrNew};
use error::{Error as OclError, Result as OclResult};
use standard::{Queue, MemLen, EventList, BufferCmd, SpatialDims};


/// A chunk of memory physically located on a device, such as a GPU.
///
/// Data is stored both remotely in a memory buffer on the device associated with 
/// `queue` and optionally in a vector (`self.vec`) in host (local) memory for 
/// convenient use as a workspace etc.
///
/// The host side vector must be manually synchronized with the device side buffer 
/// using `::fill_vec`, `::flush_vec`, etc. Data within the contained vector 
/// should generally be considered stale except immediately after a fill/flush 
/// (exception: pinned memory).
///
/// Fill/flush methods are for convenience and are equivalent to the psuedocode: 
/// read_into(self.vec) and write_from(self.vec).
///
/// ## Stability
///
/// Read/write/fill/flush methods will eventually be returning a result instead of
/// panicing upon error.
///
// TODO: Return result for reads and writes instead of panicing.
// TODO: Check that type size (sizeof(T)) is <= the maximum supported by device.
// TODO: Consider integrating an event list to help coordinate pending reads/writes.
#[derive(Debug, Clone)]
pub struct Buffer<T: OclPrm> {
    // vec: Vec<T>,
    obj_core: MemCore,
    // queue: Queue,
    command_queue_obj_core: CommandQueueCore,
    dims: SpatialDims,
    len: usize,
    _data: PhantomData<T>,
    // vec: VecOption<T>,
}

impl<T: OclPrm> Buffer<T> {
    /// Creates a new read/write Buffer with dimensions: `dims` which will use the 
    /// command queue: `queue` (and its associated device and context) for all operations.
    ///
    /// The device side buffer will be allocated a size based on the maximum workgroup 
    /// size of the device. This helps ensure that kernels do not attempt to read 
    /// from or write to memory beyond the length of the buffer (see crate level 
    /// documentation for more details about how dimensions are used). The buffer
    /// will be initialized with a sensible default value (probably `0`).
    ///
    /// ## Other Method Panics
    /// The returned Buffer contains no host side vector. Functions associated with
    /// one such as `.enqueue_flush_vec()`, `enqueue_fill_vec()`, etc. will panic.
    /// [FIXME]: Return result.
    pub fn new<D: MemLen>(dims: D, queue: &Queue) -> Buffer<T> {
        let dims: SpatialDims = dims.to_lens().into(); 
        // let len = dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");
        let len = dims.to_len();
        Buffer::_new(dims, len, queue)
    }

    pub fn newer_new<D: MemLen>(queue: &Queue, flags: Option<MemFlags>, dims: D, host_ptr: Option<&[T]>) 
            -> OclResult<Buffer<T>>
    {
        let flags = flags.unwrap_or(core::MEM_READ_WRITE);
        let dims: SpatialDims = dims.to_lens().into();
        // let len = dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");
        let len = dims.to_len();
        let obj_core = unsafe { try!(core::create_buffer(queue.context_core_as_ref(), flags, len,
            host_ptr)) };

        Ok( Buffer {
            obj_core: obj_core,
            command_queue_obj_core: queue.core_as_ref().clone(),
            dims: dims,
            len: len,
            _data: PhantomData,
        })
    }

    /// Creates a new Buffer with caller-managed buffer length, type, flags, and 
    /// initialization.
    ///
    /// - Does not optimize size. 
    ///
    /// [DOCUMENTATION OUT OF DATE]
    ///
    /// ## Examples
    /// See `examples/buffer_unchecked.rs`.
    ///
    /// ## Parameter Reference Documentation
    /// Refer to the following page for information about how to configure flags and
    /// other options for the buffer.
    /// [https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html)
    ///
    /// ## Safety
    /// No creation time checks are made to help prevent device buffer overruns, 
    /// etc. Caller should be sure to initialize the device side memory with a 
    /// write. Any host side memory pointed to by `host_ptr` is at even greater
    /// risk if used with `CL_MEM_USE_HOST_PTR` (see below).
    ///
    /// [IMPORTANT] Practically every read and write to an Buffer created in this way is
    /// potentially unsafe. Because `.enqueue_read()` and `.enqueue_write()` do not require an 
    /// unsafe block, their implied promises about safety may be broken at any time.
    ///
    /// *You need to know what you're doing and be extra careful using an Buffer created 
    /// with this method because it badly breaks Rust's usual safety promises even
    /// outside of an unsafe block.*
    ///
    /// **This is horribly un-idiomatic Rust. You have been warned.**
    ///
    /// NOTE: The above important warnings probably only apply to buffers created with 
    /// the `CL_MEM_USE_HOST_PTR` flag because the memory is considered 'pinned' but 
    /// there may also be implementation specific issues which haven't been considered 
    /// or are unknown.
    ///
    /// [FIXME]: Return result.
    /// [FIXME]: Update docs.
    pub unsafe fn new_unchecked<S>(flags: MemFlags, dims: S, host_ptr: Option<&[T]>, 
                queue: &Queue) -> Buffer<T> where S: Into<SpatialDims>
    {
        let dims = dims.into();
        let len = dims.to_len();
        let obj_core = core::create_buffer(queue.context_core_as_ref(), flags, len,
            host_ptr).expect("[FIXME: TEMPORARY]: Buffer::_new():");

        Buffer {
            obj_core: obj_core,
            command_queue_obj_core: queue.core_as_ref().clone(),
            dims: dims,
            len: len,
            _data: PhantomData,
            // vec: VecOption::None,
        }
    }

    // Consolidated constructor for Buffers without vectors.
    /// [FIXME]: Return result.
    fn _new(dims: SpatialDims, len: usize, queue: &Queue) -> Buffer<T> {
        let obj_core = unsafe { core::create_buffer::<T>(queue.context_core_as_ref(),
            core::MEM_READ_WRITE, len, None).expect("Buffer::_new()") };

        Buffer {            
            obj_core: obj_core,
            command_queue_obj_core: queue.core_as_ref().clone(),
            dims: dims,
            len: len,
            _data: PhantomData,
            // vec: VecOption::None,
        }
    }

    /// Returns a buffer command builder used to read, write, copy, etc.
    ///
    /// Run `.enq()` to enqueue the command.
    ///
    pub fn cmd<'b>(&'b self) -> BufferCmd<'b, T> {
        BufferCmd::new(&self.command_queue_obj_core, &self.obj_core, self.len)
    }


    /// Enqueues reading `data.len() * mem::size_of::<T>()` bytes from the device 
    /// buffer into `data` with a remote offset of `offset`.
    ///
    /// Setting `queue` to `None` will use the default queue set during creation.
    /// Otherwise, the queue passed will be used for this call only.
    ///
    /// Will optionally wait for events in `ewait` to finish 
    /// before reading. Will also optionally add a new event associated with
    /// the read to `enew`.
    ///
    /// [UPDATE] If the `enew` event list is `None`, the read will be blocking, otherwise
    /// returns immediately.
    ///
    /// ## Safety
    ///
    /// Bad things will happen if the memory referred to by `data` is freed and
    /// reallocated before the read completes. It's up to the caller to make 
    /// sure that the new event added to `enew` completes. Use 
    /// 'enew.last()' right after the calling `::read_async` to get a.
    /// reference to the event associated with the read. [NOTE: Improved ease
    /// of use is coming to the event api eventually]
    ///
    /// ## Errors
    ///
    /// `offset` must be less than the length of the buffer.
    ///
    /// The length of `data` must be less than the length of the buffer minus `offset`.
    ///
    /// Errors upon any OpenCL error.
    ///
    /// [UNSTABLE: Likely to be depricated in favor of `::cmd`.
    unsafe fn enqueue_read(&self, queue: Option<&Queue>, block: bool, offset: usize, data: &mut [T],
                ewait: Option<&EventList>, enew: Option<&mut ClEventPtrNew>) -> OclResult<()>
    {
        // assert!(offset < self.len(), "Buffer::read{{_async}}(): Offset out of range.");
        // assert!(data.len() <= self.len() - offset, 
        //     "Buffer::read{{_async}}(): Data length out of range.");
        if offset >= self.len() { 
            return OclError::err("Buffer::read{{_async}}(): Offset out of range."); }
        if data.len() > self.len() - offset {
            return OclError::err("Buffer::read{{_async}}(): Data length out of range."); }

        let command_queue = match queue {
            Some(q) => q.core_as_ref(),
            None => &self.command_queue_obj_core,
        };

        // let blocking_read = enew.is_none();
        core::enqueue_read_buffer(command_queue, &self.obj_core, block, 
            // offset, data, ewait.map(|el| el.core_as_ref()), enew.map(|el| el.core_as_mut()))
            offset, data, ewait, enew)
    }

    /// Enqueues writing `data.len() * mem::size_of::<T>()` bytes from `data` to the 
    /// device buffer with a remote offset of `offset`.
    ///
    /// Setting `queue` to `None` will use the default queue set during creation.
    /// Otherwise, the queue passed will be used for this call only.
    ///
    /// Will optionally wait for events in `ewait` to finish before writing. 
    /// Will also optionally add a new event associated with the write to `enew`.
    ///
    /// [UPDATE] If the `enew` event list is `None`, the write will be blocking, otherwise
    /// returns immediately.
    ///
    /// ## Data Integrity
    ///
    /// Ensure that the memory referred to by `data` is unmolested until the 
    /// write completes if passing a `enew`.
    ///
    /// ## Errors
    ///
    /// `offset` must be less than the length of the buffer.
    ///
    /// The length of `data` must be less than the length of the buffer minus `offset`.
    ///
    /// Errors upon any OpenCL error.
    /// 
    /// [UNSTABLE: Likely to be depricated in favor of `::cmd`.
    fn enqueue_write(&self, queue: Option<&Queue>, block: bool, offset: usize, data: &[T], 
                ewait: Option<&EventList>, enew: Option<&mut ClEventPtrNew>) -> OclResult<()>
    {        
        // assert!(offset < self.len(), "Buffer::write{{_async}}(): Offset out of range.");
        // assert!(data.len() <= self.len() - offset, 
        //     "Buffer::write{{_async}}(): Data length out of range.");
        if offset >= self.len() { 
            return OclError::err("Buffer::write{{_async}}(): Offset out of range."); }
        if data.len() > self.len() - offset {
            return OclError::err("Buffer::write{{_async}}(): Data length out of range."); }

        let command_queue = match queue {
            Some(q) => q.core_as_ref(),
            None => &self.command_queue_obj_core,
        };

        // let blocking_write = enew.is_none();
        core::enqueue_write_buffer(command_queue, &self.obj_core, block, 
            // offset, data, ewait.map(|el| el.core_as_ref()), enew.map(|el| el.core_as_mut()))
            offset, data, ewait, enew)
    }

    /// Reads `data.len() * mem::size_of::<T>()` bytes from the (remote) device buffer 
    /// into `data` with a remote offset of `offset` and blocks until complete.
    ///
    /// ## Errors
    ///
    /// `offset` must be less than the length of the buffer.
    ///
    /// The length of `data` must be less than the length of the buffer minus `offset`.
    ///
    /// Errors upon any OpenCL error.
    pub fn read(&self, data: &mut [T])
    {
        // Safe due to being a blocking read (right?).
        unsafe { self.enqueue_read(None, true, 0, data, None, None).expect("ocl::Buffer::read()") }
    }

    /// Writes `data.len() * mem::size_of::<T>()` bytes from `data` to the (remote) 
    /// device buffer with a remote offset of `offset` and blocks until complete.
    ///
    /// ## Panics
    ///
    /// `offset` must be less than the length of the buffer.
    ///
    /// The length of `data` must be less than the length of the buffer minus `offset`.
    ///
    /// Errors upon any OpenCL error.
    pub fn write(&self, data: &[T])
    {
        self.enqueue_write(None, true, 0, data, None, None).expect("ocl::Buffer::read()")
    }

    /// Blocks the current thread until the underlying command queue has
    /// completed all commands.
    pub fn wait(&self) {
        core::finish(&self.command_queue_obj_core).unwrap();
    }

    /// Returns the length of the Buffer.
    ///
    /// This is the length of both the device side buffer and the host side vector,
    /// if any. This may not agree with desired dataset size because it will have been
    /// rounded up to the nearest maximum workgroup size of the device on which it was
    /// created.
    #[inline]
    pub fn len(&self) -> usize {
        // debug_assert!((if let VecOption::Some(ref vec) = self.vec { vec.len() } 
        //     else { self.len }) == self.len);
        self.len
    }

    /// Returns a copy of the core buffer object reference.
    pub fn core_as_ref(&self) -> &MemCore {
        &self.obj_core
    }

    /// Changes the default queue used by this Buffer for reads and writes, etc.
    ///
    /// Returns a ref for chaining i.e.:
    ///
    /// `buffer.set_queue(queue).flush_vec(....);`
    ///
    /// [NOTE]: Even when used as above, the queue is changed permanently,
    /// not just for the one call. Changing the queue is cheap so feel free
    /// to change as often as needed.
    ///
    pub fn set_queue<'a>(&'a mut self, queue: &Queue) -> &'a mut Buffer<T> {
        // [FIXME]: Set this up:
        // assert!(queue.device == self.queue.device);
        // [/FIXME]

        self.command_queue_obj_core = queue.core_as_ref().clone();
        self
    }

    /// Returns info about this buffer.
    fn mem_info(&self, info_kind: MemInfo) -> MemInfoResult {
        match core::get_mem_object_info(&self.obj_core, info_kind) {
            Ok(res) => res,
            Err(err) => MemInfoResult::Error(Box::new(err)),
        }        
    }

    fn fmt_mem_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Buffer Mem")
            .field("Type", &self.mem_info(MemInfo::Type))
            .field("Flags", &self.mem_info(MemInfo::Flags))
            .field("Size", &self.mem_info(MemInfo::Size))
            .field("HostPtr", &self.mem_info(MemInfo::HostPtr))
            .field("MapCount", &self.mem_info(MemInfo::MapCount))
            .field("ReferenceCount", &self.mem_info(MemInfo::ReferenceCount))
            .field("Context", &self.mem_info(MemInfo::Context))
            .field("AssociatedMemobject", &self.mem_info(MemInfo::AssociatedMemobject))
            .field("Offset", &self.mem_info(MemInfo::Offset))
            .finish()
    }
}

impl<T: OclPrm> Deref for Buffer<T> {
    type Target = MemCore;

    fn deref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<T: OclPrm> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}

impl<T: OclPrm> std::fmt::Display for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_mem_info(f)
    }
}

