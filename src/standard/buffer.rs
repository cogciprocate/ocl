//! Interfaces with a buffer.

use std;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use core::{self, OclPrm, Mem as MemCore, MemFlags,
    MemInfo, MemInfoResult, ClEventPtrNew, ClWaitList};
use error::{Error as OclError, Result as OclResult};
use standard::{Queue, MemLen, SpatialDims};
use ffi::ClGlUint;

fn check_len(mem_len: usize, data_len: usize, offset: usize) -> OclResult<()> {
    if offset >= mem_len { return OclError::err(format!(
        "ocl::Buffer::enq(): Offset out of range. (mem_len: {}, data_len: {}, offset: {}",
        mem_len, data_len, offset)); }
    if data_len > (mem_len - offset) { return OclError::err(
        "ocl::Buffer::enq(): Data length exceeds buffer length."); }
    Ok(())
}

/// The type of operation to be performed by a command.
pub enum BufferCmdKind<'b, T: 'b> {
    Unspecified,
    Read { data: &'b mut [T] },
    Write { data: &'b [T] },
    Copy { dst_buffer: &'b MemCore, dst_offset: usize, len: usize },
    Fill { pattern: T, len: Option<usize> },
    CopyToImage { image: &'b MemCore, dst_origin: [usize; 3], region: [usize; 3] },
    GLAcquire,
    GLRelease,
}

impl<'b, T: 'b> BufferCmdKind<'b, T> {
    fn is_unspec(&'b self) -> bool {
        if let &BufferCmdKind::Unspecified = self {
            true
        } else {
            false
        }
    }
}

/// The 'shape' of the data to be processed, whether one or multi-dimensional.
///
/// Should really be called dimensionality or something.
///
pub enum BufferCmdDataShape {
    Lin { offset: usize },
    Rect {
        src_origin: [usize; 3],
        dst_origin: [usize; 3],
        region: [usize; 3],
        src_row_pitch: usize,
        src_slc_pitch: usize,
        dst_row_pitch: usize,
        dst_slc_pitch: usize,
    },
}

/// A buffer command builder used to enqueue reads, writes, fills, and copies.
///
/// Create one using `Buffer::cmd` or with shortcut methods such as
/// `Buffer::read` and `Buffer::write`.
///
/// ## Examples
///
/// ```text
/// // Copies one buffer to another:
/// src_buffer.cmd().copy(&dst_buffer, 0, dst_buffer.len()).enq().unwrap();
///
/// // Writes from a vector to an buffer, waiting on an event:
/// buffer.write(&src_vec).ewait(&event).enq().unwrap();
///
/// // Reads from a buffer into a vector, waiting on an event list and
/// // filling a new empty event:
/// buffer.read(&dst_vec).ewait(&event_list).enew(&empty_event).enq().unwrap();
///
/// // Reads without blocking:
/// buffer.cmd().read_async(&dst_vec).enew(&empty_event).enq().unwrap();
///
/// ```
///
pub struct BufferCmd<'b, T: 'b + OclPrm> {
    queue: &'b Queue,
    obj_core: &'b MemCore,
    block: bool,
    lock_block: bool,
    kind: BufferCmdKind<'b, T>,
    shape: BufferCmdDataShape,
    ewait: Option<&'b ClWaitList>,
    enew: Option<&'b mut ClEventPtrNew>,
    mem_len: usize,
}

/// [UNSTABLE]: All methods still in a state of tweakification.
impl<'b, T: 'b + OclPrm> BufferCmd<'b, T> {
    /// Returns a new buffer command builder associated with with the
    /// memory object `obj_core` along with a default `queue` and `mem_len`
    /// (the length of the device side buffer).
    pub fn new(queue: &'b Queue, obj_core: &'b MemCore, mem_len: usize)
            -> BufferCmd<'b, T>
    {
        BufferCmd {
            queue: queue,
            obj_core: obj_core,
            block: true,
            lock_block: false,
            kind: BufferCmdKind::Unspecified,
            shape: BufferCmdDataShape::Lin { offset: 0 },
            ewait: None,
            enew: None,
            mem_len: mem_len,
        }
    }

    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'b Queue) -> BufferCmd<'b, T> {
        self.queue = queue;
        self
    }

    /// Specifies whether or not to block thread until completion.
    ///
    /// Ignored if this is a copy, fill, or copy to image operation.
    ///
    /// ## Panics
    ///
    /// Will panic if `::read` has already been called. Use `::read_async`
    /// (unsafe) for a non-blocking read operation.
    ///
    pub fn block(mut self, block: bool) -> BufferCmd<'b, T> {
        if !block && self.lock_block {
            panic!("ocl::BufferCmd::block(): Blocking for this command has been disabled by \
                the '::read' method. For non-blocking reads use '::read_async'.");
        }
        self.block = block;
        self
    }

    /// Sets the linear offset for an operation.
    ///
    /// ## Panics
    ///
    /// The 'shape' may not have already been set to rectangular by the
    /// `::rect` function.
    pub fn offset(mut self, offset: usize) -> BufferCmd<'b, T> {
        if let BufferCmdDataShape::Rect { .. } = self.shape {
            panic!("ocl::BufferCmd::offset(): This command builder has already been set to \
                rectangular mode with '::rect`. You cannot call both '::offset' and '::rect'.");
        }

        self.shape = BufferCmdDataShape::Lin { offset: offset };
        self
    }

    /// Specifies that this command will be a blocking read operation.
    ///
    /// After calling this method, the blocking state of this command will
    /// be locked to true and a call to `::block` will cause a panic.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified.
    ///
    pub fn read(mut self, dst_data: &'b mut [T]) -> BufferCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::read(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Read { data: dst_data };
        self.block = true;
        self.lock_block = true;
        self
    }

    /// Specifies that this command will be a non-blocking, asynchronous read
    /// operation.
    ///
    /// Sets the block mode to false automatically but it may still be freely
    /// toggled back. If set back to `true` this method call becomes equivalent
    /// to calling `::read`.
    ///
    /// ## Safety
    ///
    /// Caller must ensure that the container referred to by `dst_data` lives
    /// until the call completes.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub unsafe fn read_async(mut self, dst_data: &'b mut [T]) -> BufferCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::read(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Read { data: dst_data };
        self
    }

    /// Specifies that this command will be a write operation.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn write(mut self, src_data: &'b [T]) -> BufferCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::write(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Write { data: src_data };
        self
    }

    /// Specifies that this command will be a copy operation.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Errors
    ///
    /// If this is a rectangular copy, `dst_offset` and `len` must be zero.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy(mut self, dst_buffer: &'b Buffer<T>, dst_offset: usize, len: usize)
            -> BufferCmd<'b, T>
    {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::copy(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Copy {
            dst_buffer: dst_buffer.core_as_ref(),
            dst_offset: dst_offset,
            len: len,
        };
        self
    }

    /// Specifies that this command will acquire a GL buffer.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn gl_acquire(mut self) -> BufferCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::gl_acquire(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::GLAcquire;
        self
    }

    /// Specifies that this command will release a GL buffer.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn gl_release(mut self) -> BufferCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::gl_release(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::GLRelease;
        self
    }

    /// Specifies that this command will be a copy to image.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy_to_image(mut self, image: &'b MemCore, dst_origin: [usize; 3],
                region: [usize; 3]) -> BufferCmd<'b, T>
    {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::copy_to_image(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::CopyToImage { image: image, dst_origin: dst_origin, region: region };
        self
    }

    /// Specifies that this command will be a fill.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// `pattern` is the vector or scalar value to repeat contiguously. `len`
    /// is the overall size expressed in units of sizeof(T) If `len` is `None`,
    /// the pattern will fill the entire buffer, otherwise, `len` must be
    /// divisible by sizeof(`pattern`).
    ///
    /// As an example if you want to fill the first 100 `cl_float4` sized
    /// elements of a buffer, `pattern` would be a `cl_float4` and `len` would
    /// be 400.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn fill(mut self, pattern: T, len: Option<usize>) -> BufferCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::fill(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Fill { pattern: pattern, len: len };
        self
    }

    /// Specifies that this will be a rectangularly shaped operation
    /// (the default being linear).
    ///
    /// Only valid for 'read', 'write', and 'copy' modes. Will error if used
    /// with 'fill' or 'copy to image'.
    pub fn rect(mut self, src_origin: [usize; 3], dst_origin: [usize; 3], region: [usize; 3],
                src_row_pitch: usize, src_slc_pitch: usize, dst_row_pitch: usize,
                dst_slc_pitch: usize) -> BufferCmd<'b, T>
    {
        if let BufferCmdDataShape::Lin { offset } = self.shape {
            assert!(offset == 0, "ocl::BufferCmd::rect(): This command builder has already been \
                set to linear mode with '::offset`. You cannot call both '::offset' and '::rect'.");
        }

        self.shape = BufferCmdDataShape::Rect { src_origin: src_origin, dst_origin: dst_origin,
            region: region, src_row_pitch: src_row_pitch, src_slc_pitch: src_slc_pitch,
            dst_row_pitch: dst_row_pitch, dst_slc_pitch: dst_slc_pitch };
        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait(mut self, ewait: &'b ClWaitList) -> BufferCmd<'b, T> {
        self.ewait = Some(ewait);
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt(mut self, ewait: Option<&'b ClWaitList>) -> BufferCmd<'b, T> {
        self.ewait = ewait;
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew(mut self, enew: &'b mut ClEventPtrNew) -> BufferCmd<'b, T> {
        self.enew = Some(enew);
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt(mut self, enew: Option<&'b mut ClEventPtrNew>) -> BufferCmd<'b, T> {
        self.enew = enew;
        self
    }

    // core::enqueue_copy_buffer::<f32, core::EventList>(&queue, &src_buffer, &dst_buffer,
    //     copy_range.0, copy_range.0, copy_range.1 - copy_range.0, None,
    //     None).unwrap();

    /// Enqueues this command.
    pub fn enq(self) -> OclResult<()> {
        match self.kind {
            BufferCmdKind::Read { data } => {
                match self.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.mem_len, data.len(), offset));

                        unsafe { core::enqueue_read_buffer(self.queue, self.obj_core, self.block,
                            offset, data, self.ewait, self.enew) }
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                            dst_row_pitch, dst_slc_pitch } =>
                    {
                        // Verify dims given.
                        // try!(Ok(()));

                        unsafe { core::enqueue_read_buffer_rect(self.queue, self.obj_core,
                            self.block, src_origin, dst_origin, region, src_row_pitch,
                            src_slc_pitch, dst_row_pitch, dst_slc_pitch, data,
                            self.ewait, self.enew) }
                    }
                }
            },
            BufferCmdKind::Write { data } => {
                match self.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.mem_len, data.len(), offset));
                        core::enqueue_write_buffer(self.queue, self.obj_core, self.block,
                            offset, data, self.ewait, self.enew)
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                            dst_row_pitch, dst_slc_pitch } =>
                    {
                        // Verify dims given.
                        // try!(Ok(()));

                        core::enqueue_write_buffer_rect(self.queue, self.obj_core,
                            self.block, src_origin, dst_origin, region, src_row_pitch,
                            src_slc_pitch, dst_row_pitch, dst_slc_pitch, data,
                            self.ewait, self.enew)
                    }
                }
            },
            BufferCmdKind::Copy { dst_buffer, dst_offset, len } => {
                match self.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.mem_len, len, offset));
                        core::enqueue_copy_buffer::<T>(self.queue,
                            self.obj_core, dst_buffer, offset, dst_offset, len,
                            self.ewait, self.enew)
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                            dst_row_pitch, dst_slc_pitch } =>
                    {
                        // Verify dims given.
                        // try!(Ok(()));

                        if dst_offset != 0 || len != 0 { return OclError::err(
                            "ocl::BufferCmd::enq(): For 'rect' shaped copies, destination \
                            offset and length must be zero. Ex.: \
                            'cmd().copy(&{{buf_name}}, 0, 0)..'.");
                        }
                        core::enqueue_copy_buffer_rect::<T>(self.queue, self.obj_core, dst_buffer,
                        src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                        dst_row_pitch, dst_slc_pitch, self.ewait, self.enew)
                    },
                }
            },
            BufferCmdKind::Fill { pattern, len } => {
                match self.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        let len = match len {
                            Some(l) => l,
                            None => self.mem_len,
                        };
                        try!(check_len(self.mem_len, len, offset));
                        core::enqueue_fill_buffer(self.queue, self.obj_core, pattern,
                            offset, len, self.ewait, self.enew)
                    },
                    BufferCmdDataShape::Rect { .. } => {
                        return OclError::err("ocl::BufferCmd::enq(): Rectangular fill is not a \
                            valid operation. Please use the default shape, linear.");
                    }
                }
            },
            BufferCmdKind::GLAcquire => {
                core::enqueue_acquire_gl_buffer::<T>(self.queue, self.obj_core, self.ewait, self.enew)
            },
            BufferCmdKind::GLRelease => {
                core::enqueue_release_gl_buffer::<T>(self.queue, self.obj_core, self.ewait, self.enew)
            },
            BufferCmdKind::Unspecified => return OclError::err("ocl::BufferCmd::enq(): No operation \
                specified. Use '.read(...)', 'write(...)', etc. before calling '.enq()'."),
            _ => unimplemented!(),
        }
    }
}

/// A chunk of memory physically located on a device, such as a GPU.
///
/// Data is stored remotely in a memory buffer on the device associated with
/// `queue`.
///
#[derive(Debug, Clone)]
pub struct Buffer<T: OclPrm> {
    obj_core: MemCore,
    queue: Queue,
    dims: SpatialDims,
    len: usize,
    _data: PhantomData<T>,
}

impl<T: OclPrm> Buffer<T> {
    /// Creates a new buffer
    ///
    /// [UNSTABLE]: New method, arguments still in a state of flux.
    pub fn new<D: MemLen>(queue: &Queue, flags: Option<MemFlags>, dims: D, data: Option<&[T]>)
            -> OclResult<Buffer<T>> {
        let flags = flags.unwrap_or(core::MEM_READ_WRITE);
        let dims: SpatialDims = dims.to_lens().into();
        // let len = dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");
        let len = dims.to_len();
        let obj_core = unsafe { try!(core::create_buffer(queue.context_core_as_ref(), flags, len, data)) };

        let buf = Buffer {
            obj_core: obj_core,
            queue: queue.clone(),
            dims: dims,
            len: len,
            _data: PhantomData,
        };

        // if data.is_none() { try!(buf.cmd().fill(&[Default::default()], None).enq()); }
        if data.is_none() { try!(buf.cmd().fill(Default::default(), None).enq()); }
        Ok(buf)
    }

    /// [UNTESTED]
    /// Create a buffer linked to a GL buffer object (created with openGL)
    ///
    /// ## Errors
    ///
    /// Don't forget to `.cmd().gl_acquire().enq()` before using it and
    /// `.cmd().gl_release().enq()` after.
    ///
    /// See the [`BufferCmd` docs](/ocl/ocl/build/struct.BufferCmd.html)
    /// for more info.
    pub fn from_gl_buffer<D: MemLen>(queue: &Queue, flags: Option<MemFlags>, dims: D, gl_object: ClGlUint)
            -> OclResult<Buffer<T>> {
        let flags = flags.unwrap_or(core::MEM_READ_WRITE);
        let dims: SpatialDims = dims.to_lens().into();
        let len = dims.to_len();
        let obj_core = unsafe { try!(core::create_from_gl_buffer::<T>(queue.context_core_as_ref(),
            gl_object, flags)) };

        let buf = Buffer {
            obj_core: obj_core,
            queue: queue.clone(),
            dims: dims,
            len: len,
            _data: PhantomData,
        };

        Ok(buf)
    }

    /// Returns a buffer command builder used to read, write, copy, etc.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [`BufferCmd` docs](/ocl/ocl/build/struct.BufferCmd.html)
    /// for more info.
    ///
    pub fn cmd<'b>(&'b self) -> BufferCmd<'b, T> {
        BufferCmd::new(&self.queue, &self.obj_core, self.len)
    }

    /// Returns a buffer command builder used to read.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [`BufferCmd` docs](/ocl/ocl/build/struct.BufferCmd.html)
    /// for more info.
    ///
    pub fn read<'b>(&'b self, data: &'b mut [T]) -> BufferCmd<'b, T> {
        self.cmd().read(data)
    }

    /// Returns a buffer command builder used to write.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [`BufferCmd` docs](/ocl/ocl/build/struct.BufferCmd.html)
    /// for more info.
    ///
    pub fn write<'b>(&'b self, data: &'b [T]) -> BufferCmd<'b, T> {
        self.cmd().write(data)
    }

    /// Returns the length of the Buffer.
    #[inline]
    pub fn len(&self) -> usize {
        // debug_assert!((if let VecOption::Some(ref vec) = self.vec { vec.len() }
        //     else { self.len }) == self.len);
        self.len
    }

    /// Returns info about the underlying memory object.
    pub fn mem_info(&self, info_kind: MemInfo) -> MemInfoResult {
        // match core::get_mem_object_info(&self.obj_core, info_kind) {
        //     Ok(res) => res,
        //     Err(err) => MemInfoResult::Error(Box::new(err)),
        // }
        core::get_mem_object_info(&self.obj_core, info_kind)
    }

    /// Changes the default queue used by this Buffer for reads and writes, etc.
    ///
    /// Returns a ref for chaining i.e.:
    ///
    /// ## Example
    ///
    /// `buffer.set_default_queue(queue).read(....);`
    ///
    pub fn set_default_queue<'a>(&'a mut self, queue: &Queue) -> &'a mut Buffer<T> {
        // [FIXME]: Set this up:
        // assert!(queue.device == self.queue.device);
        // [/FIXME]

        self.queue = queue.clone();
        self
    }

    /// Returns a reference to the default queue.
    pub fn default_queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    pub fn core_as_ref(&self) -> &MemCore {
        &self.obj_core
    }

    /// Formats memory info.
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

