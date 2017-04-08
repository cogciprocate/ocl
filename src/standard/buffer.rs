//! Interfaces with a buffer.

use std;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Range};
use ffi::cl_GLuint;
use core::{self, Error as OclError, Result as OclResult, OclPrm, Mem as MemCore,
    MemFlags, MemInfo, MemInfoResult, BufferRegion, MapFlags, AsMem, MemCmdRw,
    MemCmdAll, ClNullEventPtr};
use ::{Context, Queue, SpatialDims, FutureMemMap, MemMap, Event, RwVec, WriteGuard,
    FutureRwGuard, FutureReader, FutureWriter};
use standard::{ClNullEventPtrEnum, ClWaitListPtrEnum};


fn check_len(mem_len: usize, data_len: usize, offset: usize) -> OclResult<()> {
    if offset >= mem_len {
        OclError::err_string(format!("ocl::Buffer::enq(): Offset out of range. \
            (mem_len: {}, data_len: {}, offset: {}", mem_len, data_len, offset))
    } else if data_len > (mem_len - offset) {
        OclError::err_string("ocl::Buffer::enq(): Data length exceeds buffer length.")
    } else {
        Ok(())
    }
}

/// A queue or context reference.
#[derive(Debug, Clone)]
pub enum QueCtx<'o> {
    Queue(Queue),
    Context(&'o Context),
}

impl<'o> QueCtx<'o> {
    /// Returns a context regardless of the contained variant.
    pub fn context_cloned(&self) -> Context {
        match *self {
            QueCtx::Queue(ref q) => q.context(),
            QueCtx::Context(c) => c.clone(),
        }
    }
}

impl<'o> From<Queue> for QueCtx<'o> {
    fn from(q: Queue) -> QueCtx<'o> {
        QueCtx::Queue(q)
    }
}

impl<'a, 'o> From<&'a Queue> for QueCtx<'o> {
    fn from(q: &Queue) -> QueCtx<'o> {
        QueCtx::Queue(q.clone())
    }
}

impl<'o> From<&'o Context> for QueCtx<'o> {
    fn from(c: &'o Context) -> QueCtx<'o> {
        QueCtx::Context(c)
    }
}

impl<'o> From<QueCtx<'o>> for Option<Queue> {
    fn from(qc: QueCtx<'o>) -> Option<Queue> {
        match qc {
            QueCtx::Queue(q) => Some(q),
            QueCtx::Context(_) => None,
        }
    }
}


/// The type of operation to be performed by a command.
pub enum BufferCmdKind<'c, T> where T: 'c {
    Unspecified,
    Read,
    Write,
    Map,
    Copy { dst_buffer: &'c MemCore, dst_offset: Option<usize>, len: Option<usize> },
    Fill { pattern: T, len: Option<usize> },
    CopyToImage { image: &'c MemCore, dst_origin: [usize; 3], region: [usize; 3] },
    GLAcquire,
    GLRelease,
}

impl<'c, T> BufferCmdKind<'c, T> {
    fn is_unspec(&'c self) -> bool {
        if let BufferCmdKind::Unspecified = *self {
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
        src_row_pitch_bytes: usize,
        src_slc_pitch_bytes: usize,
        dst_row_pitch_bytes: usize,
        dst_slc_pitch_bytes: usize,
    },
}

/// A buffer command builder used to enqueue reads, writes, fills, and copies.
///
/// Create one by using `Buffer::cmd` or with shortcut methods such as
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
#[must_use = "commands do nothing unless enqueued"]
pub struct BufferCmd<'c, T> where T: 'c {
    queue: Option<&'c Queue>,
    obj_core: &'c MemCore,
    block: bool,
    kind: BufferCmdKind<'c, T>,
    shape: BufferCmdDataShape,
    ewait: Option<ClWaitListPtrEnum<'c>>,
    enew: Option<ClNullEventPtrEnum<'c>>,
    mem_len: usize,
}

/// [UNSTABLE]: All methods still in a state of flux.
impl<'c, T> BufferCmd<'c, T> where T: 'c + OclPrm {
    /// Returns a new buffer command builder associated with with the
    /// memory object `obj_core` along with a default `queue` and `mem_len`
    /// (the length of the device side buffer).
    pub fn new(queue: Option<&'c Queue>, obj_core: &'c MemCore, mem_len: usize)
            -> BufferCmd<'c, T>
    {
        BufferCmd {
            queue: queue,
            obj_core: obj_core,
            block: true,
            kind: BufferCmdKind::Unspecified,
            shape: BufferCmdDataShape::Lin { offset: 0 },
            ewait: None,
            enew: None,
            mem_len: mem_len,
        }
    }

    /// Specifies that this command will be a blocking read operation.
    ///
    /// After calling this method, the blocking state of this command will
    /// be locked to true and a call to `::block` will cause a panic.
    ///
    /// ### Panics
    ///
    /// The command operation kind must not have already been specified.
    ///
    /// ### More Information
    ///
    /// See [SDK][read_buffer] docs for more details.
    ///
    /// [read_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueReadBuffer.html
    pub fn read<'d, R>(mut self, dst_data: R) -> BufferReadCmd<'c, 'd, T>
            where R: Into<ReadDst<'d, T>>
    {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::read(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Read;
        let dst = dst_data.into();
        let len = dst.len();
        BufferReadCmd { cmd: self, dst: dst, range: 0..len }
    }

    /// Specifies that this command will be a non-blocking, asynchronous read
    /// operation. [DEPRICATED]
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
    #[deprecated(since="0.13.0", note="Use '::read' with '::block(false)' for unsafe asynchronous reads.")]
    pub unsafe fn read_async<'d>(mut self, dst_data: &'d mut [T]) -> BufferReadCmd<'c, 'd, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::read(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Read;
        self.block = false;
        let len = dst_data.len();
        BufferReadCmd { cmd: self, dst: dst_data.into(), range: 0..len }
    }

    /// Specifies that this command will be a write operation.
    ///
    /// ### Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    /// ### More Information
    ///
    /// See [SDK][write_buffer] docs for more details.
    ///
    /// [write_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueWriteBuffer.html
    pub fn write<'d, W>(mut self, src_data: W) -> BufferWriteCmd<'c, 'd, T> 
            where W: Into<WriteSrc<'d, T>>
    {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::write(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Write;
        let src = src_data.into();
        let len = src.len();
        BufferWriteCmd { cmd: self, src: src, range: 0..len }
    }

    /// Specifies that this command will be a map operation.
    ///
    /// If `.block(..)` has been set it will be ignored. Non-blocking map
    /// commands are enqueued using `::enq_async`.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    /// ### More Information
    ///
    /// See [SDK][map_buffer] docs for more details.
    ///
    /// [map_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueMapBuffer.html
    pub fn map(mut self) -> BufferMapCmd<'c, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::write(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Map;
        BufferMapCmd { cmd: self, flags: None, len: None }
    }


    /// Specifies that this command will be a copy operation.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// `dst_offset` defaults to `0`, `len` defaults to the full length of the
    /// source buffer.
    ///
    /// ## Errors
    ///
    /// If this is a rectangular copy, `dst_offset` and `len` must be None.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    /// ### More Information
    ///
    /// See [SDK][copy_buffer] docs for more details.
    ///
    /// [copy_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueCopyBuffer.html
    pub fn copy<'d, M>(mut self, dst_buffer: &'d M, dst_offset: Option<usize>, len: Option<usize>)
            -> BufferCmd<'c, T>
            where 'd: 'c, M: AsMem<T>
    {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::copy(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Copy {
            dst_buffer: dst_buffer.as_mem(),
            dst_offset: dst_offset,
            len: len,
        };
        self
    }

    /// Specifies that this command will be a copy to image operation.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy_to_image<'d>(mut self, image: &'d MemCore, dst_origin: [usize; 3],
            region: [usize; 3]) -> BufferCmd<'c, T>
            where 'd: 'c,
    {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::copy_to_image(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::CopyToImage { image: image, dst_origin: dst_origin, region: region };
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
    pub fn gl_acquire(mut self) -> BufferCmd<'c, T> {
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
    pub fn gl_release(mut self) -> BufferCmd<'c, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::gl_release(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::GLRelease;
        self
    }

    /// Specifies that this command will be a fill operation.
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
    pub fn fill(mut self, pattern: T, len: Option<usize>) -> BufferCmd<'c, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::fill(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Fill { pattern: pattern, len: len };
        self
    }

    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'c Queue) -> BufferCmd<'c, T> {
        self.queue = Some(queue);
        self
    }

    /// Specifies whether or not to block the current thread until completion.
    ///
    /// Ignored if this is not a read or write operation.
    ///
    /// Default is `block = true`.
    ///
    /// ## Safety
    ///
    /// When performing non-blocking reads or writes, the caller must ensure
    /// that the data being read from or written to is not accessed improperly
    /// until the command completes. Use events (`Event::wait_for`) or the
    /// command queue (`Queue::finish`) to synchronize.
    ///
    /// If possible, prefer instead to use [`::map`] with [`::enq_async`] for
    /// optimal performance and data integrity.
    ///
    /// [`::map`]: struct.BufferMapCmd.html
    /// [`::enq_async`]: struct.BufferMapCmd.html#method.enq_async
    //
    // [FIXME]: Check/fix links.
    //
    // * TODO: Should this error when calling non-rw `::enq`?
    //
    pub unsafe fn block(mut self, block: bool) -> BufferCmd<'c, T> {
        self.block = block;
        self
    }

    /// Sets the linear offset for an operation.
    ///
    /// ## Panics
    ///
    /// The 'shape' may not have already been set to rectangular by the
    /// `::rect` function.
    pub fn offset(mut self, offset: usize)  -> BufferCmd<'c, T> {
        if let BufferCmdDataShape::Rect { .. } = self.shape {
            panic!("ocl::BufferCmd::offset(): This command builder has already been set to \
                rectangular mode with '::rect`. You cannot call both '::offset' and '::rect'.");
        }

        self.shape = BufferCmdDataShape::Lin { offset: offset };
        self
    }

    /// Specifies that this will be a rectangularly shaped operation
    /// (the default being linear).
    ///
    /// Row and slice pitches must all be expressed in bytes.
    ///
    /// Only valid for 'read', 'write', and 'copy' modes. Will error if used
    /// with any other mode.
    pub fn rect(mut self, src_origin: [usize; 3], dst_origin: [usize; 3], region: [usize; 3],
                src_row_pitch_bytes: usize, src_slc_pitch_bytes: usize, dst_row_pitch_bytes: usize,
                dst_slc_pitch_bytes: usize) -> BufferCmd<'c, T>
    {
        if let BufferCmdDataShape::Lin { offset } = self.shape {
            assert!(offset == 0, "ocl::BufferCmd::rect(): This command builder has already been \
                set to linear mode with '::offset`. You cannot call both '::offset' and '::rect'.");
        }

        self.shape = BufferCmdDataShape::Rect { src_origin: src_origin, dst_origin: dst_origin,
            region: region, src_row_pitch_bytes: src_row_pitch_bytes,
            src_slc_pitch_bytes: src_slc_pitch_bytes, dst_row_pitch_bytes: dst_row_pitch_bytes,
            dst_slc_pitch_bytes: dst_slc_pitch_bytes };
        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> BufferCmd<'c, T>
            where 'e: 'c, Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.ewait = Some(ewait.into());
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt<'e, Ewl>(mut self, ewait: Option<Ewl>) -> BufferCmd<'c, T>
            where 'e: 'c, Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.ewait = ewait.map(|el| el.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew<'e, En>(mut self, enew: En) -> BufferCmd<'c, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.enew = Some(enew.into());
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt<'e, En>(mut self, enew: Option<En>) -> BufferCmd<'c, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.enew = enew.map(|e| e.into());
        self
    }

    /// Enqueues this command.
    pub fn enq(self) -> OclResult<()> {
        let queue = match self.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        match self.kind {
            BufferCmdKind::Copy { dst_buffer, dst_offset, len } => {
                match self.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        let len = len.unwrap_or(self.mem_len);
                        try!(check_len(self.mem_len, len, offset));
                        let dst_offset = dst_offset.unwrap_or(0);

                        core::enqueue_copy_buffer::<T, _, _, _>(queue,
                            self.obj_core, dst_buffer, offset, dst_offset, len,
                            self.ewait, self.enew)
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region,
                        src_row_pitch_bytes, src_slc_pitch_bytes, dst_row_pitch_bytes,
                        dst_slc_pitch_bytes } =>
                    {
                        if dst_offset.is_some() || len.is_some() { return OclError::err_string(
                            "ocl::BufferCmd::enq(): For 'rect' shaped copies, destination \
                            offset and length must be 'None'. Ex.: \
                            'cmd().copy(&{{buf_name}}, None, None)..'.");
                        }

                        core::enqueue_copy_buffer_rect::<T, _, _, _>(queue, self.obj_core,
                            dst_buffer, src_origin, dst_origin, region, src_row_pitch_bytes,
                            src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes,
                            self.ewait, self.enew)
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

                        core::enqueue_fill_buffer(queue, self.obj_core, pattern,
                            offset, len, self.ewait, self.enew, Some(&queue.device_version()))
                    },
                    BufferCmdDataShape::Rect { .. } => OclError::err_string(
                        "ocl::BufferCmd::enq(): Rectangular fill is not a valid operation. \
                        Please use the default shape, linear.")
                }
            },
            BufferCmdKind::GLAcquire => {
                core::enqueue_acquire_gl_buffer(queue, self.obj_core, self.ewait, self.enew)
            },
            BufferCmdKind::GLRelease => {
                core::enqueue_release_gl_buffer(queue, self.obj_core, self.ewait, self.enew)
            },
            BufferCmdKind::Unspecified => OclError::err_string("ocl::BufferCmd::enq(): \
                No operation specified. Use '.read(...)', 'write(...)', etc. before calling \
                '.enq()'."),
            BufferCmdKind::Map { .. } => unreachable!(),
            _ => unimplemented!(),
        }
    }
}


/// The data destination for a buffer read command.
pub enum ReadDst<'d, T> where T: 'd {
    Slice(&'d mut [T]),
    RwVec(RwVec<T>),
    Writer(FutureWriter<T>),
    None,
}

impl<'d, T> ReadDst<'d, T> {
    fn take(&mut self) -> ReadDst<'d, T> {
        ::std::mem::replace(self, ReadDst::None)
    }

    pub fn len(&self) -> usize {
        match *self {
            ReadDst::RwVec(ref rw_vec) => rw_vec.len(),
            ReadDst::Writer(ref writer) => writer.len(),
            ReadDst::Slice(ref slice) => slice.len(),
            ReadDst::None => 0,
        }
    }
}

impl<'d, T> From<&'d mut [T]> for ReadDst<'d, T>  where T: OclPrm {
    fn from(slice: &'d mut [T]) -> ReadDst<'d, T> {
        ReadDst::Slice(slice)
    }
}

impl<'d, T> From<&'d mut Vec<T>> for ReadDst<'d, T>  where T: OclPrm {
    fn from(vec: &'d mut Vec<T>) -> ReadDst<'d, T> {
        ReadDst::Slice(vec.as_mut_slice())
    }
}

impl<'d, T> From<RwVec<T>> for ReadDst<'d, T> where T: OclPrm {
    fn from(rw_vec: RwVec<T>) -> ReadDst<'d, T> {
        ReadDst::RwVec(rw_vec)
    }
}

impl<'a, 'd, T> From<&'a RwVec<T>> for ReadDst<'d, T> where T: OclPrm {
    fn from(rw_vec: &'a RwVec<T>) -> ReadDst<'d, T> {
        ReadDst::RwVec(rw_vec.clone())
    }
}

impl<'d, T> From<FutureWriter<T>> for ReadDst<'d, T> where T: OclPrm {
    fn from(writer: FutureWriter<T>) -> ReadDst<'d, T> {
        ReadDst::Writer(writer)
    }
}


/// A buffer command builder used to enqueue reads.
///
/// See [SDK][read_buffer] docs for more details.
///
/// [read_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueReadBuffer.html
#[must_use = "commands do nothing unless enqueued"]
pub struct BufferReadCmd<'c, 'd, T> where T: 'c + 'd {
    cmd: BufferCmd<'c, T>,
    dst: ReadDst<'d, T>,
    range: Range<usize>,
}

impl<'c, 'd, T> BufferReadCmd<'c, 'd, T> where T: OclPrm {
    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'c Queue) -> BufferReadCmd<'c, 'd, T> {
        self.cmd.queue = Some(queue);
        self
    }

    /// Specifies whether or not to block the current thread until completion.
    ///
    /// Ignored if this is not a read or write operation.
    ///
    /// Default is `block = true`.
    ///
    /// ## Safety
    ///
    /// When performing non-blocking reads or writes, the caller must ensure
    /// that the data being read from or written to is not accessed improperly
    /// until the command completes. Use events (`Event::wait_for`) or the
    /// command queue (`Queue::finish`) to synchronize.
    ///
    /// If possible, prefer instead to use [`::map`] with [`::enq_async`] for
    /// optimal performance and data integrity.
    ///
    /// [`::map`]: struct.BufferMapCmd.html
    /// [`::enq_async`]: struct.BufferMapCmd.html
    //
    // [FIXME]: Check/fix links.
    //
    pub unsafe fn block(mut self, block: bool) -> BufferReadCmd<'c, 'd, T> {
        self.cmd.block = block;
        self
    }

    /// Sets the linear offset for an operation.
    ///
    /// ## Panics
    ///
    /// The 'shape' may not have already been set to rectangular by the
    /// `::rect` function.
    pub fn offset(self, offset: usize) -> BufferReadCmd<'c, 'd, T> {
        BufferReadCmd { cmd: self.cmd.offset(offset), ..self }
    }

    /// Sets an offset into the destination data.
    ///
    /// Equivalent to setting the start position of a slice into the
    /// destination data (e.g. `dst_data[dst_offset..]`). Use `::len` to set
    /// the end position (resulting in `dst_data[dst_offset..len]`).
    ///
    /// Defaults to 0 if not set. Panics if `::rect` has been called.
    pub fn dst_offset(mut self, dst_offset: usize) -> BufferReadCmd<'c, 'd, T> {
        if let BufferCmdDataShape::Rect { .. } = self.cmd.shape {
            panic!("Cannot set a destination offset for a rectangular read.");
        }
        self.range.end = dst_offset + self.range.len();
        self.range.start = dst_offset;        
        self
    }

    /// Sets the total length of data to read.
    ///
    /// Equivalent to setting the end position of a slice into the destination
    /// data (e.g. `destination[..len]`). Use `::dst_offset` to set the start
    /// position (resulting in `dst_data[dst_offset..len]`).
    ///
    /// Defaults to the total length of the read destination provided. Panics
    /// if `::rect` has been called.
    pub fn len(mut self, len: usize) -> BufferReadCmd<'c, 'd, T> {
        if let BufferCmdDataShape::Rect { .. } = self.cmd.shape {
            panic!("Cannot set a length for a rectangular read.");
        }
        self.range.end = self.range.start + len;
        self
    }

    /// Specifies that this will be a rectangularly shaped operation
    /// (the default being linear).
    ///
    /// Row and slice pitches must all be expressed in bytes.
    ///
    /// Panics if `:offset`, `dst_offset`, or `::len` have been called.
    pub fn rect(mut self, src_origin: [usize; 3], dst_origin: [usize; 3], region: [usize; 3],
                src_row_pitch_bytes: usize, src_slc_pitch_bytes: usize, dst_row_pitch_bytes: usize,
                dst_slc_pitch_bytes: usize) -> BufferReadCmd<'c, 'd, T>
    {
        if let BufferCmdDataShape::Lin { offset } = self.cmd.shape {
            assert!(offset == 0, "ocl::BufferCmd::rect(): This command builder has already been \
                set to linear mode with '::offset`. You cannot call both '::offset' and '::rect'.");
        }
        if self.range.len() != self.dst.len() {
            panic!("Buffer read: Cannot call '::rect' after calling '::src_offset' or '::len'.");
        }

        self.cmd.shape = BufferCmdDataShape::Rect { src_origin: src_origin, dst_origin: dst_origin,
            region: region, src_row_pitch_bytes: src_row_pitch_bytes,
            src_slc_pitch_bytes: src_slc_pitch_bytes, dst_row_pitch_bytes: dst_row_pitch_bytes,
            dst_slc_pitch_bytes: dst_slc_pitch_bytes };

        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> BufferReadCmd<'c, 'd, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.cmd.ewait = Some(ewait.into());
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt<'e, Ewl>(mut self, ewait: Option<Ewl>) -> BufferReadCmd<'c, 'd, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.cmd.ewait = ewait.map(|el| el.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew<'e, En>(mut self, enew: En) -> BufferReadCmd<'c, 'd, T>
            // where E: Into<ClNullEventPtrEnum<'e>>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.cmd.enew = Some(enew.into());
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt<'e, En>(mut self, enew: Option<En>) -> BufferReadCmd<'c, 'd, T>
            // where E: Into<ClNullEventPtrEnum<'e>>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.cmd.enew = enew.map(|e| e.into());
        self
    }

    /// Enqueues this command, blocking the current thread until it is complete.
    ///
    /// If an `RwVec` is being used as the data destination, the current
    /// thread will be blocked until an exclusive lock can be obtained before
    /// running the command (which will also block for its duration).
    pub fn enq(mut self) -> OclResult<()> {
        let read_dst = self.dst.take();
        let range = self.range.clone();
        if range.end > read_dst.len() { return Err(OclError::from(
            "Unable to enqueue buffer read command: Invalid src_offset and/or len.")) }

        let mut enqueue_with_data = |dst: &mut [T]| {
            let queue = match self.cmd.queue {
                Some(q) => q,
                None => return Err("BufferReadCmd::enq: No queue set.".into()),
            };

            match self.cmd.kind {
                BufferCmdKind::Read => {
                    match self.cmd.shape {
                        BufferCmdDataShape::Lin { offset } => {
                            try!(check_len(self.cmd.mem_len, dst.len(), offset));

                            unsafe { core::enqueue_read_buffer(queue, self.cmd.obj_core, self.cmd.block,
                                offset, dst, self.cmd.ewait.take(), self.cmd.enew.take()) }
                        },
                        BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch_bytes, src_slc_pitch_bytes,
                                dst_row_pitch_bytes, dst_slc_pitch_bytes } =>
                        {
                            // TODO: Verify dims given (like `::check_len`).
                            unsafe { core::enqueue_read_buffer_rect(queue, self.cmd.obj_core,
                                self.cmd.block, src_origin, dst_origin, region, src_row_pitch_bytes,
                                src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes, dst,
                                self.cmd.ewait.take(), self.cmd.enew.take()) }
                        }
                    }
                },
                _ => unreachable!(),
            }
        };

        match read_dst {
            ReadDst::Slice(slice) => {
                enqueue_with_data(&mut slice[range])
            },
            ReadDst::RwVec(rw_vec) => {
                let mut guard = rw_vec.request_write().wait()
                    .map_err(|_| OclError::from("Unable to obtain lock."))?;
                enqueue_with_data(&mut guard.as_mut_slice()[range])
            },
            ReadDst::Writer(writer) => {
                let mut guard = writer.wait()
                    .map_err(|_| OclError::from("Unable to obtain lock."))?;
                enqueue_with_data(&mut guard.as_mut_slice()[range])
            }
            ReadDst::None => panic!("Invalid read destination."),
        }
    }

    /// Enqueues this command and returns a future representing its completion
    /// which resolves to a guard providing exclusive data access usable
    /// within subsequent futures.
    ///
    /// A data destination container appropriate for an asynchronous operation
    /// (such as `RwVec`) must have been passed to `::read`.
    ///
    pub fn enq_async(mut self) -> OclResult<FutureRwGuard<T, WriteGuard<T>>> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        // let rw_vec = match self.dst {
        //     ReadDst::RwVec(rw_vec) => rw_vec,
        //     _ => return Err("BufferReadCmd::enq_async: Invalid data destination kind for an
        //         asynchronous enqueue. The read destination must be a 'RwVec'.".into()),
        // };

        match self.cmd.kind {
            BufferCmdKind::Read => {
                let mut writer = match self.dst {
                    ReadDst::RwVec(rw_vec) => rw_vec.request_write(),
                    ReadDst::Writer(writer) => writer,
                    _ => return Err("BufferReadCmd::enq_async: Invalid data destination kind for an
                        asynchronous enqueue. The read destination must be a 'RwVec'.".into()),
                };
                if self.range.end > writer.len() { return Err(OclError::from(
                    "Unable to enqueue buffer read command: Invalid src_offset and/or len.")) }

                writer.create_lock_event(queue.context_ptr()?)?;             

                if let Some(wl) = self.cmd.ewait {
                    writer.set_wait_list(wl);
                }

                let dst = unsafe { &mut writer.as_mut_slice().expect("BufferReadCmd::enq_async: \
                    Invalid writer.")[self.range] };

                let mut read_event = Event::empty();

                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.cmd.mem_len, dst.len(), offset));

                        unsafe { core::enqueue_read_buffer(queue, self.cmd.obj_core, false,
                            offset, dst, writer.lock_event(), Some(&mut read_event))?; }
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region,
                        src_row_pitch_bytes, src_slc_pitch_bytes,
                            dst_row_pitch_bytes, dst_slc_pitch_bytes } =>
                    {
                        unsafe { core::enqueue_read_buffer_rect(queue, self.cmd.obj_core,
                            false, src_origin, dst_origin, region, src_row_pitch_bytes,
                            src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes,
                            dst, writer.lock_event(), Some(&mut read_event))?; }
                    }
                }

                if let Some(ref mut enew) = self.cmd.enew.take() {
                    unsafe { enew.clone_from(&read_event) }
                }

                writer.set_command_completion_event(read_event);
                Ok(writer)
            },
            _ => unreachable!(),
        }
    }
}


/// The data destination for a buffer read command.
pub enum WriteSrc<'d, T> where T: 'd {
    Slice(&'d [T]),
    RwVec(RwVec<T>),
    Reader(FutureReader<T>),
    None,
}

impl<'d, T> WriteSrc<'d, T> {
    fn take(&mut self) -> WriteSrc<'d, T> {
        ::std::mem::replace(self, WriteSrc::None)
    }

    pub fn len(&self) -> usize {
        match *self {
            WriteSrc::RwVec(ref rw_vec) => rw_vec.len(),
            WriteSrc::Reader(ref writer) => writer.len(),
            WriteSrc::Slice(slice) => slice.len(),
            WriteSrc::None => 0,
        }
    }
}

impl<'d, T> From<&'d [T]> for WriteSrc<'d, T>  where T: OclPrm {
    fn from(slice: &'d [T]) -> WriteSrc<'d, T> {
        WriteSrc::Slice(slice)
    }
}

impl<'d, T> From<&'d Vec<T>> for WriteSrc<'d, T>  where T: OclPrm {
    fn from(vec: &'d Vec<T>) -> WriteSrc<'d, T> {
        WriteSrc::Slice(vec.as_slice())
    }
}

impl<'d, T> From<RwVec<T>> for WriteSrc<'d, T> where T: OclPrm {
    fn from(rw_vec: RwVec<T>) -> WriteSrc<'d, T> {
        WriteSrc::RwVec(rw_vec)
    }
}

impl<'a, 'd, T> From<&'a RwVec<T>> for WriteSrc<'d, T> where T: OclPrm {
    fn from(rw_vec: &'a RwVec<T>) -> WriteSrc<'d, T> {
        WriteSrc::RwVec(rw_vec.clone())
    }
}

impl<'d, T> From<FutureReader<T>> for WriteSrc<'d, T> where T: OclPrm {
    fn from(reader: FutureReader<T>) -> WriteSrc<'d, T> {
        WriteSrc::Reader(reader)
    }
}


/// A buffer command builder used to enqueue writes.
///
/// See [SDK][write_buffer] docs for more details.
///
/// [write_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueWriteBuffer.html
#[must_use = "commands do nothing unless enqueued"]
pub struct BufferWriteCmd<'c, 'd, T> where T: 'c + 'd {
    cmd: BufferCmd<'c, T>,
    src: WriteSrc<'d, T>,
    range: Range<usize>,
 }

impl<'c, 'd, T> BufferWriteCmd<'c, 'd, T> where T: OclPrm {
    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'c Queue) -> BufferWriteCmd<'c, 'd, T> {
        self.cmd.queue = Some(queue);
        self
    }

    /// Specifies whether or not to block the current thread until completion.
    ///
    /// Ignored if this is not a read or write operation.
    ///
    /// Default is `block = true`.
    ///
    /// ## Safety
    ///
    /// When performing non-blocking reads or writes, the caller must ensure
    /// that the data being read from or written to is not accessed improperly
    /// until the command completes. Use events (`Event::wait_for`) or the
    /// command queue (`Queue::finish`) to synchronize.
    ///
    /// If possible, prefer instead to use [`::map`] with [`::enq_async`] for
    /// optimal performance and data integrity.
    ///
    /// [`::map`]: struct.BufferMapCmd.html
    /// [`::enq_async`]: struct.BufferMapCmd.html
    //
    // [FIXME]: Check/fix links.
    //
    pub unsafe fn block(mut self, block: bool) -> BufferWriteCmd<'c, 'd, T> {
        self.cmd.block = block;
        self
    }

    /// Sets the linear offset for an operation.
    ///
    /// ## Panics
    ///
    /// The 'shape' may not have already been set to rectangular by the
    /// `::rect` function.
    pub fn offset(self, offset: usize)  -> BufferWriteCmd<'c, 'd, T> {
        BufferWriteCmd { cmd: self.cmd.offset(offset), ..self }
    }

    /// Sets an offset into the source data.
    ///
    /// Equivalent to setting the start position of a slice into the source
    /// data (e.g. `src_data[src_offset..]`). Use `::len` to set the end
    /// position (resulting in `src_data[dst_offset..len]`).
    ///
    /// Defaults to 0 if not set. Panics if `::rect` has been called.
    pub fn src_offset(mut self, src_offset: usize) -> BufferWriteCmd<'c, 'd, T> {
        if let BufferCmdDataShape::Rect { .. } = self.cmd.shape {
            panic!("Cannot set a source offset for a rectangular write.");
        }
        self.range.end = src_offset + self.range.len();
        self.range.start = src_offset;
        self
    }

    /// Sets the total length of data to write.
    ///
    /// Equivalent to setting the end position of a slice into the source
    /// data (e.g. `src_data[..len]`). Use `::src_offset` to set the
    /// start position (resulting in `src_data[src_offset..len]`).
    ///
    /// Defaults to the length of the write source provided. Panics if
    /// `::rect` has been called.
    pub fn len(mut self, len: usize) -> BufferWriteCmd<'c, 'd, T> {
        if let BufferCmdDataShape::Rect { .. } = self.cmd.shape {
            panic!("Cannot set a length for a rectangular write.");
        }
        self.range.end = self.range.start + len;
        self
    }

    /// Specifies that this will be a rectangularly shaped operation
    /// (the default being linear).
    ///
    /// Row and slice pitches must all be expressed in bytes.
    ///
    /// Panics if `:offset`, `src_offset`, or `::len` have been called.
    pub fn rect(mut self, src_origin: [usize; 3], dst_origin: [usize; 3], region: [usize; 3],
                src_row_pitch_bytes: usize, src_slc_pitch_bytes: usize, dst_row_pitch_bytes: usize,
                dst_slc_pitch_bytes: usize) -> BufferWriteCmd<'c, 'd, T>
    {
        if let BufferCmdDataShape::Lin { offset } = self.cmd.shape {
            assert!(offset == 0, "ocl::BufferCmd::rect(): This command builder has already been \
                set to linear mode with '::offset`. You cannot call both '::offset' and '::rect'.");
        }
        if self.range.len() != self.src.len() {
            panic!("Buffer write: Cannot call '::rect' after calling '::src_offset' or '::len'.");
        }

        self.cmd.shape = BufferCmdDataShape::Rect { src_origin: src_origin, dst_origin: dst_origin,
            region: region, src_row_pitch_bytes: src_row_pitch_bytes,
            src_slc_pitch_bytes: src_slc_pitch_bytes, dst_row_pitch_bytes: dst_row_pitch_bytes,
            dst_slc_pitch_bytes: dst_slc_pitch_bytes };

        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> BufferWriteCmd<'c, 'd, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.cmd.ewait = Some(ewait.into());
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt<'e, Ewl>(mut self, ewait: Option<Ewl>) -> BufferWriteCmd<'c, 'd, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.cmd.ewait = ewait.map(|el| el.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew<'e, En>(mut self, enew: En) -> BufferWriteCmd<'c, 'd, T>
            // where E: Into<ClNullEventPtrEnum<'e>>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.cmd.enew = Some(enew.into());
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt<'e, En>(mut self, enew: Option<En>) -> BufferWriteCmd<'c, 'd, T>
            // where E: Into<ClNullEventPtrEnum<'e>>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.cmd.enew = enew.map(|e| e.into());
        self
    }

    /// Enqueues this command, blocking the current thread until it is complete.
    ///
    /// If an `RwVec` is being used as the data destination, the current
    /// thread will be blocked until an exclusive lock can be obtained before
    /// running the command (which will also block).
    pub fn enq(mut self) -> OclResult<()> {
        let write_src = self.src.take();
        let range = self.range.clone();
        if range.end > write_src.len() { return Err(OclError::from(
            "Unable to enqueue buffer write command: Invalid src_offset and/or len.")) }

        let mut enqueue_with_data = |src: &[T]| {
            let queue = match self.cmd.queue {
                Some(q) => q,
                None => return Err("BufferCmd::enq: No queue set.".into()),
            };

            match self.cmd.kind {
                BufferCmdKind::Write => {
                    match self.cmd.shape {
                        BufferCmdDataShape::Lin { offset } => {
                            try!(check_len(self.cmd.mem_len, src.len(), offset));

                            core::enqueue_write_buffer(queue, self.cmd.obj_core, self.cmd.block,
                                offset, src, self.cmd.ewait.take(), self.cmd.enew.take())
                        },
                        BufferCmdDataShape::Rect { src_origin, dst_origin, region,
                            src_row_pitch_bytes, src_slc_pitch_bytes, dst_row_pitch_bytes,
                            dst_slc_pitch_bytes } =>
                        {
                            core::enqueue_write_buffer_rect(queue, self.cmd.obj_core,
                                self.cmd.block, src_origin, dst_origin, region, src_row_pitch_bytes,
                                src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes,
                                src, self.cmd.ewait.take(), self.cmd.enew.take())
                        }
                    }
                },
                _ => unreachable!(),
            }
        };

        match write_src {
            WriteSrc::Slice(slice) => {
                enqueue_with_data(&slice[range])
            },
            WriteSrc::RwVec(rw_vec) => {
                let guard = rw_vec.request_read().wait()
                    .map_err(|_| OclError::from("Unable to obtain lock."))?;
                enqueue_with_data(&guard.as_slice()[range])
            },
            WriteSrc::Reader(reader) => {
                let guard = reader.wait()
                    .map_err(|_| OclError::from("Unable to obtain lock."))?;
                enqueue_with_data(&guard.as_slice()[range])
            },
            WriteSrc::None => panic!("Invalid read destination."),
        }
    }

    /// Enqueues this command and returns a future representing its completion
    /// which resolves to a guard providing exclusive data access usable
    /// within subsequent futures.
    ///
    /// A data destination container appropriate for an asynchronous operation
    /// (such as `RwVec`) must have been passed to `::write`.
    pub fn enq_async(mut self) -> OclResult<FutureRwGuard<T, WriteGuard<T>>> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        // let rw_vec = match self.src {
        //     WriteSrc::RwVec(rw_vec) => rw_vec,
        //     _ => return Err("BufferWriteCmd::enq_async: Invalid data destination kind for an
        //         asynchronous enqueue. The read destination must be a 'RwVec'.".into()),
        // };

        match self.cmd.kind {
            BufferCmdKind::Write => {
                // let mut reader = rw_vec.request_read().upgrade_after_command();

                let mut reader = match self.src {
                    WriteSrc::RwVec(rw_vec) => rw_vec.request_read().upgrade_after_command(),
                    WriteSrc::Reader(reader) => reader.upgrade_after_command(),
                    _ => return Err("BufferWriteCmd::enq_async: Invalid data destination kind for an
                        asynchronous enqueue. The read destination must be a 'RwVec'.".into()),
                };
                if self.range.end > reader.len() { return Err(OclError::from(
                    "Unable to enqueue buffer write command: Invalid src_offset and/or len.")) }
                
                reader.create_lock_event(queue.context_ptr()?)?;

                if let Some(wl) = self.cmd.ewait {
                    reader.set_wait_list(wl);
                }

                let src = unsafe { &reader.as_mut_slice().expect("BufferWriteCmd::enq_async: \
                    Invalid reader.")[self.range] };

                let mut write_event = Event::empty();

                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.cmd.mem_len, src.len(), offset));

                        core::enqueue_write_buffer(queue, self.cmd.obj_core, false,
                            offset, src, reader.lock_event(), Some(&mut write_event))?;
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region,
                            src_row_pitch_bytes, src_slc_pitch_bytes,
                                dst_row_pitch_bytes, dst_slc_pitch_bytes } =>
                    {
                        core::enqueue_write_buffer_rect(queue, self.cmd.obj_core,
                            false, src_origin, dst_origin, region, src_row_pitch_bytes,
                            src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes,
                            src, reader.lock_event(), Some(&mut write_event))?;
                    }
                }

                if let Some(ref mut enew) = self.cmd.enew.take() {
                    unsafe { enew.clone_from(&write_event) }
                }

                reader.set_command_completion_event(write_event);
                Ok(reader)
            },
            _ => unreachable!(),
        }
    }
}


/// A command builder used to enqueue a map command.
///
/// See [SDK][map_buffer] docs for more details.
///
/// [map_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueMapBuffer.html
#[must_use = "commands do nothing unless enqueued"]
pub struct BufferMapCmd<'c, T> where T: 'c {
    cmd: BufferCmd<'c, T>,
    flags: Option<MapFlags>,
    len: Option<usize> ,
}

impl<'c, T> BufferMapCmd<'c, T> where T: OclPrm {
    /// Specifies the flags to be used for this map command.
    ///
    /// Flags can also be specified using the `::read`, `::write`, and
    /// `::write_invalidate` methods instead.
    ///
    /// See [SDK] docs for more details.
    ///
    /// [SDK]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueMapBuffer.html
    //
    // * TODO: Add links to methods listed above.
    pub fn flags(mut self, flags: MapFlags) -> BufferMapCmd<'c, T> {
        self.flags = Some(flags);
        self
    }

    /// Specifies that the memory object is being mapped for reading.
    ///
    /// Sets the flag to be used for this map command to `[CL_]MAP_READ`.
    ///
    /// This is the fastest way to move data from device to host for many use
    /// cases when used with buffers created with the `MEM_ALLOC_HOST_PTR` or
    /// `MEM_USE_HOST_PTR` flags.
    pub fn read(mut self) -> BufferMapCmd<'c, T> {
        self.flags = Some(::flags::MAP_READ);
        self
    }

    /// Specifies that the memory object is being mapped for writing.
    ///
    /// Sets the flag to be used for this map command to `[CL_]MAP_WRITE`.
    ///
    /// This is not the most efficient method of transferring data from host
    /// to device due to the memory being synchronized beforehand. Prefer
    /// `::write_invalidate` unless you need the memory region to be updated
    /// (e.g. if you are only writing to particular portions of the data, and
    /// will not be overwriting the entire contents, etc.). Use this with
    /// buffers created with the `MEM_ALLOC_HOST_PTR` or `MEM_USE_HOST_PTR`
    /// flags for best performance.
    pub fn write(mut self) -> BufferMapCmd<'c, T> {
        self.flags = Some(::flags::MAP_WRITE);
        self
    }

    /// Specifies that the memory object is being mapped for writing and that
    /// the local (host) memory region may contain stale data that must be
    /// completely overwritten before unmapping.
    ///
    /// Sets the flag to be used for this map command to
    /// `[CL_]MAP_WRITE_INVALIDATE_REGION`.
    ///
    /// This option may provide a substantial performance improvement when
    /// writing and is the fastest method for moving data in bulk from host to
    /// device memory when used with buffers created with the
    /// `MEM_ALLOC_HOST_PTR` or `MEM_USE_HOST_PTR` flags. Only use this when
    /// you will be overwriting the entire contents of the mapped region
    /// otherwise you will send stale or junk data to the device.
    pub fn write_invalidate(mut self) -> BufferMapCmd<'c, T> {
        self.flags = Some(::flags::MAP_WRITE_INVALIDATE_REGION);
        self
    }

    /// Specifies the length of the region to map.
    ///
    /// If unspecified the entire buffer will be mapped.
    //
    // * TODO: Consider taking an `Into<SpatialDims>` argument and possibly
    //   renaming method.
    pub fn len(mut self, len: usize) -> BufferMapCmd<'c, T> {
        self.len = Some(len);
        self
    }

    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'c Queue) -> BufferMapCmd<'c, T> {
        self.cmd.queue = Some(queue);
        self
    }

    /// Sets the linear offset for an operation.
    ///
    /// ## Panics
    ///
    /// The 'shape' may not have already been set to rectangular by the
    /// `::rect` function.
    pub fn offset(self, offset: usize)  -> BufferMapCmd<'c, T> {
        BufferMapCmd { cmd: self.cmd.offset(offset), ..self }
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> BufferMapCmd<'c, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.cmd.ewait = Some(ewait.into());
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt<'e, Ewl>(mut self, ewait: Option<Ewl>) -> BufferMapCmd<'c, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.cmd.ewait = ewait.map(|el| el.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew<'e, En>(mut self, enew: En) -> BufferMapCmd<'c, T>
            // where E: Into<ClNullEventPtrEnum<'e>>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.cmd.enew = Some(enew.into());
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt<'e, En>(mut self, enew: Option<En>) -> BufferMapCmd<'c, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.cmd.enew = enew.map(|e| e.into());
        self
    }

    /// Enqueues a map command.
    pub fn enq(mut self) -> OclResult<MemMap<T>> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        let flags = self.flags.unwrap_or(MapFlags::empty());

        if let BufferCmdKind::Map = self.cmd.kind {
            if let BufferCmdDataShape::Lin { offset } = self.cmd.shape {
                let len = match self.len {
                    Some(l) => l,
                    None => self.cmd.mem_len,
                };

                check_len(self.cmd.mem_len, len, offset)?;

                unsafe {
                    let mm_core = core::enqueue_map_buffer::<T, _, _, _>(queue,
                        self.cmd.obj_core, true, flags, offset, len, self.cmd.ewait.take(),
                        self.cmd.enew.take())?;

                    let unmap_event = None;

                    Ok(MemMap::new(mm_core, len, None, unmap_event, self.cmd.obj_core.clone(),
                        queue.clone()))
                }
            } else {
                OclError::err_string("ocl::BufferCmd::enq_map(): A rectangular map is \
                    not a valid operation. Please use the default shape, linear.")
            }
        } else {
            unreachable!();
        }
    }

    /// Enqueues a map command and returns a future representing the
    /// completion of that map command and containing a reference to the
    /// mapped memory.
    pub fn enq_async(mut self) -> OclResult<FutureMemMap<T>> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        let flags = self.flags.unwrap_or(MapFlags::empty());

        if let BufferCmdKind::Map = self.cmd.kind {
            if let BufferCmdDataShape::Lin { offset } = self.cmd.shape {
                let len = match self.len {
                    Some(l) => l,
                    None => self.cmd.mem_len,
                };

                check_len(self.cmd.mem_len, len, offset)?;

                let future = unsafe {
                    let mut map_event = Event::empty();

                    let mm_core = core::enqueue_map_buffer::<T, _, _, _>(queue,
                        self.cmd.obj_core, false, flags, offset, len, self.cmd.ewait.take(),
                        Some(&mut map_event))?;

                    // If a 'new/null event' has been set, copy pointer
                    // into it and increase refcount (to 2).
                    if let Some(ref mut self_enew) = self.cmd.enew.take() {
                        // map_event/self_enew refcount: 2
                        self_enew.clone_from(&map_event)
                    }

                    FutureMemMap::new(mm_core, len, map_event, self.cmd.obj_core.clone(),
                        queue.clone())
                };

                Ok(future)
            } else {
                OclError::err_string("ocl::BufferMapCmd::enq_async(): A rectangular map is \
                    not a valid operation. Please use the default shape, linear.")
            }
        } else {
            unreachable!();
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
    queue: Option<Queue>, 
    origin: Option<SpatialDims>,
    dims: SpatialDims,
    len: usize,
    flags: MemFlags,
    _data: PhantomData<T>,
}

impl<T: OclPrm> Buffer<T> {
    /// Returns a new buffer builder.
    ///
    /// This is the preferred (and forward compatible) way to create a buffer.
    pub fn builder<'a>() -> BufferBuilder<'a, T> {
        BufferBuilder::new()
    }

    /// Creates a new buffer. **[NOTE]: Use `::builder` instead now.**
    ///
    /// See the [`BufferBuilder`] and [SDK] documentation for argument
    /// details.
    ///
    /// [UNSTABLE]: Arguments may still be in a state of flux. It is
    /// recommended to use `::builder` instead.
    ///
    /// [`BufferBuilder`]: struct.BufferBuilder.html
    /// [SDK]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
    ///
    //
    // * TODO: Consider removing `fill_val` and leaving filling completely to
    //   the builder.
    //
    pub fn new<'e, 'o, D, Q, En>(que_ctx: Q, flags_opt: Option<MemFlags>, dims: D, 
            host_data: Option<&[T]>, fill_val: Option<(T, Option<En>)>) -> OclResult<Buffer<T>>
            where D: Into<SpatialDims>, Q: Into<QueCtx<'o>>, En: Into<ClNullEventPtrEnum<'e>>
    {
        if host_data.is_some() && fill_val.is_some() { panic!("ocl::Buffer::new: Cannot initialize a \
            buffer ('fill_val') when the 'data' argument is 'Some(...)'.") };

        let flags = flags_opt.unwrap_or(::flags::MEM_READ_WRITE);
        let dims: SpatialDims = dims.into();
        let len = dims.to_len();
        let que_ctx = que_ctx.into();

        let obj_core = match que_ctx {
            QueCtx::Queue(ref q) => unsafe { core::create_buffer(&q.context(),
                flags, len, host_data)? },
            QueCtx::Context(c) => unsafe { core::create_buffer(c,
                flags, len, host_data)? },
        };

        let buf = Buffer {
            obj_core: obj_core,
            queue: que_ctx.into(),
            origin: None,
            dims: dims,
            len: len,
            flags: flags,
            _data: PhantomData,
        };

        // Fill with `fill_val` if specified, blocking if the associated event is `None`.
        //
        // TODO: Move this functionality to builder.
        if let Some((val, enew_opt)) = fill_val {
            match enew_opt {
                Some(enew) => buf.cmd().fill(val, None).enew(enew.into()).enq()?,
                None => {
                    let mut new_event = Event::empty();
                    buf.cmd().fill(val, None).enew(&mut new_event).enq()?;
                    new_event.wait_for()?;
                }
            }
        }

        Ok(buf)
    }

    /// Creates a buffer linked to a previously created OpenGL buffer object.
    ///
    /// [UNTESTED]
    ///
    /// ### Errors
    ///
    /// Don't forget to `.cmd().gl_acquire().enq()` before using it and
    /// `.cmd().gl_release().enq()` after.
    ///
    /// See the [`BufferCmd` docs](struct.BufferCmd.html)
    /// for more info.
    ///
    pub fn from_gl_buffer<'o, D, Q>(que_ctx: Q, flags_opt: Option<MemFlags>, dims: D, 
            gl_object: cl_GLuint) -> OclResult<Buffer<T>>
            where D: Into<SpatialDims>, Q: Into<QueCtx<'o>>
    {
        let flags = flags_opt.unwrap_or(core::MEM_READ_WRITE);
        let dims: SpatialDims = dims.into();
        let len = dims.to_len();
        let que_ctx = que_ctx.into();

        // let obj_core = match que_ctx.context_core() {
        //     Some(ref cc) => unsafe { core::create_from_gl_buffer(cc, gl_object, flags)? },
        //     None => panic!("ocl::Buffer::new: A context or default queue must be set."),
        // };

        let obj_core = match que_ctx {
            QueCtx::Queue(ref q) => unsafe { core::create_from_gl_buffer(&q.context(), gl_object, flags)? },
            QueCtx::Context(c) => unsafe { core::create_from_gl_buffer(c, gl_object, flags)? },
        };

        let buf = Buffer {
            obj_core: obj_core,
            queue: que_ctx.into(),
            origin: None,
            dims: dims,
            len: len,
            _data: PhantomData,
            flags: flags,
        };

        Ok(buf)
    }

    /// Returns a command builder used to read, write, copy, etc.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](struct.BufferCmd)
    /// for more details.
    ///
    ///
    #[inline]
    pub fn cmd<'c>(&'c self) -> BufferCmd<'c, T> {
        BufferCmd::new(self.queue.as_ref(), &self.obj_core, self.len)
    }

    /// Returns a command builder used to read data.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](struct.BufferCmd#method.read)
    /// for more details.
    ///
    #[inline]
    pub fn read<'c, 'd, R>(&'c self, dst: R) -> BufferReadCmd<'c, 'd, T>
            where 'd: 'c, R: Into<ReadDst<'d, T>>
    {
        self.cmd().read(dst)
    }

    /// Returns a command builder used to write data.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](struct.BufferCmd#method.write)
    /// for more details.
    ///
    #[inline]
    pub fn write<'c, 'd, W>(&'c self, src: W) -> BufferWriteCmd<'c, 'd, T>
            where 'd: 'c, W: Into<WriteSrc<'d, T>>
    {
        self.cmd().write(src)
    }

    /// Returns a command builder used to map data for reading or writing.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](struct.BufferCmd#method.map)
    /// for more details.
    ///
    #[inline]
    pub fn map<'c>(&'c self) -> BufferMapCmd<'c, T> {
        self.cmd().map()
    }

    /// Specifies that this command will be a copy operation.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](struct.BufferCmd#method.copy)
    /// for more details.
    ///
    #[inline]
    pub fn copy<'c, M>(&'c self, dst_buffer: &'c M, dst_offset: Option<usize>, len: Option<usize>)
            -> BufferCmd<'c, T>
            where M: AsMem<T>
    {
        self.cmd().copy(dst_buffer, dst_offset, len)
    }

    /// Returns the origin of the sub-buffer within its buffer if this is a
    /// sub-buffer.
    #[inline]
    pub fn origin(&self) -> Option<&SpatialDims> {
        self.origin.as_ref()
    }

    /// Returns the dimensions of the buffer.
    #[inline]
    pub fn dims(&self) -> &SpatialDims {
        &self.dims
    }

    /// Returns the length of the buffer.
    ///
    /// Equivalent to `::dims().to_len()`.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if this is a sub-buffer.
    #[inline]
    pub fn is_sub_buffer(&self) -> bool {
        self.origin.is_some()
    }

    /// Returns info about the underlying memory object.
    #[inline]
    pub fn mem_info(&self, info_kind: MemInfo) -> MemInfoResult {
        core::get_mem_object_info(&self.obj_core, info_kind)
    }

    /// Changes the default queue used by this Buffer for reads and writes, etc.
    ///
    /// Returns a mutable reference for optional chaining i.e.:
    ///
    /// ### Example
    ///
    /// `buffer.set_default_queue(queue).read(....);`
    ///
    #[inline]
    pub fn set_default_queue<'a>(&'a mut self, queue: Queue) -> &'a mut Buffer<T> {
        // [FIXME]: Update this to check whether new queue.device is within
        // context or matching existing queue.
        // assert!(queue.device() == self.que_ctx.queue().device());
        self.queue = Some(queue);
        self
    }

    /// Returns a reference to the default queue.
    ///
    #[inline]
    pub fn default_queue(&self) -> Option<&Queue> {
        self.queue.as_ref()
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    ///
    #[deprecated(since="0.13.0", note="Use `::core` instead.")]
    #[inline]
    pub fn core_as_ref(&self) -> &MemCore {
        &self.obj_core
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    ///
    #[inline]
    pub fn core(&self) -> &MemCore {
        &self.obj_core
    }

    /// Returns the memory flags used during the creation of this buffer.
    ///
    /// Saves the cost of having to look them up using `::mem_info`.
    ///
    #[inline]
    pub fn flags(&self) -> MemFlags {
        self.flags
    }

    /// Creates a new sub-buffer.
    ///
    /// ### Flags (adapted from [SDK])
    ///
    /// [NOTE]: Flags described below can be found in the [`ocl::flags`] module
    /// or within the [`MemFlags`][mem_flags] type (example:
    /// [`MemFlags::new().read_write()`]).
    ///
    /// `flags`: A bit-field that is used to specify allocation and usage
    /// information about the sub-buffer memory object being created and is
    /// described in the table below. If the `MEM_READ_WRITE`, `MEM_READ_ONLY`
    /// or `MEM_WRITE_ONLY` values are not specified in flags, they are
    /// inherited from the corresponding memory access qualifers associated
    /// with buffer. The `MEM_USE_HOST_PTR`, `MEM_ALLOC_HOST_PTR` and
    /// `MEM_COPY_HOST_PTR` values cannot be specified in flags but are
    /// inherited from the corresponding memory access qualifiers associated
    /// with buffer. If `MEM_COPY_HOST_PTR` is specified in the memory access
    /// qualifier values associated with buffer it does not imply any
    /// additional copies when the sub-buffer is created from buffer. If the
    /// `MEM_HOST_WRITE_ONLY`, `MEM_HOST_READ_ONLY` or `MEM_HOST_NO_ACCESS`
    /// values are not specified in flags, they are inherited from the
    /// corresponding memory access qualifiers associated with buffer.
    ///
    /// ### Offset and Dimensions
    ///
    /// `origin` and `dims` set up the region of the sub-buffer within the
    ///  original buffer and must not fall beyond the boundaries of it.
    ///
    /// `origin` must be a multiple of the `DeviceInfo::MemBaseAddrAlign`
    /// otherwise you will get a `CL_MISALIGNED_SUB_BUFFER_OFFSET` error. To
    /// determine, use `Device::mem_base_addr_align` for the device associated
    /// with the queue which will be use with this sub-buffer.
    ///
    /// [SDK]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateSubBuffer.html
    /// [`ocl::flags`]: flags/index.html
    /// [mem_flags]: struct.MemFlags.html
    /// [`MemFlags::new().read_write()`] struct.MemFlags.html#method.read_write
    ///
    pub fn create_sub_buffer<Do, Ds>(&self, flags_opt: Option<MemFlags>, origin: Do,
        dims: Ds) -> OclResult<Buffer<T>>
        where Do: Into<SpatialDims>, Ds: Into<SpatialDims>
    {
        let flags = flags_opt.unwrap_or(::flags::MEM_READ_WRITE);

        // Check flags here to preempt a somewhat vague OpenCL runtime error message:
        assert!(!flags.contains(::flags::MEM_USE_HOST_PTR) &&
            !flags.contains(::flags::MEM_ALLOC_HOST_PTR) &&
            !flags.contains(::flags::MEM_COPY_HOST_PTR),
            "'MEM_USE_HOST_PTR', 'MEM_ALLOC_HOST_PTR', or 'MEM_COPY_HOST_PTR' flags may \
            not be specified when creating a sub-buffer. They will be inherited from \
            the containing buffer.");

        let origin = origin.into();
        let dims = dims.into();

        let buffer_len = self.dims().to_len();
        let origin_ofs = origin.to_len();
        let len = dims.to_len();

        if origin_ofs > buffer_len {
            return OclError::err_string(format!("Buffer::create_sub_buffer: Origin ({:?}) is outside of the \
                dimensions of the source buffer ({:?}).", origin, self.dims()));
        }

        if origin_ofs + len > buffer_len {
            return OclError::err_string(format!("Buffer::create_sub_buffer: Sub-buffer region (origin: '{:?}', \
                dims: '{:?}') exceeds the dimensions of the source buffer ({:?}).", origin, dims,
                self.dims()));
        }

        let obj_core = core::create_sub_buffer::<T>(self, flags,
            &BufferRegion::new(origin_ofs, len))?;

        Ok(Buffer {
            obj_core: obj_core,
            queue: self.default_queue().cloned(),
            origin: Some(origin),
            dims: dims,
            len: len,
            flags: flags,
            _data: PhantomData,
        })
    }

    /// Formats memory info.
    #[inline]
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

impl<T: OclPrm> AsRef<MemCore> for Buffer<T> {
    fn as_ref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<T: OclPrm> AsMut<MemCore> for Buffer<T> {
    fn as_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}

impl<T: OclPrm> AsMem<T> for Buffer<T> {
    fn as_mem(&self) -> &MemCore {
        &self.obj_core
    }
}

// impl<'a, T: OclPrm> AsMem<T> for &'a mut Buffer<T> {
//     fn as_mem(&self) -> &MemCore {
//         &self.obj_core
//     }
// }

impl<T: OclPrm> std::fmt::Display for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_mem_info(f)
    }
}

unsafe impl<'a, T> MemCmdRw for Buffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdRw for &'a Buffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdRw for &'a mut Buffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for Buffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for &'a Buffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for &'a mut Buffer<T> where T: OclPrm {}


/// A buffer builder.
///
/// * TODO: Add examples and details. For now see project examples folder.
///
#[must_use = "builders do nothing unless '::build' is called"]
#[derive(Debug)]
pub struct BufferBuilder<'a, T> where T: 'a {
    queue_option: Option<QueCtx<'a>>,
    flags: Option<MemFlags>,
    host_data: Option<&'a [T]>,
    dims: Option<SpatialDims>,
    fill_val: Option<(T, Option<ClNullEventPtrEnum<'a>>)>
}

impl<'a, T> BufferBuilder<'a, T> where T: 'a + OclPrm {
    /// Returns a new buffer builder.
    pub fn new() -> BufferBuilder<'a, T> {
        BufferBuilder {
            queue_option: None,
            flags: None,
            host_data: None,
            dims: None,
            fill_val: None,
        }
    }

    /// Sets the context with which to associate the buffer.
    ///
    /// May not be used in combination with `::queue` (use one or the other).
    pub fn context<'o>(mut self, context: &'o Context) -> BufferBuilder<'a, T>
            where 'o: 'a
    {
        assert!(self.queue_option.is_none());
        self.queue_option = Some(QueCtx::Context(context));
        self
    }

    /// Sets the default queue.
    ///
    /// If this is set, the context associated with the `default_queue` will
    /// be used when creating the buffer (use one or the other).
    pub fn queue<'b>(mut self, default_queue: Queue) -> BufferBuilder<'a, T> {
        assert!(self.queue_option.is_none());
        self.queue_option = Some(QueCtx::Queue(default_queue));
        self
    }

    /// Sets the flags used when creating the buffer.
    ///
    /// Defaults to `flags::MEM_READ_WRITE` aka. `MemFlags::new().read_write()`
    /// if this is not set. See the [SDK Docs] for more information about
    /// flags. Note that the `host_ptr` mentioned in the [SDK Docs] is
    /// equivalent to the slice optionally passed as the `host_data` argument.
    /// Also note that the names of all flags in this library have the `CL_`
    /// prefix removed for brevity.
    ///
    /// [SDK Docs]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
    pub fn flags<'b>(mut self, flags: MemFlags) -> BufferBuilder<'a, T> {
        self.flags = Some(flags);
        self
    }

    /// A slice use to designate a region of memory for use in combination of
    /// one of the two following flags:
    ///
    /// * `flags::MEM_USE_HOST_PTR` aka. `MemFlags::new().use_host_ptr()`:
    ///   * This flag is valid only if `host_data` is not `None`. If
    ///     specified, it indicates that the application wants the OpenCL
    ///     implementation to use memory referenced by `host_data` as the
    ///     storage bits for the memory object (buffer/image).
    ///   * OpenCL implementations are allowed to cache the buffer contents
    ///     pointed to by `host_data` in device memory. This cached copy can
    ///     be used when kernels are executed on a device.
    ///   * The result of OpenCL commands that operate on multiple buffer
    ///     objects created with the same `host_data` or overlapping host
    ///     regions is considered to be undefined.
    ///   * Refer to the [description of the alignment][align_rules] rules for
    ///     `host_data` for memory objects (buffer and images) created using
    ///     `MEM_USE_HOST_PTR`.
    ///   * `MEM_ALLOC_HOST_PTR` and `MEM_USE_HOST_PTR` are mutually exclusive.
    ///
    /// * `flags::MEM_COPY_HOST_PTR` aka. `MemFlags::new().copy_host_ptr()`
    ///   * This flag is valid only if `host_data` is not NULL. If specified, it
    ///     indicates that the application wants the OpenCL implementation to
    ///     allocate memory for the memory object and copy the data from
    ///     memory referenced by `host_data`.
    ///   * CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually
    ///     exclusive.
    ///   * CL_MEM_COPY_HOST_PTR can be used with CL_MEM_ALLOC_HOST_PTR to
    ///     initialize the contents of the cl_mem object allocated using
    ///     host-accessible (e.g. PCIe) memory.
    ///
    /// Note: Descriptions adapted from:
    /// [https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html][create_buffer].
    ///
    ///
    /// [align_rules]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/dataTypes.html
    /// [create_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
    pub fn host_data<'d>(mut self, host_data: &'d [T]) -> BufferBuilder<'a, T>
            where 'd: 'a
    {
        self.host_data = Some(host_data);
        self
    }

    /// Sets the dimensions for this buffer.
    ///
    /// Typically a single integer value to set the total length is used
    /// however up to three dimensions may be specified in order to more
    /// easily coordinate with kernel work sizes.
    ///
    /// Note that although sizes in the standard OpenCL API are expressed in
    /// bytes, sizes, lengths, and dimensions in this library are always
    /// specified in `bytes / sizeof(T)` (like everything else in Rust) unless
    /// otherwise noted.
    pub fn dims<'b, D>(mut self, dims: D) -> BufferBuilder<'a, T>
            where D: Into<SpatialDims>
    {
        self.dims = Some(dims.into());
        self
    }

    /// Allows the caller to automatically fill the buffer with a value (such
    /// as zero) immediately after creation.
    ///
    /// Platforms that have trouble with `clEnqueueFillBuffer` such as
    /// [pocl](http://portablecl.org/) should not use this option and should
    /// handle initializing buffers manually (using a kernel or copy host data
    /// flag).
    ///
    /// The `enew` argument is provided to allow an empty event to be
    /// associated with the `fill` command which will be enqueued after
    /// creation and just before returning the new buffer. It is up to the
    /// caller to ensure that the command has completed before performing any
    /// other operations on the buffer. Failure to do so may cause the fill
    /// command to run **after** subsequently queued commands if multiple or
    /// out-of-order queues are being used. Passing `None` for `enew` (use
    /// `None::<()>` to avoid the the type inference error) will cause the
    /// fill command to block before returning the new buffer and is the safe
    /// option if you don't want to worry about it.
    ///
    /// ### Examples
    ///
    /// * TODO: Provide examples once this stabilizes.
    ///
    ///
    /// [UNSTABLE]: May be changed or removed.
    pub fn fill_val<'b, 'e, En>(mut self, fill_val: T, enew: Option<En>)
            -> BufferBuilder<'a, T>
            where 'e: 'a, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.fill_val = Some((fill_val, enew.map(|e| e.into())));
        self
    }

    /// Creates a buffer and returns it.
    ///
    /// Dimensions and either a context or default queue must be specified
    /// before calling `::build`.
    pub fn build(self) -> OclResult<Buffer<T>> {
        match self.queue_option {
            Some(qo) => {
                let dims = match self.dims {
                    Some(d) => d,
                    None => panic!("ocl::BufferBuilder::build: The dimensions must be set with '.dims(...)'."),
                };

                Buffer::new(qo, self.flags, dims, self.host_data, self.fill_val)
            },
            None => panic!("ocl::BufferBuilder::build: A context or default queue must be set \
                with '.context(...)' or '.queue(...)'."),
        }
    }
}
