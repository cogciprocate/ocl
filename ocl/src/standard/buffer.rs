//! Interfaces with a buffer.

use std;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Range};
use crate::core::{self, Error as OclCoreError, Result as OclCoreResult, OclPrm, Mem as MemCore,
    MemFlags, MemInfo, MemInfoResult, BufferRegion, MapFlags, AsMem, MemCmdRw, MemCmdAll,
    ClNullEventPtr};
use crate::{Context, Queue, FutureMemMap, MemMap, Event, RwVec, FutureReadGuard, FutureWriteGuard,
    SpatialDims};
use crate::standard::{ClNullEventPtrEnum, ClWaitListPtrEnum, HostSlice};
use crate::error::{Error as OclError, Result as OclResult};

#[cfg(not(feature="opencl_vendor_mesa"))]
use crate::ffi::cl_GLuint;


fn check_len(mem_len: usize, data_len: usize, offset: usize) -> OclResult<()> {
    if offset >= mem_len {
        Err(format!("ocl::Buffer::enq(): Offset out of range. \
            (mem_len: {}, data_len: {}, offset: {}", mem_len, data_len, offset).into())
    } else if data_len > (mem_len - offset) {
        Err("ocl::Buffer::enq(): Data length exceeds buffer length.".into())
    } else {
        Ok(())
    }
}


/// A buffer command error.
#[derive(Debug, Fail)]
pub enum BufferCmdError {
    #[fail(display = "A rectangular map is not a valid operation. \
        Please use the default shape, linear.")]
    RectUnavailable,
    #[fail(display = "No queue specified.")]
    NoQueue,
    #[fail(display = "Buffer already mapped.")]
    AlreadyMapped,
    #[fail(display = "Unable to map this buffer. Must create with either the \
        MEM_USE_HOST_PTR or MEM_ALLOC_HOST_PTR flag.")]
    MapUnavailable,
    #[fail(display = "ocl-core error: {}", _0)]
    Ocl(#[cause] OclCoreError)
}


impl From<OclCoreError> for BufferCmdError {
    fn from(err: OclCoreError) -> BufferCmdError {
        BufferCmdError::Ocl(err)
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
/// ```rust,ignore
/// // Copies one buffer to another:
/// src_buffer.copy(&dst_buffer, 0, dst_buffer.len()).enq().unwrap();
///
/// // Writes from a vector to an buffer, waiting on an event:
/// buffer.write(&src_vec).ewait(&event).enq().unwrap();
///
/// // Reads from a buffer into a vector, waiting on an event list and
/// // filling a new empty event:
/// buffer.read(&dst_vec).ewait(&event_list).enew(&mut empty_event).enq().unwrap();
///
/// // Reads without blocking:
/// buffer.read(&dst_vec).block(false).enew(&mut empty_event).enq().unwrap();
///
/// ```
///
#[must_use = "commands do nothing unless enqueued"]
pub struct BufferCmd<'c, T> where T: 'c + OclPrm {
    buffer: &'c Buffer<T>,
    queue: Option<&'c Queue>,
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
    fn new(buffer: &'c Buffer<T>, queue: Option<&'c Queue>, /*obj_core: &'c MemCore,*/ mem_len: usize)
            -> BufferCmd<'c, T> {
        BufferCmd {
            buffer,
            queue,
            block: true,
            kind: BufferCmdKind::Unspecified,
            shape: BufferCmdDataShape::Lin { offset: 0 },
            ewait: None,
            enew: None,
            mem_len,
        }
    }

    /// Specifies that this command will be a read operation.
    ///
    /// After calling this method, the blocking state of this command will
    /// be unchanged.
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
            where R: Into<ReadDst<'d, T>> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::read(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Read;
        let dst = dst_data.into();
        let len = dst.len();
        BufferReadCmd { cmd: self, dst: dst, range: 0..len }
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
            where W: Into<WriteSrc<'d, T>> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::write(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Write;
        let src = src_data.into();
        let len = src.len();
        BufferWriteCmd { cmd: self, src: src, range: 0..len }
    }

    /// Specifies that this command will be a map operation.
    ///
    /// Enqueuing a map command will map a region of a buffer into the host
    /// address space and return a [`MemMap`] or [`FutureMemMap`], allowing
    /// access to this mapped region. Accessing memory via a [`MemMap`] is
    /// exactly like using a [slice].
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
    /// [`MemMap`]: struct.MemMap.html
    /// [`FutureMemMap`]: async/struct.FutureMemMap.html
    /// [slice]: https://doc.rust-lang.org/std/primitive.slice.html
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
            where 'd: 'c, M: AsMem<T> {
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
            where 'd: 'c {
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
    ///
    /// Overrides the buffer's default queue if one is set. If no default
    /// queue is set, this method **must** be called before enqueuing the
    /// command.
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
    /// [`::map`]: builders/struct.BufferMapCmd.html
    /// [`::enq_async`]: builders/struct.BufferMapCmd.html#method.enq_async
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
                dst_slc_pitch_bytes: usize) -> BufferCmd<'c, T> {
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

    /// Specifies an event or list of events to wait on before the command
    /// will run.
    ///
    /// When events generated using the `::enew` method of **other**,
    /// previously enqueued commands are passed here (either individually or
    /// as part of an [`EventList`]), this command will not execute until
    /// those commands have completed.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event_list = EventList::new();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event_list).enq()?;
    /// // Read from a buffer using `queue_2`, ensuring the read does not begin until
    /// // after the kernel command has completed:
    /// buffer.read(rwvec.clone()).queue(&queue_2).ewait(&event_list).enq_async()?;
    /// ```
    ///
    /// [`EventList`]: struct.EventList.html
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> BufferCmd<'c, T>
            where 'e: 'c, Ewl: Into<ClWaitListPtrEnum<'e>> {
        self.ewait = Some(ewait.into());
        self
    }

    /// Specifies the destination to store a new, optionally created event
    /// associated with this command.
    ///
    /// The destination can be a mutable reference to an empty event (created
    /// using [`Event::empty`]) or a mutable reference to an event list.
    ///
    /// After this command is enqueued, the event in the destination can be
    /// passed to the `::ewait` method of another command. Doing so will cause
    /// the other command to wait until this command has completed before
    /// executing.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event = Event::empty();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event).enq()?;
    /// // Read from a buffer using `queue_2`, ensuring the read does not begin until
    /// // after the kernel command has completed:
    /// buffer.read(rwvec.clone()).queue(&queue_2).ewait(&event).enq_async()?;
    /// ```
    ///
    /// [`Event::empty`]: struct.Event.html#method.empty
    pub fn enew<'e, En>(mut self, enew: En) -> BufferCmd<'c, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>> {
        self.enew = Some(enew.into());
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
                        check_len(self.mem_len, len, offset)?;
                        let dst_offset = dst_offset.unwrap_or(0);

                        core::enqueue_copy_buffer::<T, _, _, _>(queue,
                            &self.buffer.obj_core, dst_buffer, offset, dst_offset, len,
                            self.ewait, self.enew).map_err(OclError::from)
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region,
                        src_row_pitch_bytes, src_slc_pitch_bytes, dst_row_pitch_bytes,
                        dst_slc_pitch_bytes } =>
                    {
                        if dst_offset.is_some() || len.is_some() { return Err(
                            "ocl::BufferCmd::enq(): For 'rect' shaped copies, destination \
                            offset and length must be 'None'. Ex.: \
                            'cmd().copy(&{{buf_name}}, None, None)..'.".into());
                        }

                        core::enqueue_copy_buffer_rect::<T, _, _, _>(queue, &self.buffer.obj_core,
                            dst_buffer, src_origin, dst_origin, region, src_row_pitch_bytes,
                            src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes,
                            self.ewait, self.enew).map_err(OclError::from)
                    },
                }
            },

            #[cfg(not(feature="opencl_vendor_mesa"))]
            BufferCmdKind::Fill { pattern, len } => {
                match self.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        let len = match len {
                            Some(l) => l,
                            None => self.mem_len,
                        };

                        check_len(self.mem_len, len, offset)?;

                        core::enqueue_fill_buffer(queue, &self.buffer.obj_core, pattern,
                            offset, len, self.ewait, self.enew, Some(&queue.device_version()))
                            .map_err(OclError::from)
                    },
                    BufferCmdDataShape::Rect { .. } => Err(
                        "ocl::BufferCmd::enq(): Rectangular fill is not a valid operation. \
                        Please use the default shape, linear.".into())
                }
            },
            #[cfg(not(feature="opencl_vendor_mesa"))]
            BufferCmdKind::GLAcquire => {
                let buf_slc = unsafe { std::slice::from_raw_parts(&self.buffer.obj_core, 1) };
                core::enqueue_acquire_gl_objects(queue, buf_slc, self.ewait, self.enew).map_err(OclError::from)
            },

            #[cfg(not(feature="opencl_vendor_mesa"))]
            BufferCmdKind::GLRelease => {
                let buf_slc = unsafe { std::slice::from_raw_parts(&self.buffer.obj_core, 1) };
                core::enqueue_release_gl_objects(queue, buf_slc, self.ewait, self.enew).map_err(OclError::from)
            },

            BufferCmdKind::Unspecified => Err("ocl::BufferCmd::enq(): \
                No operation specified. Use '.read(...)', 'write(...)', etc. before calling \
                '.enq()'.".into()),
            BufferCmdKind::Map { .. } => unreachable!(),
            _ => unimplemented!(),
        }
    }
}


/// The data destination for a buffer read command.
pub enum ReadDst<'d, T> where T: 'd {
    Slice(&'d mut [T]),
    RwVec(RwVec<T>),
    Writer(FutureWriteGuard<Vec<T>>),
    None,
}

impl<'d, T> ReadDst<'d, T> {
    fn take(&mut self) -> ReadDst<'d, T> {
        ::std::mem::replace(self, ReadDst::None)
    }

    pub fn len(&self) -> usize {
        match *self {
            ReadDst::RwVec(ref rw_vec) => rw_vec.len_stale(),
            ReadDst::Writer(ref writer) => unsafe { (*writer.as_ptr()).len() },
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

impl<'d, T> From<FutureWriteGuard<Vec<T>>> for ReadDst<'d, T> where T: OclPrm {
    fn from(writer: FutureWriteGuard<Vec<T>>) -> ReadDst<'d, T> {
        ReadDst::Writer(writer)
    }
}


/// A buffer command builder used to enqueue reads.
///
/// See [SDK][read_buffer] docs for more details.
///
/// [read_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueReadBuffer.html
#[must_use = "commands do nothing unless enqueued"]
pub struct BufferReadCmd<'c, 'd, T> where T: 'c + 'd + OclPrm {
    cmd: BufferCmd<'c, T>,
    dst: ReadDst<'d, T>,
    range: Range<usize>,
}

impl<'c, 'd, T> BufferReadCmd<'c, 'd, T> where T: OclPrm {
    /// Specifies a queue to use for this call only.
    ///
    /// Overrides the buffer's default queue if one is set. If no default
    /// queue is set, this method **must** be called before enqueuing the
    /// command.
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
    /// [`::map`]: builders/struct.BufferMapCmd.html
    /// [`::enq_async`]: builders/struct.BufferMapCmd.html
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
                dst_slc_pitch_bytes: usize) -> BufferReadCmd<'c, 'd, T> {
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

    /// Specifies an event or list of events to wait on before the command
    /// will run.
    ///
    /// When events generated using the `::enew` method of **other**,
    /// previously enqueued commands are passed here (either individually or
    /// as part of an [`EventList`]), this command will not execute until
    /// those commands have completed.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event_list = EventList::new();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event_list).enq()?;
    /// // Read from a buffer using `queue_2`, ensuring the read does not begin until
    /// // after the kernel command has completed:
    /// buffer.read(rwvec.clone()).queue(&queue_2).ewait(&event_list).enq_async()?;
    /// ```
    ///
    /// [`EventList`]: struct.EventList.html
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> BufferReadCmd<'c, 'd, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>> {
        self.cmd.ewait = Some(ewait.into());
        self
    }

    /// Specifies the destination to store a new, optionally created event
    /// associated with this command.
    ///
    /// The destination can be a mutable reference to an empty event (created
    /// using [`Event::empty`]) or a mutable reference to an event list.
    ///
    /// After this command is enqueued, the event in the destination can be
    /// passed to the `::ewait` method of another command. Doing so will cause
    /// the other command to wait until this command has completed before
    /// executing.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event = Event::empty();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event).enq()?;
    /// // Read from a buffer using `queue_2`, ensuring the read does not begin until
    /// // after the kernel command has completed:
    /// buffer.read(rwvec.clone()).queue(&queue_2).ewait(&event).enq_async()?;
    /// ```
    ///
    /// [`Event::empty`]: struct.Event.html#method.empty
    pub fn enew<'e, En>(mut self, enew: En) -> BufferReadCmd<'c, 'd, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>> {
        self.cmd.enew = Some(enew.into());
        self
    }

    /// Enqueues this command, blocking the current thread until it is complete.
    ///
    /// If an `RwVec` is being used as the data destination, the current
    /// thread will be blocked until an exclusive lock can be obtained before
    /// running the command (which will also block for its duration).
    //
    // NOTE: Could use deferred initialization for the guard slice instead of closure.
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
                            check_len(self.cmd.mem_len, dst.len(), offset)?;
                            unsafe {
                                core::enqueue_read_buffer(queue, &self.cmd.buffer.obj_core,
                                    self.cmd.block, offset, dst, self.cmd.ewait.take(),
                                    self.cmd.enew.take()).map_err(OclError::from)
                            }
                        },
                        BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch_bytes,
                                src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes } =>
                        {
                            // TODO: Verify dims given (like `::check_len`).
                            unsafe {
                                core::enqueue_read_buffer_rect(queue, &self.cmd.buffer.obj_core,
                                    self.cmd.block, src_origin, dst_origin, region,
                                    src_row_pitch_bytes, src_slc_pitch_bytes, dst_row_pitch_bytes,
                                    dst_slc_pitch_bytes, dst, self.cmd.ewait.take(),
                                    self.cmd.enew.take()).map_err(OclError::from)
                            }
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
                let mut guard = rw_vec.write().wait()
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
    pub fn enq_async(mut self) -> OclResult<FutureWriteGuard<Vec<T>>> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        match self.cmd.kind {
            BufferCmdKind::Read => {
                let mut writer = match self.dst {
                    ReadDst::RwVec(rw_vec) => rw_vec.write(),
                    ReadDst::Writer(writer) => writer,
                    _ => return Err("BufferReadCmd::enq_async: Invalid data destination kind for an
                        asynchronous enqueue. The read destination must be a 'RwVec'.".into()),
                };
                let writer_len = unsafe { (*writer.as_ptr()).len() };
                if self.range.end > writer_len {
                    return Err(OclError::from("Unable to enqueue buffer read command: \
                        Invalid src_offset and/or len."))
                }

                writer.create_lock_event(queue.context_ptr()?)?;

                if let Some(wl) = self.cmd.ewait {
                    writer.set_lock_wait_events(wl);
                }

                // let dst = unsafe { &mut writer.as_mut_slice().expect("BufferReadCmd::enq_async: \
                //     Invalid writer.")[self.range] };
                let dst = unsafe {
                    &mut ::std::slice::from_raw_parts_mut(
                        (*writer.as_mut_ptr()).as_mut_ptr(), writer_len)[self.range]
                };

                let mut read_event = Event::empty();

                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        check_len(self.cmd.mem_len, dst.len(), offset)?;

                        unsafe { core::enqueue_read_buffer(queue, &self.cmd.buffer.obj_core, false,
                            offset, dst, writer.lock_event(), Some(&mut read_event))?; }
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region,
                        src_row_pitch_bytes, src_slc_pitch_bytes,
                            dst_row_pitch_bytes, dst_slc_pitch_bytes } =>
                    {
                        unsafe { core::enqueue_read_buffer_rect(queue, &self.cmd.buffer.obj_core,
                            false, src_origin, dst_origin, region, src_row_pitch_bytes,
                            src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes,
                            dst, writer.lock_event(), Some(&mut read_event))?; }
                    }
                }

                if let Some(ref mut enew) = self.cmd.enew.take() {
                    unsafe { enew.clone_from(&read_event) }
                }

                writer.set_command_wait_event(read_event);
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
    Reader(FutureReadGuard<Vec<T>>),
    None,
}

impl<'d, T> WriteSrc<'d, T> {
    fn take(&mut self) -> WriteSrc<'d, T> {
        ::std::mem::replace(self, WriteSrc::None)
    }

    pub fn len(&self) -> usize {
        match *self {
            WriteSrc::RwVec(ref rw_vec) => rw_vec.len_stale(),
            WriteSrc::Reader(ref writer) => unsafe { (*writer.as_ptr()).len() },
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

impl<'d, T> From<FutureReadGuard<Vec<T>>> for WriteSrc<'d, T> where T: OclPrm {
    fn from(reader: FutureReadGuard<Vec<T>>) -> WriteSrc<'d, T> {
        WriteSrc::Reader(reader)
    }
}


/// A buffer command builder used to enqueue writes.
///
/// See [SDK][write_buffer] docs for more details.
///
/// [write_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueWriteBuffer.html
#[must_use = "commands do nothing unless enqueued"]
pub struct BufferWriteCmd<'c, 'd, T> where T: 'c + 'd + OclPrm {
    cmd: BufferCmd<'c, T>,
    src: WriteSrc<'d, T>,
    range: Range<usize>,
 }

impl<'c, 'd, T> BufferWriteCmd<'c, 'd, T> where T: OclPrm {
    /// Specifies a queue to use for this call only.
    ///
    /// Overrides the buffer's default queue if one is set. If no default
    /// queue is set, this method **must** be called before enqueuing the
    /// command.
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
    /// [`::map`]: builders/struct.BufferMapCmd.html
    /// [`::enq_async`]: builders/struct.BufferMapCmd.html
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
                dst_slc_pitch_bytes: usize) -> BufferWriteCmd<'c, 'd, T> {
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

    /// Specifies an event or list of events to wait on before the command
    /// will run.
    ///
    /// When events generated using the `::enew` method of **other**,
    /// previously enqueued commands are passed here (either individually or
    /// as part of an [`EventList`]), this command will not execute until
    /// those commands have completed.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event_list = EventList::new();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event_list).enq()?;
    /// // Write to a buffer using `queue_2`, ensuring the write does not begin until
    /// // after the kernel command has completed:
    /// buffer.write(rwvec.clone()).queue(&queue_2).ewait(&event_list).enq_async()?;
    /// ```
    ///
    /// [`EventList`]: struct.EventList.html
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> BufferWriteCmd<'c, 'd, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>> {
        self.cmd.ewait = Some(ewait.into());
        self
    }

    /// Specifies the destination to store a new, optionally created event
    /// associated with this command.
    ///
    /// The destination can be a mutable reference to an empty event (created
    /// using [`Event::empty`]) or a mutable reference to an event list.
    ///
    /// After this command is enqueued, the event in the destination can be
    /// passed to the `::ewait` method of another command. Doing so will cause
    /// the other command to wait until this command has completed before
    /// executing.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event = Event::empty();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event).enq()?;
    /// // Write to a buffer using `queue_2`, ensuring the write does not begin until
    /// // after the kernel command has completed:
    /// buffer.write(rwvec.clone()).queue(&queue_2).ewait(&event).enq_async()?;
    /// ```
    ///
    /// [`Event::empty`]: struct.Event.html#method.empty
    pub fn enew<'e, En>(mut self, enew: En) -> BufferWriteCmd<'c, 'd, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>> {
        self.cmd.enew = Some(enew.into());
        self
    }

    /// Enqueues this command, blocking the current thread until it is complete.
    ///
    /// If an `RwVec` is being used as the data destination, the current
    /// thread will be blocked until an exclusive lock can be obtained before
    /// running the command (which will also block).
    //
    // NOTE: Could use deferred initialization for the guard slice instead of closure.
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
                            check_len(self.cmd.mem_len, src.len(), offset)?;

                            unsafe {
                                core::enqueue_write_buffer(queue, &self.cmd.buffer.obj_core, self.cmd.block,
                                    offset, src, self.cmd.ewait.take(), self.cmd.enew.take())
                                    .map_err(OclError::from)
                            }
                        },
                        BufferCmdDataShape::Rect { src_origin, dst_origin, region,
                            src_row_pitch_bytes, src_slc_pitch_bytes, dst_row_pitch_bytes,
                            dst_slc_pitch_bytes } =>
                        {
                            unsafe {
                                core::enqueue_write_buffer_rect(queue, &self.cmd.buffer.obj_core,
                                    self.cmd.block, src_origin, dst_origin, region, src_row_pitch_bytes,
                                    src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes,
                                    src, self.cmd.ewait.take(), self.cmd.enew.take())
                                    .map_err(OclError::from)
                            }
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
                let guard = rw_vec.read().wait()
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
    /// which resolves to a read guard usable within subsequent futures.
    ///
    /// A data destination container appropriate for an asynchronous operation
    /// (such as `RwVec`) must have been passed to `::write`.
    ///
    /// The returned future must be resolved.
    ///
    pub fn enq_async(mut self) -> OclResult<FutureReadGuard<Vec<T>>> {
        match self.cmd.kind {
            BufferCmdKind::Write => {
                let mut reader = match self.src {
                    WriteSrc::RwVec(rw_vec) => rw_vec.read(),
                    WriteSrc::Reader(reader) => reader,
                    _ => return Err("BufferWriteCmd::enq_async: Invalid data destination kind for an
                        asynchronous enqueue. The read destination must be a 'RwVec'.".into()),
                };
                let reader_len = unsafe { (*reader.as_ptr()).len() };
                if self.range.end > reader_len { return Err(OclError::from(
                    "Unable to enqueue buffer write command: Invalid src_offset and/or len.")) }

                if let Some(wl) = self.cmd.ewait {
                    reader.set_lock_wait_events(wl);
                }

                let queue = match self.cmd.queue {
                    Some(q) => q,
                    None => return Err("BufferCmd::enq: No queue set.".into()),
                };

                reader.create_lock_event(queue.context_ptr()?)?;

                let src = unsafe {
                    &::std::slice::from_raw_parts((*reader.as_ptr()).as_ptr(), reader_len)[self.range]
                };

                let mut write_event = Event::empty();

                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        check_len(self.cmd.mem_len, src.len(), offset)?;
                        unsafe {
                            core::enqueue_write_buffer(queue, &self.cmd.buffer.obj_core, false,
                                offset, src, reader.lock_event(), Some(&mut write_event))?;
                        }
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region,
                            src_row_pitch_bytes, src_slc_pitch_bytes,
                                dst_row_pitch_bytes, dst_slc_pitch_bytes } =>
                    {
                        unsafe {
                            core::enqueue_write_buffer_rect(queue, &self.cmd.buffer.obj_core,
                                false, src_origin, dst_origin, region, src_row_pitch_bytes,
                                src_slc_pitch_bytes, dst_row_pitch_bytes, dst_slc_pitch_bytes,
                                src, reader.lock_event(), Some(&mut write_event))?;
                        }
                    }
                }

                if let Some(ref mut enew) = self.cmd.enew.take() {
                    unsafe { enew.clone_from(&write_event) }
                }

                reader.set_command_wait_event(write_event);
                Ok(reader)
            },
            _ => unreachable!(),
        }
    }

    /// Enqueues this command and returns a future representing its
    /// completion.
    ///
    /// The returned future resolves to a write guard providing exclusive data
    /// access available within subsequent futures. This is important when a
    /// write must occur at the correct time in the global read/write order.
    ///
    /// A data destination container appropriate for an asynchronous operation
    /// (such as `RwVec`) must have been passed to `::write`.
    ///
    /// The returned future must be resolved.
    ///
    pub fn enq_async_then_write(self) -> OclResult<FutureWriteGuard<Vec<T>>> {
        // NOTE: The precise point in time at which `::upgrade_after_command`
        // is called does not matter since a read request will have already
        // been enqueued in the RwVec's queue. The upgrade can be requested at
        // any time before the read guard is destroyed and have the exact same
        // effect. That read request will lock out any possibility of a write
        // request interfering with the global r/w order.
        self.enq_async().map(|read_guard| read_guard.upgrade_after_command())
    }
}


/// A command builder used to enqueue a map command.
///
/// Enqueuing a map command will map a region of a buffer into the host
/// address space and return a [`MemMap`] or [`FutureMemMap`], allowing access
/// to this mapped region. Accessing memory via a [`MemMap`] is exactly like
/// using a [slice].
///
/// See [SDK][map_buffer] docs for more details.
///
/// [map_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueMapBuffer.html
/// [`MemMap`]: struct.MemMap.html
/// [`FutureMemMap`]: async/struct.FutureMemMap.html
/// [slice]: https://doc.rust-lang.org/std/primitive.slice.html
#[must_use = "commands do nothing unless enqueued"]
pub struct BufferMapCmd<'c, T> where T: 'c + OclPrm {
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
    // * TODO: Sort out the `BufferBuilder::host_slice`/`::flags` situation.
    //   Possibly create separate methods, `use_host_ptr` and
    //   `copy_host_slice`.
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
        self.flags = Some(crate::flags::MAP_READ);
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
        self.flags = Some(crate::flags::MAP_WRITE);
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
        self.flags = Some(crate::flags::MAP_WRITE_INVALIDATE_REGION);
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
    ///
    /// Overrides the buffer's default queue if one is set. If no default
    /// queue is set, this method **must** be called before enqueuing the
    /// command.
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

    /// Specifies an event or list of events to wait on before the command
    /// will run.
    ///
    /// When events generated using the `::enew` method of **other**,
    /// previously enqueued commands are passed here (either individually or
    /// as part of an [`EventList`]), this command will not execute until
    /// those commands have completed.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event_list = EventList::new();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event_list).enq()?;
    /// // Map a buffer using `queue_2`, ensuring the map does not begin until
    /// // after the kernel command has completed:
    /// buffer.map().queue(&queue_2).ewait(&event_list).enq_async()?;
    /// ```
    ///
    /// [`EventList`]: struct.EventList.html
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> BufferMapCmd<'c, T>
            where 'e: 'c,  Ewl: Into<ClWaitListPtrEnum<'e>> {
        self.cmd.ewait = Some(ewait.into());
        self
    }

    /// Specifies the destination to store a new, optionally created event
    /// associated with this command.
    ///
    /// The destination can be a mutable reference to an empty event (created
    /// using [`Event::empty`]) or a mutable reference to an event list.
    ///
    /// After this command is enqueued, the event in the destination can be
    /// passed to the `::ewait` method of another command. Doing so will cause
    /// the other command to wait until this command has completed before
    /// executing.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event = Event::empty();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event).enq()?;
    /// // Map a buffer using `queue_2`, ensuring the map does not begin until
    /// // after the kernel command has completed:
    /// buffer.map().queue(&queue_2).ewait(&event).enq_async()?;
    /// ```
    ///
    /// [`Event::empty`]: struct.Event.html#method.empty
    pub fn enew<'e, En>(mut self, enew: En) -> BufferMapCmd<'c, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>> {
        self.cmd.enew = Some(enew.into());
        self
    }

    /// Returns operation details.
    #[inline]
    fn enq_details(&mut self) -> OclResult<(usize, usize, Queue, MapFlags,
            Option<ClWaitListPtrEnum<'c>>, Option<ClNullEventPtrEnum<'c>>)> {
        if let BufferCmdKind::Map = self.cmd.kind {
            if let BufferCmdDataShape::Lin { offset } = self.cmd.shape {
                let len = match self.len {
                    Some(l) => l,
                    None => self.cmd.mem_len,
                };

                check_len(self.cmd.mem_len, len, offset)?;

                let queue = match self.cmd.queue {
                    Some(q) => q.clone(),
                    None => return Err(BufferCmdError::NoQueue.into()),
                };

                let flags = self.flags.unwrap_or(MapFlags::empty());

                Ok((offset, len, queue, flags, self.cmd.ewait.take(), self.cmd.enew.take()))
            } else {
                Err(BufferCmdError::RectUnavailable.into())
            }
        } else {
            unreachable!();
        }
    }

    /// Enqueues a map command, blocking the current thread until it
    /// completes and returns a reference to the mapped memory.
    ///
    /// ## Safety
    ///
    /// The caller must ensure that either only one mapping of a buffer exists
    /// at a time or that, if simultaneously mapping for the purposes of
    /// sub-region access or whole-buffer aliasing, no two mappings will allow
    /// writes to the same memory region at the same time. Use atomics or some
    /// other synchronization mechanism to ensure this.
    pub unsafe fn enq(mut self) -> OclResult<MemMap<T>> {
        let (offset, len, queue, flags, ewait, enew, /*is_mapped*/) = self.enq_details()?;

        let mm_core = core::enqueue_map_buffer::<T, _, _, _>(&queue,
            &self.cmd.buffer.obj_core, true, flags, offset, len, ewait, enew)?;

        let unmap_event = None;

        Ok(MemMap::new(mm_core, len, None, unmap_event, self.cmd.buffer.obj_core.clone(),
            queue))
    }

    /// Enqueues a map command and returns a future representing the
    /// completion of that map command.
    ///
    /// The returned future will resolve to a reference to the mapped memory.
    ///
    /// ## Safety
    ///
    /// The caller must ensure that either only one mapping of a buffer exists
    /// at a time or that, if simultaneously mapping for the purposes of
    /// sub-region access or whole-buffer aliasing, no two mappings will allow
    /// writes to the same memory region at the same time. Use atomics or some
    /// other synchronization mechanism to ensure this.
    pub unsafe fn enq_async(mut self) -> OclResult<FutureMemMap<T>> {
        let (offset, len, queue, flags, ewait, enew, /*is_mapped*/) = self.enq_details()?;

        let mut map_event = Event::empty();

        let mm_core = core::enqueue_map_buffer::<T, _, _, _>(&queue,
            &self.cmd.buffer.obj_core, false, flags, offset, len, ewait,
            Some(&mut map_event))?;

        // If a 'new/null event' has been set, copy pointer
        // into it and increase refcount (to 2).
        if let Some(mut self_enew) = enew {
            // map_event/self_enew refcount: 2
            self_enew.clone_from(&map_event)
        }

        Ok(FutureMemMap::new(mm_core, len, map_event,
            self.cmd.buffer.obj_core.clone(), queue, /*is_mapped*/))

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
    len: usize,
    offset: Option<usize>,
    _data: PhantomData<T>,
}

impl<T: OclPrm> Buffer<T> {
    /// Returns a new buffer builder.
    ///
    /// This is the preferred (and forward compatible) way to create a buffer.
    pub fn builder<'a>() -> BufferBuilder<'a, T> {
        BufferBuilder::new()
    }

    /// Creates a new buffer.
    ///
    /// [UNSTABLE]: Arguments may still be in a state of flux. It is
    /// recommended to use `::builder` instead.
    ///
    /// See the [`BufferBuilder`] and [SDK] documentation for argument
    /// details.
    ///
    /// ### Safety
    ///
    /// Incorrectly using flags and/or host_slice is unsafe.
    ///
    /// [`BufferBuilder`]: builders/struct.BufferBuilder.html
    /// [SDK]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
    ///
    pub unsafe fn new<'e, 'o, Q, D>(que_ctx: Q, flags: MemFlags, len: D,
            host_slice: Option<&[T]>) -> OclResult<Buffer<T>>
            where Q: Into<QueCtx<'o>>, D: Into<SpatialDims> {
        // let flags = flags_opt.unwrap_or(::flags::MEM_READ_WRITE);
        let len = len.into().to_len();
        let que_ctx = que_ctx.into();

        let ctx_owned;
        let ctx_ref = match que_ctx {
            QueCtx::Queue(ref q) => {
                ctx_owned = q.context();
                &ctx_owned
            },
            QueCtx::Context(c) => c,
        };

        let obj_core = core::create_buffer(ctx_ref, flags, len, host_slice)?;

        debug_assert!({
            let size_info = match core::get_mem_object_info(&obj_core, MemInfo::Size)? {
                MemInfoResult::Size(len_bytes) => len_bytes,
                _ => unreachable!(),
            };
            size_info >= (::std::mem::size_of::<T>() * len)
        });

        let buf = Buffer {
            obj_core,
            queue: que_ctx.into(),
            len,
            offset: None,
            _data: PhantomData,
        };

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
    /// See the [`BufferCmd` docs](builders/struct.BufferCmd.html)
    /// for more info.
    ///
    #[cfg(not(feature="opencl_vendor_mesa"))]
    pub fn from_gl_buffer<'o, Q>(que_ctx: Q, flags_opt: Option<MemFlags>, gl_object: cl_GLuint)
            -> OclResult<Buffer<T>>
            where Q: Into<QueCtx<'o>> {
        let flags = flags_opt.unwrap_or(core::MEM_READ_WRITE);
        let que_ctx = que_ctx.into();

        let obj_core = match que_ctx {
            QueCtx::Queue(ref q) => unsafe { core::create_from_gl_buffer(&q.context(), gl_object, flags)? },
            QueCtx::Context(c) => unsafe { core::create_from_gl_buffer(c, gl_object, flags)? },
        };

        let len = match core::get_mem_object_info(&obj_core, MemInfo::Size)? {
            MemInfoResult::Size(len_bytes) => len_bytes / ::std::mem::size_of::<T>(),
            _ => unreachable!(),
        };

        let buf = Buffer {
            obj_core,
            queue: que_ctx.into(),
            len,
            offset: None,
            _data: PhantomData,
        };

        Ok(buf)
    }

    /// Returns a command builder used to read, write, copy, etc.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](builders/struct.BufferCmd)
    /// for more details.
    ///
    ///
    #[inline]
    pub fn cmd<'c>(&'c self) -> BufferCmd<'c, T> {
        BufferCmd::new(self, self.queue.as_ref(), /*&self.obj_core,*/ self.len())
    }

    /// Returns a command builder used to read data.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](builders/struct.BufferCmd#method.read)
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
    /// See the [command builder documentation](builders/struct.BufferCmd#method.write)
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
    /// Enqueuing a map command will map a region of a buffer into the host
    /// address space and return a [`MemMap`] or [`FutureMemMap`], allowing
    /// access to this mapped region. Accessing memory via a [`MemMap`] is
    /// exactly like using a [slice].
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// ### More Information
    ///
    /// See the [command builder
    /// documentation](builders/struct.BufferCmd#method.map) or
    /// [official SDK][map_buffer] for more details.
    ///
    /// [map_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueMapBuffer.html
    /// [`MemMap`]: struct.MemMap.html
    /// [`FutureMemMap`]: async/struct.FutureMemMap.html
    /// [slice]: https://doc.rust-lang.org/std/primitive.slice.html
    #[inline]
    pub fn map<'c>(&'c self) -> BufferMapCmd<'c, T> {
        self.cmd().map()
    }

    /// Specifies that this command will be a copy operation.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](builders/struct.BufferCmd#method.copy)
    /// for more details.
    ///
    #[inline]
    pub fn copy<'c, M>(&'c self, dst_buffer: &'c M, dst_offset: Option<usize>, len: Option<usize>)
            -> BufferCmd<'c, T>
            where M: AsMem<T> {
        self.cmd().copy(dst_buffer, dst_offset, len)
    }

    // /// Returns the origin of the sub-buffer within its buffer if this is a
    // /// sub-buffer.
    // #[inline]
    // pub fn origin(&self) -> Option<&SpatialDims> {
    //     self.origin.as_ref()
    // }

    /// Returns the offset of the sub-buffer within its buffer if this is a
    /// sub-buffer.
    #[inline]
    pub fn offset(&self) -> Option<usize> {
        // if self.is_sub_buffer() {
        //     match self.mem_info(MemInfo::Offset)? {
        //         MemInfoResult::Offset(off) => Ok(Some(off)),
        //         _ => unreachable!(),
        //     }
        // } else {
        //     Ok(None)
        // }
        self.offset
    }

    /// Returns the length of the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    // /// Returns the length of the buffer.
    // ///
    // /// Equivalent to `::dims().to_len()`.
    // #[inline]
    // pub fn len(&self) -> usize {
    //     self.dims.to_len()
    // }

    /// Returns true if this is a sub-buffer.
    #[inline]
    pub fn is_sub_buffer(&self) -> bool {
        // match self.mem_info(MemInfo::AssociatedMemobject) {
        //     MemInfoResult::AssociatedMemobject(Some(_)) => true,
        //     MemInfoResult::AssociatedMemobject(None) => false,
        //     _ => unreachable!(),
        // }
        debug_assert!({
            let is_sub_buffer = match self.mem_info(MemInfo::AssociatedMemobject).unwrap() {
                MemInfoResult::AssociatedMemobject(Some(_)) => true,
                MemInfoResult::AssociatedMemobject(None) => panic!("Buffer::is_sub_buffer"),
                _ => unreachable!(),
            };
            self.offset.is_some() == is_sub_buffer
        });
        self.offset.is_some()
    }

    /// Returns info about the underlying memory object.
    #[inline]
    pub fn mem_info(&self, info_kind: MemInfo) -> OclCoreResult<MemInfoResult> {
        core::get_mem_object_info(&self.obj_core, info_kind)
    }

    /// Changes the default queue used by this buffer for all subsequent
    /// command enqueue operations (reads, writes, etc.).
    ///
    /// The default queue is the queue which will be used when enqueuing
    /// commands if no queue is specified.
    ///
    /// Without a default queue:
    ///
    /// ```rust,ignore
    /// buffer.read(data).queue(&queue).enq()?;
    /// ```
    ///
    /// With a default queue:
    ///
    /// ```rust,ignore
    /// buffer.set_default_queue(queue.clone());
    /// buffer.read(data).enq()?;
    /// ```
    ///
    /// The default queue can also be set when creating a buffer by using the
    /// [`BufferBuilder::queue`] method.
    ///
    /// This method returns a mutable reference for optional chaining i.e.:
    ///
    /// ```rust,ignore
    /// buffer.set_default_queue(queue).read(....)...;
    /// ```
    ///
    /// [`BufferBuilder::queue`]: builders/struct.BufferBuilder.html#method.queue
    //
    // TODO: Allow `Option<Queue>` (to unset queue)?
    //
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
    /// The default queue is the queue which will be used when enqueuing
    /// commands if no queue is specified.
    #[inline]
    pub fn default_queue(&self) -> Option<&Queue> {
        self.queue.as_ref()
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    ///
    #[inline]
    pub fn as_core(&self) -> &MemCore {
        &self.obj_core
    }

    /// Returns the memory flags used during the creation of this buffer.
    ///
    #[inline]
    pub fn flags(&self) -> OclResult<MemFlags> {
        match self.mem_info(MemInfo::Flags)? {
            MemInfoResult::Flags(flags) => Ok(flags),
            _ => unreachable!(),
        }
    }

    // /// Returns a reference to the `AtomicBool` tracking whether or not this
    // /// buffer is mapped.
    // ///
    // /// If `None` is returned, this buffer is not able to be mapped.
    // pub fn is_mapped(&self) -> Option<&Arc<AtomicBool>> {
    //     self.is_mapped.as_ref()
    // }

    /// Creates a new sub-buffer from a region of this buffer.
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
    /// `offset` and `len` set up the region of the sub-buffer within the
    ///  original buffer and must not fall beyond the boundaries of it.
    ///
    /// `offset` must be a multiple of the `DeviceInfo::MemBaseAddrAlign`
    /// otherwise you will get a `CL_MISALIGNED_SUB_BUFFER_OFFSET` error. To
    /// determine, use `Device::mem_base_addr_align` for the device associated
    /// with the queue which will be use with this sub-buffer.
    ///
    /// [SDK]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateSubBuffer.html
    /// [`ocl::flags`]: flags/index.html
    /// [mem_flags]: flags/struct.MemFlags.html
    /// [`MemFlags::new().read_write()`] flags/struct.MemFlags.html#method.read_write
    ///
    pub fn create_sub_buffer<Do, Dl>(&self, flags_opt: Option<MemFlags>, offset: Do,
            len: Dl) -> OclResult<Buffer<T>>
            where Do: Into<SpatialDims>, Dl: Into<SpatialDims> {
        let flags = flags_opt.unwrap_or(crate::flags::MEM_READ_WRITE);

        // Check flags here to preempt a somewhat vague OpenCL runtime error message:
        assert!(!flags.contains(crate::flags::MEM_USE_HOST_PTR) &&
            !flags.contains(crate::flags::MEM_ALLOC_HOST_PTR) &&
            !flags.contains(crate::flags::MEM_COPY_HOST_PTR),
            "'MEM_USE_HOST_PTR', 'MEM_ALLOC_HOST_PTR', or 'MEM_COPY_HOST_PTR' flags may \
            not be specified when creating a sub-buffer. They will be inherited from \
            the containing buffer.");

        let offset = offset.into().to_len();
        let len = len.into().to_len();

        let buffer_len = self.len();
        // let offsets = origin.to_len();
        // let len = dims.to_len();

        if offset > buffer_len {
            return Err(format!("Buffer::create_sub_buffer: Origin ({:?}) is outside of the \
                dimensions of the source buffer ({:?}).", offset, buffer_len).into());
        }

        if offset + len > buffer_len {
            return Err(format!("Buffer::create_sub_buffer: Sub-buffer region (origin: '{:?}', \
                len: '{:?}') exceeds the dimensions of the source buffer ({:?}).",
                offset, len, buffer_len).into());
        }

        let obj_core = core::create_sub_buffer::<T>(self, flags,
            &BufferRegion::new(offset, len))?;

        Ok(Buffer {
            obj_core: obj_core,
            queue: self.default_queue().cloned(),
            len,
            // Share mapped status with super-buffer:
            // is_mapped: self.is_mapped.clone(),
            offset: Some(offset),
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
// * TODO: Add examples and details. For now see project examples folder.
// * TODO: Consider converting this to a re-usable builder.
//
#[must_use = "builders do nothing unless '::build' is called"]
#[derive(Debug)]
pub struct BufferBuilder<'a, T> where T: OclPrm {
    queue_option: Option<QueCtx<'a>>,
    flags: Option<MemFlags>,
    host_slice: HostSlice<'a, T>,
    len: usize,
    fill_val: Option<(T, Option<ClNullEventPtrEnum<'a>>)>
}

impl<'a, T> BufferBuilder<'a, T> where T: 'a + OclPrm {
    /// Returns a new buffer builder.
    pub fn new() -> BufferBuilder<'a, T> {
        BufferBuilder {
            queue_option: None,
            flags: None,
            host_slice: HostSlice::None,
            len: 0,
            fill_val: None,
        }
    }

    /// Sets the context with which to associate the buffer.
    ///
    /// May not be used in conjunction with [`::queue`] (use one or the other).
    ///
    /// [`::queue`]: builders/struct.BufferBuilder.html#method.queue
    pub fn context<'o>(mut self, context: &'o Context) -> BufferBuilder<'a, T>
            where 'o: 'a {
        assert!(self.queue_option.is_none(), "A context or queue has already been set.");
        self.queue_option = Some(QueCtx::Context(context));
        self
    }

    /// Specifies the default queue used to be used by the buffer for all
    /// command enqueue operations (reads, writes, etc.).
    ///
    /// The default queue is the queue which will be used when enqueuing
    /// commands if no queue is specified.
    ///
    /// Without a default queue:
    ///
    /// ```rust,ignore
    /// buffer.read(data).queue(&queue).enq()?;
    /// ```
    ///
    /// With a default queue:
    ///
    /// ```rust,ignore
    /// buffer.read(data).enq()?;
    /// ```
    ///
    /// If this is set, the context associated with the `default_queue` will
    /// be used when creating the buffer. Attempting to specify the context
    /// separately (by calling [`::context`]) will cause a panic.
    ///
    /// [`::context`]: builders/struct.BufferBuilder.html#method.context
    pub fn queue<'b>(mut self, default_queue: Queue) -> BufferBuilder<'a, T> {
        assert!(self.queue_option.is_none(), "A context or queue has already been set.");
        self.queue_option = Some(QueCtx::Queue(default_queue));
        self
    }

    /// Sets the flags used when creating the buffer.
    ///
    /// Defaults to `flags::MEM_READ_WRITE` aka.
    /// `MemFlags::new().read_write()` if this is not set. See the [SDK Docs]
    /// for more information about flags. Note that the names of all flags in
    /// this library have the `CL_` prefix removed for brevity.
    ///
    /// ### Panics
    ///
    /// Due to its unsafety, setting the
    /// `MEM_USE_HOST_PTR`/`MemFlags::new()::use_host_ptr()` flag will cause a
    /// panic. Use the `::use_host_slice` method instead.
    ///
    /// [SDK Docs]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
    pub fn flags<'b>(mut self, flags: MemFlags) -> BufferBuilder<'a, T> {
        assert!(!flags.contains(MemFlags::new().use_host_ptr()),
            "The `BufferBuilder::flags` method may not be used to set the \
            `MEM_USE_HOST_PTR` flag. Use the `::use_host_ptr` method instead.");
        self.flags = Some(flags);
        self
    }

    /// Specifies a region of host memory to use as storage for the buffer.
    ///
    /// OpenCL implementations are allowed to cache the buffer contents
    /// pointed to by `host_slice` in device memory. This cached copy can be
    /// used when kernels are executed on a device.
    ///
    /// The result of OpenCL commands that operate on multiple buffer objects
    /// created with the same `host_slice` or overlapping host regions is
    /// considered to be undefined
    ///
    /// Refer to the [description of the alignment][align_rules] rules for
    /// `host_slice` for memory objects (buffer and images) created using
    /// this method.
    ///
    /// Automatically sets the `flags::MEM_USE_HOST_PTR` aka.
    /// `MemFlags::new().use_host_ptr()` flag.
    ///
    /// ### Panics
    ///
    /// `::copy_host_slice` or `::use_host_slice` must not have already been
    /// called.
    ///
    /// ### Safety
    ///
    /// The caller must ensure that `host_slice` lives until the buffer is
    /// destroyed. The caller must also ensure that only one buffer uses
    /// `host_slice` and that it is not tampered with inappropriately.
    ///
    /// [align_rules]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/dataTypes.html
    pub unsafe fn use_host_slice<'d>(mut self, host_slice: &'d [T]) -> BufferBuilder<'a, T>
            where 'd: 'a {
        assert!(self.host_slice.is_none(), "BufferBuilder::use_host_slice: \
            A host slice has already been specified.");
        self.host_slice = HostSlice::Use(host_slice);
        self
    }

    /// Specifies a region of memory to copy into the buffer upon creation.
    ///
    /// Automatically sets the `flags::MEM_COPY_HOST_PTR` aka.
    /// `MemFlags::new().copy_host_ptr()` flag.
    ///
    /// ### Panics
    ///
    /// `::copy_host_slice` or `::use_host_slice` must not have already been
    /// called.
    ///
    pub fn copy_host_slice<'d>(mut self, host_slice: &'d [T]) -> BufferBuilder<'a, T>
            where 'd: 'a {
        assert!(self.host_slice.is_none(), "BufferBuilder::copy_host_slice: \
            A host slice has already been specified.");
        self.host_slice = HostSlice::Copy(host_slice);
        self
    }

    /// Sets the length for this buffer.
    ///
    /// Note that although sizes in the standard OpenCL API are expressed in
    /// bytes, sizes, lengths, and dimensions in this library are always
    /// specified in `bytes / sizeof(T)` (like everything else in Rust) unless
    /// otherwise noted.
    pub fn len<'b, D>(mut self, len: D) -> BufferBuilder<'a, T>
            where D: Into<SpatialDims> {
        self.len = len.into().to_len();
        self
    }

    /// Allows the caller to automatically fill the buffer with a value (such
    /// as zero) immediately after creation.
    ///
    /// Use [`::fill_event`] to set an event associated with the completion of
    /// the fill command if you want it to execute asynchronously (it will
    /// otherwise block the calling thread).
    ///
    /// Platforms that have trouble with `clEnqueueFillBuffer` such as
    /// [pocl](http://portablecl.org/) should not use this option and should
    /// handle initializing buffers manually (using a kernel or copy host data
    /// flag).
    ///
    /// ### Examples
    ///
    /// * TODO: Provide examples once this stabilizes.
    ///
    /// [UNSTABLE]: May be changed or removed.
    ///
    /// [`::fill_event`]: builders/struct.BufferBuilder.html#method.fill_event
    pub fn fill_val(mut self, fill_val: T) -> BufferBuilder<'a, T> {
        self.fill_val = Some((fill_val, None));
        self
    }

    /// Specifies the (empty) event to use for association with the completion
    /// of the fill command.
    ///
    /// `enew` specifies an empty event (generally a `&mut Event`) to be
    /// associated with the fill command which will be enqueued after creation
    /// and just before returning the new buffer. It is up to the caller to
    /// ensure that the command has completed before performing any other
    /// operations on the buffer. Failure to do so may cause the fill command
    /// to run **after** subsequently queued commands if multiple or
    /// out-of-order queues are being used.
    ///
    /// Not calling this method at all will cause the fill command to block
    /// before returning the new buffer and is the safe option if you don't
    /// want to worry about it.
    ///
    pub fn fill_event<'b, 'e, En>(mut self, enew: En) -> BufferBuilder<'a, T>
            where 'e: 'a, En: Into<ClNullEventPtrEnum<'e>> {
        match self.fill_val {
            Some(ref fv) => assert!(fv.1.is_some(), "Buffer::fill_event: Fill event already set."),
            None => panic!("Buffer::fill_event: Fill value must be set first"),
        }
        self.fill_val = self.fill_val.take().map(|fv| (fv.0, Some(enew.into())));
        self
    }

    /// Creates a buffer and returns it.
    ///
    /// Dimensions and either a context or default queue must be specified
    /// before calling `::build`.
    pub fn build(self) -> OclResult<Buffer<T>> {
        let mut flags = match self.flags {
            Some(f) => f,
            None => MemFlags::new().read_write(),
        };

        let host_slice = match self.host_slice {
            HostSlice::Use(hs) => {
                flags.insert(MemFlags::new().use_host_ptr());
                Some(hs)
            }
            HostSlice::Copy(hs) => {
                if self.fill_val.is_some() {
                    panic!("ocl::BufferBuilder::build: Cannot create a buffer with both
                        'copy_host_slice' and 'fill_val' specified. Use one or the other.");
                }

                flags.insert(MemFlags::new().copy_host_ptr());
                Some(hs)
            },
            HostSlice::None => None,
        };

        let qc = match self.queue_option {
            Some(qc) => qc,
            None => panic!("ocl::BufferBuilder::build: A context or default queue must be set \
                with '.context(...)' or '.queue(...)'."),
        };

        let len = match self.len {
            0 => panic!("ocl::BufferBuilder::build: The length must be set with \
                '.len(...)' and cannot be zero."),
            l @ _ => l,
        };

        let device_ver = match qc {
            QueCtx::Queue(ref queue) => Some(queue.device_version()),
            QueCtx::Context(_) => None,
        };

        let buf = unsafe { Buffer::new(qc, flags, len, host_slice)? };

        // Fill buffer if `fill_val` and a queue have been specified,
        // blocking if the `fill_event` is `None`.
        if let Some((val, fill_event)) = self.fill_val {
            match device_ver {
                Some(dv) => {
                    if dv >= [1, 2].into() {
                        match fill_event {
                            Some(enew) => buf.cmd().fill(val, None).enew(enew).enq()?,
                            None => {
                                let mut new_event = Event::empty();
                                buf.cmd().fill(val, None).enew(&mut new_event).enq()?;
                                new_event.wait_for()?;
                            }
                        }
                    } else {
                        let fill_vec = vec![val; buf.len()];
                        match fill_event {
                            Some(enew) => buf.cmd().write(&fill_vec).enew(enew).enq()?,
                            None => {
                                let mut new_event = Event::empty();
                                buf.cmd().write(&fill_vec).enew(&mut new_event).enq()?;
                                new_event.wait_for()?;
                            }
                        }
                    }
                },
                None => panic!("ocl::BufferBuilder::build: A queue must be specified \
                    for this builder with `::queue` when using `::fill_val`."),
            }
        }

        Ok(buf)
    }
}
