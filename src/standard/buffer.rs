//! Interfaces with a buffer.

use std;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
#[cfg(feature = "experimental_async_rw")]
use futures::{task, Future, Poll, Async};
use ffi::cl_GLuint;
use core::{self, Error as OclError, Result as OclResult, OclPrm, Mem as MemCore,
    MemFlags, MemInfo, MemInfoResult, BufferRegion, MapFlags, AsMem, MemCmdRw,
    MemCmdAll, Event as EventCore, ClNullEventPtr};
use ::{Context, Queue, SpatialDims, FutureMemMap, MemMap};
use standard::{ClNullEventPtrEnum, ClWaitListPtrEnum};
#[cfg(feature = "experimental_async_rw")]
use standard::{Event, _unpark_task, box_raw_void};



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


#[allow(dead_code)]
#[cfg(feature = "experimental_async_rw")]
pub struct ReadCompletion<'d, T> where T: 'd {
    event: Event,
    data: &'d mut [T],
}

#[allow(dead_code)]
#[cfg(feature = "experimental_async_rw")]
impl<'d, T> ReadCompletion<'d, T> where T: 'd + OclPrm {
    pub fn new(event: Event, data: &'d mut [T]) -> ReadCompletion<'d, T> {
        ReadCompletion {
            event: event,
            data: data,
        }
    }
}

/// Non-blocking, proper implementation.
#[cfg(feature = "event_callbacks")]
#[cfg(feature = "experimental_async_rw")]
impl<'d, T> Future for ReadCompletion<'d, T> where T: 'd + OclPrm {
    type Item = ();
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.event.is_complete() {
            Ok(true) => {
                Ok(Async::Ready(()))
            }
            Ok(false) => {
                let task_ptr = box_raw_void(task::park());
                unsafe { self.event.set_callback(Some(_unpark_task), task_ptr)?; };
                Ok(Async::NotReady)
            },
            Err(err) => Err(err),
        }
    }
}

/// Blocking implementation (yuk).
#[cfg(not(feature = "event_callbacks"))]
#[cfg(feature = "experimental_async_rw")]
impl<'d, T> Future for ReadCompletion<'d, T> {
    type Item = &'d mut [T];
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.event.wait_for()?;
        Ok(Async::Ready(()))
    }
}


/// The type of operation to be performed by a command.
pub enum BufferCmdKind<'c, T> where T: 'c {
    Unspecified,
    // Read { data: &'c mut [T] },
    // Write { data: &'c [T] },
    // Map { flags: Option<MapFlags>, len: Option<usize> },
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
        src_row_pitch: usize,
        src_slc_pitch: usize,
        dst_row_pitch: usize,
        dst_slc_pitch: usize,
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
//
// pub struct BufferCmd<'b, T: 'b + OclPrm> {
//     queue: &'b Queue,
//     obj_core: &'b MemCore,
//     block: bool,
//     lock_block: bool,
//     kind: BufferCmdKind<'b, T>,
//     shape: BufferCmdDataShape,
//     ewait: Option<&'b ClWaitListPtr>,
//     enew: Option<&'b mut ClNullEventPtr>,
//     mem_len: usize,
// }
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

/// [UNSTABLE]: All methods still in a state of tweakification.
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
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified.
    ///
    pub fn read<'d>(mut self, dst_data: &'d mut [T]) -> BufferReadCmd<'c, 'd, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::read(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Read;
        BufferReadCmd { cmd: self, data: dst_data }
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
    #[deprecated(since="0.13.0", note="Use '::read' with '::block(false)' for unsafe asynchronous reads.")]
    pub unsafe fn read_async<'d>(mut self, dst_data: &'d mut [T]) -> BufferReadCmd<'c, 'd, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::read(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Read;
        self.block = false;
        BufferReadCmd { cmd: self, data: dst_data }
    }

    /// Specifies that this command will be a write operation.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn write<'d>(mut self, src_data: &'d [T]) -> BufferWriteCmd<'c, 'd, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::write(): Operation kind \
            already set for this command.");
        self.kind = BufferCmdKind::Write;
        BufferWriteCmd { cmd: self, data: src_data }
    }

    /// Specifies that this command will be a map operation.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
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
    /// ## Errors
    ///
    /// If this is a rectangular copy, `dst_offset` and `len` must be zero.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy<M>(mut self, dst_buffer: &'c M, dst_offset: Option<usize>, len: Option<usize>)
            -> BufferCmd<'c, T>
            where M: AsMem<T>
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
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy_to_image(mut self, image: &'c MemCore, dst_origin: [usize; 3],
                region: [usize; 3]) -> BufferCmd<'c, T>
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
    /// [`::enq_async`]: struct.BufferMapCmd.html
    //
    // [FIXME]: Check/fix links.
    //
    // [TODO]: Should this error when calling non-rw `::enq`?
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
    /// Only valid for 'read', 'write', and 'copy' modes. Will error if used
    /// with any other mode.
    pub fn rect(mut self, src_origin: [usize; 3], dst_origin: [usize; 3], region: [usize; 3],
                src_row_pitch: usize, src_slc_pitch: usize, dst_row_pitch: usize,
                dst_slc_pitch: usize) -> BufferCmd<'c, T>
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
    ///
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
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                            dst_row_pitch, dst_slc_pitch } =>
                    {
                        if dst_offset.is_some() || len.is_some() { return OclError::err_string(
                            "ocl::BufferCmd::enq(): For 'rect' shaped copies, destination \
                            offset and length must be 'None'. Ex.: \
                            'cmd().copy(&{{buf_name}}, None, None)..'.");
                        }

                        core::enqueue_copy_buffer_rect::<T, _, _, _>(queue, self.obj_core, dst_buffer,
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

                        core::enqueue_fill_buffer(queue, self.obj_core, pattern,
                            offset, len, self.ewait, self.enew, Some(&queue.device_version()))
                    },
                    BufferCmdDataShape::Rect { .. } => OclError::err_string("ocl::BufferCmd::enq(): \
                        Rectangular fill is not a valid operation. Please use the default shape, linear.")
                }
            },
            BufferCmdKind::GLAcquire => {
                core::enqueue_acquire_gl_buffer(queue, self.obj_core, self.ewait, self.enew)
            },
            BufferCmdKind::GLRelease => {
                core::enqueue_release_gl_buffer(queue, self.obj_core, self.ewait, self.enew)
            },
            BufferCmdKind::Unspecified => OclError::err_string("ocl::BufferCmd::enq(): No operation \
                specified. Use '.read(...)', 'write(...)', etc. before calling '.enq()'."),
            BufferCmdKind::Map { .. } => OclError::err_string("ocl::BufferCmd::enq(): \
                For map operations use '::enq_map()' instead."),
            _ => unimplemented!(),
        }
    }
}


/// A buffer command builder used to enqueue reads.
pub struct BufferReadCmd<'c, 'd, T> where T: 'c + 'd {
    cmd: BufferCmd<'c, T>,
    data: &'d mut [T],
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
    pub fn offset(self, offset: usize)  -> BufferReadCmd<'c, 'd, T> {
        BufferReadCmd { cmd: self.cmd.offset(offset), ..self }
    }

    /// Specifies that this will be a rectangularly shaped operation
    /// (the default being linear).
    ///
    /// Only valid for 'read', 'write', and 'copy' modes. Will error if used
    /// with any other mode.
    pub fn rect(mut self, src_origin: [usize; 3], dst_origin: [usize; 3], region: [usize; 3],
                src_row_pitch: usize, src_slc_pitch: usize, dst_row_pitch: usize,
                dst_slc_pitch: usize) -> BufferReadCmd<'c, 'd, T>
    {
        if let BufferCmdDataShape::Lin { offset } = self.cmd.shape {
            assert!(offset == 0, "ocl::BufferCmd::rect(): This command builder has already been \
                set to linear mode with '::offset`. You cannot call both '::offset' and '::rect'.");
        }

        self.cmd.shape = BufferCmdDataShape::Rect { src_origin: src_origin, dst_origin: dst_origin,
            region: region, src_row_pitch: src_row_pitch, src_slc_pitch: src_slc_pitch,
            dst_row_pitch: dst_row_pitch, dst_slc_pitch: dst_slc_pitch };

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

    /// Enqueues this command.
    ///
    pub fn enq(self) -> OclResult<()> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        match self.cmd.kind {
            BufferCmdKind::Read => {
                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.cmd.mem_len, self.data.len(), offset));

                        unsafe { core::enqueue_read_buffer(queue, self.cmd.obj_core, self.cmd.block,
                            offset, self.data, self.cmd.ewait, self.cmd.enew) }
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                            dst_row_pitch, dst_slc_pitch } =>
                    {
                        // Verify dims given.
                        // try!(Ok(()));

                        unsafe { core::enqueue_read_buffer_rect(queue, self.cmd.obj_core,
                            self.cmd.block, src_origin, dst_origin, region, src_row_pitch,
                            src_slc_pitch, dst_row_pitch, dst_slc_pitch, self.data,
                            self.cmd.ewait, self.cmd.enew) }
                    }
                }
            },
            _ => Err("ocl::BufferReadCmd::enq(): Invalid command kind.".into()),
        }
    }

    /// Enqueues this command asynchronously.
    ///
    #[allow(unused_unsafe)]
    #[cfg(feature = "experimental_async_rw")]
    pub unsafe fn enq_unsafely(mut self) -> OclResult<ReadCompletion<'d, T>> {
        let queue = match self.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        match self.cmd.kind {
            BufferCmdKind::Read => {
                // let data = unsafe { std::mem::replace(data, std::mem::uninitialized()) };
                let mut read_event = EventCore::null();

                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.cmd.mem_len, self.data.len(), offset));

                        unsafe { core::enqueue_read_buffer(self.cmd.queue, self.cmd.obj_core, false,
                            offset, self.data, self.cmd.ewait.take(), Some(&mut read_event))?; }
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                            dst_row_pitch, dst_slc_pitch } =>
                    {
                        unsafe { core::enqueue_read_buffer_rect(self.cmd.queue, self.cmd.obj_core,
                            false, src_origin, dst_origin, region, src_row_pitch,
                            src_slc_pitch, dst_row_pitch, dst_slc_pitch, self.data,
                            self.cmd.ewait.take(), Some(&mut read_event))?; }
                    }
                }

                if let Some(ref mut self_enew) = self.enew.take() {
                    // Should be equivalent to `.clone().into_raw()` [TODO]: test.
                    // core::retain_event(&read_event)?;
                    // *(self_enew.alloc_new()) = *(read_event.as_ptr_ref());
                    // read_event/self_enew refcount: 2

                    unsafe { *(self_enew.alloc_new()) = read_event.clone().into_raw(); }
                }

                Ok(ReadCompletion::new(Event::from(read_event), self.data))
            },
            _ => Err("ocl::BufferReadCmd::enq_async(): Invalid command kind.".into()),
        }
    }
}

// impl<'c, 'd, T> Deref for BufferReadCmd<'c, 'd, T> where T: OclPrm {
//     type Target = BufferCmd<'c, T>;

//     #[inline] fn deref(&self) -> &BufferCmd<'c, T> { &self.cmd }
// }

// impl<'c, 'd, T> DerefMut for BufferReadCmd<'c, 'd, T> where T: OclPrm{
//     #[inline] fn deref_mut(&mut self) -> &mut BufferCmd<'c, T> { &mut self.cmd }
// }



/// A buffer command builder used to enqueue writes.
pub struct BufferWriteCmd<'c, 'd, T> where T: 'c + 'd {
    cmd: BufferCmd<'c, T>,
    data: &'d [T],
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

    /// Specifies that this will be a rectangularly shaped operation
    /// (the default being linear).
    ///
    /// Only valid for 'read', 'write', and 'copy' modes. Will error if used
    /// with any other mode.
    pub fn rect(mut self, src_origin: [usize; 3], dst_origin: [usize; 3], region: [usize; 3],
                src_row_pitch: usize, src_slc_pitch: usize, dst_row_pitch: usize,
                dst_slc_pitch: usize) -> BufferWriteCmd<'c, 'd, T>
    {
        if let BufferCmdDataShape::Lin { offset } = self.cmd.shape {
            assert!(offset == 0, "ocl::BufferCmd::rect(): This command builder has already been \
                set to linear mode with '::offset`. You cannot call both '::offset' and '::rect'.");
        }

        self.cmd.shape = BufferCmdDataShape::Rect { src_origin: src_origin, dst_origin: dst_origin,
            region: region, src_row_pitch: src_row_pitch, src_slc_pitch: src_slc_pitch,
            dst_row_pitch: dst_row_pitch, dst_slc_pitch: dst_slc_pitch };

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

    /// Enqueues this command.
    pub fn enq(self) -> OclResult<()> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        match self.cmd.kind {
            BufferCmdKind::Write => {
                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.cmd.mem_len, self.data.len(), offset));

                        core::enqueue_write_buffer(queue, self.cmd.obj_core, self.cmd.block,
                            offset, self.data, self.cmd.ewait, self.cmd.enew)
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                            dst_row_pitch, dst_slc_pitch } =>
                    {
                        core::enqueue_write_buffer_rect(queue, self.cmd.obj_core,
                            self.cmd.block, src_origin, dst_origin, region, src_row_pitch,
                            src_slc_pitch, dst_row_pitch, dst_slc_pitch, self.data,
                            self.cmd.ewait, self.cmd.enew)
                    }
                }
            },
            _ => Err("ocl::BufferWriteCmd::enq(): Invalid command kind.".into()),
        }
    }

    /// Enqueues this command.
    #[allow(unused_unsafe)]
    #[cfg(feature = "experimental_async_rw")]
    pub fn enq_unsafely(self) -> OclResult<()> {
        let queue = match self.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        match self.cmd.kind {
            BufferCmdKind::Write => {
                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        try!(check_len(self.cmd.mem_len, self.data.len(), offset));

                        core::enqueue_write_buffer(self.cmd.queue, self.cmd.obj_core, false,
                            offset, self.data, self.cmd.ewait, self.cmd.enew)
                    },
                    BufferCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
                            dst_row_pitch, dst_slc_pitch } =>
                    {
                        core::enqueue_write_buffer_rect(self.cmd.queue, self.cmd.obj_core,
                            false, src_origin, dst_origin, region, src_row_pitch,
                            src_slc_pitch, dst_row_pitch, dst_slc_pitch, self.data,
                            self.cmd.ewait, self.cmd.enew)
                    }
                }
            },
            _ => Err("ocl::BufferWriteCmd::enq_async(): Invalid command kind.".into()),
        }
    }
}

// impl<'c, 'd, T> Deref for BufferWriteCmd<'c, 'd, T> where T: OclPrm {
//     type Target = BufferCmd<'c, T>;

//     #[inline] fn deref(&self) -> &BufferCmd<'c, T> { &self.cmd }
// }

// impl<'c, 'd, T> DerefMut for BufferWriteCmd<'c, 'd, T> where T: OclPrm{
//     #[inline] fn deref_mut(&mut self) -> &mut BufferCmd<'c, T> { &mut self.cmd }
// }


/// A buffer command builder used to enqueue maps.
pub struct BufferMapCmd<'c, T> where T: 'c {
    cmd: BufferCmd<'c, T>,
    flags: Option<MapFlags>,
    len: Option<usize> ,
}

impl<'c, T> BufferMapCmd<'c, T> where T: OclPrm {
    pub fn flags(mut self, flags: MapFlags) -> BufferMapCmd<'c, T> {
         self.flags = Some(flags);

        self
    }

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
    ///
    /// For all other operation types use `::map` instead.
    ///
    pub fn enq(mut self) -> OclResult<MemMap<T>> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        match self.cmd.kind {
            BufferCmdKind::Map => {
                match self.cmd.shape {
                    BufferCmdDataShape::Lin { offset } => {
                        let len = match self.len {
                            Some(l) => l,
                            None => self.cmd.mem_len,
                        };

                        check_len(self.cmd.mem_len, len, offset)?;
                        let flags = self.flags.unwrap_or(MapFlags::empty());

                        unsafe {
                            let mm_core = core::enqueue_map_buffer::<T, _, _, _>(queue,
                                self.cmd.obj_core, true, flags, offset, len, self.cmd.ewait.take(),
                                self.cmd.enew.take())?;

                            let unmap_event = None;

                            Ok(MemMap::new(mm_core, len, unmap_event, self.cmd.obj_core.clone(),
                                queue.core().clone()))
                        }
                    },
                    BufferCmdDataShape::Rect { .. } => {
                        OclError::err_string("ocl::BufferCmd::enq_map(): A rectangular map is not a valid \
                            operation. Please use the default shape, linear.")
                    },
                }
            },
            BufferCmdKind::Unspecified => OclError::err_string("ocl::BufferCmd::enq_map(): No operation \
                specified. Use '::map', before calling '::enq_map'."),
            _ => OclError::err_string("ocl::BufferCmd::enq_map(): For non-map operations use '::enq' instead."),
        }
    }

    /// Enqueues a map command and returns a future representing the
    /// completion of that map command and containing a reference to the
    /// mapped memory.
    ///
    /// For all other operation types use `::map` instead.
    ///
    pub fn enq_async(mut self) -> OclResult<FutureMemMap<T>> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("BufferCmd::enq: No queue set.".into()),
        };

        match self.cmd.kind {
            BufferCmdKind::Map => {
                if let BufferCmdDataShape::Lin { offset } = self.cmd.shape {
                    let len = match self.len {
                        Some(l) => l,
                        None => self.cmd.mem_len,
                    };

                    check_len(self.cmd.mem_len, len, offset)?;

                    let flags = self.flags.unwrap_or(MapFlags::empty());

                    let future = unsafe {
                        let mut map_event = EventCore::null();

                        let mm_core = core::enqueue_map_buffer::<T, _, _, _>(queue,
                            self.cmd.obj_core, false, flags, offset, len, self.cmd.ewait.take(),
                            Some(&mut map_event))?;

                        // If a 'new/null event' has been set, copy pointer
                        // into it and increase refcount (to 2).
                        if let Some(ref mut self_enew) = self.cmd.enew.take() {
                            // // Should be equivalent to `.clone().into_raw()` [TODO]: test.
                            // core::retain_event(&map_event)?;
                            // *(self_enew.alloc_new()) = *(map_event.as_ptr_ref());
                            // // map_event/self_enew refcount: 2

                            *(self_enew.alloc_new()) = map_event.clone().into_raw();
                        }

                        FutureMemMap::new(mm_core, len, map_event, self.cmd.obj_core.clone(),
                            queue.core().clone())

                    };

                    Ok(future)
                } else {
                    OclError::err_string("ocl::BufferCmd::enq_map(): A rectangular map is not a valid \
                        operation. Please use the default shape, linear.")
                }
            },
            BufferCmdKind::Unspecified => OclError::err_string("ocl::BufferCmd::enq_map(): No operation \
                specified. Use '::map', before calling '::enq_map'."),
            _ => OclError::err_string("ocl::BufferCmd::enq_map(): For non-map operations use '::enq' instead."),
        }
    }
}

// impl<'c, T> Deref for BufferMapCmd<'c, T> where T: OclPrm {
//     type Target = BufferCmd<'c, T>;

//     #[inline] fn deref(&self) -> &BufferCmd<'c, T> { &self.cmd }
// }

// impl<'c, T> DerefMut for BufferMapCmd<'c, T> where T: OclPrm {
//     #[inline] fn deref_mut(&mut self) -> &mut BufferCmd<'c, T> { &mut self.cmd }
// }


#[derive(Debug, Clone)]
pub enum QueCtx {
    Queue(Queue),
    Context(Context),
}

impl QueCtx {
    // pub fn context(&self) -> Option<Context> {
    //     match *self {
    //         QueCtx::Queue(ref q) => Some(q.context()),
    //         QueCtx::Context(ref c) => Some(c.clone()),
    //         QueCtx::None => None,
    //     }
    // }

    pub fn queue(&self) -> Option<&Queue> {
        match *self {
            QueCtx::Queue(ref q) => Some(q),
            QueCtx::Context(_) => None,
        }
    }
}

impl From<Queue> for QueCtx {
    fn from(q: Queue) -> QueCtx {
        QueCtx::Queue(q)
    }
}

impl<'a> From<&'a Queue> for QueCtx {
    fn from(q: &Queue) -> QueCtx {
        QueCtx::Queue(q.clone())
    }
}

impl From<Context> for QueCtx {
    fn from(c: Context) -> QueCtx {
        QueCtx::Context(c)
    }
}

impl<'a> From<&'a Context> for QueCtx {
    fn from(c: &Context) -> QueCtx {
        QueCtx::Context(c.clone())
    }
}


/// A buffer builder.
#[derive(Debug)]
pub struct BufferBuilder<'a, T> where T: 'a {
    queue_option: Option<QueCtx>,
    flags: Option<MemFlags>,
    dims: Option<SpatialDims>,
    host_data: Option<&'a [T]>,
    fill_val: Option<(T, Option<ClNullEventPtrEnum<'a>>)>
}

impl<'a, T> BufferBuilder<'a, T> where T: 'a + OclPrm {
    pub fn new() -> BufferBuilder<'a, T> {
        BufferBuilder {
            queue_option: None,
            flags: None,
            dims: None,
            host_data: None,
            fill_val: None,
        }
    }

    /// Sets the context with which to associate the buffer.
    ///
    /// May not be used in combination with `::queue` (use one or the other).
    pub fn context<'b>(mut self, context: Context) -> BufferBuilder<'a, T> {
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

    pub fn fill_val<'b, 'e, En>(mut self, fill_val: T, enew: Option<En>)
            -> BufferBuilder<'a, T>
            where 'e: 'a, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.fill_val = Some((fill_val, enew.map(|e| e.into())));
        self
    }

    pub fn build(mut self) -> OclResult<Buffer<T>> {
        match self.queue_option {
            Some(qo) => {
                let dims = match self.dims {
                    Some(d) => d,
                    None => panic!("ocl::BufferBuilder::build: The dimensions must be set with '.dims(...)'."),
                };

                Buffer::new(qo.clone(), self.flags.take(), dims, self.host_data.take(), self.fill_val.take())
            },
            None => panic!("ocl::BufferBuilder::build: A context or default queue must be set \
                with '.context(...)' or '.queue(...)'."),
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
    pub fn new<'e, D, Q, En>(que_ctx: Q, flags_opt: Option<MemFlags>, dims: D,
            host_data: Option<&[T]>, fill_val: Option<(T, Option<En>)>) -> OclResult<Buffer<T>>
            where D: Into<SpatialDims>, Q: Into<QueCtx>, En: Into<ClNullEventPtrEnum<'e>>
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
            QueCtx::Context(ref c) => unsafe { core::create_buffer(c,
                flags, len, host_data)? },
        };

        let buf = Buffer {
            obj_core: obj_core,
            queue: que_ctx.queue().cloned(),
            dims: dims,
            len: len,
            flags: flags,
            _data: PhantomData,
        };

        if let Some((val, en)) = fill_val {
            let enew = en.map(|e| e.into());

            // Create a new event and use it then copy that created event into enew if it exists.

            buf.cmd()
                .fill(val, None)
                .enew_opt(enew)
                .enq()?;



        }

        Ok(buf)
    }

    /// [UNTESTED]
    /// Creates a buffer linked to a previously created OpenGL buffer object.
    ///
    ///
    /// ### Errors
    ///
    /// Don't forget to `.cmd().gl_acquire().enq()` before using it and
    /// `.cmd().gl_release().enq()` after.
    ///
    /// See the [`BufferCmd` docs](struct.BufferCmd.html)
    /// for more info.
    ///
    pub fn from_gl_buffer<D, Q>(que_ctx: Q, flags_opt: Option<MemFlags>, dims: D,
            gl_object: cl_GLuint) -> OclResult<Buffer<T>>
            where D: Into<SpatialDims>, Q: Into<QueCtx>
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
            QueCtx::Context(ref c) => unsafe { core::create_from_gl_buffer(c, gl_object, flags)? },
        };

        let buf = Buffer {
            obj_core: obj_core,
            queue: que_ctx.queue().cloned(),
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
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn cmd<'c>(&'c self) -> BufferCmd<'c, T> {
        BufferCmd::new(self.queue.as_ref(), &self.obj_core, self.len)
    }

    /// Returns a command builder used to read data.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn read<'c, 'd>(&'c self, data: &'d mut [T]) -> BufferReadCmd<'c, 'd, T>
            where 'd: 'c
    {
        self.cmd().read(data)
    }

    /// Returns a command builder used to write data.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn write<'c, 'd>(&'c self, data: &'d [T]) -> BufferWriteCmd<'c, 'd, T>
            where 'd: 'c
    {
        self.cmd().write(data)
    }

    /// Returns a command builder used to map data for reading or writing.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn map<'c>(&'c self) -> BufferMapCmd<'c, T> {
        self.cmd().map()
    }

    /// Specifies that this command will be a copy operation.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn copy<'c, M>(&'c self, dst_buffer: &'c M, dst_offset: Option<usize>, len: Option<usize>)
            -> BufferCmd<'c, T>
            where M: AsMem<T>
    {
        self.cmd().copy(dst_buffer, dst_offset, len)
    }

    /// Returns the length of the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the dimensions of the buffer.
    #[inline]
    pub fn dims(&self) -> &SpatialDims {
        &self.dims
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

    /// Creates a new sub-buffer and returns it if successful.
    ///
    /// See [`SubBuffer::new`][subbuf_new] for more information about arguments.
    ///
    /// [subbuf_new]: struct.SubBuffer.html#method.new
    #[inline]
    pub fn create_sub_buffer<D: Into<SpatialDims>>(&self, flags: Option<MemFlags>, origin: D,
        size: D) -> OclResult<SubBuffer<T>>
    {
        SubBuffer::new(self, flags, origin, size)
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

impl<'a, T: OclPrm> AsMem<T> for &'a mut Buffer<T> {
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


/// A subsection of buffer memory physically located on a device, such as a
/// GPU.
///
#[derive(Debug, Clone)]
pub struct SubBuffer<T: OclPrm> {
    obj_core: MemCore,
    queue: Option<Queue>,
    origin: SpatialDims,
    size: SpatialDims,
    len: usize,
    flags: MemFlags,
    _data: PhantomData<T>,
}

impl<T: OclPrm> SubBuffer<T> {
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
    /// `origin` and `size` set up the region of the sub-buffer within the
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
    pub fn new<D: Into<SpatialDims>>(buffer: &Buffer<T>, flags_opt: Option<MemFlags>, origin: D,
        size: D) -> OclResult<SubBuffer<T>>
    {
        let flags = flags_opt.unwrap_or(::flags::MEM_READ_WRITE);
        let origin: SpatialDims = origin.into();
        let size: SpatialDims = size.into();

        let buffer_len = buffer.dims().to_len();
        let origin_len = origin.to_len();
        let size_len = size.to_len();

        if origin_len > buffer_len {
            return OclError::err_string(format!("SubBuffer::new: Origin ({:?}) is outside of the \
                dimensions of the source buffer ({:?}).", origin, buffer.dims()));
        }

        if origin_len + size_len > buffer_len {
            return OclError::err_string(format!("SubBuffer::new: Sub-buffer region (origin: '{:?}', \
                size: '{:?}') exceeds the dimensions of the source buffer ({:?}).", origin, size,
                buffer.dims()));
        }

        let obj_core = core::create_sub_buffer::<T>(buffer, flags,
            &BufferRegion::new(origin.to_len(), size.to_len()))?;

        Ok(SubBuffer {
            obj_core: obj_core,
            queue: buffer.default_queue().cloned(),
            origin: origin,
            size: size,
            len: size_len,
            flags: flags,
            _data: PhantomData,
        })
    }


    /// Returns a command builder used to read, write, copy, etc.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn cmd<'c>(&'c self) -> BufferCmd<'c, T> {
        BufferCmd::new(self.queue.as_ref(), &self.obj_core, self.len)
    }

    /// Returns a command builder used to read data.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn read<'c, 'd>(&'c self, data: &'d mut [T]) -> BufferReadCmd<'c, 'd, T>
            where 'd: 'c
    {
        self.cmd().read(data)
    }

    /// Returns a command builder used to write data.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn write<'c, 'd>(&'c self, data: &'d [T]) -> BufferWriteCmd<'c, 'd, T>
            where 'd: 'c
    {
        self.cmd().write(data)
    }

    /// Returns a command builder used to map data for reading or writing.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn map<'c>(&'c self) -> BufferMapCmd<'c, T> {
        self.cmd().map()
    }

    /// Specifies that this command will be a copy operation.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the command builder documentation linked in the function signature
    /// for more information.
    ///
    #[inline]
    pub fn copy<'b, 'c, M>(&'c self, dst_buffer: &'b M, dst_offset: Option<usize>, len: Option<usize>)
            -> BufferCmd<'c, T>
            where 'b: 'c, M: AsMem<T>
    {
        self.cmd().copy(dst_buffer, dst_offset, len)
    }

    /// Returns the origin of the sub-buffer within the buffer.
    #[inline]
    pub fn origin(&self) -> &SpatialDims {
        &self.origin
    }

    /// Returns the dimensions of the sub-buffer.
    #[inline]
    pub fn dims(&self) -> &SpatialDims {
        &self.size
    }

    /// Returns the length of the sub-buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns info about the underlying memory object.
    #[inline]
    pub fn mem_info(&self, info_kind: MemInfo) -> MemInfoResult {
        core::get_mem_object_info(&self.obj_core, info_kind)
    }

    /// Changes the default queue used by this sub-buffer for reads and writes, etc.
    ///
    /// Returns a mutable reference for optional chaining i.e.:
    ///
    /// ### Example
    ///
    /// `buffer.set_default_queue(queue).read(....);`
    ///
    #[inline]
    pub fn set_default_queue<'a>(&'a mut self, queue: Queue) -> &'a mut SubBuffer<T> {
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
    #[deprecated(since="0.13.0", note="Use `::core` instead.")]
    #[inline]
    pub fn core_as_ref(&self) -> &MemCore {
        &self.obj_core
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn core(&self) -> &MemCore {
        &self.obj_core
    }

    /// Returns the memory flags used during the creation of this sub-buffer.
    ///
    /// Saves the cost of having to look them up using `::mem_info`.
    ///
    #[inline]
    pub fn flags(&self) -> MemFlags {
        self.flags
    }

    /// Formats memory info.
    #[inline]
    fn fmt_mem_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("SubBuffer Mem")
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

impl<T: OclPrm> Deref for SubBuffer<T> {
    type Target = MemCore;

    fn deref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<T: OclPrm> DerefMut for SubBuffer<T> {
    fn deref_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}

impl<T: OclPrm> AsRef<MemCore> for SubBuffer<T> {
    fn as_ref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<T: OclPrm> AsMut<MemCore> for SubBuffer<T> {
    fn as_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}

impl<'a, T: OclPrm> AsMem<T> for SubBuffer<T> {
    fn as_mem(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<'a, T: OclPrm> AsMem<T> for &'a mut SubBuffer<T> {
    fn as_mem(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<T: OclPrm> std::fmt::Display for SubBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_mem_info(f)
    }
}

unsafe impl<'a, T> MemCmdRw for SubBuffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdRw for &'a SubBuffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdRw for &'a mut SubBuffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for SubBuffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for &'a SubBuffer<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for &'a mut SubBuffer<T> where T: OclPrm {}