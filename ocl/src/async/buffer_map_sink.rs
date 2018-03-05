//! High performance buffer writes.

#![allow(unused_imports, dead_code)]

use std::ops::{Deref, DerefMut};
use futures::{Future, Poll, Async};
use core::{self, OclPrm, Mem as MemCore, MemMap as MemMapCore,
    MemFlags, MapFlags, ClNullEventPtr, ClWaitListPtr, AsMem};
use standard::{Event, EventList, Queue, Buffer, ClWaitListPtrEnum, ClNullEventPtrEnum};
use async::{Error as OclError, Result as OclResult};



/// A view of memory mapped by `clEnqueueMap{...}`.
///
///
/// [UNSTABLE]: Still in a state of flux: ~90% stable
///
//
#[derive(Debug)]
pub struct SinkMapGuard<T> where T: OclPrm {
    mem_map: MemMapCore<T>,
    len: usize,
    buffer: MemCore,
    queue: Queue,
    unmap_event: Option<Event>,
}

impl<T> SinkMapGuard<T>  where T: OclPrm {
    pub unsafe fn new(mem_map: MemMapCore<T>, len: usize, unmap_event: Option<Event>,
            buffer: MemCore, queue: Queue) -> SinkMapGuard<T> {
        SinkMapGuard {
            mem_map,
            len,
            buffer,
            queue,
            unmap_event,
        }
    }

    /// Returns a reference to the unmap target event if it has been set.
    pub fn unmap_event(&self) -> Option<&Event> {
        self.unmap_event.as_ref()
    }

    /// Returns a pointer to the host mapped memory.
    #[inline] pub fn as_ptr(&self) -> *const T { self.mem_map.as_ptr() }

    /// Returns a mutable pointer to the host mapped memory.
    #[inline] pub fn as_mut_ptr(&mut self) -> *mut T { self.mem_map.as_mut_ptr() }

    /// Returns a reference to the internal core command queue.
    #[inline] pub fn queue(&self) -> &Queue { &self.queue }

    /// Enqueues an unmap command for the memory mapping immediately.
    fn unmap(&mut self) -> OclResult<()> {
        let mut origin_event_opt = if self.unmap_event.is_some() {
            Some(Event::empty())
        } else {
            None
        };

        core::enqueue_unmap_mem_object(&self.queue, &self.buffer, &self.mem_map,
            None::<&Event>, origin_event_opt.as_mut())?;

        if let Some(origin_event) = origin_event_opt {
            if let Some(unmap_user_event) = self.unmap_event.take() {
                #[cfg(not(feature = "async_block"))]
                unsafe { origin_event.register_event_relay(unmap_user_event)?; }

                #[cfg(feature = "async_block")]
                origin_event.wait_for()?;
                #[cfg(feature = "async_block")]
                unmap_user_event.set_complete()?;
            }
        }

        Ok(())
    }
}

impl<T: OclPrm> Drop for SinkMapGuard<T> {
    fn drop(&mut self) {
        self.unmap().expect("error dropping `SinkMapGuard`");
    }
}

impl<T> Deref for SinkMapGuard<T> where T: OclPrm {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { self.mem_map.as_slice(self.len) }
    }
}

impl<T> DerefMut for SinkMapGuard<T> where T: OclPrm {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { self.mem_map.as_slice_mut(self.len) }
    }
}

impl<T: OclPrm> AsMem for SinkMapGuard<T> {
    fn as_mem(&self) -> &MemCore {
        self.mem_map.as_mem()
    }
}


/// A future that resolves to a `SinkMapGuard`.
///
#[must_use = "futures do nothing unless polled"]
#[derive(Debug)]
pub struct FutureSinkMapGuard<T: OclPrm> {
    mem_map: Option<MemMapCore<T>>,
    len: usize,
    map_event: Event,
    unmap_event: Option<Event>,
    buffer: Option<MemCore>,
    queue: Option<Queue>,
    callback_is_set: bool,
}

impl<T: OclPrm> FutureSinkMapGuard<T> {
    /// Returns a new `FutureMemMap`.
    fn new(mem_map: MemMapCore<T>, len: usize, map_event: Event, buffer: MemCore,
            queue: Queue) -> FutureSinkMapGuard<T> {
        FutureSinkMapGuard {
            mem_map: Some(mem_map),
            len: len,
            map_event: map_event,
            unmap_event: None,
            buffer: Some(buffer),
            queue: Some(queue),
            callback_is_set: false,
        }
    }

    /// Create an event which will be triggered (set complete) after this
    /// future resolves into a `SinkMapGuard` **and** after that
    /// `SinkMapGuard` is dropped.
    ///
    /// The returned event can be added to the wait list of subsequent OpenCL
    /// commands with the expectation that when all preceeding futures are
    /// complete, the event will automatically be 'triggered' by having its
    /// status set to complete, causing those commands to execute. This can be
    /// used to inject host side code in amongst OpenCL commands without
    /// thread blocking or extra delays of any kind.
    ///
    /// [UNSTABLE]: This method may be renamed or otherwise changed.
    pub fn create_unmap_event(&mut self) -> OclResult<&mut Event> {
        if let Some(ref queue) = self.queue {
            let uev = Event::user(&queue.context())?;
            self.unmap_event = Some(uev);
            Ok(self.unmap_event.as_mut().unwrap())
        } else {
            Err("FutureSinkMapGuard::create_unmap_event: No queue found!".into())
        }
    }

    /// Specifies an event which will be triggered (set complete) after this
    /// future resolves into a `SinkMapGuard` **and** after that `SinkMapGuard` is dropped
    /// or manually unmapped.
    ///
    /// See `::create_unmap_event`.
    pub fn enew_unmap<En>(mut self, mut enew: En) -> FutureSinkMapGuard<T>
            where En: ClNullEventPtr {
        {
            let unmap_event = self.create_unmap_event()
                .expect("FutureSinkMapGuard::enew_unmap");
            unsafe { enew.clone_from(unmap_event); }
        }
        self
    }

    /// Optionally specifies a queue to be used for the unmap command which
    /// will occur when the `SinkMapGuard` is dropped.
    ///
    /// If no unmap queue is specified, the same queue used during the map
    /// will be used.
    pub fn set_unmap_queue(&mut self, queue: Queue) {
        self.queue = Some(queue)
    }

    /// Returns the unmap event if it has been created.
    #[inline]
    pub fn unmap_event(&self) -> Option<&Event> {
        self.unmap_event.as_ref()
    }

    /// Blocks the current thread until the OpenCL command is complete and an
    /// appropriate lock can be obtained on the underlying data.
    pub fn wait(self) -> OclResult<SinkMapGuard<T>> {
        <Self as Future>::wait(self)
    }

    /// Resolves this `FutureSinkMapGuard` into a `SinkMapGuard`.
    fn resolve(&mut self) -> OclResult<SinkMapGuard<T>> {
        match (self.mem_map.take(), self.buffer.take(), self.queue.take()) {
            (Some(mem_map), Some(buffer), Some(queue)) => {
                unsafe { Ok(SinkMapGuard::new(mem_map, self.len, self.unmap_event.take(),
                    buffer, queue)) }
            },
            _ => Err("FutureSinkMapGuard::create_unmap_event: No queue and/or buffer found!".into()),
        }
    }
}

// impl<T: OclPrm> Future for FutureSinkMapGuard<T> {
//     type Item = ();
//     type Error = OclError;

//     #[inline]
//     fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
//         self.future_guard.poll().map(|res| res.map(|_read_guard| ()))
//     }
// }

#[cfg(not(feature = "async_block"))]
impl<T> Future for FutureSinkMapGuard<T> where T: OclPrm + 'static {
    type Item = SinkMapGuard<T>;
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        // println!("Polling FutureSinkMapGuard...");
        match self.map_event.is_complete() {
            Ok(true) => {
                self.resolve().map(|mm| Async::Ready(mm))
            }
            Ok(false) => {
                if !self.callback_is_set {
                    self.map_event.set_unpark_callback()?;
                    self.callback_is_set = true;
                }
                Ok(Async::NotReady)
            },
            Err(err) => Err(err.into()),
        }
    }
}

/// Blocking implementation.
#[cfg(feature = "async_block")]
impl<T: OclPrm> Future for FutureSinkMapGuard<T> {
    type Item = SinkMapGuard<T>;
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        // println!("Polling FutureSinkMapGuard...");
        let _ = self.callback_is_set;
        self.map_event.wait_for()?;
        self.resolve().map(|mm| Async::Ready(mm))
    }
}


/// A flush command builder.
///
#[must_use = "commands do nothing unless enqueued"]
#[derive(Debug)]
pub struct SinkMapCmd<'c, T> where T: 'c + OclPrm {
    sink: &'c BufferMapSink<T>,
    map_queue: Option<&'c Queue>,
    unmap_queue: Option<Queue>,
    offset: usize,
    len: usize,
    ewait: Option<ClWaitListPtrEnum<'c>>,
    enew: Option<ClNullEventPtrEnum<'c>>,
}

impl<'c, T> SinkMapCmd<'c, T> where T: OclPrm {
    /// Returns a new flush command builder.
    fn new(sink: &'c BufferMapSink<T>, map_queue: &'c Queue, offset: usize, len: usize)
            -> SinkMapCmd<'c, T> {
        SinkMapCmd {
            sink,
            map_queue: Some(map_queue),
            unmap_queue: None,
            offset,
            len,
            ewait: None,
            enew: None,
        }
    }

    /// Specifies a queue to use for the this map call only.
    pub fn queue<'q, Q>(mut self, queue: &'q Q) -> SinkMapCmd<'c, T>
            where 'q: 'c, Q: 'q + AsRef<Queue> {
        self.map_queue = Some(queue.as_ref());
        self
    }

    /// Specifies a queue to use for the the unmap command which will be
    /// called when the resolved `SinkMapGuard` is eventually dropped.
    pub fn unmap_queue(mut self, unmap_queue: Queue) -> SinkMapCmd<'c, T> {
        self.unmap_queue = Some(unmap_queue);
        self
    }

    /// Specifies the offset to use for this map only.
    pub fn offset(mut self, offset: usize) -> SinkMapCmd<'c, T> {
        assert!(offset + self.len <= self.sink.buffer().len());
        self.offset = offset;
        self
    }

    /// Specifies the offset to use for this map only.
    pub fn len(mut self, len: usize) -> SinkMapCmd<'c, T> {
        assert!(self.offset + len <= self.sink.buffer().len());
        self.len = len;
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
    pub fn ewait<Ewl>(mut self, ewait: Ewl) -> SinkMapCmd<'c, T>
            where Ewl: Into<ClWaitListPtrEnum<'c>> {
        self.ewait = Some(ewait.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with the unmap command of the eventual `SinkMapGuard`.
    pub fn enew_unmap<En>(mut self, enew: En) -> SinkMapCmd<'c, T>
            where En: Into<ClNullEventPtrEnum<'c>> {
        self.enew = Some(enew.into());
        self
    }

    /// Enqueues this command.
    pub fn enq(mut self) -> OclResult<FutureSinkMapGuard<T>> {
        let buffer_core = self.sink.buffer.core().clone();

        let map_queue = match self.map_queue {
            Some(q) => q,
            None => panic!("SinkMapCmd::enq: No queue set."),
        };

        let mut map_event = Event::empty();

        let mem_map = unsafe {
            core::enqueue_map_buffer::<T, _, _, _>(map_queue, &buffer_core, false,
                MapFlags::new().write_invalidate_region(), self.offset,
                self.len, self.ewait, Some(&mut map_event))?
        };

        let unmap_queue = match self.unmap_queue {
            Some(q) => q,
            None => map_queue.clone(),
        };

        // // Ensure that the unmap and (re)map finish before the future resolves.
        // future_read.set_command_wait_event(map_event);

        let mut future_guard = FutureSinkMapGuard::new(mem_map, self.len, map_event,
            buffer_core, unmap_queue);

        // Copy the tail/conclusion event.
        if let Some(ref mut enew) = self.enew {
            unsafe { enew.clone_from(future_guard.create_unmap_event()?) };
        }

        Ok(future_guard)
    }
}


/// Represents mapped memory and allows frames of data to be written from
/// host-accessible mapped memory region into its associated device-visible
/// buffer in a repeated fashion.
///
/// This represents the fastest possible method for continuously writing
/// frames of data to a device.
///
#[derive(Debug)]
pub struct BufferMapSink<T: OclPrm> {
    buffer: Buffer<T>,
    queue: Queue,
    default_offset: usize,
    default_len: usize,
}

impl<T: OclPrm> BufferMapSink<T> {
    /// Returns a new `BufferMapSink`.
    ///
    /// The current thread will be blocked while the buffer is initialized
    /// upon calling this function.
    pub fn new(queue: Queue, len: usize) -> OclResult<BufferMapSink<T>> {
        let buffer = Buffer::<T>::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().alloc_host_ptr().host_write_only())
            .dims(len)
            .fill_val(T::default())
            .build()?;

        unsafe { BufferMapSink::from_buffer(buffer, None, 0, len) }
    }

    /// Returns a new `BufferMapSink`.
    ///
    /// ## Safety
    ///
    /// `buffer` must not have the same region mapped more than once.
    ///
    pub unsafe fn from_buffer(mut buffer: Buffer<T>, queue: Option<Queue>,
            default_offset: usize, default_len: usize) -> OclResult<BufferMapSink<T>> {
        let buf_flags = buffer.flags()?;
        assert!(buf_flags.contains(MemFlags::new().alloc_host_ptr()) ||
            buf_flags.contains(MemFlags::new().use_host_ptr()),
            "A buffer sink must be created with a buffer that has either \
            the MEM_ALLOC_HOST_PTR` or `MEM_USE_HOST_PTR flag.");
        assert!(!buf_flags.contains(MemFlags::new().host_no_access()) &&
            !buf_flags.contains(MemFlags::new().host_read_only()),
            "A buffer sink may not be created with a buffer that has either the \
            `MEM_HOST_NO_ACCESS` or `MEM_HOST_READ_ONLY` flags.");
        assert!(default_offset + default_len <= buffer.len());

        let queue = match queue {
            Some(q) => {
                buffer.set_default_queue(q.clone());
                q
            },
            None => {
                buffer.default_queue()
                    .expect("A buffer sink must be created with a queue.").clone()
            },
        };

        Ok(BufferMapSink {
            buffer,
            queue,
            default_offset,
            default_len,
        })
    }

    /// Returns a command builder which, when enqueued, will return a future
    /// resolving to an accessible mapped memory region.
    pub fn write<'c>(&'c self) -> SinkMapCmd<'c, T> {
        SinkMapCmd::new(self, &self.queue, self.default_offset, self.default_len)
    }

    /// Returns a reference to the internal buffer.
    #[inline]
    pub fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }

    /// Returns a reference to the internal offset.
    #[inline]
    pub fn default_offset(&self) -> usize {
        self.default_offset
    }

    /// Returns the length of the memory region.
    #[inline]
    pub fn default_len(&self) -> usize {
        self.default_len
    }
}
