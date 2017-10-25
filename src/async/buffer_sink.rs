//! High performance buffer writes.

use std::ops::{Deref, DerefMut};
use futures::{Future, Poll};
use core::{self, Result as OclResult, OclPrm, MemMap as MemMapCore,
    MemFlags, MapFlags, ClNullEventPtr};
use standard::{Event, EventList, Queue, Buffer, ClWaitListPtrEnum, ClNullEventPtrEnum};
use async::{Error as AsyncError, OrderLock, FutureGuard, ReadGuard, WriteGuard};


#[must_use = "futures do nothing unless polled"]
#[derive(Debug)]
pub struct FutureFlush<T: OclPrm> {
    future_guard: FutureGuard<Inner<T>, ReadGuard<Inner<T>>>,
}

impl<T: OclPrm> FutureFlush<T> {
    fn new(future_guard: FutureGuard<Inner<T>, ReadGuard<Inner<T>>>) -> FutureFlush<T> {
        FutureFlush { future_guard: future_guard }
    }

    pub fn future_guard(&mut self) -> &mut FutureGuard<Inner<T>, ReadGuard<Inner<T>>> {
        &mut self.future_guard
    }
}

impl<T: OclPrm> Future for FutureFlush<T> {
    type Item = ();
    type Error = AsyncError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.future_guard.poll().map(|res| res.map(|_read_guard| ()))
    }
}


/// A flush command builder.
///
#[must_use = "commands do nothing unless enqueued"]
#[derive(Debug)]
pub struct FlushCmd<'c, T> where T: 'c + OclPrm {
    queue: Option<&'c Queue>,
    sink: &'c BufferSink<T>,
    ewait: Option<ClWaitListPtrEnum<'c>>,
    enew: Option<ClNullEventPtrEnum<'c>>,
}

impl<'c, T> FlushCmd<'c, T> where T: OclPrm {
    /// Returns a new unmap command builder.
    fn new(sink: &'c BufferSink<T>) -> FlushCmd<'c, T> {
        FlushCmd {
            queue: None,
            sink: sink,
            ewait: None,
            enew: None,
        }
    }

    /// Specifies a queue to use for this call only.
    pub fn queue<'q, Q>(mut self, queue: &'q Q) -> FlushCmd<'c, T>
        where 'q: 'c, Q: 'q + AsRef<Queue>
    {
        self.queue = Some(queue.as_ref());
        self
    }


    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<Ewl>(mut self, ewait: Ewl) -> FlushCmd<'c, T>
            where Ewl: Into<ClWaitListPtrEnum<'c>>
    {
        self.ewait = Some(ewait.into());
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt<Ewl>(mut self, ewait: Option<Ewl>) -> FlushCmd<'c, T>
            where Ewl: Into<ClWaitListPtrEnum<'c>> {
        self.ewait = ewait.map(|el| el.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew<En>(mut self, enew: En) -> FlushCmd<'c, T>
            where En: Into<ClNullEventPtrEnum<'c>> {
        self.enew = Some(enew.into());
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt<En>(mut self, enew: Option<En>) -> FlushCmd<'c, T>
            where En: Into<ClNullEventPtrEnum<'c>>
    {
        self.enew = enew.map(|e| e.into());
        self
    }

    /// Enqueues this command.
    ///
    pub fn enq(mut self) -> OclResult<FutureFlush<T>> {
        let mut future_read = self.sink.clone().lock.read();
        if let Some(wl) = self.ewait {
            future_read.set_lock_wait_events(wl);
        }

        let mut unmap_event = Event::empty();
        let mut map_event = Event::empty();

        unsafe {
            let inner = &*self.sink.lock.as_ptr();
            let default_queue = inner.buffer.default_queue().unwrap();
            let buffer = inner.buffer.core();

            let unmap_queue = match self.queue {
                Some(q) => q,
                None => default_queue,
            };

            future_read.create_lock_event(unmap_queue.context_ptr()?)?;

            core::enqueue_unmap_mem_object::<T, _, _, _>(unmap_queue, buffer, &inner.memory,
                future_read.lock_event(), Some(&mut unmap_event))?;

            let map_flags = MapFlags::new().write_invalidate_region();
            core::enqueue_map_buffer::<T, _, _, _>(default_queue, buffer, false, map_flags,
                inner.default_offset, inner.default_len, Some(&unmap_event), Some(&mut map_event))?;
        }

        // Copy the tail/conclusion event.
        if let Some(ref mut enew) = self.enew {
            unsafe { enew.clone_from(&map_event) }
        }

        // Ensure that the unmap and (re)map finish.
        future_read.set_command_wait_event(map_event.clone());

        Ok(FutureFlush::new(future_read))
    }
}


#[derive(Debug)]
pub struct Inner<T: OclPrm> {
    buffer: Buffer<T>,
    memory: MemMapCore<T>,
    default_offset: usize,
    default_len: usize,
}

impl<T: OclPrm> Inner<T> {
    /// Returns the internal buffer.
    pub fn buffer(self: &Inner<T>) -> &Buffer<T> {
        &self.buffer
    }

    /// Returns the internal memory mapping.
    pub fn memory(self: &Inner<T>) -> &MemMapCore<T> {
        &self.memory
    }

    /// Returns the default offset.
    pub fn default_offset(self: &Inner<T>) -> usize {
        self.default_offset
    }
}

impl<T: OclPrm> Deref for Inner<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        unsafe { self.memory.as_slice(self.default_len) }
    }
}

impl<T: OclPrm> DerefMut for Inner<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { self.memory.as_slice_mut(self.default_len) }
    }
}

impl<T: OclPrm> Drop for Inner<T> {
    /// Drops the `Inner` enqueuing an unmap and blocking until it
    /// completes.
    fn drop(&mut self) {
        let mut new_event = Event::empty();
        core::enqueue_unmap_mem_object::<T, _, _, _>(self.buffer.default_queue().unwrap(),
            &self.buffer, &self.memory, None::<&EventList>, Some(&mut new_event)).unwrap();
        new_event.wait_for().unwrap();
    }
}


/// Represents mapped memory and allows frames of data to be 'flushed'
/// (written) from host-accessible mapped memory region into its associated
/// device-visible buffer in a repeated fashion.
///
/// This represents the fastest possible method for continuously writing
/// frames of data to a device.
///
#[derive(Clone, Debug)]
pub struct BufferSink<T: OclPrm> {
    lock: OrderLock<Inner<T>>,
}

impl<T: OclPrm> BufferSink<T> {
    /// Returns a new `BufferSink`.
    ///
    /// The current thread will be blocked while the buffer is initialized
    /// upon calling this function.
    pub fn new(queue: Queue, len: usize) -> OclResult<BufferSink<T>> {
        let buffer = Buffer::<T>::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().alloc_host_ptr().host_write_only())
            .dims(len)
            .fill_val(T::default())
            .build()?;

        unsafe { BufferSink::from_buffer(buffer, queue, 0, len) }
    }

    /// Returns a new `BufferSink`.
    ///
    /// ## Safety
    ///
    /// `buffer` must not have the same region mapped more than once.
    ///
    pub unsafe fn from_buffer(mut buffer: Buffer<T>, queue: Queue, default_offset: usize,
            default_len: usize) -> OclResult<BufferSink<T>> {
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

        buffer.set_default_queue(queue);
        let map_flags = MapFlags::new().write_invalidate_region();

        let memory = core::enqueue_map_buffer::<T, _, _, _>(buffer.default_queue().unwrap(),
            buffer.core(), true, map_flags, default_offset, default_len, None::<&EventList>, None::<&mut Event>)?;

        let inner = Inner {
            buffer,
            memory,
            default_offset,
            default_len
        };

        Ok(BufferSink {
            lock: OrderLock::new(inner),
        })
    }

    /// Returns a new `FutureGuard` which will resolve into a a `ReadGuard`.
    pub fn read(self) -> FutureGuard<Inner<T>, ReadGuard<Inner<T>>> {
        self.lock.read()

    }

    /// Returns a new `FutureGuard` which will resolve into a a `WriteGuard`.
    pub fn write(self) -> FutureGuard<Inner<T>, WriteGuard<Inner<T>>> {
        self.lock.write()
    }

    /// Returns a command builder which once enqueued will flush data from the
    /// mapped memory region to the device.
    pub fn flush(&self) -> FlushCmd<T> {
        FlushCmd::new(self)
    }

    /// Returns a reference to the internal buffer.
    pub fn buffer(&self) -> &Buffer<T> {
        unsafe { &(*self.lock.as_mut_ptr()).buffer }
    }

    /// Returns a reference to the internal memory mapping.
    pub fn memory(&self) -> &MemMapCore<T> {
        unsafe { &(*self.lock.as_mut_ptr()).memory }
    }

    /// Returns a reference to the internal offset.
    pub fn default_offset(&self) -> usize {
        unsafe { (*self.lock.as_mut_ptr()).default_offset }
    }

    /// Returns a mutable slice into the contained memory region.
    ///
    /// Used by buffer command builders when preparing future read and write
    /// commands.
    ///
    /// Do not use unless you are 100% certain that there will be no other
    /// reads or writes for the entire access duration (only possible if
    /// manually manipulating the lock status).
    pub unsafe fn as_mut_slice(&self) -> &mut [T] {
        let ptr = (*self.lock.as_mut_ptr()).memory.as_mut_ptr();
        let default_len = (*self.lock.as_ptr()).default_len;
        ::std::slice::from_raw_parts_mut(ptr, default_len)
    }

    /// Returns the length of the memory region.
    pub fn default_len(&self) -> usize {
        unsafe { (*self.lock.as_ptr()).default_len }
    }

    /// Returns a pointer address to the internal memory region, usable as a
    /// unique identifier.
    pub fn id(&self) -> usize {
        unsafe { (*self.lock.as_ptr()).memory.as_ptr() as usize }
    }
}

impl<T: OclPrm> From<OrderLock<Inner<T>>> for BufferSink<T> {
    fn from(order_lock: OrderLock<Inner<T>>) -> BufferSink<T> {
        BufferSink { lock: order_lock }
    }
}