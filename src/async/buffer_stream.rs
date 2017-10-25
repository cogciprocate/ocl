//! High performance buffer reads.

use std::ops::Deref;
use futures::{Future, Poll};
use core::{self, Result as OclResult, OclPrm, MemMap as MemMapCore,
    MemFlags, MapFlags, ClNullEventPtr};
use standard::{Event, EventList, Queue, Buffer};
use async::{Error as AsyncError, OrderLock, FutureGuard, ReadGuard, WriteGuard};


#[must_use = "futures do nothing unless polled"]
#[derive(Debug)]
pub struct FutureFlood<T: OclPrm> {
    future_guard: FutureGuard<Inner<T>, WriteGuard<Inner<T>>>,
}

impl<T: OclPrm> FutureFlood<T> {
    fn new(future_guard: FutureGuard<Inner<T>, WriteGuard<Inner<T>>>) -> FutureFlood<T> {
        FutureFlood { future_guard: future_guard }
    }

    pub fn future_guard(&mut self) -> &mut FutureGuard<Inner<T>, WriteGuard<Inner<T>>> {
        &mut self.future_guard
    }
}

impl<T: OclPrm> Future for FutureFlood<T> {
    type Item = ();
    type Error = AsyncError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.future_guard.poll().map(|res| res.map(|_write_guard| ()))
    }
}


#[derive(Debug)]
pub struct Inner<T: OclPrm> {
    buffer: Buffer<T>,
    memory: MemMapCore<T>,
    offset: usize,
    len: usize,
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

    /// Returns the internal offset.
    pub fn offset(self: &Inner<T>) -> usize {
        self.offset
    }
}

impl<T: OclPrm> Deref for Inner<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        unsafe { self.memory.as_slice(self.len) }
    }
}


/// Represents mapped memory and allows frames of data to be 'flooded' (read)
/// from a device-visible `Buffer` into its associated host-accessible mapped
/// memory region in a repeated fashion.
///
/// This represents the fastest possible method for reading data from an
/// OpenCL device.
///
#[derive(Clone, Debug)]
pub struct BufferStream<T: OclPrm> {
    lock: OrderLock<Inner<T>>,
}

impl<T: OclPrm> BufferStream<T> {
    /// Returns a new `BufferStream`.
    ///
    /// The current thread will be blocked while the buffer is initialized
    /// upon calling this function.
    pub fn new(queue: Queue, len: usize) -> OclResult<BufferStream<T>> {
        let buffer = Buffer::<T>::builder()
            .queue(queue.clone())
            .flags(MemFlags::new().alloc_host_ptr().host_read_only())
            .dims(len)
            .fill_val(T::default())
            .build()?;

        unsafe { BufferStream::from_buffer(buffer, queue, 0, len) }
    }

    /// Returns a new `BufferStream`.
    ///
    /// ## Safety
    ///
    /// `buffer` must not have the same region mapped more than once.
    ///
    /// ## Note
    ///
    /// The buffer will initially be mapped then unmapped, blocking the
    /// current thread until those operations complete upon calling this
    /// function.
    ///
    pub unsafe fn from_buffer(mut buffer: Buffer<T>, queue: Queue, offset: usize, len: usize)
            -> OclResult<BufferStream<T>> {
        let buf_flags = buffer.flags()?;
        assert!(buf_flags.contains(MemFlags::new().alloc_host_ptr()) ||
            buf_flags.contains(MemFlags::new().use_host_ptr()),
            "A buffer sink must be created with a buffer that has either \
            the MEM_ALLOC_HOST_PTR` or `MEM_USE_HOST_PTR flag.");
        assert!(!buf_flags.contains(MemFlags::new().host_no_access()) &&
            !buf_flags.contains(MemFlags::new().host_write_only()),
            "A buffer sink may not be created with a buffer that has either the \
            `MEM_HOST_NO_ACCESS` or `MEM_HOST_WRITE_ONLY` flags.");
        assert!(offset + len <= buffer.len());

        buffer.set_default_queue(queue);
        let map_flags = MapFlags::new().read();

        let mut map_event = Event::empty();
        let mut unmap_event = Event::empty();

        let memory = core::enqueue_map_buffer::<T, _, _, _>(buffer.default_queue().unwrap(),
            buffer.core(), false, map_flags, offset, len, None::<&EventList>, Some(&mut map_event))?;

        core::enqueue_unmap_mem_object::<T, _, _, _>(buffer.default_queue().unwrap(),
            &buffer, &memory, Some(&map_event), Some(&mut unmap_event)).unwrap();

        unmap_event.wait_for().unwrap();

        let inner = Inner {
            buffer,
            memory,
            offset,
            len
        };

        Ok(BufferStream {
            lock: OrderLock::new(inner),
        })
    }

    /// Returns a new `FutureGuard` which will resolve into a a `ReadGuard`.
    pub fn read(self) -> FutureGuard<Inner<T>, ReadGuard<Inner<T>>> {
        self.lock.read()

    }

    /// Floods the mapped memory region with fresh data from the device.
    pub fn flood<Ew, En>(&self, wait_events: Option<Ew>, mut flood_event: Option<En>)
            -> OclResult<FutureFlood<T>>
            where Ew: Into<EventList>, En: ClNullEventPtr {
        let mut future_write = self.clone().lock.write();
        if let Some(wl) = wait_events {
            future_write.set_lock_wait_events(wl);
        }

        let mut map_event = Event::empty();
        let mut unmap_event = Event::empty();

        unsafe {
            let inner = &*self.lock.as_ptr();
            let queue = inner.buffer.default_queue().unwrap();
            let buffer = inner.buffer.core();

            // Ensure that we have a read lock before the command can run.
            future_write.create_lock_event(queue.context_ptr()?)?;

            let map_flags = MapFlags::new().read();
            core::enqueue_map_buffer::<T, _, _, _>(queue, buffer, false, map_flags,
                inner.offset, inner.len, future_write.lock_event(), Some(&mut map_event))?;

            core::enqueue_unmap_mem_object::<T, _, _, _>(queue, buffer, &inner.memory,
                Some(&map_event), Some(&mut unmap_event))?;
        }

        // Copy the tail/conclusion event.
        if let Some(ref mut enew) = flood_event {
            unsafe { enew.clone_from(&unmap_event) }
        }

        // Ensure that the map and unmap finish.
        future_write.set_command_wait_event(unmap_event.clone());

        Ok(FutureFlood::new(future_write))
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
    pub fn offset(&self) -> usize {
        unsafe { (*self.lock.as_mut_ptr()).offset }
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
        let len = (*self.lock.as_ptr()).len;
        ::std::slice::from_raw_parts_mut(ptr, len)
    }

    /// Returns the length of the memory region.
    pub fn len(&self) -> usize {
        unsafe { (*self.lock.as_ptr()).len }
    }

    /// Returns a pointer address to the internal memory region, usable as a
    /// unique identifier.
    pub fn id(&self) -> usize {
        unsafe { (*self.lock.as_ptr()).memory.as_ptr() as usize }
    }
}

impl<T: OclPrm> From<OrderLock<Inner<T>>> for BufferStream<T> {
    fn from(order_lock: OrderLock<Inner<T>>) -> BufferStream<T> {
        BufferStream { lock: order_lock }
    }
}