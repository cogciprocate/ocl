//! High performance buffer reads.

use std::ops::Deref;
use futures::{Future, Poll};
use core::{self, OclPrm, MemMap as MemMapCore,
    MemFlags, MapFlags, ClNullEventPtr};
use standard::{Event, EventList, Queue, Buffer, ClWaitListPtrEnum, ClNullEventPtrEnum};
use async::{OrderLock, FutureGuard, ReadGuard, WriteGuard};
use error::{Error as OclError, Result as OclResult};


#[must_use = "futures do nothing unless polled"]
#[derive(Debug)]
pub struct FutureFlood<T: OclPrm> {
    future_guard: FutureGuard<Inner<T>, WriteGuard<Inner<T>>>,
}

impl<T: OclPrm> FutureFlood<T> {
    fn new(future_guard: FutureGuard<Inner<T>, WriteGuard<Inner<T>>>) -> FutureFlood<T> {
        FutureFlood { future_guard: future_guard }
    }

    /// Returns a mutable reference to the contained `FutureGuard`.
    pub fn future_guard(&mut self) -> &mut FutureGuard<Inner<T>, WriteGuard<Inner<T>>> {
        &mut self.future_guard
    }
}

impl<T: OclPrm> Future for FutureFlood<T> {
    type Item = ();
    type Error = OclError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.future_guard.poll().map(|res| res.map(|_write_guard| ()))
    }
}


/// A flood command builder.
///
#[must_use = "commands do nothing unless enqueued"]
#[derive(Debug)]
pub struct FloodCmd<'c, T> where T: 'c + OclPrm {
    queue: Option<&'c Queue>,
    // stream: &'c BufferStream<T>,
    stream: BufferStream<T>,
    offset: usize,
    len: usize,
    ewait: Option<ClWaitListPtrEnum<'c>>,
    enew: Option<ClNullEventPtrEnum<'c>>,
}

impl<'c, T> FloodCmd<'c, T> where T: OclPrm {
    /// Returns a new flood command builder.
    fn new(stream: BufferStream<T>) -> FloodCmd<'c, T> {
        let offset = stream.default_offset();
        let len = stream.default_len();
        FloodCmd {
            queue: None,
            stream: stream,
            offset,
            len,
            ewait: None,
            enew: None,
        }
    }

    /// Specifies a queue to use for this call only.
    pub fn queue<'q, Q>(mut self, queue: &'q Q) -> FloodCmd<'c, T>
            where 'q: 'c, Q: 'q + AsRef<Queue> {
        self.queue = Some(queue.as_ref());
        self
    }

    /// Specifies the offset to use for this map only.
    pub fn offset(mut self, offset: usize) -> FloodCmd<'c, T> {
        assert!(offset + self.len <= self.stream.buffer().len());
        self.offset = offset;
        self
    }

    /// Specifies the offset to use for this map only.
    pub fn len(mut self, len: usize) -> FloodCmd<'c, T> {
        assert!(self.offset + len <= self.stream.buffer().len());
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
    pub fn ewait<Ewl>(mut self, ewait: Ewl) -> FloodCmd<'c, T>
            where Ewl: Into<ClWaitListPtrEnum<'c>> {
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
    pub fn enew<En>(mut self, enew: En) -> FloodCmd<'c, T>
            where En: Into<ClNullEventPtrEnum<'c>> {
        self.enew = Some(enew.into());
        self
    }

    /// Enqueues this command.
    pub fn enq(mut self) -> OclResult<FutureFlood<T>> {
        let inner = unsafe { &*self.stream.lock.as_ptr() };

        let buffer = inner.buffer.as_core();

        let queue = match self.queue {
            Some(q) => q,
            None => inner.buffer.default_queue().unwrap(),
        };

        let mut future_write = self.stream.lock.write();
        if let Some(wl) = self.ewait {
            future_write.set_lock_wait_events(wl);
        }

        // Ensure that we have a read lock before the command can run.
        future_write.create_lock_event(queue.context_ptr()?)?;

        let mut map_event = Event::empty();
        let mut unmap_event = Event::empty();

        unsafe {
            let memory = core::enqueue_map_buffer::<T, _, _, _>(queue, buffer, false,
                MapFlags::new().read(), self.offset, self.len,
                future_write.lock_event(), Some(&mut map_event))?;

            debug_assert!(memory.as_ptr() == inner.memory.as_ptr());

            core::enqueue_unmap_mem_object::<T, _, _, _>(queue, buffer, &memory,
                Some(&map_event), Some(&mut unmap_event))?;

            // Copy the tail/conclusion event.
            if let Some(ref mut enew) = self.enew {
                enew.clone_from(&unmap_event);
            }
        }

        // Ensure that the map and unmap finish before the future resolves.
        future_write.set_command_wait_event(unmap_event);

        Ok(FutureFlood::new(future_write))
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
            .len(len)
            .fill_val(T::default())
            .build()?;

        unsafe { BufferStream::from_buffer(buffer, None, 0, len) }
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
    pub unsafe fn from_buffer(mut buffer: Buffer<T>, queue: Option<Queue>, default_offset: usize,
            default_len: usize) -> OclResult<BufferStream<T>> {
        let buf_flags = buffer.flags()?;
        assert!(buf_flags.contains(MemFlags::new().alloc_host_ptr()) ||
            buf_flags.contains(MemFlags::new().use_host_ptr()),
            "A buffer stream must be created with a buffer that has either \
            the MEM_ALLOC_HOST_PTR` or `MEM_USE_HOST_PTR flag.");
        assert!(!buf_flags.contains(MemFlags::new().host_no_access()) &&
            !buf_flags.contains(MemFlags::new().host_write_only()),
            "A buffer stream may not be created with a buffer that has either the \
            `MEM_HOST_NO_ACCESS` or `MEM_HOST_WRITE_ONLY` flags.");
        assert!(default_offset + default_len <= buffer.len());

        match queue {
            Some(q) => { buffer.set_default_queue(q); },
            None => assert!(buffer.default_queue().is_some(),
                "A buffer stream must be created with a queue."),
        }

        let map_flags = MapFlags::new().read();

        let mut map_event = Event::empty();
        let mut unmap_event = Event::empty();

        let memory = core::enqueue_map_buffer::<T, _, _, _>(buffer.default_queue().unwrap(),
            buffer.as_core(), false, map_flags, default_offset, default_len, None::<&EventList>,
                Some(&mut map_event))?;

        core::enqueue_unmap_mem_object::<T, _, _, _>(buffer.default_queue().unwrap(),
            &buffer, &memory, Some(&map_event), Some(&mut unmap_event)).unwrap();

        unmap_event.wait_for().unwrap();

        let inner = Inner {
            buffer,
            memory,
            default_offset,
            default_len
        };

        Ok(BufferStream {
            lock: OrderLock::new(inner),
        })
    }

    /// Returns a new `FutureGuard` which will resolve into a a `ReadGuard`.
    pub fn read(self) -> FutureGuard<Inner<T>, ReadGuard<Inner<T>>> {
        self.lock.read()

    }

    /// Returns a command builder which, when enqueued, floods the mapped
    /// memory region with fresh data from the device.
    pub fn flood<'c>(self) -> FloodCmd<'c, T> {
        FloodCmd::new(self)
    }

    /// Returns a reference to the internal buffer.
    #[inline]
    pub fn buffer(&self) -> &Buffer<T> {
        unsafe { &(*self.lock.as_mut_ptr()).buffer }
    }

    /// Returns a reference to the internal memory mapping.
    #[inline]
    pub fn memory(&self) -> &MemMapCore<T> {
        unsafe { &(*self.lock.as_mut_ptr()).memory }
    }

    /// Returns a reference to the default offset.
    #[inline]
    pub fn default_offset(&self) -> usize {
        unsafe { (*self.lock.as_mut_ptr()).default_offset }
    }

    /// Returns the length of the memory region.
    #[inline]
    pub fn default_len(&self) -> usize {
        unsafe { (*self.lock.as_ptr()).default_len }
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
        let len = (*self.lock.as_ptr()).default_len;
        ::std::slice::from_raw_parts_mut(ptr, len)
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