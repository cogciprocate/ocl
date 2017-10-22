#![allow(unused_imports, dead_code)]

extern crate qutex;

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use futures::{Future, Poll, Async};
use futures::sync::oneshot::{self, Receiver};
use core::{self, Result as OclResult, OclPrm, MemMap as MemMapCore, Mem as MemCore, AsMem,
    ClWaitListPtr, ClNullEventPtr, ClContextPtr, MemFlags, MapFlags};
use standard::{ClWaitListPtrEnum, ClNullEventPtrEnum, Event, EventList, Queue, Buffer,};
use async::{Error as AsyncError, Result as AsyncResult};
// pub use self::qutex::{Request, Guard, FutureGuard, Qutex};
pub use self::qutex::{ReadGuard as QrwReadGuard, WriteGuard as QrwWriteGuard,
    FutureReadGuard as QrwFutureReadGuard, FutureWriteGuard as QrwFutureWriteGuard, QrwLock,
    QrwRequest, RequestKind};


const PRINT_DEBUG: bool = false;


fn print_debug(id: usize, msg: &str) {
    if PRINT_DEBUG {
        println!("###### [{}] {} (thread: {})", id, msg,
            ::std::thread::current().name().unwrap_or("<unnamed>"));
    }
}

// /// Extracts an `RwVec` from a guard of either type.
// //
// // This saves us two unnecessary atomic stores (the reference count of lock
// // going up then down when releasing or up/downgrading) which would occur if
// // we were to clone then drop.
// unsafe fn extract_rw_vec<T, G: Guard<T>>(guard: G) -> QrwLock<T> {
//     let rw_vec = ::std::ptr::read(guard.lock());
//     ::std::mem::forget(guard);
//     rw_vec
// }



// /// A read or write guard for an `RwVec`.
// pub trait Guard<T> {
//     fn new(sink: RwVec<T>, release_event: Option<Event>) -> Self;
// }


// /// Allows access to the data contained within just like a mutex guard.
// #[derive(Debug)]
// pub struct SinkGuard<T> {
//     sink: BufferSink<T>,
//     release_event: Option<Event>,
// }

// impl<T> SinkGuard<T> {
//     /// Returns a new `SinkGuard`.
//     fn new(sink: BufferSink<T>, release_event: Option<Event>) -> SinkGuard<T> {
//         print_debug(sink.id(), "SinkGuard::new: Write lock acquired");
//         SinkGuard {
//             sink: sink,
//             release_event: release_event,
//         }
//     }

//     /// Triggers the release event and releases the lock held by this `SinkGuard`
//     /// before returning the original `BufferSink`.
//     //
//     // * NOTE: This could be done without refcount incr/decr (see `qrw_lock::extract_lock`).
//     pub fn release(guard: SinkGuard<T>) -> BufferSink<T> {
//         print_debug(guard.sink.id(), "SinkGuard::release: Releasing write lock");
//         guard.sink.clone()
//     }

//     /// Returns a reference to the event previously set using
//     /// `create_release_event` on the `FutureGuard` which preceded this
//     /// `SinkGuard`. The event can be manually 'triggered' by calling
//     /// `...set_complete()...` or used normally (as a wait event) by
//     /// subsequent commands. If the event is not manually completed it will be
//     /// automatically set complete when this `SinkGuard` is dropped.
//     pub fn release_event(guard: &SinkGuard<T>) -> Option<&Event> {
//         guard.release_event.as_ref()
//     }

//     /// Triggers the release event by setting it complete.
//     fn complete_release_event(guard: &SinkGuard<T>) {
//         if let Some(ref e) = guard.release_event {
//             if !e.is_complete().expect("SinkGuard::complete_release_event") {
//                 print_debug(guard.sink.id(), "SinkGuard::complete_release_event: \
//                     Setting release event complete");
//                 e.set_complete().expect("SinkGuard::complete_release_event");
//             }
//         }
//     }
// }

// impl<T> Deref for SinkGuard<T> {
//     type Target = Vec<T>;

//     fn deref(&self) -> &Vec<T> {
//         unsafe { &*self.sink.lock.as_ptr() }
//     }
// }

// impl<T> DerefMut for SinkGuard<T> {
//     fn deref_mut(&mut self) -> &mut Vec<T> {
//         unsafe { &mut *self.sink.lock.as_mut_ptr() }
//     }
// }

// impl<T> Drop for SinkGuard<T> {
//     fn drop(&mut self) {
//         print_debug(self.sink.id(), "SinkGuard::drop: Dropping and releasing SinkGuard");
//         unsafe { self.sink.lock.release_write_lock() };
//         Self::complete_release_event(self);
//     }
// }

// // impl<T> SinkGuard<T> for SinkGuard<T> {
// //     fn new(sink: BufferSink<T>, release_event: Option<Event>) -> SinkGuard<T> {
// //         SinkGuard::new(sink, release_event)
// //     }
// // }





// /// The polling stage of a `FutureGuard`.
// #[derive(Debug, PartialEq)]
// enum Stage {
//     Marker,
//     QrwLock,
//     Command,
// }


// /// A future that resolves to a read or write guard after ensuring that the
// /// data being guarded is appropriately locked during the execution of an
// /// OpenCL command.
// ///
// /// 1. Waits until both an exclusive data lock can be obtained **and** all
// ///    prerequisite OpenCL commands have completed.
// /// 2. Triggers an OpenCL command, remaining locked while the command
// ///    executes.
// /// 3. Returns a guard which provides exclusive (write) or shared (read)
// ///    access to the locked data.
// ///
// #[must_use = "futures do nothing unless polled"]
// #[derive(Debug)]
// pub struct FutureGuard<T, G> {
//     sink: Option<BufferSink<T>>,
//     lock_rx: Option<Receiver<()>>,
//     wait_list: Option<EventList>,
//     lock_event: Option<Event>,
//     command_completion: Option<Event>,
//     upgrade_after_command: bool,
//     upgrade_rx: Option<Receiver<()>>,
//     release_event: Option<Event>,
//     stage: Stage,
//     _guard: PhantomData<G>,
// }

// impl<T, G> FutureGuard<T> where G: SinkGuard<T> {
//     /// Returns a new `FutureGuard`.
//     fn new(sink: BufferSink<T>, lock_rx: Receiver<()>) -> FutureGuard<T> {
//         FutureGuard {
//             sink: Some(sink),
//             lock_rx: Some(lock_rx),
//             wait_list: None,
//             lock_event: None,
//             command_completion: None,
//             upgrade_after_command: false,
//             upgrade_rx: None,
//             release_event: None,
//             stage: Stage::Marker,
//             _guard: PhantomData,
//         }
//     }

//     /// Sets an event wait list.
//     ///
//     /// Setting a wait list will cause this `FutureGuard` to wait until
//     /// contained events have their status set to complete before obtaining a
//     /// lock on the guarded internal `Vec`.
//     ///
//     /// [UNSTABLE]: This method may be renamed or otherwise changed at any time.
//     pub fn set_wait_list<L: Into<EventList>>(&mut self, wait_list: L) {
//         assert!(self.wait_list.is_none(), "Wait list has already been set.");
//         self.wait_list = Some(wait_list.into());
//     }

//     /// Sets a command completion event.
//     ///
//     /// If a command completion event corresponding to the read or write
//     /// command being executed in association with this `FutureGuard` is
//     /// specified before this `FutureGuard` is polled it will cause this
//     /// `FutureGuard` to suffix itself with an additional future that will
//     /// wait until the command completion event completes before resolving
//     /// into an `SinkGuard`.
//     ///
//     /// Not specifying a command completion event will cause this
//     /// `FutureGuard` to resolve into an `SinkGuard` immediately after the
//     /// lock is obtained (indicated by the optionally created lock event).
//     ///
//     /// TODO: Reword this.
//     /// [UNSTABLE]: This method may be renamed or otherwise changed at any time.
//     pub fn set_command_completion_event(&mut self, command_completion: Event) {
//         assert!(self.command_completion.is_none(), "Command completion event has already been set.");
//         self.command_completion = Some(command_completion);
//     }

//     /// Creates an event which will be triggered when a lock is obtained on
//     /// the guarded internal `Vec`.
//     ///
//     /// The returned event can be added to the wait list of subsequent OpenCL
//     /// commands with the expectation that when all preceding futures are
//     /// complete, the event will automatically be 'triggered' by having its
//     /// status set to complete, causing those commands to execute. This can be
//     /// used to inject host side code in amongst OpenCL commands without
//     /// thread blocking or extra delays of any kind.
//     pub fn create_lock_event<C: ClContextPtr>(&mut self, context: C) -> AsyncResult<&Event> {
//         assert!(self.lock_event.is_none(), "Lock event has already been created.");
//         self.lock_event = Some(Event::user(context)?);
//         Ok(self.lock_event.as_mut().unwrap())
//     }

//     /// Creates an event which will be triggered after this future resolves
//     /// **and** the ensuing `SinkGuard` is dropped or manually released.
//     ///
//     /// The returned event can be added to the wait list of subsequent OpenCL
//     /// commands with the expectation that when all preceding futures are
//     /// complete, the event will automatically be 'triggered' by having its
//     /// status set to complete, causing those commands to execute. This can be
//     /// used to inject host side code in amongst OpenCL commands without
//     /// thread blocking or extra delays of any kind.
//     pub fn create_release_event<C: ClContextPtr>(&mut self, context: C) -> AsyncResult<&Event> {
//         assert!(self.release_event.is_none(), "Release event has already been created.");
//         self.release_event = Some(Event::user(context)?);
//         Ok(self.release_event.as_ref().unwrap())
//     }

//     /// Returns a reference to the event previously created with
//     /// `::create_lock_event` which will trigger (be completed) when the wait
//     /// events are complete and the lock is locked.
//     pub fn lock_event(&self) -> Option<&Event> {
//         self.lock_event.as_ref()
//     }

//     /// Returns a reference to the event previously created with
//     /// `::create_release_event` which will trigger (be completed) when a lock
//     /// is obtained on the guarded internal `Vec`.
//     pub fn release_event(&self) -> Option<&Event> {
//         self.release_event.as_ref()
//     }

//     /// Blocks the current thread until the OpenCL command is complete and an
//     /// appropriate lock can be obtained on the underlying data.
//     pub fn wait(self) -> AsyncResult<G> {
//         <Self as Future>::wait(self)
//     }

//     /// Returns a mutable pointer to the data contained within the internal
//     /// `Vec`, bypassing all locks and protections.
//     pub unsafe fn as_mut_ptr(&self) -> Option<*mut T> {
//         self.sink.as_ref().map(|sink| (*sink.lock.as_mut_ptr()).as_mut_ptr())
//     }

//     /// Returns a mutable slice to the data contained within the internal
//     /// `Vec`, bypassing all locks and protections.
//     pub unsafe fn as_mut_slice<'a, 'b>(&'a self) -> Option<&'b mut [T]> {
//         self.as_mut_ptr().map(|ptr| {
//             ::std::slice::from_raw_parts_mut(ptr, self.len())
//         })
//     }

//     /// Returns the length of the internal `Vec`.
//     pub fn len(&self) -> usize {
//         unsafe { (*self.sink.as_ref().expect("FutureGuard::len: No BufferSink found.")
//             .lock.as_ptr()).len() }
//     }

//     /// The 'id' of the associated `BufferSink`.
//     pub fn id(&self) -> usize {
//         self.sink.as_ref().expect("FutureGuard::id: No BufferSink found.").id()
//     }

//     /// Polls the wait events until all requisite commands have completed then
//     /// polls the lock queue.
//     fn poll_wait_events(&mut self) -> AsyncResult<Async<G>> {
//         debug_assert!(self.stage == Stage::Marker);
//         print_debug(self.sink.as_ref().unwrap().id(), "FutureGuard::poll_wait_events: Called");

//         // Check completion of wait list, if it exists:
//         if let Some(ref mut wait_list) = self.wait_list {
//             // if PRINT_DEBUG { println!("###### [{}] FutureGuard::poll_wait_events: \
//             //     Polling wait_events (thread: {})...", self.sink.as_ref().unwrap().id(),
//             //     ::std::thread::current().name().unwrap_or("<unnamed>")); }

//             if let Async::NotReady = wait_list.poll()? {
//                 return Ok(Async::NotReady);
//             }

//         }

//         self.stage = Stage::QrwLock;
//         self.poll_lock()
//     }

//     /// Polls the lock until we have obtained a lock then polls the command
//     /// event.
//     #[cfg(not(feature = "async_block"))]
//     fn poll_lock(&mut self) -> AsyncResult<Async<G>> {
//         debug_assert!(self.stage == Stage::QrwLock);
//         print_debug(self.sink.as_ref().unwrap().id(), "FutureGuard::poll_lock: Called");

//         // Move the queue along:
//         unsafe { self.sink.as_ref().unwrap().lock.process_queues(); }

//         // Check for completion of the lock rx:
//         if let Some(ref mut lock_rx) = self.lock_rx {
//             match lock_rx.poll() {
//                 // If the poll returns `Async::Ready`, we have been popped from
//                 // the front of the lock queue and we now have exclusive access.
//                 // Otherwise, return the `NotReady`. The rx (oneshot channel) will
//                 // arrange for this task to be awakened when it's ready.
//                 Ok(status) => {
//                     // if PRINT_DEBUG { println!("###### [{}] FutureGuard::poll_lock: status: {:?}, \
//                     //     (thread: {}).", self.sink.as_ref().unwrap().id(), status,
//                     //     ::std::thread::current().name().unwrap_or("<unnamed>")); }
//                     match status {
//                         Async::Ready(_) => {
//                             if let Some(ref lock_event) = self.lock_event {
//                                 lock_event.set_complete()?
//                             }
//                             self.stage = Stage::Command;
//                         },
//                         Async::NotReady => return Ok(Async::NotReady),
//                     }
//                 },
//                 // Err(e) => return Err(e.into()),
//                 Err(e) => panic!("FutureGuard::poll_lock: {:?}", e),
//             }
//         } else {
//             unreachable!();
//         }

//         self.poll_command()
//     }


//     /// Polls the lock until we have obtained a lock then polls the command
//     /// event.
//     #[cfg(feature = "async_block")]
//     fn poll_lock(&mut self) -> AsyncResult<Async<G>> {
//         debug_assert!(self.stage == Stage::QrwLock);
//         print_debug(self.sink.as_ref().unwrap().id(), "FutureGuard::poll_lock: Called");

//         // Move the queue along:
//         unsafe { self.sink.as_ref().unwrap().lock.process_queues(); }

//         // Wait until completion of the lock rx:
//         self.lock_rx.take().wait()?;

//         if let Some(ref lock_event) = self.lock_event {
//             lock_event.set_complete()?
//         }

//         self.stage = Stage::Command;
//         // if PRINT_DEBUG { println!("###### [{}] FutureGuard::poll_lock: Moving to command stage.",
//         //     self.sink.as_ref().unwrap().id()); }
//         return self.poll_command();
//     }

//     /// Polls the command event until it is complete then returns an `SinkGuard`
//     /// which can be safely accessed immediately.
//     fn poll_command(&mut self) -> AsyncResult<Async<G>> {
//         debug_assert!(self.stage == Stage::Command);
//         print_debug(self.sink.as_ref().unwrap().id(), "FutureGuard::poll_command: Called");

//         if let Some(ref mut command_completion) = self.command_completion {
//             // if PRINT_DEBUG { println!("###### [{}] FutureGuard::poll_command: Polling command \
//             //     completion event (thread: {}).", self.sink.as_ref().unwrap().id(), ::std::thread::current().name()
//             //     .unwrap_or("<unnamed>")); }

//             if let Async::NotReady = command_completion.poll()? {
//                 return Ok(Async::NotReady);
//             }
//         }

//         // Set cmd event to `None` so it doesn't get waited on unnecessarily
//         // when this `FutureGuard` drops.
//         self.command_completion = None;

//         if self.upgrade_after_command {
//             self.stage = Stage::Upgrade;
//             self.poll_upgrade()
//         } else {
//             Ok(Async::Ready(self.into_guard()))
//         }
//     }

//     /// Polls the lock until it has been upgraded.
//     ///
//     /// Only used if `::upgrade_after_command` has been called.
//     ///
//     #[cfg(not(feature = "async_block"))]
//     fn poll_upgrade(&mut self) -> AsyncResult<Async<G>> {
//         debug_assert!(self.stage == Stage::Upgrade);
//         debug_assert!(self.upgrade_after_command);
//         print_debug(self.sink.as_ref().unwrap().id(), "FutureGuard::poll_upgrade: Called");

//         // unsafe { self.sink.as_ref().unwrap().lock.process_queues() }

//         if self.upgrade_rx.is_none() {
//             match unsafe { self.sink.as_ref().unwrap().lock.upgrade_read_lock() } {
//                 Ok(_) => {
//                     print_debug(self.sink.as_ref().unwrap().id(),
//                         "FutureGuard::poll_upgrade: Write lock acquired. Upgrading immediately.");
//                     Ok(Async::Ready(self.into_guard()))
//                 },
//                 Err(rx) => {
//                     self.upgrade_rx = Some(rx);
//                     match self.upgrade_rx.as_mut().unwrap().poll() {
//                         Ok(res) => {
//                             // print_debug(self.sink.as_ref().unwrap().id(),
//                             //     "FutureGuard::poll_upgrade: Channel completed. Upgrading.");
//                             // Ok(res.map(|_| self.into_guard()))
//                             match res {
//                                 Async::Ready(_) => {
//                                     print_debug(self.sink.as_ref().unwrap().id(),
//                                         "FutureGuard::poll_upgrade: Channel completed. Upgrading.");
//                                     Ok(Async::Ready(self.into_guard()))
//                                 },
//                                 Async::NotReady => {
//                                     print_debug(self.sink.as_ref().unwrap().id(),
//                                         "FutureGuard::poll_upgrade: Upgrade rx not ready.");
//                                     Ok(Async::NotReady)
//                                 },
//                             }
//                         },
//                         // Err(e) => Err(e.into()),
//                         Err(e) => panic!("FutureGuard::poll_upgrade: {:?}", e),
//                    }
//                 },
//             }
//         } else {
//             // Check for completion of the upgrade rx:
//             match self.upgrade_rx.as_mut().unwrap().poll() {
//                 Ok(status) => {
//                     print_debug(self.sink.as_ref().unwrap().id(),
//                         &format!("FutureGuard::poll_upgrade: Status: {:?}", status));
//                     Ok(status.map(|_| self.into_guard()))
//                 },
//                 // Err(e) => Err(e.into()),
//                 Err(e) => panic!("FutureGuard::poll_upgrade: {:?}", e),
//             }
//         }
//     }

//     /// Polls the lock until it has been upgraded.
//     ///
//     /// Only used if `::upgrade_after_command` has been called.
//     ///
//     #[cfg(feature = "async_block")]
//     fn poll_upgrade(&mut self) -> AsyncResult<Async<G>> {
//         debug_assert!(self.stage == Stage::Upgrade);
//         debug_assert!(self.upgrade_after_command);
//         print_debug(self.sink.as_ref().unwrap().id(), "FutureGuard::poll_upgrade: Called");

//         match unsafe { self.sink.as_ref().unwrap().lock.upgrade_read_lock() } {
//             Ok(_) => Ok(Async::Ready(self.into_guard())),
//             Err(rx) => {
//                 self.upgrade_rx = Some(rx);
//                 self.upgrade_rx.take().unwrap().wait()?;
//                 Ok(Async::Ready(self.into_guard()))
//             }
//         }
//     }

//     /// Resolves this `FutureGuard` into the appropriate result guard.
//     fn into_guard(&mut self) -> G {
//         print_debug(self.sink.as_ref().unwrap().id(), "FutureGuard::into_guard: All polling complete");
//         G::new(self.sink.take().unwrap(), self.release_event.take())
//     }
// }

// impl<T, G> Future for FutureGuard<T> where G: SinkGuard<T> {
//     type Item = G;
//     type Error = AsyncError;

//     #[inline]
//     fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
//         if self.sink.is_some() {
//             match self.stage {
//                 Stage::Marker => self.poll_wait_events(),
//                 Stage::QrwLock => self.poll_lock(),
//                 Stage::Command => self.poll_command(),
//                 Stage::Upgrade => self.poll_upgrade(),
//             }
//         } else {
//             Err("FutureGuard::poll: Task already completed.".into())
//         }
//     }
// }

// impl<T, G> Drop for FutureGuard<T> {
//     /// Drops this FutureGuard.
//     ///
//     /// Blocks the current thread until the command associated with this
//     /// `FutureGuard` (represented by the command completion event)
//     /// completes. This ensures that the underlying `Vec` is not dropped
//     /// before the command completes (which would cause obvious problems).
//     fn drop(&mut self) {
//         if let Some(ref ccev) = self.command_completion {
//             // println!("###### FutureGuard::drop: Event ({:?}) incomplete...", ccev);
//             // panic!("###### FutureGuard::drop: Event ({:?}) incomplete...", ccev);
//             ccev.wait_for().expect("Error waiting on command completion event \
//                 while dropping 'FutureGuard'");
//         }
//         if let Some(ref rev) = self.release_event {
//             rev.set_complete().expect("Error setting release event complete \
//                 while dropping 'FutureGuard'");
//         }
//     }
// }



#[derive(Debug)]
struct Inner<T: OclPrm> {
    buffer: Buffer<T>,
    memory: MemMapCore<T>,
    offset: usize,
    len: usize,
}


/// Represents mapped memory and allows frames of data to be 'flushed'
/// (written) from host-accessible mapped memory region into its associated
/// device-visible buffer in a repeated fashion.
///
/// This represents the fastest possible method for continuously writing
/// buffer-sized frames of data to a device.
#[derive(Debug)]
pub struct BufferSink<T: OclPrm> {
    // buffer: Buffer<T>,
    // memory: MemMapCore<T>,
    // offset: usize,
    // len: usize,
    lock: QrwLock<Inner<T>>,
}

impl<T: OclPrm> BufferSink<T> {
    /// Returns a new `BufferSink`.
    ///
    /// ## Safety
    ///
    /// `buffer` must not have the same region mapped more than once.
    ///
    pub unsafe fn new(mut buffer: Buffer<T>, queue: Queue, offset: usize, len: usize)
            -> OclResult<BufferSink<T>> {
        // TODO: Ensure that these checks are complete enough.
        let buf_flags = buffer.flags()?;
        assert!(buf_flags.contains(MemFlags::new().alloc_host_ptr()) ||
            buf_flags.contains(MemFlags::new().use_host_ptr()),
            "A buffer sink must be created with a buffer that has either \
            the MEM_ALLOC_HOST_PTR` or `MEM_USE_HOST_PTR flag.");
        assert!(!buf_flags.contains(MemFlags::new().host_no_access()) &&
            !buf_flags.contains(MemFlags::new().host_read_only()),
            "A buffer sink may not be created with a buffer that has either the \
            `MEM_HOST_NO_ACCESS` or `MEM_HOST_READ_ONLY` flags.");

        let map_flags = MapFlags::new().write_invalidate_region();
        buffer.set_default_queue(queue);

        let memory = core::enqueue_map_buffer::<T, _, _, _>(buffer.default_queue().unwrap(),
            buffer.core(), true, map_flags, offset, len, None::<&EventList>, None::<&mut Event>)?;

        let inner = Inner {
            buffer,
            memory,
            offset,
            len
        };

        Ok(BufferSink {
            lock: QrwLock::new(inner),
        })
    }

    // pub fn write(&self) -> FutureGuard<T> {

    // }


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

impl<T: OclPrm> Clone for BufferSink<T> {
    #[inline]
    fn clone(&self) -> BufferSink<T> {
        BufferSink {
            lock: self.lock.clone(),
        }
    }
}

impl<T: OclPrm> Drop for Inner<T> {
    /// Drops the `Inner` enqueuing an unmap and blocking until it
    /// completes.
    fn drop(&mut self) {
        let mut new_event = Event::empty();
        core::enqueue_unmap_mem_object::<T, _, _, _>(self.buffer.default_queue().unwrap(), &self.buffer,
            &self.memory, None::<&EventList>, Some(&mut new_event)).unwrap();
        new_event.wait_for().unwrap();
    }
}
