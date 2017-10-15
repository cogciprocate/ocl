// use std::sync::Arc;
// use std::sync::atomic::AtomicBool;
use futures::{Future, Poll, Async};
use core::{OclPrm, MemMap as MemMapCore, Mem};
use standard::{Event, Queue, EventList};
use super::{Error as AsyncError, Result as AsyncResult, MemMap};


/// A future which resolves to a `MemMap` as soon as its creating command
/// completes.
///
/// [UNSTABLE]: This type's methods may be renamed or otherwise changed at any time.
#[must_use = "futures do nothing unless polled"]
#[derive(Debug)]
pub struct FutureMemMap<T: OclPrm> {
    core: Option<MemMapCore<T>>,
    len: usize,
    map_event: Event,
    unmap_wait_list: Option<EventList>,
    unmap_target_event: Option<Event>,
    buffer: Option<Mem>,
    queue: Option<Queue>,
    callback_is_set: bool,
    // buffer_is_mapped: Option<Arc<AtomicBool>>,
}

impl<T: OclPrm> FutureMemMap<T> {
    /// Returns a new `FutureMemMap`.
    pub unsafe fn new(core: MemMapCore<T>, len: usize, map_event: Event, buffer: Mem, queue: Queue,
            /*buffer_is_mapped: Arc<AtomicBool>*/) -> FutureMemMap<T> {
        FutureMemMap {
            core: Some(core),
            len: len,
            map_event: map_event,
            unmap_wait_list: None,
            unmap_target_event: None,
            buffer: Some(buffer),
            queue: Some(queue),
            callback_is_set: false,
            // buffer_is_mapped: Some(buffer_is_mapped),
        }
    }

    /// Set an event wait list for the unmap command.
    ///
    /// Setting a wait list here will disallow any wait list from being set
    /// later if/when calling unmap manually.
    ///
    /// [UNSTABLE]: This method may be renamed or otherwise changed.
    pub fn set_unmap_wait_list<El>(&mut self, wait_list: El) where El: Into<EventList> {
        self.unmap_wait_list = Some(wait_list.into())
    }

    /// Create an event which will be triggered (set complete) after this
    /// future resolves into a `MemMap` **and** after that `MemMap` is dropped
    /// or manually unmapped.
    ///
    /// The returned event can be added to the wait list of subsequent OpenCL
    /// commands with the expectation that when all preceeding futures are
    /// complete, the event will automatically be 'triggered' by having its
    /// status set to complete, causing those commands to execute. This can be
    /// used to inject host side code in amongst OpenCL commands without
    /// thread blocking or extra delays of any kind.
    ///
    /// [UNSTABLE]: This method may be renamed or otherwise changed.
    pub fn create_unmap_target_event(&mut self) -> AsyncResult<&mut Event> {
        if let Some(ref queue) = self.queue {
            let uev = Event::user(&queue.context())?;
            self.unmap_target_event = Some(uev);
            Ok(self.unmap_target_event.as_mut().unwrap())
        } else {
            Err("FutureMemMap::create_unmap_target_event: No queue found!".into())
        }
    }

    /// Blocks the current thread until the OpenCL command is complete and an
    /// appropriate lock can be obtained on the underlying data.
    pub fn wait(self) -> AsyncResult<MemMap<T>> {
        <Self as Future>::wait(self)
    }

    /// Returns the unmap event if it has been created.
    ///
    /// [UNSTABLE]: This method may be renamed or otherwise changed.
    #[inline]
    pub fn unmap_target_event(&self) -> Option<&Event> {
        self.unmap_target_event.as_ref()
    }

    /// Resolves this `FutureMemMap` into a `MemMap`.
    fn to_mapped_mem(&mut self) -> AsyncResult<MemMap<T>> {
        let joined = self.core.take().and_then(|core| {
            self.buffer.take().and_then(|buf| {
                self.queue.take().and_then(|queue| {
                    Some((core, buf, queue))
                })
            })
        });

        match joined {
            Some((core, buffer, queue)) => {
                // TODO: Add `buffer_is_mapped` to list of joined stuff.
                unsafe { Ok(MemMap::new(core, self.len, self.unmap_wait_list.take(),
                    self.unmap_target_event.take(), buffer, queue,
                    /*self.buffer_is_mapped.take().unwrap()*/)) }
            },
            _ => Err("FutureMemMap::create_unmap_target_event: No queue and/or buffer found!".into()),
        }
    }
}

#[cfg(not(feature = "async_block"))]
impl<T> Future for FutureMemMap<T> where T: OclPrm + 'static {
    type Item = MemMap<T>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        // println!("Polling FutureMemMap...");
        match self.map_event.is_complete() {
            Ok(true) => {
                self.to_mapped_mem().map(|mm| Async::Ready(mm))
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
impl<T: OclPrm> Future for FutureMemMap<T> {
    type Item = MemMap<T>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        // println!("Polling FutureMemMap...");
        let _ = self.callback_is_set;
        self.map_event.wait_for()?;
        self.to_mapped_mem().map(|mm| Async::Ready(mm))
    }
}

unsafe impl<T: OclPrm> Send for FutureMemMap<T> {}
unsafe impl<T: OclPrm> Sync for FutureMemMap<T> {}



