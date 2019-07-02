// use std::sync::Arc;
// use std::sync::atomic::AtomicBool;
use futures::{Future, Poll, Async};
use crate::core::{OclPrm, MemMap as MemMapCore, Mem, ClNullEventPtr};
use crate::r#async::MemMap;
use crate::error::{Error as OclError, Result as OclResult};
use crate::{Event, Queue, EventList};


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
    unmap_wait_events: Option<EventList>,
    unmap_event: Option<Event>,
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
            unmap_wait_events: None,
            unmap_event: None,
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
    pub fn set_unmap_wait_events<El>(&mut self, wait_events: El) where El: Into<EventList> {
        self.unmap_wait_events = Some(wait_events.into())
    }

    /// Set an event wait list for the unmap command.
    ///
    /// See `::set_unmap_wait_events`.
    pub fn ewait_unmap<L: Into<EventList>>(mut self, wait_events: L) -> FutureMemMap<T> {
        self.set_unmap_wait_events(wait_events);
        self
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
    pub fn create_unmap_event(&mut self) -> OclResult<&mut Event> {
        if let Some(ref queue) = self.queue {
            let uev = Event::user(&queue.context())?;
            self.unmap_event = Some(uev);
            Ok(self.unmap_event.as_mut().unwrap())
        } else {
            Err("FutureMemMap::create_unmap_event: No queue found!".into())
        }
    }

    /// Specifies an event which will be triggered (set complete) after this
    /// future resolves into a `MemMap` **and** after that `MemMap` is dropped
    /// or manually unmapped.
    ///
    /// See `::create_unmap_event`.
    pub fn enew_unmap<En>(mut self, mut enew: En) -> FutureMemMap<T>
            where En: ClNullEventPtr {
        {
            let unmap_event = self.create_unmap_event()
                .expect("FutureMemMap::enew_unmap");
            unsafe { enew.clone_from(unmap_event); }
        }
        self
    }

    /// Specifies the queue to be used for the unmap command.
    pub fn set_unmap_queue(&mut self, queue: Queue) {
        self.queue = Some(queue)
    }

    /// Specifies the queue to be used for the unmap command.
    pub fn with_unmap_queue(mut self, queue: Queue) -> FutureMemMap<T> {
        self.set_unmap_queue(queue);
        self
    }

    /// Returns the unmap event if it has been created.
    ///
    /// [UNSTABLE]: This method may be renamed or otherwise changed.
    #[inline]
    pub fn unmap_event(&self) -> Option<&Event> {
        self.unmap_event.as_ref()
    }

    /// Blocks the current thread until the OpenCL command is complete and an
    /// appropriate lock can be obtained on the underlying data.
    pub fn wait(self) -> OclResult<MemMap<T>> {
        <Self as Future>::wait(self)
    }

    /// Resolves this `FutureMemMap` into a `MemMap`.
    fn to_mapped_mem(&mut self) -> OclResult<MemMap<T>> {
        match (self.core.take(), self.buffer.take(), self.queue.take()) {
            (Some(core), Some(buffer), Some(queue)) => {
                // TODO: Add `buffer_is_mapped` to list of joined stuff.
                unsafe { Ok(MemMap::new(core, self.len, self.unmap_wait_events.take(),
                    self.unmap_event.take(), buffer, queue,
                    /*self.buffer_is_mapped.take().unwrap()*/)) }
            },
            _ => Err("FutureMemMap::create_unmap_event: No queue and/or buffer found!".into()),
        }
    }
}

#[cfg(not(feature = "async_block"))]
impl<T> Future for FutureMemMap<T> where T: OclPrm + 'static {
    type Item = MemMap<T>;
    type Error = OclError;

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
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        // println!("Polling FutureMemMap...");
        let _ = self.callback_is_set;
        self.map_event.wait_for()?;
        self.to_mapped_mem().map(|mm| Async::Ready(mm))
    }
}

unsafe impl<T: OclPrm> Send for FutureMemMap<T> {}
unsafe impl<T: OclPrm> Sync for FutureMemMap<T> {}



