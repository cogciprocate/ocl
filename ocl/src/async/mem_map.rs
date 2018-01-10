// use std::sync::Arc;
// use std::sync::atomic::AtomicBool;
use std::ops::{Deref, DerefMut};
use core::{self, OclPrm, ClWaitListPtr, ClNullEventPtr, MemMap as MemMapCore, Mem as MemCore, AsMem};
use standard::{ClWaitListPtrEnum, ClNullEventPtrEnum, Event, EventList, Queue};
use error::{Result as OclResult};


/// An unmap command builder.
///
/// [UNSTABLE]
#[must_use = "commands do nothing unless enqueued"]
#[derive(Debug)]
pub struct MemUnmapCmd<'c, T> where T: 'c + OclPrm {
    queue: Option<&'c Queue>,
    mem_map: &'c mut MemMap<T>,
    ewait: Option<ClWaitListPtrEnum<'c>>,
    enew: Option<ClNullEventPtrEnum<'c>>,
}

impl<'c, T> MemUnmapCmd<'c, T> where T: OclPrm {
    /// Returns a new unmap command builder.
    fn new(mem_map: &'c mut MemMap<T>) -> MemUnmapCmd<'c, T>
    {
        MemUnmapCmd {
            queue: None,
            mem_map: mem_map,
            ewait: None,
            enew: None,
        }
    }

    /// Specifies a queue to use for this call only.
    pub fn queue<'q, Q>(mut self, queue: &'q Q) -> MemUnmapCmd<'c, T>
            where 'q: 'c, Q: 'q + AsRef<Queue> {
        self.queue = Some(queue.as_ref());
        self
    }


    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<Ewl>(mut self, ewait: Ewl) -> MemUnmapCmd<'c, T>
            where Ewl: Into<ClWaitListPtrEnum<'c>> {
        self.ewait = Some(ewait.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew<En>(mut self, enew: En) -> MemUnmapCmd<'c, T>
            where En: Into<ClNullEventPtrEnum<'c>> {
        self.enew = Some(enew.into());
        self
    }

    /// Enqueues this command.
    ///
    pub fn enq(self) -> OclResult<()> {
        self.mem_map.enqueue_unmap(self.queue, self.ewait, self.enew)
    }
}


/// A view of memory mapped by `clEnqueueMap{...}`.
///
///
/// [UNSTABLE]: Still in a state of flux: ~90% stable
///
//
// [NOTE]: Do not derive/impl `Clone`. Will not be thread safe without a mutex.
//
#[derive(Debug)]
pub struct MemMap<T> where T: OclPrm {
    core: MemMapCore<T>,
    len: usize,
    buffer: MemCore,
    queue: Queue,
    unmap_wait_events: Option<EventList>,
    unmap_event: Option<Event>,
    is_unmapped: bool,
    // buffer_is_mapped: Arc<AtomicBool>
}

impl<T> MemMap<T>  where T: OclPrm {
    pub unsafe fn new(core: MemMapCore<T>, len: usize, unmap_wait_events: Option<EventList>,
            unmap_event: Option<Event>, buffer: MemCore, queue: Queue,
            /*buffer_is_mapped: Arc<AtomicBool>*/) -> MemMap<T> {
        MemMap {
            core: core,
            len: len,
            buffer: buffer,
            queue: queue,
            unmap_wait_events: unmap_wait_events,
            unmap_event: unmap_event,
            is_unmapped: false,
            // buffer_is_mapped,
        }
    }

    /// Returns an unmap command builder.
    ///
    /// Call `::enq` on it to enqueue the unmap command.
    pub fn unmap<'c>(&'c mut self) -> MemUnmapCmd<'c, T> {
        MemUnmapCmd::new(self)
    }

    /// Enqueues an unmap command for this memory object immediately.
    ///
    /// Prefer `::unmap` for a more stable interface as this function may
    /// change at any time.
    pub fn enqueue_unmap<Ewl, En>(&mut self, queue: Option<&Queue>, ewait_opt: Option<Ewl>,
            mut enew_opt: Option<En>) -> OclResult<()>
            where En: ClNullEventPtr, Ewl: ClWaitListPtr
    {
        if !self.is_unmapped {
            assert!(!(ewait_opt.is_some() && self.unmap_wait_events.is_some()),
                "MemMap::enqueue_unmap: Cannot set an event wait list for the unmap command \
                when the 'unmap_wait_events' has already been set.");

            let mut origin_event_opt = if self.unmap_event.is_some() || enew_opt.is_some() {
                Some(Event::empty())
            } else {
                None
            };

            core::enqueue_unmap_mem_object(queue.unwrap_or(&self.queue), &self.buffer,
                &self.core, ewait_opt.and(self.unmap_wait_events.as_ref()), origin_event_opt.as_mut())?;

            self.is_unmapped = true;

            if let Some(origin_event) = origin_event_opt {
                if let Some(ref mut enew) = enew_opt {
                    unsafe { enew.clone_from(&origin_event) }
                }

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
        } else {
            Err("ocl_core::- ::unmap: Already unmapped.".into())
        }
    }

    /// Returns a reference to the unmap target event if it has been set.
    pub fn unmap_event(&self) -> Option<&Event> {
        self.unmap_event.as_ref()
    }

    /// Returns a reference to the unmap wait event list if it has been set.
    pub fn unmap_wait_events(&self) -> Option<&EventList> {
        self.unmap_wait_events.as_ref()
    }

    /// Returns true if an unmap command has already been enqueued, causing
    /// the memory referenced by this `MemMap` to become invalid.
    #[inline] pub fn is_unmapped(&self) -> bool { self.is_unmapped }

    /// Returns a pointer to the host mapped memory.
    #[inline] pub fn as_ptr(&self) -> *const T { self.core.as_ptr() }

    /// Returns a mutable pointer to the host mapped memory.
    #[inline] pub fn as_mut_ptr(&mut self) -> *mut T { self.core.as_mut_ptr() }

    /// Returns a reference to the internal core command queue.
    #[inline] pub fn queue(&self) -> &Queue { &self.queue }
}

impl<T> Deref for MemMap<T> where T: OclPrm {
    type Target = [T];

    fn deref(&self) -> &[T] {
        assert!(!self.is_unmapped, "Mapped memory has been unmapped and cannot be accessed.");
        unsafe { self.core.as_slice(self.len) }
    }
}

impl<T> DerefMut for MemMap<T> where T: OclPrm {
    fn deref_mut(&mut self) -> &mut [T] {
        assert!(!self.is_unmapped, "Mapped memory has been unmapped and cannot be accessed.");
        unsafe { self.core.as_slice_mut(self.len) }
    }
}

impl<T: OclPrm> Drop for MemMap<T> {
    fn drop(&mut self) {
        if !self.is_unmapped {
            self.enqueue_unmap::<&Event, &mut Event>(None, None, None).ok();
        }
    }
}

impl<T: OclPrm> AsMem<T> for MemMap<T> {
    fn as_mem(&self) -> &MemCore {
        self.core.as_mem()
    }
}

// impl<'a, T: OclPrm> AsMem<T> for &'a mut MemMap<T> {
//     fn as_mem(&self) -> &MemCore {
//         self.core.as_mem()
//     }
// }