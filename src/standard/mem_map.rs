
// #![allow(dead_code, unused_imports)]

use std::mem;
// use std::ptr;
// use std::slice;
use std::ops::{Deref, DerefMut};
// use std::thread::Thread;
// use std::sync::Arc;
// use libc::c_void;

// use futures;
// use futures::{Future, Poll, Async};
// use futures::sync::oneshot::{self, Sender};
// use futures::task::{self, /*Task, UnparkEvent, EventSet*/};

// use ffi::{cl_event, cl_mem};
use core::{self, Event as EventCore, OclPrm, ClWaitListPtr, ClNullEventPtr,
    MemMap as MemMapCore, Mem, CommandQueue, /*CommandQueueInfo, CommandQueueInfoResult*/};
// use standard::{box_raw_void, _unpark_task};
use async::{/*Error as AsyncError,*/ Result as AsyncResult};



///////////////////////////////////// KEEP: ///////////////////////////////////////

// pub struct MemCmdUnmap<'b> {
//     queue: &'b CommandQueue,
//     obj_core: &'b MemCore,
//     ewait: Option<ClWaitListPtrEnum<'b>>,
//     enew: Option<ClNullEventPtrEnum<'b>>,
// }

// impl<'b> MemCmdUnmap<'b> {
//     fn new(queue: &'b Queue, obj_core: &'b MemCore) -> MemCmdUnmap<'b>
//     {
//         MemCmdUnmap {
//             queue: queue,
//             obj_core: obj_core,
//             block: true,
//             lock_block: false,
//             kind: MemCmdUnmapKind::Unspecified,
//             shape: MemCmdUnmapDataShape::Lin { offset: 0 },
//             ewait: None,
//             enew: None,
//             mem_len: mem_len,
//         }
//     }

//     /// Specifies a queue to use for this call only.
//     pub fn queue(mut self, queue: &'b Queue) -> MemCmdUnmap<'b> {
//         self.queue = queue;
//         self
//     }


//     /// Specifies a list of events to wait on before the command will run.
//     pub fn ewait<EWL>(mut self, ewait: EWL) -> MemCmdUnmap<'b> where EWL: Into<ClWaitListPtrEnum<'b>> {
//         self.ewait = Some(ewait.into());
//         self
//     }

//     /// Specifies a list of events to wait on before the command will run or
//     /// resets it to `None`.
//     pub fn ewait_opt<EWL>(mut self, ewait: Option<EWL>) -> MemCmdUnmap<'b> where EWL: Into<ClWaitListPtrEnum<'b>> {
//         self.ewait = ewait.map(|el| el.into());
//         self
//     }

//     /// Specifies the destination for a new, optionally created event
//     /// associated with this command.
//     pub fn enew<NE>(mut self, enew: NE) -> MemCmdUnmap<'b>
//             where NE: Into<ClNullEventPtrEnum<'b>>
//     {
//         self.enew = Some(enew.into());
//         self
//     }

//     /// Specifies a destination for a new, optionally created event
//     /// associated with this command or resets it to `None`.
//     pub fn enew_opt<NE>(mut self, enew: Option<NE>) -> MemCmdUnmap<'b>
//             where NE: Into<ClNullEventPtrEnum<'b>>
//     {
//         self.enew = enew.map(|e| e.into());
//         self
//     }

//     /// Enqueues this command.
//     ///
//     pub fn enq(self) -> AsyncResult<()> {

//     }
// }

//////////////////////////////////////////////////////////////////////////////



/// A view of memory mapped by `clEnqueueMap{...}`.
///
///
///
/// ### [UNSTABLE]
///
/// Still in a state of flux but is ~80% stable.
///
//
// [NOTE]: Do not derive/impl `Clone`. Will not be thread safe without a mutex.
//
#[derive(Debug)]
pub struct MemMap<T: OclPrm> {
    core: MemMapCore<T>,
    len: usize,
    buffer: Mem,
    queue: CommandQueue,
    unmap_target: Option<EventCore>,
    callback_is_set: bool,
    is_unmapped: bool,
}

impl<T> MemMap<T>  where T: OclPrm {
    pub unsafe fn new(core: MemMapCore<T>, len: usize, unmap_target: Option<EventCore>,
        buffer: Mem, queue: CommandQueue) -> MemMap<T>
    {
        MemMap {
            core: core,
            len: len,
            buffer: buffer,
            queue: queue,
            unmap_target: unmap_target,
            callback_is_set: false,
            is_unmapped: false,
        }
    }

    // [TODO]: MAKE THIS A COMMANDBUILDERTHINGY:
    pub fn unmap() {
        unimplemented!();
    }

    /// Enqueues an unmap command for this memory object immediately.
    ///
    //
    // [NOTE]: Passing `enew_opt` is yet untested.
    pub fn enqueue_unmap<Ewl, En>(&mut self, queue: Option<&CommandQueue>, ewait_opt: Option<Ewl>,
            mut enew_opt: Option<En>) -> AsyncResult<()>
            where En: ClNullEventPtr, Ewl: ClWaitListPtr
    {
        if !self.is_unmapped {
            let mut origin_event_opt = if self.unmap_target.is_some() || enew_opt.is_some() {
                Some(EventCore::null())
            } else {
                None
            };

            // print!("MemMap::enqueue_unmap: 'core::enqueue_unmap_mem_object' (PRE): \n\
            //     {t}{t}Unmapping with: \n\
            //     {t}{t}- ewait_opt: {:?}, \n\
            //     {t}{t}- origin_event_opt: {:?}",
            //     &ewait_opt, &origin_event_opt, t="  ");

            core::enqueue_unmap_mem_object(queue.unwrap_or(&self.queue), &self.buffer,
                &self.core, ewait_opt, origin_event_opt.as_mut())?;

            // println!(" --> (POST): {:?}", &origin_event_opt);

            self.is_unmapped = true;

            if let Some(origin_event) = origin_event_opt {
                // origin_event refcount: 1

                // If enew_opt is `Some`, update its internal event ptr.
                if let Some(ref mut enew) = enew_opt {
                    // println!("- ::enqueue_unmap: 'Some(ref mut enew) = enew_opt'.");
                    unsafe {
                        // Should be equivalent to `.clone().into_raw()` [TODO]: test.
                        core::retain_event(&origin_event)?;
                        *(enew.alloc_new()) = *(origin_event.as_ptr_ref());
                        // origin_event/enew refcount: 2
                        // println!("- ::enqueue_unmap: '*(enew.alloc_new()) = *(origin_event.as_ptr_ref())' has been set.");
                    }
                }

                if !cfg!(not(feature = "event_callbacks")) {
                    // Async version:
                    if self.unmap_target.is_some() {

                        // // [DEBUG]:
                        // println!("Registering event trigger (complete: {})...", origin_event.is_complete().unwrap());

                        #[cfg(feature = "event_callbacks")]
                        self.register_event_trigger(&origin_event)?;

                        // `origin_event` will be reconstructed by the callback
                        // function using `UserEvent::from_raw` and `::drop`
                        // will be run there. Do not also run it here.
                        #[cfg(feature = "event_callbacks")]
                        mem::forget(origin_event);
                    }
                } else {
                    // Blocking version:
                    if let Some(ref mut um_tar) = self.unmap_target {
                        origin_event.wait_for()?;
                        um_tar.set_complete()?;
                    }
                }
            }

            // core::flush(&self.queue)?;
            // println!("MemMap::enqueue_unmap: Queue flushed.");

            // println!("- ::enqueue_unmap: Returning 'Ok(())'....");

            Ok(())
        } else {
            Err("ocl_core::- ::unmap: Already unmapped.".into())
        }
    }

    #[cfg(feature = "event_callbacks")]
    fn register_event_trigger(&mut self, event: &EventCore) -> AsyncResult<()> {
        debug_assert!(self.is_unmapped && self.unmap_target.is_some());

        if !self.callback_is_set {
            if let Some(ref ev) = self.unmap_target {
                unsafe {
                    let unmap_target_ptr = ev.clone().into_raw();
                    event.set_callback(Some(core::_complete_user_event), unmap_target_ptr)?;

                    // core::flush(&self.queue)?;

                    // // [DEBUG]:
                    // println!("Callback set from trigger: {:?} to target: {:?}", event, unmap_target_ptr);

                }

                self.callback_is_set = true;
                Ok(())
            } else {
                panic!("- ::register_event_trigger: No unmap event target \
                    has been configured with this MemMap.");
            }
        } else {
            Err("Callback already set.".into())
        }
    }

    pub fn get_unmap_target(&self) -> Option<&EventCore> {
        self.unmap_target.as_ref()
    }

    // #[inline]
    // pub fn is_accessible(&self) -> AsyncResult<bool> {
    //     // self.map_event.is_complete().map(|cmplt| cmplt && !self.is_unmapped)
    //     Ok(!self.is_unmapped)
    // }

    #[inline] pub fn is_unmapped(&self) -> bool { self.is_unmapped }
    #[inline] pub fn as_ptr(&self) -> *const T { self.core.as_ptr() }
    #[inline] pub fn as_mut_ptr(&mut self) -> *mut T { self.core.as_mut_ptr() }
    #[inline] pub fn queue(&self) -> &CommandQueue { &self.queue }
}

impl<T> Deref for MemMap<T> where T: OclPrm {
    type Target = [T];

    fn deref(&self) -> &[T] {
        assert!(!self.is_unmapped, "Mapped memory inaccessible. Check with '::is_accessable'
            before attempting to access.");
        // unsafe { slice::from_raw_parts(self.core.as_ptr(), self.len) }
        unsafe { self.core.as_slice(self.len) }
    }
}

impl<T> DerefMut for MemMap<T> where T: OclPrm {
    fn deref_mut(&mut self) -> &mut [T] {
        assert!(!self.is_unmapped, "Mapped memory inaccessible. Check with '::is_accessable'
            before attempting to access.");
        // unsafe { slice::from_raw_parts_mut(self.core.as_ptr(), self.len) }
        unsafe { self.core.as_slice_mut(self.len) }
    }
}

impl<T: OclPrm> Drop for MemMap<T> {
    #[cfg(feature = "event_callbacks")]
    fn drop(&mut self) {

        // // [DEBUG]:
        // println!("Dropping MemMap.");

        if !self.is_unmapped {
            self.enqueue_unmap::<&EventCore, &mut EventCore>(None, None, None).ok();
            // core::flush(&self.queue).unwrap();
            // core::finish(&self.queue).unwrap();
        }

        // // [DEBUG]:
        // println!("MemMap::drop: Unmap enqueued.");
    }

    #[cfg(not(feature = "event_callbacks"))]
    fn drop(&mut self) {
        assert!(self.is_unmapped, "ocl_core::MemMap: '::drop' called while still mapped. \
            Call '::unmap' before allowing this 'MemMap' to fall out of scope.");
    }
}



