
// #![allow(dead_code, unused_imports)]

use std::mem;
// use std::ptr;
// use std::slice;
use std::ops::{Deref, DerefMut};
// use std::thread::Thread;
// use std::sync::Arc;
// use libc::c_void;

// use futures;
use futures::{Future, Poll, Async};
// use futures::sync::oneshot::{self, Sender};
use futures::task::{self, /*Task, UnparkEvent, EventSet*/};

// use ffi::{cl_event, cl_mem};
use core::{self, Event as EventCore, OclPrm, ClWaitListPtr, ClNullEventPtr,
    MemMap as MemMapCore, Mem, CommandQueue, CommandQueueInfo, CommandQueueInfoResult,
    /*CommandExecutionStatus, MemObjectType, ImageChannelOrder, ImageChannelDataType,
    ContextProperty, PlatformId,*/};
use standard::{box_raw_void, _unpark_task};
// use ::{Error as OclError, Result as OclResult};
use super::{Error as AsyncError, Result as AsyncResult};


// pub struct EventListTrigger {
//     wait_events: EventList,
//     completion_event: UserEvent,
//     callback_is_set: bool,
// }

// pub struct EventTrigger {
//     wait_event: Event,
//     completion_event: UserEvent,
//     callback_is_set: bool,
// }

// impl EventTrigger {
//     pub fn new(wait_event: Event, completion_event: UserEvent) -> EventTrigger {
//         EventTrigger {
//             wait_event: wait_event,
//             completion_event: completion_event ,
//             callback_is_set: false,
//         }
//     }
// }

#[must_use = "futures do nothing unless polled"]
pub struct FutureMemMap<T: OclPrm> {
    core: Option<MemMapCore<T>>,
    len: usize,
    map_event: EventCore,
    unmap_target: Option<EventCore>,
    buffer: Option<Mem>,
    queue: Option<CommandQueue>,
    callback_is_set: bool,

}

impl<T: OclPrm> FutureMemMap<T> {
    pub unsafe fn new(core: MemMapCore<T>, len: usize, map_event: EventCore, buffer: Mem, queue: CommandQueue)
            -> FutureMemMap<T>
    {
        FutureMemMap {
            core: Some(core),
            len: len,
            map_event: map_event,
            unmap_target: None,
            buffer: Some(buffer),
            queue: Some(queue),
            callback_is_set: false,
        }
    }

    #[cfg(not(feature = "disable_event_callbacks"))]
    pub fn create_unmap_event(&mut self) -> AsyncResult<&mut EventCore> {
        if let Some(ref queue) = self.queue {
            let context = match core::get_command_queue_info(queue,
                    CommandQueueInfo::Context)
            {
                CommandQueueInfoResult::Context(ctx) => ctx,
                CommandQueueInfoResult::Error(err) => return Err((*err).into()),
                _ => unreachable!(),
            };

            match EventCore::user(&context) {
                Ok(uev) => {
                    self.unmap_target = Some(uev);
                    Ok(self.unmap_target.as_mut().unwrap())
                }
                Err(err) => Err(err.into())
            }
        } else {
            Err("FutureMemMap::create_unmap_target: No queue found!".into())
        }
    }

    pub fn to_mapped_mem(&mut self) -> AsyncResult<MemMap<T>> {
        let joined = self.core.take().and_then(|core| {
            self.buffer.take().and_then(|buf| {
                self.queue.take().and_then(|queue| {
                    Some((core, buf, queue))
                })
            })
        });

        match joined {
            Some((core, buffer, queue)) => {
                unsafe { Ok(MemMap::new(core, self.len,
                    self.unmap_target.take(), buffer, queue )) }
            },
            _ => Err("FutureMemMap::create_unmap_target: No queue and/or buffer found!".into()),
        }
    }

    /// Returns the unmap event if it has been created.
    #[inline]
    pub fn get_unmap_target(&self) -> Option<&EventCore> {
        self.unmap_target.as_ref()
    }
}

#[cfg(not(feature = "disable_event_callbacks"))]
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
                    unsafe {
                        // println!("Setting callback...");
                        self.map_event.set_callback(Some(_unpark_task),
                            box_raw_void(task::park()))?;
                    }
                    // println!("Task callback is set for event: {:?}.", self.map_event);
                    self.callback_is_set = true;
                }

                Ok(Async::NotReady)
            },
            Err(err) => Err(err.into()),
        }
    }
}

/// Blocking implementation.
#[cfg(feature = "disable_event_callbacks")]
impl<T: OclPrm> Future for FutureMemMap<T> {
    type Item = MemMap<T>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        // loop {
        //     match self.map_event.is_complete() {
        //         Ok(true) => return self.to_mapped_mem().map(|mm| Async::Ready(mm)),
        //         Ok(false) => {
        //             // task::park();

        //             // sleep somehow?
        //             continue;
        //         },
        //         Err(err) => return Err(err),
        //     };
        // }

        self.map_event.wait_for()?;
        self.to_mapped_mem().map(|mm| Async::Ready(mm))
    }
}

unsafe impl<T: OclPrm> Send for FutureMemMap<T> {}
unsafe impl<T: OclPrm> Sync for FutureMemMap<T> {}





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

                if !cfg!(feature = "disable_event_callbacks") {
                    // Async version:
                    if self.unmap_target.is_some() {

                        // // [DEBUG]:
                        // println!("Registering event trigger (complete: {})...", origin_event.is_complete().unwrap());

                        #[cfg(not(feature = "disable_event_callbacks"))]
                        self.register_event_trigger(&origin_event)?;

                        // `origin_event` will be reconstructed by the callback
                        // function using `UserEvent::from_raw` and `::drop`
                        // will be run there. Do not also run it here.
                        #[cfg(not(feature = "disable_event_callbacks"))]
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

    #[cfg(not(feature = "disable_event_callbacks"))]
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
    #[cfg(not(feature = "disable_event_callbacks"))]
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

    #[cfg(feature = "disable_event_callbacks")]
    fn drop(&mut self) {
        assert!(self.is_unmapped, "ocl_core::MemMap: '::drop' called while still mapped. \
            Call '::unmap' before allowing this 'MemMap' to fall out of scope.");
    }
}

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







//#############################################################################
//#############################################################################
//########################### FAILED EXPERIMENTS ##############################
//#############################################################################
//#############################################################################

// #[cfg(not(feature = "disable_event_callbacks"))]
// extern "C" fn _unpark_task<T: OclPrm>(event_ptr: ffi::cl_event, event_status: i32,
//         user_data: *mut c_void)
// {
//     // let (_, _, _) = (event_ptr, event_status, user_data);
//     let _ = event_ptr;

//     if event_status == CommandExecutionStatus::Complete as i32 && !user_data.is_null() {
//         // let future = user_data as *mut _ as *mut FutureMemMap<T>;
//         // let task_ptr = user_data as *mut _ as *mut Arc<Task>;
//         let tx_ptr = user_data as *mut _ as *mut Sender<T>;

//         unsafe {
//             println!("Unparking task via callback...");

//             // (*task).unpark();
//             // if (*task).is_current() {
//             //     (*task).unpark();
//             // } else {
//             //     panic!("futures::_unpark_task: Task is not current.");
//             // }

//             // (*future).poll();

//             // self.to_mapped_mem().map(|mm| Async::Ready(mm))

//             // (*future).task.as_ref().unwrap().unpark();

//             // let task = Arc::from_raw(task_ptr);
//             // task.unpark();

//             let tx = Box::from_raw(tx_ptr);
//             tx.complete(Default::default());
//         }
//     } else {
//         panic!("Wake up user data is null or event is not complete.");
//     }
// }


// #[cfg(not(feature = "disable_event_callbacks"))]
// extern "C" fn _dummy(_: ffi::cl_event, _: i32, _: *mut c_void) {}


// #[cfg(not(feature = "disable_event_callbacks"))]
// impl<T: OclPrm> Future for FutureMemMap<T> {
//     type Item = MemMap<T>;
//     type Error = AsyncError;

//     fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
//         println!("Polling FutureMemMap...");

//         loop {
//             match self.map_event.is_complete() {
//                 Ok(true) => {
//                     if self.task.is_none() {
//                         println!("Task completed on first poll.");
//                         // return self.to_mapped_mem().map(|mm| Async::Ready(mm));
//                     } else {
//                         println!("Unsetting callback...");
//                         unsafe { self.map_event.set_callback::<T>(Some(_dummy), None)?; }
//                         self.task = None;
//                         // println!("Task complete but waiting on callback.");
//                         // continue;
//                     }

//                     return self.to_mapped_mem().map(|mm| Async::Ready(mm));
//                 }
//                 Ok(false) => {
//                     if self.task.is_none() {
//                         println!("Task incomplete...");
//                         // self.task = Some(Arc::new(task::park()));

//                         // let unpark = Arc::new(ThreadUnpark::new(thread::current()));

//                         unsafe {
//                             println!("Setting event callback...");

//                             // // [`FutureMemMap<T>`]:
//                             // let self_ptr = self as *mut _ as *mut c_void;
//                             // self.map_event.set_callback_with_ptr(Some(_unpark_task::<T>), self_ptr)?;

//                             // // [`Arc<Task>`]:
//                             // let task = Arc::into_raw(self.task.clone().unwrap()) as *mut _ as *mut c_void;
//                             // self.map_event.set_callback_with_ptr(Some(_unpark_task::<T>), task)?;

//                             let (tx, rx) = oneshot::channel::<()>();
//                             let tx_ptr = Box::into_raw(Box::new(tx)) as *mut _ as *mut c_void;
//                             self.map_event.set_callback_with_ptr(Some(_unpark_task::<T>), tx_ptr)?;

//                             // rx.wait().unwrap();
//                             // let mm = self.to_mapped_mem().map(|mm| Async::Ready(mm))?;
//                             // return Ok(rx.and_then(|| mm));
//                         }

//                         // return self.to_mapped_mem().map(|mm| Async::Ready(mm))

//                         // self.task = Some(task::park());
//                         // continue;
//                         panic!("Whatever.");
//                     } else {
//                         println!("Task incomplete, already parked.");
//                     }

//                     // continue;
//                     return Ok(Async::NotReady);
//                 },
//                 Err(err) => return Err(err),
//             }
//         }
//     }
// }

