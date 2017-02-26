
use futures::{task, Future, Poll, Async};

use core::{self, Event as EventCore, OclPrm, MemMap as MemMapCore, Mem, CommandQueue,
    CommandQueueInfo, CommandQueueInfoResult};
use standard::{box_raw_void, _unpark_task, MemMap, Event};
use super::{Error as AsyncError, Result as AsyncResult};



#[must_use = "futures do nothing unless polled"]
pub struct FutureMemMap<T: OclPrm> {
    core: Option<MemMapCore<T>>,
    len: usize,
    map_event: Event,
    unmap_target: Option<Event>,
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
            map_event: map_event.into(),
            unmap_target: None,
            buffer: Some(buffer),
            queue: Some(queue),
            callback_is_set: false,
        }
    }

    #[cfg(feature = "event_callbacks")]
    pub fn create_unmap_event(&mut self) -> AsyncResult<&mut Event> {
        if let Some(ref queue) = self.queue {
            let context = match core::get_command_queue_info(queue,
                    CommandQueueInfo::Context)
            {
                CommandQueueInfoResult::Context(ctx) => ctx,
                CommandQueueInfoResult::Error(err) => return Err((*err).into()),
                _ => unreachable!(),
            };

            // match EventCore::user(&context) {
            //     Ok(uev) => {
            //         self.unmap_target = Some(uev.into());
            //         Ok(self.unmap_target.as_mut().unwrap())
            //     }
            //     Err(err) => Err(err.into())
            // }

            let uev = EventCore::user(&context)?;
            self.unmap_target = Some(uev.into());
            Ok(self.unmap_target.as_mut().unwrap())

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
    pub fn get_unmap_target(&self) -> Option<&Event> {
        self.unmap_target.as_ref()
    }
}

#[cfg(feature = "event_callbacks")]
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
                        self.map_event.set_callback(_unpark_task,
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
#[cfg(not(feature = "event_callbacks"))]
impl<T: OclPrm> Future for FutureMemMap<T> {
    type Item = MemMap<T>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.map_event.wait_for()?;
        self.to_mapped_mem().map(|mm| Async::Ready(mm))
    }
}

unsafe impl<T: OclPrm> Send for FutureMemMap<T> {}
unsafe impl<T: OclPrm> Sync for FutureMemMap<T> {}



