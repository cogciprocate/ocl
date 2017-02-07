#![allow(dead_code)]

use functions;
use futures::{Future, Poll, Async};
use futures::task;
use ::{Error as OclError, Result as OclResult, Event, UserEvent, Context, OclPrm, MappedMem,
    MappedMemPtr, Mem, CommandQueue, CommandQueueInfo, CommandQueueInfoResult};


pub struct FutureMappedMem<T: OclPrm> {
    ptr: MappedMemPtr<T>,
    len: usize,
    map_event: Event,
    unmap_event: Option<UserEvent>,
    buffer: Option<Mem>,
    queue: Option<CommandQueue>,
}

impl<T: OclPrm> FutureMappedMem<T> {
    pub unsafe fn new(ptr: *mut T, len: usize, map_event: Event, buffer: Mem, queue: CommandQueue)
            -> FutureMappedMem<T>
    {
        FutureMappedMem {
            ptr: MappedMemPtr::new(ptr),
            len: len,
            map_event: map_event,
            unmap_event: None,
            buffer: Some(buffer),
            queue: Some(queue),
        }
    }

    pub fn create_unmap_event(&mut self) -> OclResult<&mut UserEvent> {
        if let Some(ref queue) = self.queue {
            let context = match functions::get_command_queue_info(queue,
                    CommandQueueInfo::Context)
            {
                CommandQueueInfoResult::Context(ctx) => ctx,
                CommandQueueInfoResult::Error(err) => return Err(*err),
                _ => unreachable!(),
            };

            match UserEvent::new(&context) {
                Ok(uev) => {
                    self.unmap_event = Some(uev);
                    Ok(self.unmap_event.as_mut().unwrap())
                }
                Err(err) => Err(err)
            }
        } else {
            Err("FutureMappedMem::create_unmap_event: No queue found!".into())
        }
    }
}


impl<T: OclPrm> Future for FutureMappedMem<T> {
    type Item = MappedMem<T>;
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        println!("Polling FutureMappedMem...");

        loop {
            match self.map_event.is_complete() {
                Ok(true) => unsafe {
                    return match self.buffer.take().map(|buf| self.queue.take().map(|qu| (buf, qu))) {
                        Some(Some((buffer, queue))) => {
                            Ok(Async::Ready(MappedMem::new(self.ptr.as_ptr(), self.len,
                                self.unmap_event.take(), buffer, queue )))
                        },
                        _ => Err("FutureMappedMem::create_unmap_event: \
                            No queue and/or buffer found!".into()),
                    }
                },
                Ok(false) => {
                    // Ok(Async::NotReady)
                    task::park();
                    continue;
                },
                Err(err) => return Err(err),
            };
        }
    }
}

unsafe impl<T: OclPrm> Send for FutureMappedMem<T> {}
unsafe impl<T: OclPrm> Sync for FutureMappedMem<T> {}
