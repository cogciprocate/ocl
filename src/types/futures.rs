#![allow(dead_code)]

use futures::{Future, Poll, Async};
use futures::task;
use ::{Error as OclError, Result as OclResult, Event, UserEvent, Context, OclPrm, MappedMem,
    MappedMemPtr};


pub struct FutureMappedMem<T: OclPrm> {
    ptr: MappedMemPtr<T>,
    len: usize,
    map_event: Event,
    unmap_event: Option<UserEvent>,
}

impl<T: OclPrm> FutureMappedMem<T> {
    pub unsafe fn new(ptr: *mut T, len: usize, map_event: Event) -> FutureMappedMem<T> {
        FutureMappedMem {
            ptr: MappedMemPtr::new(ptr),
            len: len,
            map_event: map_event,
            unmap_event: None,
        }
    }

    pub fn create_unmap_event(&mut self, context: &Context) -> OclResult<&mut UserEvent> {
        match UserEvent::new(context) {
            Ok(uev) => {
                self.unmap_event = Some(uev);
                Ok(self.unmap_event.as_mut().unwrap())
            }
            Err(err) => Err(err)
        }
    }
}


impl<T: OclPrm> Future for FutureMappedMem<T> {
    type Item = MappedMem<T>;
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        println!("Polling FutureMappedMem...");

        while let Ok(false) = self.map_event.is_complete() {
            task::park();
        }

        match self.map_event.is_complete() {
            Ok(true) => unsafe {
                Ok(Async::Ready(MappedMem::new(self.ptr.as_ptr(), self.len, self.unmap_event.take())))
            },
            Ok(false) => {
                println!("Not Ready -- Unreachable?.");
                Ok(Async::NotReady)
            },
            Err(err) => Err(err),
        }

        // unsafe { Ok(Async::Ready(MappedMem::new(self.ptr.as_ptr(), self.len, self.unmap_event.take()))) }
    }
}

unsafe impl<T: OclPrm> Send for FutureMappedMem<T> {}
unsafe impl<T: OclPrm> Sync for FutureMappedMem<T> {}
