#![allow(unused_imports, dead_code)]

// use std::ops::{Deref, DerefMut};
use standard::{_unpark_task, box_raw_void};
use futures::{task, Future, Poll, Async};
use async::{RwVec, RwLockWriteGuard, Error as AsyncError};
use ::{OclPrm, /*Error as OclError,*/ Event};

#[allow(dead_code)]
pub struct ReadCompletion<T> {
    event: Event,
    // data: Option<RwLockWriteGuard<'d, Vec<T>>>,
    data: Option<RwVec<T>>
}

#[allow(dead_code)]
impl<T> ReadCompletion<T> where T: OclPrm {
    pub fn new(event: Event, data: RwVec<T>) -> ReadCompletion<T> {
        ReadCompletion {
            event: event,
            data: Some(data),
        }
    }
}

// impl<T> Deref for ReadCompletion<T> where T: OclPrm {
//     type Target = [T];

//     fn deref(&self) -> &[T] {
//         self.data.as_slice()
//     }
// }

// impl<T> DerefMut for ReadCompletion<T> where T: OclPrm {
//     fn deref_mut(&mut self) -> &mut [T] {
//         self.data.as_mut_slice()
//     }
// }

/// Non-blocking, proper implementation.
#[cfg(feature = "event_callbacks")]
impl<T> Future for ReadCompletion<T> where T: OclPrm {
    type Item = RwVec<T>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.event.is_complete() {
            Ok(true) => {
                self.data.take()
                    .ok_or(AsyncError::from("ReadCompletion::poll: Data has already been taken."))
                    .map(|d| {
                        unsafe { d.raw_unlock_write(); }
                        Async::Ready(d)
                    })
            },
            Ok(false) => {
                let task_ptr = box_raw_void(task::park());
                unsafe { self.event.set_callback(_unpark_task, task_ptr)?; };
                Ok(Async::NotReady)
            },
            Err(err) => Err(err.into()),
        }
    }
}

/// Blocking implementation (yuk).
#[cfg(not(feature = "event_callbacks"))]
impl<T> Future for ReadCompletion<T> {
    type Item = RwLockWriteGuard<'d, Vec<T>>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.event.wait_for()?;
        Ok(Async::Ready(()))
    }
}
