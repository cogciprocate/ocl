// use std::ops::{Deref, DerefMut};
use standard::{_unpark_task, box_raw_void};
use futures::{task, Future, Poll, Async};
use async::{/*RwLockReadGuard,*/ RwLockWriteGuard, Error as AsyncError};
use ::{OclPrm, /*Error as OclError,*/ Event};

#[allow(dead_code)]
pub struct ReadCompletion<'d, T> where T: 'd {
    event: Event,
    data: Option<RwLockWriteGuard<'d, Vec<T>>>,
}

#[allow(dead_code)]
impl<'d, T> ReadCompletion<'d, T> where T: 'd + OclPrm {
    pub fn new(event: Event, data: RwLockWriteGuard<'d, Vec<T>>) -> ReadCompletion<'d, T> {
        ReadCompletion {
            event: event,
            data: Some(data),
        }
    }
}

// impl<'d, T> Deref for ReadCompletion<'d, T> where T: OclPrm {
//     type Target = [T];

//     fn deref(&self) -> &[T] {
//         self.data.as_slice()
//     }
// }

// impl<'d, T> DerefMut for ReadCompletion<'d, T> where T: OclPrm {
//     fn deref_mut(&mut self) -> &mut [T] {
//         self.data.as_mut_slice()
//     }
// }

/// Non-blocking, proper implementation.
#[cfg(feature = "event_callbacks")]
impl<'d, T> Future for ReadCompletion<'d, T> where T: 'd + OclPrm {
    type Item = RwLockWriteGuard<'d, Vec<T>>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.event.is_complete() {
            Ok(true) => {
                Ok(Async::Ready(self.data.take().unwrap()))
            }
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
impl<'d, T> Future for ReadCompletion<'d, T> {
    type Item = RwLockWriteGuard<'d, Vec<T>>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.event.wait_for()?;
        Ok(Async::Ready(()))
    }
}
