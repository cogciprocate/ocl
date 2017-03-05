#![allow(unused_imports, dead_code)]

use std::ops::{Deref, DerefMut};
use standard::{_unpark_task, box_raw_void};
use futures::{task, Future, Poll, Async};
use async::{Error as AsyncError, Result as AsyncResult, RwVec, };
use ::{OclPrm, /*Error as OclError,*/ Event, Queue};

/// [UNSTABLE]
pub struct ReadCompletion<T> {
    data: RwVec<T>,
    triggerable_event: Option<Event>,

}

impl<T> ReadCompletion<T> where T: OclPrm {
    /// Returns a new `ReadCompletion`.
    fn new(data: RwVec<T>, triggerable_event: Option<Event>)
            -> ReadCompletion<T>
    {
        ReadCompletion {
            data: data,
            triggerable_event: triggerable_event,
        }
    }

    /// Returns a reference to the event previously set using
    /// `create_triggerable_event` on the `FutureReadCompletion` which
    /// preceeded this `ReadCompletion`. The event can be manually 'triggered'
    /// by calling `event.set_complete()...` or used normally (as a wait
    /// event) by subsequent commands. If the event is not manually completed
    /// it will be automatically set complete when this `ReadCompletion` is
    /// dropped.
    pub fn triggerable_event(&self) -> Option<&Event> {
        self.triggerable_event.as_ref()
    }
}

impl<T> Deref for ReadCompletion<T> where T: OclPrm {
    type Target = RwVec<T>;

    fn deref(&self) -> &RwVec<T> {
        &self.data
    }
}

impl<T> DerefMut for ReadCompletion<T> where T: OclPrm {
    fn deref_mut(&mut self) -> &mut RwVec<T> {
        &mut self.data
    }
}

impl<T> Drop for ReadCompletion<T> {
    /// Completes the attached triggerable event if it is not already
    /// complete.
    fn drop(&mut self) {
        if let Some(ref e) = self.triggerable_event {
            if !e.is_complete().expect("ReadCompletion::drop") {
                e.set_complete().expect("ReadCompletion::drop");
            }
        }
    }
}


/// [UNSTABLE]
#[allow(dead_code)]
pub struct FutureReadCompletion<T> {
    read_event: Event,
    triggerable_event: Option<Event>,
    data: Option<RwVec<T>>
}

#[allow(dead_code)]
impl<T> FutureReadCompletion<T> where T: OclPrm {
    pub fn new(data: RwVec<T>, read_event: Event) -> FutureReadCompletion<T> {
        FutureReadCompletion {
            data: Some(data),
            read_event: read_event,
            triggerable_event: None,
        }
    }

    /// Creates an event which can be 'triggered' later by having its status
    /// set to complete.
    ///
    /// This event can be added immediately to the wait list of subsequent
    /// commands with the expectation that when all chained futures are
    /// complete, the event will automatically be 'triggered' (set to
    /// complete), causing those commands to execute. This can be used to
    /// inject host side code in amongst device side commands (kernels, etc.).
    pub fn create_triggerable_event(&mut self, queue: &Queue) -> AsyncResult<&mut Event> {
        let uev = Event::user(&queue.context())?;
        self.triggerable_event = Some(uev);
        Ok(self.triggerable_event.as_mut().unwrap())
    }

    /// Returns the callback event if it has been created.
    #[inline]
    pub fn get_triggerable_event(&self) -> Option<&Event> {
        self.triggerable_event.as_ref()
    }
}


/// Non-blocking, proper implementation.
#[cfg(feature = "event_callbacks")]
impl<T> Future for FutureReadCompletion<T> where T: OclPrm {
    // type Item = RwVec<T>;
    type Item = ReadCompletion<T>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.read_event.is_complete() {
            Ok(true) => {
                self.data.take()
                    .ok_or(AsyncError::from("FutureReadCompletion::poll: Data has already been taken."))
                    .map(|d| {
                        println!("Raw-unlocking write.");
                        // unsafe { d.raw_unlock_write(); }
                        Async::Ready(ReadCompletion::new(d, self.triggerable_event.take()))
                    })
            },
            Ok(false) => {
                let task_ptr = box_raw_void(task::park());
                unsafe { self.read_event.set_callback(_unpark_task, task_ptr)?; };
                Ok(Async::NotReady)
            },
            Err(err) => Err(err.into()),
        }
    }
}

/// Blocking implementation (yuk).
#[cfg(not(feature = "event_callbacks"))]
impl<T> Future for FutureReadCompletion<T> {
    type Item = RwLockWriteGuard<'d, Vec<T>>;
    type Error = AsyncError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.read_event.wait_for()?;
        Ok(Async::Ready(()))
    }
}
