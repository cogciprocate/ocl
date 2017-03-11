//! A mutex-like lock which can be shared between threads and can interact
//! with OpenCL events.
//!
//!

#![allow(unused_imports, dead_code)]

extern crate qutex;

// use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
// use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;
use std::cell::UnsafeCell;
use ffi::{cl_event, c_void};
use futures::{task, Future, Poll, Canceled, Async};
use futures::sync::{mpsc, oneshot};
use crossbeam::sync::SegQueue;
use ::{OclPrm, Event, Result as OclResult};
use async::{Error as AsyncError, Result as AsyncResult};
use standard::{self, ClWaitListPtrEnum};

pub use self::qutex::qutex::{Request, Guard, FutureGuard, Qutex};



// TODO: RENAME?
pub struct LockGuard<T> {
    lock: Lock<T>,
}



/// Like a `FutureGuard` but additionally waits on an OpenCL event.
pub struct PendingGuard<T> {
    lock: Option<Lock<T>>,
    rx: oneshot::Receiver<()>,
    wait_event: Event,
    trigger_event: Event,
    // rx_is_complete: bool,
    // ev_is_complete: bool,
}

impl<T> PendingGuard<T> {
    fn new(lock: Option<Lock<T>>, rx: oneshot::Receiver<()>, wait_event: Event) -> OclResult<PendingGuard<T>> {
        let trigger_event = Event::user(&wait_event.context()?)?;

        Ok(PendingGuard {
            lock: lock,
            rx: rx,
            wait_event: wait_event,
            trigger_event: trigger_event,
            // rx_is_complete: false,
            // ev_is_complete: false,
        })
    }

    pub fn trigger_event(&self) -> &Event {
        &self.trigger_event
    }

    pub fn wait(self) -> AsyncResult<LockGuard<T>> {
        <Self as Future>::wait(self)
    }

    pub unsafe fn cell_mut(&mut self) -> Option<&mut T> {
        // self.lock.as_ref().map(|l| &mut *l.inner.cell.get())
        self.lock.as_mut().and_then(|l| l.get_mut())
    }
}

impl<T> Future for PendingGuard<T> {
    type Item = LockGuard<T>;
    type Error = AsyncError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        if self.lock.is_some() {
            unsafe { self.lock.as_ref().unwrap().process_queue(); }

            if !self.wait_event.is_complete()? {
                let task_ptr = standard::box_raw_void(task::park());
                    unsafe { self.wait_event.set_callback(standard::_unpark_task, task_ptr)?; };
                    return Ok(Async::NotReady);
            }

            match self.rx.poll() {
                Ok(status) => Ok(status.map(|_| {
                    LockGuard { lock: self.lock.take().unwrap() }
                })),
                Err(e) => return Err(e.into()),
            }
        } else {
            Err("PendingGuard::poll: Task already completed.".into())
        }
    }
}

#[derive(Clone)]
pub struct Lock<T> {
    qutex: Qutex<T>,
}

impl<T> Lock<T> {
        /// Creates and returns a new `Lock`.
    #[inline]
    pub fn new(val: T) -> Lock<T> {
        Lock {
            qutex: Qutex::new(val)
        }
    }

    pub fn lock_pending_event(&self, wait_event: Event) -> OclResult<PendingGuard<T>> {
        let (tx, rx) = oneshot::channel();
        // self.inner.queue.push(Request { tx: tx, wait_event: Some(wait_event.clone()) });
        unsafe { self.qutex.push_request(Request::new(tx)); }
        // PendingGuard { lock: Some((*self).clone()), rx: rx, wait_event: wait_event }
        PendingGuard::new(Some((*self).clone().into()), rx, wait_event)
    }
}

impl<T> From<Qutex<T>> for Lock<T> {
    fn from(q: Qutex<T>) -> Lock<T> {
        Lock { qutex: q }
    }
}

impl<T> Deref for Lock<T> {
    type Target = Qutex<T>;

    fn deref(&self) -> &Qutex<T> {
        &self.qutex
    }
}

impl<T> DerefMut for Lock<T> {
    fn deref_mut(&mut self) -> &mut Qutex<T> {
        &mut self.qutex
    }
}