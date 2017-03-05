//! A read/write locking `Vec` which can be shared between threads and can
//! interact with OpenCL events.
//!
//!

#![allow(unused_imports, dead_code)]

// use std::marker::PhantomData;
use std::sync::Arc;
use ffi::{cl_event, c_void};
// use futures::sync::mpsc::{self, Sender, Receiver};
use futures::{Future, Poll, Canceled};
use futures::sync::{mpsc, oneshot};
use crossbeam::sync::SegQueue;
use async::{Lock, Error as AsyncError, Result as AsyncResult};
use ::{OclPrm, Event, Result as OclResult};




type FutureReadGuard<T> = FutureWriteGuard<T>;

pub struct FutureWriteGuard<T> {
    rw_vec: RwVec<T>,
    rx: oneshot::Receiver<()>,
}

impl<T> Future for FutureWriteGuard<T> where T: OclPrm {
    type Item = ();
    type Error = AsyncError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.rx.poll().map_err(|e| e.into())
    }
}


pub struct Request {
    tx: oneshot::Sender<()>,
}


struct Inner<T> {
    lock: Lock<Vec<T>>,
    queue: SegQueue<Request>,
    // queue: mpsc::UnboundedReceiver<Request>,
    // queue_tx: mpsc::UnboundedSender<Request>,
    thing: bool,
}

impl<T> From<Vec<T>> for Inner<T> where T: OclPrm {
    #[inline]
    fn from(vec: Vec<T>) -> Inner<T> {
        // let (queue_tx, queue) = mpsc::unbounded();

        Inner {
            lock: Lock::new(vec),
            queue: SegQueue::new(),
            // queue: queue,
            // queue_tx: queue_tx,
            thing: false,
            // injection: Lock::new(Injection::new()),
        }
    }
}


#[derive(Clone)]
pub struct RwVec<T> {
    inner: Arc<Inner<T>>,
}

impl<T> RwVec<T> where T: OclPrm {
    /// Creates and returns a new `RwVec`.
    #[inline]
    pub fn new() -> RwVec<T> {
        RwVec {
            inner: Arc::new(Inner::from(Vec::new())),
        }
    }

    /// Creates and returns a new `RwVec` initialized to the specified length
    /// with zeros.
    ///
    #[inline]
    pub fn init(len: usize) -> RwVec<T> {
        RwVec {
            inner: Arc::new(Inner::from(vec![Default::default(); len])),
        }
    }

    /// Returns a new `ReadGuard` which can be used like a future.
    #[inline]
    pub fn read(&self) -> AsyncResult<FutureReadGuard<T>> {
        self.write()
    }

    /// Returns a new `WriteGuard` which can be used like a future.
    pub fn write(&self) -> AsyncResult<FutureWriteGuard<T>> {
        let (tx, rx) = oneshot::channel();
        self.inner.queue.push(Request { tx: tx });
        // self.inner.queue_tx.send(Request { tx: tx })?;
        Ok(FutureWriteGuard { rw_vec: self.clone(), rx: rx })
    }

    /// Returns a mutable reference to the inner `Vec` if there are currently
    /// no other copies of this `RwVec`.
    ///
    /// Since this call borrows the inner lock mutably, no actual locking needs to
    /// take place---the mutable borrow statically guarantees no locks exist.
    ///
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut Vec<T>> {
        Arc::get_mut(&mut self.inner).map(|inn| inn.lock.get_mut())
    }
}


impl<T> From<Vec<T>> for RwVec<T> where T: OclPrm {
    #[inline]
    fn from(vec: Vec<T>) -> RwVec<T> {
        RwVec {
            inner: Arc::new(Inner::from(vec))
        }
    }
}