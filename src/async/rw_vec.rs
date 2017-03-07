//! A read/write locking `Vec` which can be shared between threads and can
//! interact with OpenCL events.
//!
//!

#![allow(unused_imports, dead_code)]

// use std::marker::PhantomData;
use std::sync::Arc;
// use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;
use std::cell::UnsafeCell;
use ffi::{cl_event, c_void};
// use futures::sync::mpsc::{self, Sender, Receiver};
use futures::{Future, Poll, Canceled, Async};
use futures::sync::{mpsc, oneshot};
use crossbeam::sync::SegQueue;
use async::{LockSimple, TryLockSimple, Error as AsyncError, Result as AsyncResult};
use ::{OclPrm, Event, Result as OclResult};



pub struct Guard<T> {
    rw_vec: RwVec<T>,
}


pub struct FutureGuard<T> {
    rw_vec: Option<RwVec<T>>,
    rx: oneshot::Receiver<()>,
    // rx: oneshot::Receiver<RwVec<T>>,
}

impl<T> Future for FutureGuard<T> where T: OclPrm {
    // type Item = RwVec<T>;
    // type Item = TryLock<Vec<T>>;
    type Item = Guard<T>;
    type Error = AsyncError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        if self.rw_vec.is_some() {
            // self.rw_vec.as_ref().unwrap().process_queue();

            match self.rx.poll() {
                Ok(status) => Ok(status.map(|_| {
                    Guard { rw_vec: self.rw_vec.take().unwrap() }
                })),
                Err(e) => Err(e.into()),
            }
        } else {
            Err("FutureGuard::poll: Task already completed.".into())
        }

        // self.rx.poll().map_err(|e| e.into())
    }
}


pub struct Request {
    tx: oneshot::Sender<()>,
    // tx: oneshot::Sender<RwVec<T>>,
}


struct Inner<T> {
    // lock: Lock<Vec<T>>,
    state: AtomicUsize,
    cell: UnsafeCell<Vec<T>>,
    queue: SegQueue<Request>,
    // pending: Option<Request>,
    // request_pending: AtomicBool,
    // queue: mpsc::UnboundedReceiver<Request>,
    // queue_tx: mpsc::UnboundedSender<Request>,
}

impl<T> From<Vec<T>> for Inner<T> where T: OclPrm {
    #[inline]
    fn from(vec: Vec<T>) -> Inner<T> {
        // let (queue_tx, queue) = mpsc::unbounded();

        Inner {
            // lock: Lock::new(vec),
            state: AtomicUsize::new(0),
            cell: UnsafeCell::new(vec),
            queue: SegQueue::new(),
            // pending: None,
            // request_pending: AtomicBool::new(false),
            // queue: queue,
            // queue_tx: queue_tx,
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
            inner: Arc::new(Inner::from(Vec::<T>::new())),
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

    fn process_queue<'a>(&'a self) -> Option<TryLockSimple<'a, Vec<T>>> {
        // if !self.inner.request_pending.load(SeqCst) {
        //     if let Some(req) = self.inner.queue.try_pop() {
        //         req.tx.complete(());
        //     }
        // }

        // if let Some(handle) = self.inner.lock.try_lock() {
        //     if let Some(req) = self.inner.queue.try_pop() {
        //         req.tx.complete(handle);
        //     }
        // }

        // match self.inner.lock.try_lock() {
        //     Some(v) =>
        //     None => {
        //         if let Some(req) = self.inner.queue.try_pop() {
        //             req.tx.complete(());
        //         }
        //     }
        // }

        // self.inner.lock.try_lock()
        None
    }

    // /// Returns a new `ReadGuard` which can be used like a future.
    // #[inline]
    // pub fn read(&self) -> AsyncResult<FutureReadGuard<T>> {
    //     self.write()
    // }

    /// Returns a new `Guard` which can be used like a future.
    pub fn lock(&self) -> FutureGuard<T> {
        let (tx, rx) = oneshot::channel();
        self.inner.queue.push(Request { tx: tx });
        self.process_queue();
        // self.inner.queue_tx.send(Request { tx: tx })?;

        FutureGuard { rw_vec: Some(self.clone()), rx: rx }
        // Ok(FutureGuard { rx: rx })
    }

    /// Returns a mutable reference to the inner `Vec` if there are currently
    /// no other copies of this `RwVec`.
    ///
    /// Since this call borrows the inner lock mutably, no actual locking needs to
    /// take place---the mutable borrow statically guarantees no locks exist.
    ///
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut Vec<T>> {
        Arc::get_mut(&mut self.inner).map(|inn| unsafe { &mut *inn.cell.get() })
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


#[cfg(test)]
mod tests {
    #![allow(unused_variables, unused_imports, dead_code)]
    use super::*;

    #[test]
    fn cycle() {
        let vec = RwVec::from(vec![999i32; 128]);
        let future_guard = vec.lock();
        let guard = future_guard.wait().unwrap();
    }
}