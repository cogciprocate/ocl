//! A read/write locking `Vec` which can be shared between threads and can
//! interact with OpenCL events.
//!
//!

#![allow(unused_imports, dead_code)]

// use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
// use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;
use std::cell::UnsafeCell;
use ffi::{cl_event, c_void};
use futures::{Future, Poll, Canceled, Async};
use futures::sync::{mpsc, oneshot};
use crossbeam::sync::SegQueue;
use async::{Error as AsyncError, Result as AsyncResult};
use ::{OclPrm, Event, Result as OclResult};


pub struct Guard<T> {
    lock: Lock<T>,
}

impl<T> Deref for Guard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.inner.cell.get() }
    }
}

impl<T> DerefMut for Guard<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.inner.cell.get() }
    }
}

impl<T> Drop for Guard<T> {
    fn drop(&mut self) {
        // [TODO]: Consider using `Ordering::Acquire`.
        self.lock.inner.state.store(0, SeqCst);
        self.lock.process_queue();
    }
}


pub struct FutureGuard<T> {
    lock: Option<Lock<T>>,
    rx: oneshot::Receiver<()>,
}

impl<T> Future for FutureGuard<T> {
    type Item = Guard<T>;
    type Error = AsyncError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        if self.lock.is_some() {
            self.lock.as_ref().unwrap().process_queue();

            match self.rx.poll() {
                Ok(status) => Ok(status.map(|_| {
                    Guard { lock: self.lock.take().unwrap() }
                })),
                Err(e) => Err(e.into()),
            }
        } else {
            Err("FutureGuard::poll: Task already completed.".into())
        }
    }
}


pub struct Request {
    tx: oneshot::Sender<()>,
}


struct Inner<T> {
    state: AtomicUsize,
    cell: UnsafeCell<T>,
    queue: SegQueue<Request>,
}

impl<T> From<T> for Inner<T> {
    #[inline]
    fn from(val: T) -> Inner<T> {
        Inner {
            state: AtomicUsize::new(0),
            cell: UnsafeCell::new(val),
            queue: SegQueue::new(),
        }
    }
}


pub struct Lock<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Lock<T> {
    /// Creates and returns a new `Lock`.
    #[inline]
    pub fn new(val: T) -> Lock<T> {
        Lock {
            inner: Arc::new(Inner::from(val)),
        }
    }

    fn process_queue(&self) {
        // [TODO]: Consider using `Ordering::Acquire`.
        match self.inner.state.load(SeqCst) {
            // Unlocked:
            0 => {
                if let Some(req) = self.inner.queue.try_pop() {
                    req.tx.complete(());
                    self.inner.state.store(1, SeqCst);
                }
            },
            // Locked:
            1 => (),
            // Something else:
            n => panic!("Lock::process_queue: inner.state: {}.", n),
        }

    }

    /// Returns a new `Guard` which can be used like a future.
    pub fn lock(&self) -> FutureGuard<T> {
        let (tx, rx) = oneshot::channel();
        self.inner.queue.push(Request { tx: tx });
        FutureGuard { lock: Some((*self).clone()), rx: rx }
    }

    /// Returns a mutable reference to the inner `Vec` if there are currently
    /// no other copies of this `Lock`.
    ///
    /// Since this call borrows the inner lock mutably, no actual locking needs to
    /// take place---the mutable borrow statically guarantees no locks exist.
    ///
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut T> {
        Arc::get_mut(&mut self.inner).map(|inn| unsafe { &mut *inn.cell.get() })
    }
}

impl<T> From<T> for Lock<T> {
    #[inline]
    fn from(val: T) -> Lock<T> {
        Lock {
            inner: Arc::new(Inner::from(val))
        }
    }
}

impl<T> Clone for Lock<T> {
    #[inline]
    fn clone(&self) -> Lock<T> {
        Lock {
            inner: self.inner.clone(),
        }
    }
}


#[cfg(test)]
mod tests {
    #![allow(unused_variables, unused_imports, dead_code)]
    use super::*;

    #[test]
    fn cycle() {
        let val = Lock::from(999i32);

        println!("Reading val...");
        {
            let future_guard = val.lock();
            let guard = future_guard.wait().unwrap();
            println!("val: {}", *guard);
        }

        println!("Storing new val...");
        {
            let future_guard = val.lock();
            let mut guard = future_guard.wait().unwrap();
            *guard = 5;
        }

        println!("Reading val...");
        {
            let future_guard = val.lock();
            let guard = future_guard.wait().unwrap();
            println!("val: {}", *guard);
        }
    }

    #[test]
    fn concurrent() {
        let val = Lock::from(10000i32);

        let fg0 = val.lock();
        let fg1 = val.lock();
        let fg2 = val.lock();

        println!("Reading val 0...");
        {
            let guard = fg0.wait().unwrap();
            println!("val: {}", *guard);
        }

        println!("Reading val 1...");
        {
            let guard = fg1.wait().unwrap();
            println!("val: {}", *guard);
        }

        println!("Reading val 2...");
        {
            let guard = fg2.wait().unwrap();
            println!("val: {}", *guard);
        }

        // println!("Storing new val...");
        // {
        //     let mut guard = fg.wait().unwrap();
        //     *guard = 55;
        // }

        // println!("Reading val...");
        // {
        //     let guard = fg.wait().unwrap();
        //     println!("val: {}", *guard);
        // }
    }
}