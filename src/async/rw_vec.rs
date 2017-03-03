//! A read/write locking `Vec` which can be shared between threads and can
//! interact with OpenCL events.
//!
//!
//
// Some documentation adapted from:
// `https://amanieu.github.io/parking_lot/parking_lot/struct.RwLock.html`.

#![allow(unused_imports, dead_code)]

use ::{OclPrm};
use std::sync::Arc;
use parking_lot::RwLock;
pub use parking_lot::{RwLockReadGuard, RwLockWriteGuard};


/// A locking `Vec`.
// #[derive(Clone)]
pub struct RwVec<T> {
    lock: Arc<RwLock<Vec<T>>>,
}

impl<T> RwVec<T> where T: OclPrm {
    /// Creates and returns a new `RwVec`.
    pub fn new() -> RwVec<T> {
        RwVec {
            lock: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Creates and returns a new `RwVec` initialized to the specified length
    /// with the default value for `T`.
    pub fn init(len: usize) -> RwVec<T> {
        RwVec {
            lock: Arc::new(RwLock::new(vec![Default::default(); len])),
        }
    }

    /// Locks this `RwVec` with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// The calling thread will be blocked until there are no more writers
    /// which hold the lock. There may be other readers currently inside the
    /// lock when this method returns.
    ///
    /// Note that attempts to recursively acquire a read lock on a RwLock when
    /// the current thread already holds one may result in a deadlock.
    ///
    /// Returns an RAII guard which will release this thread's shared access
    /// once it is dropped.
    #[inline]
    pub fn read(&self) -> RwLockReadGuard<Vec<T>> {
        self.lock.read()
    }

    /// Attempts to acquire this `RwVec` with shared read access.
    ///
    /// If the access could not be granted at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned which will release the shared access
    /// when it is dropped.
    ///
    /// This function does not block.
    #[inline]
    pub fn try_read(&self) -> Option<RwLockReadGuard<Vec<T>>> {
        self.lock.try_read()
    }

    /// Locks this `RwVec` with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// This function will not return while other writers or other readers
    /// currently have access to the lock.
    ///
    /// Returns an RAII guard which will drop the write access of this `RwVec`
    /// when dropped.
    #[inline]
    pub fn write(&self) -> RwLockWriteGuard<Vec<T>> {
        self.lock.write()
    }

    /// Attempts to lock this `RwVec` with exclusive write access.
    ///
    /// If the lock could not be acquired at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned which will release the lock when
    /// it is dropped.
    ///
    /// This function does not block.
    #[inline]
    pub fn try_write(&self) -> Option<RwLockWriteGuard<Vec<T>>> {
        self.lock.try_write()
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this call borrows the `RwLock` mutably, no actual locking needs to
    /// take place---the mutable borrow statically guarantees no locks exist.
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut Vec<T>> {
        Arc::get_mut(&mut self.lock).map(|l| l.get_mut())
    }

    /// Releases exclusive write access of the `RwVec`.
    ///
    /// # Safety
    ///
    /// This function must only be called if the `RwVec` was locked using
    /// `raw_write` or `raw_try_write`, or if an `RwLockWriteGuard` from this
    /// `RwVec` was leaked (e.g. with `mem::forget`). The `RwVec` must be locked
    /// with exclusive write access.
    #[inline]
    pub unsafe fn raw_unlock_write(&self) {
        self.lock.raw_unlock_write();
    }

    /// Returns a cloned RwVec which refers to the same underlying data.
    ///
    /// This function does not create a new `Vec` nor does it allocate
    /// anything, it simply increases the internal reference count and returns
    /// a copied reference. Locking the cloned `RwVec` will also lock it's
    /// source.
    ///
    /// To clone underlying data, first lock (`::read` or `::write`) then
    /// clone using the returned guard.
    ///
    #[inline]
    pub fn clone(&self) -> RwVec<T> {
        RwVec {
            lock: self.lock.clone(),
        }
    }
}


impl<T> From<Vec<T>> for RwVec<T> where T: OclPrm {
    #[inline]
    fn from(vec: Vec<T>) -> RwVec<T> {
        RwVec {
            lock: Arc::new(RwLock::new(vec))
        }
    }
}

// impl<'d, T> Deref for FutureReadCompletion<'d, T> where T: OclPrm {
//     type Target = [T];

//     fn deref(&self) -> &[T] {
//         self.data.as_slice()
//     }
// }

// impl<'d, T> DerefMut for FutureReadCompletion<'d, T> where T: OclPrm {
//     fn deref_mut(&mut self) -> &mut [T] {
//         self.data.as_mut_slice()
//     }
// }