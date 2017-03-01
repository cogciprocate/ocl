#![allow(unused_imports, dead_code)]

use ::{OclPrm};
// use std::sync::Arc;
use parking_lot::RwLock;
pub use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

pub struct RwVec<T> {
    // lock: Arc<RwLock<Vec<T>>>,
    lock: RwLock<Vec<T>>,
}

impl<T> RwVec<T> where T: OclPrm {
    /// Creates and returns a new `RwVec` initialized with the default value of `T`.
    pub fn new(len: usize) -> RwVec<T> {
        RwVec {
            // lock: Arc::new(RwLock::new(vec![Default::default(); len]))
            lock: RwLock::new(vec![Default::default(); len])
        }
    }

    #[inline]
    pub fn read(&self) -> RwLockReadGuard<Vec<T>> {
        self.lock.read()
    }

    /// Attempts to acquire this rwlock with shared read access.
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

    /// Locks this rwlock with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// This function will not return while other writers or other readers
    /// currently have access to the lock.
    ///
    /// Returns an RAII guard which will drop the write access of this rwlock
    /// when dropped.
    #[inline]
    pub fn write(&self) -> RwLockWriteGuard<Vec<T>> {
        self.lock.write()
    }

    /// Attempts to lock this rwlock with exclusive write access.
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
    // pub fn get_mut(&mut self) -> Option<&mut Vec<T>> {
    pub fn get_mut(&mut self) -> &mut Vec<T> {
        // Arc::get_mut(&mut self.lock).map(|l| l.get_mut())
        self.lock.get_mut()
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