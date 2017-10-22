//! A mutex-like lock which can be shared between threads and can interact
//! with OpenCL events.
//!
//!
//! TODO: Add doc links.
//
//


use std::ops::{Deref, DerefMut};
use async::{OrderLock, FutureGuard, ReadGuard, WriteGuard};
use async::qutex::QrwLock;


pub type FutureRwGuard<T, G> = FutureGuard<Vec<T>, G>;
pub type FutureReadGuard<T> = FutureRwGuard<T, ReadGuard<Vec<T>>>;
pub type FutureWriteGuard<T> = FutureRwGuard<T, WriteGuard<Vec<T>>>;


/// A locking `Vec` which interoperates with OpenCL events and Rust futures to
/// provide exclusive access to data.
///
/// Calling `::read` or `::write` returns a future which will resolve into a
/// `RwGuard`.
///
/// ## Platform Compatibility
///
/// Some CPU device/platform combinations have synchronization problems when
/// accessing an `RwVec` from multiple threads. Known platforms with problems
/// are 2nd and 4th gen Intel Core processors (Sandy Bridge and Haswell) with
/// Intel OpenCL CPU drivers. Others may be likewise affected. Run the
/// `device_check.rs` example to determine if your device/platform is
/// affected. AMD platform drivers are known to work properly on the
/// aforementioned CPUs so use those instead if possible.
#[derive(Clone, Debug)]
pub struct RwVec<T> {
    lock: OrderLock<Vec<T>>,
}

impl<T> RwVec<T> {
    /// Creates and returns a new `RwVec`.
    #[inline]
    pub fn new() -> RwVec<T> {
        RwVec {
            lock: OrderLock::new(Vec::new())
        }
    }

    /// Returns a new `FutureRwGuard` which will resolve into a a `RwGuard`.
    pub fn read(self) -> FutureGuard<Vec<T>, ReadGuard<Vec<T>>> {
        self.lock.read()

    }

    /// Returns a new `FutureRwGuard` which will resolve into a a `RwGuard`.
    pub fn write(self) -> FutureGuard<Vec<T>, WriteGuard<Vec<T>>> {
        self.lock.write()
    }

    /// Returns a mutable slice into the contained `Vec`.
    ///
    /// Used by buffer command builders when preparing future read and write
    /// commands.
    ///
    /// Do not use unless you are 100% certain that there will be no other
    /// reads or writes for the entire access duration (only possible if
    /// manually manipulating the lock status).
    pub unsafe fn as_mut_slice(&self) -> &mut [T] {
        let ptr = (*self.lock.as_mut_ptr()).as_mut_ptr();
        let len = (*self.lock.as_ptr()).len();
        ::std::slice::from_raw_parts_mut(ptr, len)
    }

    /// Returns the length of the internal `Vec`.
    pub fn len(&self) -> usize {
        unsafe { (*self.lock.as_ptr()).len() }
    }

    /// Returns a pointer address to the internal array, usable as a unique
    /// identifier.
    ///
    /// Note that resizing the `Vec` will likely change the address. Also, the
    /// same 'id' could be reused by another `RwVec` created after this one is
    /// dropped.
    pub fn id(&self) -> usize {
        unsafe { (*self.lock.as_ptr()).as_ptr() as usize }
    }
}

impl<T> From<QrwLock<Vec<T>>> for RwVec<T> {
    fn from(q: QrwLock<Vec<T>>) -> RwVec<T> {
        RwVec { lock: OrderLock::from(q) }
    }
}

impl<T> From<Vec<T>> for RwVec<T> {
    fn from(vec: Vec<T>) -> RwVec<T> {
        RwVec { lock: OrderLock::from(vec) }
    }
}

impl<T> Deref for RwVec<T> {
    type Target = OrderLock<Vec<T>>;

    #[inline]
    fn deref(&self) -> &OrderLock<Vec<T>> {
        &self.lock
    }
}

impl<T> DerefMut for RwVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut OrderLock<Vec<T>> {
        &mut self.lock
    }
}