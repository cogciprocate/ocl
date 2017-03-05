#![allow(unused_imports, dead_code)]

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::AtomicBool;

pub struct TryLock<'a, T: 'a> {
    __ptr: &'a Lock<T>,
}

impl<'a, T> Deref for TryLock<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.__ptr.data.get() }
    }
}

impl<'a, T> DerefMut for TryLock<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.__ptr.data.get() }
    }
}

impl<'a, T> Drop for TryLock<'a, T> {
    fn drop(&mut self) {
        self.__ptr.locked.store(false, SeqCst);
    }
}


/// A basic lock.
// #[derive(Debug)]
pub struct Lock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for Lock<T> {}
unsafe impl<T: Send> Sync for Lock<T> {}

impl<T> Lock<T> {
    pub fn new(t: T) -> Lock<T> {
        Lock {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(t),
        }
    }

    pub fn try_lock(&self) -> Option<TryLock<T>> {
        if !self.locked.swap(true, SeqCst) {
            Some(TryLock { __ptr: self })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.data.get() }
    }
}

