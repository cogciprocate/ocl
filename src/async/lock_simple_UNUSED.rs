#![allow(unused_imports, dead_code)]

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::AtomicBool;

pub struct TryLockSimple<'a, T: 'a> {
    __ptr: &'a LockSimple<T>,
}

impl<'a, T> Deref for TryLockSimple<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        unsafe { &*self.__ptr.data.get() }
    }
}

impl<'a, T> DerefMut for TryLockSimple<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.__ptr.data.get() }
    }
}

impl<'a, T> Drop for TryLockSimple<'a, T> {
    fn drop(&mut self) {
        self.__ptr.locked.store(false, SeqCst);
    }
}


/// A basic lock.
// #[derive(Debug)]
pub struct LockSimple<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for LockSimple<T> {}
unsafe impl<T: Send> Sync for LockSimple<T> {}

impl<T> LockSimple<T> {
    pub fn new(t: T) -> LockSimple<T> {
        LockSimple {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(t),
        }
    }

    pub fn try_lock(&self) -> Option<TryLockSimple<T>> {
        if !self.locked.swap(true, SeqCst) {
            Some(TryLockSimple { __ptr: self })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.data.get() }
    }
}

