//! A read/write locking `Vec` which can be shared between threads and can
//! interact with OpenCL events.
//!
//!
//
// Some documentation adapted from:
// `https://amanieu.github.io/parking_lot/parking_lot/struct.RwLock.html`.

#![allow(unused_imports, dead_code)]

use std::sync::Arc;
use parking_lot::RwLock;
use ffi::{cl_event, c_void};
use ::{OclPrm, Event, Result as OclResult};
pub use parking_lot::{RwLockReadGuard, RwLockWriteGuard};


extern "C" fn _wait_event_complete(_: cl_event, _: i32, rw_vec: *mut c_void) {

}

struct Injection {
    wait_event: Option<Event>,
    wait_lock_mode: LockMode,
    wait_trigger: Option<Event>,
    unlock_trigger: Option<Event>,
}

impl Injection {
    fn new() -> Injection {
        Injection {
            wait_event: None,
            wait_lock_mode: LockMode::None,
            wait_trigger: None,
            unlock_trigger: None,
        }
    }
}


// * TODO: Add wait event to this.
enum LockMode {
    Read,
    Write,
    None,
}


// Ensure wait marker is removed as soon as it completes.
struct Inner<T> {
    lock: RwLock<Vec<T>>,
    injection: RwLock<Injection>,
}

impl<T> Inner<T> where T: OclPrm {
    #[inline]
    fn new() -> Inner<T> {
        Inner::from(Vec::new())
    }
}

impl<T> From<Vec<T>> for Inner<T> where T: OclPrm {
    #[inline]
    fn from(vec: Vec<T>) -> Inner<T> {
        Inner {
            lock: RwLock::new(vec),
            injection: RwLock::new(Injection::new()),
        }
    }
}


/// A locking `Vec`.
pub struct RwVec<T> {
    inner: Arc<Inner<T>>,
}

impl<T> RwVec<T> where T: OclPrm {
    /// Creates and returns a new `RwVec`.
    #[inline]
    pub fn new() -> RwVec<T> {
        RwVec {
            inner: Arc::new(Inner::new()),
        }
    }

    /// Creates and returns a new `RwVec` initialized to the specified length
    /// with the default value for `T` (zeros).
    ///
    #[inline]
    pub fn init(len: usize) -> RwVec<T> {
        RwVec {
            inner: Arc::new(Inner::from(vec![Default::default(); len])),
        }
    }

    /// Blocks until the wait marker completes.
    fn wait_for_marker(&self) {
        if let Some(ref e) = self.inner.wait_event {
            e.wait_for().unwrap();
        }
    }

    fn lock_after(&mut self, wait_event: Event, lock_mode: LockMode) -> OclResult<&Event> {
        if self.inner.wait_event.is_some() { return Err("RwVec::...lock_after: Wait event already set.".into()); }
        let context = wait_event.context()?;
        self.inner.wait_event = Some(wait_event);

        ///// SET UP CALLBACK

        self.inner.wait_lock_mode = lock_mode;
        self.inner.wait_trigger = Some(Event::user(&context)?);
        Ok(self.inner.wait_trigger.as_ref().unwrap())
    }


    pub fn read_lock_after(&mut self, wait_event: Event) -> OclResult<&Event> {
        self.lock_after(wait_event, LockMode::Read)
    }

    pub fn write_lock_after(&mut self, wait_event: Event) -> OclResult<&Event> {
        self.lock_after(wait_event, LockMode::Write)
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
    ///
    #[inline]
    pub fn read(&self) -> RwLockReadGuard<Vec<T>> {
        self.wait_for_marker();
        self.inner.lock.read()
    }

    /// Attempts to acquire this `RwVec` with shared read access.
    ///
    /// If the access could not be granted at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned which will release the shared access
    /// when it is dropped.
    ///
    /// This function does not block.
    ///
    #[inline]
    pub fn try_read(&self) -> Option<RwLockReadGuard<Vec<T>>> {
        if self.inner.injection.wait_event.is_some() { return None; }
        self.inner.lock.try_read()
    }

    /// Locks this `RwVec` with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// This function will not return while other writers or other readers
    /// currently have access to the lock.
    ///
    /// Returns an RAII guard which will drop the write access of this `RwVec`
    /// when dropped.
    ///
    #[inline]
    pub fn write(&self) -> RwLockWriteGuard<Vec<T>> {
        self.wait_for_marker();
        self.inner.lock.write()
    }

    /// Attempts to lock this `RwVec` with exclusive write access.
    ///
    /// If the lock could not be acquired at this time, then `None` is returned.
    /// Otherwise, an RAII guard is returned which will release the lock when
    /// it is dropped.
    ///
    /// This function does not block.
    ///
    #[inline]
    pub fn try_write(&self) -> Option<RwLockWriteGuard<Vec<T>>> {
        if self.inner.injection.wait_event.is_some() { return None; }
        self.inner.lock.try_write()
    }

    /// Returns a mutable reference to the inner `Vec` if there are currently no other
    ///
    /// Since this call borrows the `RwLock` mutably, no actual locking needs to
    /// take place---the mutable borrow statically guarantees no locks exist.
    ///
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut Vec<T>> {
        Arc::get_mut(&mut self.inner).map(|inn| inn.lock.get_mut())
    }

    /// Releases exclusive write access of the `RwVec`.
    ///
    /// # Safety
    ///
    /// This function must only be called if the `RwVec` was locked using
    /// `raw_write` or `raw_try_write`, or if an `RwLockWriteGuard` from this
    /// `RwVec` was leaked (e.g. with `mem::forget`). The `RwVec` must be locked
    /// with exclusive write access.
    ///
    #[inline]
    pub unsafe fn raw_unlock_write(&self) {
        self.inner.lock.raw_unlock_write();
    }

    /// Returns a cloned `RwVec` which refers to the same underlying data.
    ///
    /// This function does not create a new `Vec` nor does it allocate
    /// anything, it simply increases the internal reference count and returns
    /// a copied reference exactly the same as any other ocl type (`Buffer`,
    /// `Queue`, etc.). Locking the cloned `RwVec` will also lock its source.
    ///
    /// To clone underlying data, first lock (`::read` or `::write`) then
    /// clone using the reference available from the returned guard.
    ///
    #[inline]
    pub fn clone(&self) -> RwVec<T> {
        RwVec {
            inner: self.inner.clone(),
        }
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