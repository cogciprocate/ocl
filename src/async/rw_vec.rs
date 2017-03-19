//! A mutex-like lock which can be shared between threads and can interact
//! with OpenCL events.
//!

extern crate qutex;

use std::ops::{Deref, DerefMut};
use futures::{Future, Poll, Async};
use futures::sync::oneshot;
use core::ClContextPtr;
use ::{Event, Result as OclResult, Queue};
use async::{Error as AsyncError, Result as AsyncResult};
pub use self::qutex::{Request, Guard, FutureGuard, Qutex};

/// Allows access to the data contained within a lock just like a mutex guard.
///
pub struct RwGuard<T> {
    rw_vec: RwVec<T>,
    unlock_event: Option<Event>,
}

impl<T> RwGuard<T> {
    /// Returns a new `RwGuard`.
    fn new(rw_vec: RwVec<T>, unlock_event: Option<Event>) -> RwGuard<T> {
        RwGuard {
            rw_vec: rw_vec,
            unlock_event: unlock_event,
        }
    }

    /// Triggers the unlock event and releases the lock held by this `RwGuard`
    /// before returning the original `RwVec`.
    pub fn unlock(self) -> RwVec<T> {
        self.rw_vec.clone()
    }

    /// Returns a reference to the event previously set using
    /// `create_unlock_event` on the `PendingRwGuard` which preceeded this
    /// `RwGuard`. The event can be manually 'triggered' by calling
    /// `...set_complete()...` or used normally (as a wait event) by
    /// subsequent commands. If the event is not manually completed it will be
    /// automatically set complete when this `RwGuard` is dropped.
    pub fn unlock_event(&self) -> Option<&Event> {
        self.unlock_event.as_ref()
    }
}

impl<T> Deref for RwGuard<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        unsafe { &*self.rw_vec.qutex.as_ptr() }
    }
}

impl<T> DerefMut for RwGuard<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        unsafe { &mut *self.rw_vec.qutex.as_mut_ptr() }
    }
}

impl<T> Drop for RwGuard<T> {
    fn drop(&mut self) {
        unsafe { self.rw_vec.qutex.unlock().expect("Error dropping RwGuard") };

        if let Some(ref de) = self.unlock_event {
            if !de.is_complete().expect("ReadCompletion::drop") {
                de.set_complete().expect("ReadCompletion::drop");
            }
        }
    }
}


#[derive(PartialEq)]
enum Stage {
    Marker,
    Qutex,
    Command,
}


/// A future that resolves to an `RwGuard` after ensuring that the data being
/// guarded is appropriately locked during the execution of an OpenCL command.
///
/// 1. Waits until both an exclusive data lock can be obtained **and** all
///    prerequisite OpenCL commands have completed.
/// 2. Triggers an OpenCL command, remaining locked until the command finishes
///    execution.
/// 3. Returns an `RwGuard` which provides exclusive access to the locked
///    data.
/// 
#[must_use = "futures do nothing unless polled"]
pub struct PendingRwGuard<T> {
    rw_vec: Option<RwVec<T>>,
    rx: oneshot::Receiver<()>,
    wait_event: Option<Event>,
    command_trigger: Event,
    command_completion: Option<Event>,
    unlock_event: Option<Event>,
    stage: Stage,
    // len: usize,
}

impl<T> PendingRwGuard<T> {
    pub fn new<C: ClContextPtr>(rw_vec: RwVec<T>, rx: oneshot::Receiver<()>, context: C,
            wait_event: Option<Event>) -> OclResult<PendingRwGuard<T>>
    {
        let command_trigger = Event::user(context)?;

        // let len = unsafe { (*rw_vec.qutex.as_ptr()).len() };

        Ok(PendingRwGuard {
            rw_vec: Some(rw_vec),
            rx: rx,
            wait_event: wait_event,
            command_trigger: command_trigger,
            command_completion: None,
            unlock_event: None,
            stage: Stage::Marker,
            // len: len,
        })
    }

    /// Sets the command completion event.
    ///
    /// A command completion event corresponding to the read or write command
    /// being executed in association with this `PendingRwGuard` must be
    /// specified before this `PendingRwGuard` can be polled.
    pub fn set_command_completion_event(&mut self, command_completion: Event) {
        self.command_completion = Some(command_completion);
    }

    /// Creates an event which can be 'triggered' later by having its status
    /// set to complete.
    ///    
    /// This event can be added to the wait list of subsequent OpenCL commands
    /// with the expectation that when all chained futures are complete, the
    /// event will automatically be 'triggered' (set to complete), causing
    /// those commands to execute. This can be used to inject host side code
    /// in amongst OpenCL commands without thread blocking or delays of any
    /// kind.
    ///
    /// This event will be triggered after this future resolves **and** the
    /// resulting `RwGuard` is dropped.
    pub fn create_unlock_event(&mut self, queue: &Queue) -> AsyncResult<&mut Event> {
        let uev = Event::user(&queue.context())?;
        self.unlock_event = Some(uev);
        Ok(self.unlock_event.as_mut().unwrap())
    }

    /// Returns a reference to the event previously set using
    /// `create_unlock_event`.
    pub fn unlock_event(&self) -> Option<&Event> {
        self.unlock_event.as_ref()
    }

    /// Returns a reference to the event which will trigger when the wait
    /// marker is complete and the qutex is locked.
    pub fn command_trigger_event(&self) -> &Event {
        &self.command_trigger
    }

    /// Blocks the current thread until the OpenCL command is complete and a 
    pub fn wait(self) -> AsyncResult<RwGuard<T>> {
        <Self as Future>::wait(self)
    }

    /// Returns a mutable pointer to the data contained within the internal
    /// `Vec`, bypassing all locks and protections.
    pub unsafe fn as_mut_ptr(&self) -> Option<*mut T> {
        self.rw_vec.as_ref().map(|rw_vec| (*rw_vec.qutex.as_mut_ptr()).as_mut_ptr())
    }

    /// Returns a mutable slice to the data contained within the internal
    /// `Vec`, bypassing all locks and protections.
    pub unsafe fn as_mut_slice<'a, 'b>(&'a self) -> Option<&'b mut [T]> {
        self.as_mut_ptr().map(|ptr| {
            ::std::slice::from_raw_parts_mut(ptr, self.len())
        })
    }

    /// Returns the length of the internal `Vec`.
    pub fn len(&self) -> usize {
        unsafe { (*self.rw_vec.as_ref().expect("PendingRwGuard::len: No RwVec found.")
            .qutex.as_ptr()).len() }
    }

    /// Polls the wait marker event until all requisite commands have
    /// completed then polls the qutex queue.
    fn poll_marker(&mut self) -> AsyncResult<Async<RwGuard<T>>> {
        debug_assert!(self.stage == Stage::Marker);

        // Check completion of wait event, if it exists:
        if let Some(ref wait_event) = self.wait_event {
            if !wait_event.is_complete()? {
                // let task_ptr = standard::box_raw_void(task::park());
                // unsafe { wait_event.set_callback(standard::_unpark_task, task_ptr)?; };
                wait_event.set_unpark_callback()?;
                return Ok(Async::NotReady);
            }
        }

        self.stage = Stage::Qutex;
        self.poll_qutex()
    }

    /// Polls the qutex until we have obtained a lock then polls the command
    /// event.
    fn poll_qutex(&mut self) -> AsyncResult<Async<RwGuard<T>>> {
        debug_assert!(self.stage == Stage::Qutex);

        // Move the queue along:
        unsafe { self.rw_vec.as_ref().unwrap().qutex.process_queue()
            .expect("Error polling PendingRwGuard"); }

        // Check for completion of the rx:
        match self.rx.poll() {
            // If the poll returns `Async::Ready`, we have been popped from
            // the front of the qutex queue and we now have exclusive access.
            // Otherwise, return the `NotReady`. The rx (oneshot channel) will
            // arrange for this task to be awakened when it's ready.
            Ok(status) => {
                match status {
                    Async::Ready(_) => {
                        self.command_trigger.set_complete()?;
                        self.stage = Stage::Command;
                        self.poll_command()
                    },
                    Async::NotReady => Ok(Async::NotReady),
                }
            },
            Err(e) => return Err(e.into()),
        }
    }

    /// Polls the command event until it is complete then returns an `RwGuard`
    /// which can be safely accessed immediately.
    fn poll_command(&mut self) -> AsyncResult<Async<RwGuard<T>>> {
        debug_assert!(self.stage == Stage::Command);

        match self.command_completion {
            Some(ref command_completion) => {
                if !command_completion.is_complete()? {
                    command_completion.set_unpark_callback()?;
                    return Ok(Async::NotReady);
                } else {
                    Ok(Async::Ready(RwGuard::new(self.rw_vec.take().unwrap(), self.unlock_event.take())))
                }                
            },
            None => Err("PendingRwGuard::poll_command: No command event set. A command completion \
                event must be specified using '::set_command_command_completion_event'.".into()),
        }
    }
}

impl<T> Future for PendingRwGuard<T> {
    type Item = RwGuard<T>;
    type Error = AsyncError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        if self.rw_vec.is_some() {
            match self.stage {
                Stage::Marker => self.poll_marker(),
                Stage::Qutex => self.poll_qutex(),
                Stage::Command => self.poll_command(),
            }            
        } else {
            Err("PendingRwGuard::poll: Task already completed.".into())
        }
    }
}

/// A locking `Vec` which interoperates with OpenCL events and Rust futures to
/// provide exclusive access to data.
///
/// Calling `::lock` or `::lock_pending_event` returns a future which will
/// resolve into a `RwGuard`.
///
pub struct RwVec<T> {
    qutex: Qutex<Vec<T>>,
}

impl<T> RwVec<T> {
    /// Creates and returns a new `RwVec`.
    #[inline]
    pub fn new() -> RwVec<T> {
        RwVec {
            qutex: Qutex::new(Vec::new())
        }
    }

    /// Returns a new `FutureGuard` which can be used as a future and will
    /// resolve into a `Guard`.
    pub fn lock(self) -> FutureGuard<Vec<T>> {
        self.qutex.lock()
    }

    /// Returns a new `PendingRwGuard` which will resolve into a a `RwGuard`.
    pub fn lock_pending_event<C>(self, context: C, wait_event: Option<Event>) 
            -> OclResult<PendingRwGuard<T>>
            where C: ClContextPtr
    {
        let (tx, rx) = oneshot::channel();
        unsafe { self.qutex.push_request(Request::new(tx)); }
        PendingRwGuard::new(self.into(), rx, context, wait_event)
    }

    pub unsafe fn as_mut_slice(&self) -> &mut [T] {
        let ptr = (*self.qutex.as_mut_ptr()).as_mut_ptr();
        let len = (*self.qutex.as_ptr()).len();
        ::std::slice::from_raw_parts_mut(ptr, len)
    }
}

impl<T> From<Qutex<Vec<T>>> for RwVec<T> {
    fn from(q: Qutex<Vec<T>>) -> RwVec<T> {
        RwVec { qutex: q }
    }
}

impl<T> From<Vec<T>> for RwVec<T> {
    fn from(vec: Vec<T>) -> RwVec<T> {
        RwVec { qutex: Qutex::new(vec) }
    }
}

impl<T> Clone for RwVec<T> {
    #[inline]
    fn clone(&self) -> RwVec<T> {
        RwVec {
            qutex: self.qutex.clone(),
        }
    }
}

// impl<T> Deref for RwVec<T> {
//     type Target = Qutex<Vec<T>>;

//     fn deref(&self) -> &Qutex<Vec<T>> {
//         &self.qutex
//     }
// }

// impl<T> DerefMut for RwVec<T> {
//     fn deref_mut(&mut self) -> &mut Qutex<Vec<T>> {
//         &mut self.qutex
//     }
// }