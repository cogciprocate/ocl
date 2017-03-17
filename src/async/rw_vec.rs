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
pub use self::qutex::qutex::{Request, Guard, FutureGuard, Qutex};

// Allows access to the data contained within a lock just like a mutex guard.
pub struct RwGuard<T> {
    rw_vec: RwVec<T>,
    drop_event: Option<Event>,
}

impl<T> RwGuard<T> {
    fn new(rw_vec: RwVec<T>, drop_event: Option<Event>) -> RwGuard<T> {
        RwGuard {
            rw_vec: rw_vec,
            drop_event: drop_event,
        }
    }

    /// Returns a reference to the event previously set using
    /// `create_drop_event` on the `PendingRwGuard` which preceeded this
    /// `RwGuard`. The event can be manually 'triggered' by calling
    /// `...set_complete()...` or used normally (as a wait event) by
    /// subsequent commands. If the event is not manually completed it will be
    /// automatically set complete when this `ReadCompletion` is dropped.
    pub fn drop_event(&self) -> Option<&Event> {
        self.drop_event.as_ref()
    }
}

impl<T> Deref for RwGuard<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        unsafe { &*self.rw_vec.as_ptr() }
    }
}

impl<T> DerefMut for RwGuard<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        unsafe { &mut *self.rw_vec.as_mut_ptr() }
    }
}

impl<T> Drop for RwGuard<T> {
    fn drop(&mut self) {
        unsafe { self.rw_vec.unlock().expect("Error dropping RwGuard") };

        if let Some(ref de) = self.drop_event {
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


/// Like a `FutureGuard` but additionally waits on an OpenCL event.
#[must_use = "futures do nothing unless polled"]
pub struct PendingRwGuard<T> {
    rw_vec: Option<RwVec<T>>,
    rx: oneshot::Receiver<()>,
    wait_event: Option<Event>,
    command_trigger: Event,
    command_completion: Option<Event>,
    drop_event: Option<Event>,
    stage: Stage,
    len: usize,
}

impl<T> PendingRwGuard<T> {
    pub fn new<C: ClContextPtr>(rw_vec: RwVec<T>, rx: oneshot::Receiver<()>, context: C,
            wait_event: Option<Event>) -> OclResult<PendingRwGuard<T>>
    {
        let command_trigger = Event::user(context)?;

        let len = unsafe { (*rw_vec.as_ptr()).len() };

        Ok(PendingRwGuard {
            rw_vec: Some(rw_vec),
            rx: rx,
            wait_event: wait_event,
            command_trigger: command_trigger,
            command_completion: None,
            drop_event: None,
            stage: Stage::Marker,
            len: len,
        })
    }

    /// Sets the command completion event.
    pub fn set_command_completion(&mut self, command_completion: Event) {
        self.command_completion = Some(command_completion);
    }

    /// Creates an event which can be 'triggered' later by having its status
    /// set to complete.
    ///
    /// This event can be added immediately to the wait list of subsequent
    /// commands with the expectation that when all chained futures are
    /// complete, the event will automatically be 'triggered' (set to
    /// complete), causing those commands to execute. This can be used to
    /// inject host side code in amongst device side commands (kernels, etc.).
    pub fn create_drop_event(&mut self, queue: &Queue) -> AsyncResult<&mut Event> {
        let uev = Event::user(&queue.context())?;
        self.drop_event = Some(uev);
        Ok(self.drop_event.as_mut().unwrap())
    }

    /// Returns a reference to the event previously set using
    /// `create_drop_event`.
    pub fn drop_event(&self) -> Option<&Event> {
        self.drop_event.as_ref()
    }

    /// Returns a reference to the event which will trigger when the wait
    /// marker is complete and the qutex is locked.
    pub fn command_trigger(&self) -> &Event {
        &self.command_trigger
    }

    pub fn wait(self) -> AsyncResult<RwGuard<T>> {
        <Self as Future>::wait(self)
    }

    pub unsafe fn as_mut_ptr(&self) -> Option<*mut T> {
        self.rw_vec.as_ref().map(|rw_vec| (*rw_vec.as_mut_ptr()).as_mut_ptr())
    }

    pub fn len(&self) -> usize {
        self.len
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
        unsafe { self.rw_vec.as_ref().unwrap().process_queue()
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
                    // let task_ptr = standard::box_raw_void(task::park());
                    // unsafe { command_completion.set_callback(standard::_unpark_task, task_ptr)?; };
                    command_completion.set_unpark_callback()?;
                    return Ok(Async::NotReady);
                } else {
                    Ok(Async::Ready(RwGuard::new(self.rw_vec.take().unwrap(), self.drop_event.take())))
                }                
            },
            None => Err("PendingRwGuard::poll_command: No command event set.".into()),
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

#[derive(Clone)]
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

    pub fn lock_pending_event<C>(self, context: C, wait_event: Option<Event>) 
            -> OclResult<PendingRwGuard<T>>
            where C: ClContextPtr
    {
        let (tx, rx) = oneshot::channel();
        unsafe { self.qutex.push_request(Request::new(tx)); }
        PendingRwGuard::new(self.into(), rx, context, wait_event)
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

impl<T> Deref for RwVec<T> {
    type Target = Qutex<Vec<T>>;

    fn deref(&self) -> &Qutex<Vec<T>> {
        &self.qutex
    }
}

impl<T> DerefMut for RwVec<T> {
    fn deref_mut(&mut self) -> &mut Qutex<Vec<T>> {
        &mut self.qutex
    }
}