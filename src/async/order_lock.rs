//! A mutex-like lock with a conserved global ordering which can be shared
//! between threads and can interact with OpenCL events.
//!
//!
//! TODO: Add doc links.
//
//

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use futures::{Future, Poll, Async};
use futures::sync::oneshot::{self, Receiver};
use core::ClContextPtr;
use ::{Event, EventList};
use async::{Error as AsyncError, Result as AsyncResult};
use async::qutex::{QrwLock, QrwRequest, RequestKind};


const PRINT_DEBUG: bool = false;


/// Prints a debugging message.
fn print_debug(id: usize, msg: &str) {
    if PRINT_DEBUG {
        println!("###### [{}] {} (thread: {})", id, msg,
            ::std::thread::current().name().unwrap_or("<unnamed>"));
    }
}

/// Extracts an `OrderLock` from a guard of either type.
//
// This saves us two unnecessary atomic stores (the reference count of lock
// going up then down when releasing or up/downgrading) which would occur if
// we were to clone then drop.
unsafe fn extract_order_lock<V, G: OrderGuard<V>>(guard: G) -> OrderLock<V> {
    let order_lock = ::std::ptr::read(guard.order_lock());
    guard.forget();
    order_lock
}


/// A read or write guard for an `OrderLock`.
pub trait OrderGuard<V> where Self: ::std::marker::Sized {
    fn new(order_lock: OrderLock<V>, release_event: Option<Event>) -> Self;
    fn order_lock(&self) -> &OrderLock<V>;

    unsafe fn forget(self) {
        ::std::mem::forget(self);
    }
}


// /// A guard that releases its lock and completes its release event
// /// immediately.
// pub struct VoidGuard();

// impl<V> OrderGuard<V> for VoidGuard {
//     fn new(_order_lock: OrderLock<V>, release_event: Option<Event>) -> VoidGuard {
//         // order_lock.lock.release_read_lock();

//         if let Some(ref e) = release_event {
//             if !e.is_complete().expect("VoidGuard::new") {
//                 print_debug(0, "VoidGuard::new: setting release event complete");
//                 e.set_complete().expect("VoidGuard::new");
//             }
//         }

//         VoidGuard()
//     }

//     fn order_lock(&self) -> &OrderLock<V> {
//         panic!("`::order_lock` not implemented for `VoidGuard`.");
//     }
// }


/// Allows access to the data contained within a lock just like a mutex guard.
#[derive(Debug)]
pub struct ReadGuard<V> {
    order_lock: OrderLock<V>,
    release_event: Option<Event>,
}

impl<V> ReadGuard<V> {
    /// Returns a new `ReadGuard`.
    fn new(order_lock: OrderLock<V>, release_event: Option<Event>) -> ReadGuard<V> {
        print_debug(order_lock.id(), "ReadGuard::new: read lock acquired");
        ReadGuard {
            order_lock: order_lock,
            release_event: release_event,
        }
    }

    /// Returns a reference to the event previously set using
    /// `create_release_event` on the `FutureGuard` which preceded this
    /// `ReadGuard`. The event can be manually 'triggered' by calling
    /// `...set_complete()...` or used normally (as a wait event) by
    /// subsequent commands. If the event is not manually completed it will be
    /// automatically set complete when this `ReadGuard` is dropped.
    pub fn release_event(guard: &ReadGuard<V>) -> Option<&Event> {
        guard.release_event.as_ref()
    }

    /// Triggers the release event and releases the lock held by this `ReadGuard`
    /// before returning the original `OrderLock`.
    pub fn release(mut guard: ReadGuard<V>) -> OrderLock<V> {
        print_debug(guard.order_lock.id(), "WriteGuard::release: releasing read lock");
        unsafe {
            Self::release_components(&mut guard);
            extract_order_lock(guard)
        }
    }

    /// Triggers the release event by setting it complete.
    fn complete_release_event(guard: &mut ReadGuard<V>) {
        if let Some(ref e) = guard.release_event.take() {
            if !e.is_complete().expect("ReadGuard::complete_release_event") {
                print_debug(guard.order_lock.id(), "ReadGuard::complete_release_event: \
                    setting release event complete");
                e.set_complete().expect("ReadGuard::complete_release_event");
            }
        }
    }

    /// Releases the lock and completes the release event.
    unsafe fn release_components(guard: &mut ReadGuard<V>) {
        guard.order_lock.lock.release_read_lock();
        Self::complete_release_event(guard);
    }
}

impl<V> Deref for ReadGuard<V> {
    type Target = V;

    fn deref(&self) -> &V {
        unsafe { &*self.order_lock.lock.as_ptr() }
    }
}

impl<V> Drop for ReadGuard<V> {
    fn drop(&mut self) {
        print_debug(self.order_lock.id(), "dropping and releasing ReadGuard");
        unsafe { Self::release_components(self) }
    }
}

impl<V> OrderGuard<V> for ReadGuard<V> {
    fn new(order_lock: OrderLock<V>, release_event: Option<Event>) -> ReadGuard<V> {
        ReadGuard::new(order_lock, release_event)
    }

    fn order_lock(&self) -> &OrderLock<V> {
        &self.order_lock
    }
}


/// Allows access to the data contained within just like a mutex guard.
#[derive(Debug)]
pub struct WriteGuard<V> {
    order_lock: OrderLock<V>,
    release_event: Option<Event>,
}

impl<V> WriteGuard<V> {
    /// Returns a new `WriteGuard`.
    fn new(order_lock: OrderLock<V>, release_event: Option<Event>) -> WriteGuard<V> {
        print_debug(order_lock.id(), "WriteGuard::new: Write lock acquired");
        WriteGuard {
            order_lock: order_lock,
            release_event: release_event,
        }
    }

    /// Returns a reference to the event previously set using
    /// `create_release_event` on the `FutureGuard` which preceded this
    /// `WriteGuard`. The event can be manually 'triggered' by calling
    /// `...set_complete()...` or used normally (as a wait event) by
    /// subsequent commands. If the event is not manually completed it will be
    /// automatically set complete when this `WriteGuard` is dropped.
    pub fn release_event(guard: &WriteGuard<V>) -> Option<&Event> {
        guard.release_event.as_ref()
    }

    /// Triggers the release event and releases the lock held by this `WriteGuard`
    /// before returning the original `OrderLock`.
    pub fn release(mut guard: WriteGuard<V>) -> OrderLock<V> {
        print_debug(guard.order_lock.id(), "WriteGuard::release: Releasing write lock");
        unsafe {
            Self::release_components(&mut guard);
            extract_order_lock(guard)
        }
    }

    /// Triggers the release event by setting it complete.
    fn complete_release_event(guard: &mut WriteGuard<V>) {
        if let Some(ref e) = guard.release_event.take() {
            if !e.is_complete().expect("WriteGuard::complete_release_event") {
                print_debug(guard.order_lock.id(), "WriteGuard::complete_release_event: \
                    Setting release event complete");
                e.set_complete().expect("WriteGuard::complete_release_event");
            }
        }
    }

    /// Releases the lock and completes the release event.
    unsafe fn release_components(guard: &mut WriteGuard<V>) {
        guard.order_lock.lock.release_write_lock();
        Self::complete_release_event(guard);
    }
}

impl<V> Deref for WriteGuard<V> {
    type Target = V;

    fn deref(&self) -> &V {
        unsafe { &*self.order_lock.lock.as_ptr() }
    }
}

impl<V> DerefMut for WriteGuard<V> {
    fn deref_mut(&mut self) -> &mut V {
        unsafe { &mut *self.order_lock.lock.as_mut_ptr() }
    }
}

impl<V> Drop for WriteGuard<V> {
    fn drop(&mut self) {
        print_debug(self.order_lock.id(), "WriteGuard::drop: Dropping and releasing WriteGuard");
        unsafe { Self::release_components(self) }
    }
}

impl<V> OrderGuard<V> for WriteGuard<V> {
    fn new(order_lock: OrderLock<V>, release_event: Option<Event>) -> WriteGuard<V> {
        WriteGuard::new(order_lock, release_event)
    }

    fn order_lock(&self) -> &OrderLock<V> {
        &self.order_lock
    }
}


/// The polling stage of a `FutureGuard`.
#[derive(Debug, PartialEq)]
enum Stage {
    WaitEvents,
    LockQueue,
    Command,
    Upgrade,
}


/// A future that resolves to a read or write guard after ensuring that the
/// data being guarded is appropriately locked during the execution of an
/// OpenCL command.
///
/// 1. Waits until both an exclusive data lock can be obtained **and** all
///    prerequisite OpenCL commands have completed.
/// 2. Triggers an OpenCL command, remaining locked while the command
///    executes.
/// 3. Returns a guard which provides exclusive (write) or shared (read)
///    access to the locked data.
///
#[must_use = "futures do nothing unless polled"]
#[derive(Debug)]
pub struct FutureGuard<V, G> {
    order_lock: Option<OrderLock<V>>,
    lock_rx: Option<Receiver<()>>,
    wait_list: Option<EventList>,
    lock_event: Option<Event>,
    command_completion: Option<Event>,
    upgrade_after_command: bool,
    upgrade_rx: Option<Receiver<()>>,
    release_event: Option<Event>,
    stage: Stage,
    _guard: PhantomData<G>,
}

impl<V, G> FutureGuard<V, G> where G: OrderGuard<V> {
    /// Returns a new `FutureGuard`.
    fn new(order_lock: OrderLock<V>, lock_rx: Receiver<()>) -> FutureGuard<V, G> {
        FutureGuard {
            order_lock: Some(order_lock),
            lock_rx: Some(lock_rx),
            wait_list: None,
            lock_event: None,
            command_completion: None,
            upgrade_after_command: false,
            upgrade_rx: None,
            release_event: None,
            stage: Stage::WaitEvents,
            _guard: PhantomData,
        }
    }

    /// Sets an event wait list.
    ///
    /// Setting a wait list will cause this `FutureGuard` to wait until
    /// contained events have their status set to complete before obtaining a
    /// lock on the guarded internal value.
    ///
    /// [UNSTABLE]: This method may be renamed or otherwise changed at any time.
    pub fn set_wait_list<L: Into<EventList>>(&mut self, wait_list: L) {
        assert!(self.wait_list.is_none(), "Wait list has already been set.");
        self.wait_list = Some(wait_list.into());
    }

    /// Sets a command completion event.
    ///
    /// If a command completion event corresponding to the read or write
    /// command being executed in association with this `FutureGuard` is
    /// specified before this `FutureGuard` is polled it will cause this
    /// `FutureGuard` to suffix itself with an additional future that will
    /// wait until the command completion event completes before resolving
    /// into an `OrderGuard`.
    ///
    /// Not specifying a command completion event will cause this
    /// `FutureGuard` to resolve into an `OrderGuard` immediately after the
    /// lock is obtained (indicated by the optionally created lock event).
    ///
    /// TODO: Reword this.
    /// [UNSTABLE]: This method may be renamed or otherwise changed at any time.
    pub fn set_command_completion_event(&mut self, command_completion: Event) {
        assert!(self.command_completion.is_none(), "Command completion event has already been set.");
        self.command_completion = Some(command_completion);
    }

    /// Creates an event which will be triggered when a lock is obtained on
    /// the guarded internal value.
    ///
    /// The returned event can be added to the wait list of subsequent OpenCL
    /// commands with the expectation that when all preceding futures are
    /// complete, the event will automatically be 'triggered' by having its
    /// status set to complete, causing those commands to execute. This can be
    /// used to inject host side code in amongst OpenCL commands without
    /// thread blocking or extra delays of any kind.
    pub fn create_lock_event<C: ClContextPtr>(&mut self, context: C) -> AsyncResult<&Event> {
        assert!(self.lock_event.is_none(), "Lock event has already been created.");
        self.lock_event = Some(Event::user(context)?);
        Ok(self.lock_event.as_mut().unwrap())
    }

    /// Creates an event which will be triggered after this future resolves
    /// **and** the ensuing `OrderGuard` is dropped or manually released.
    ///
    /// The returned event can be added to the wait list of subsequent OpenCL
    /// commands with the expectation that when all preceding futures are
    /// complete, the event will automatically be 'triggered' by having its
    /// status set to complete, causing those commands to execute. This can be
    /// used to inject host side code in amongst OpenCL commands without
    /// thread blocking or extra delays of any kind.
    pub fn create_release_event<C: ClContextPtr>(&mut self, context: C) -> AsyncResult<&Event> {
        assert!(self.release_event.is_none(), "Release event has already been created.");
        self.release_event = Some(Event::user(context)?);
        Ok(self.release_event.as_ref().unwrap())
    }

    /// Returns a reference to the event previously created with
    /// `::create_lock_event` which will trigger (be completed) when the wait
    /// events are complete and the lock is locked.
    pub fn lock_event(&self) -> Option<&Event> {
        self.lock_event.as_ref()
    }

    /// Returns a reference to the event previously created with
    /// `::create_release_event` which will trigger (be completed) when a lock
    /// is obtained on the guarded internal value.
    pub fn release_event(&self) -> Option<&Event> {
        self.release_event.as_ref()
    }

    /// Blocks the current thread until the OpenCL command is complete and an
    /// appropriate lock can be obtained on the underlying data.
    pub fn wait(self) -> AsyncResult<G> {
        <Self as Future>::wait(self)
    }

    /// Returns a mutable pointer to the data contained within the internal
    /// value, bypassing all locks and protections.
    ///
    /// ## Panics
    ///
    /// This future must not have already resolved into a guard.
    ///
    pub fn as_ptr(&self) -> *const V {
        self.order_lock.as_ref().map(|order_lock| order_lock.lock.as_ptr())
            .expect("FutureGuard::as_ptr: No OrderLock found.")
    }

    /// Returns a mutable pointer to the data contained within the internal
    /// value, bypassing all locks and protections.
    ///
    /// ## Panics
    ///
    /// This future must not have already resolved into a guard.
    ///
    pub fn as_mut_ptr(&self) -> *mut V {
        self.order_lock.as_ref().map(|order_lock| order_lock.lock.as_mut_ptr())
            .expect("FutureGuard::as_mut_ptr: No OrderLock found.")
    }

    // /// The 'id' of the associated `OrderLock`.
    // pub fn id(&self) -> usize {
    //     self.order_lock.as_ref().expect("FutureGuard::id: No OrderLock found.").id()
    // }

    /// Returns a reference to the `OrderLock` used to create this future.
    pub fn order_lock(&self) -> &OrderLock<V> {
        self.order_lock.as_ref().expect("FutureGuard::order_lock: No OrderLock found.")
    }

    /// Polls the wait events until all requisite commands have completed then
    /// polls the lock queue.
    fn poll_wait_events(&mut self) -> AsyncResult<Async<G>> {
        debug_assert!(self.stage == Stage::WaitEvents);
        print_debug(self.order_lock.as_ref().unwrap().id(), "FutureGuard::poll_wait_events: Called");

        // Check completion of wait list, if it exists:
        if let Some(ref mut wait_list) = self.wait_list {
            // if PRINT_DEBUG { println!("###### [{}] FutureGuard::poll_wait_events: \
            //     Polling wait_events (thread: {})...", self.order_lock.as_ref().unwrap().id(),
            //     ::std::thread::current().name().unwrap_or("<unnamed>")); }

            if let Async::NotReady = wait_list.poll()? {
                return Ok(Async::NotReady);
            }

        }

        self.stage = Stage::LockQueue;
        self.poll_lock()
    }

    /// Polls the lock until we have obtained a lock then polls the command
    /// event.
    #[cfg(not(feature = "async_block"))]
    fn poll_lock(&mut self) -> AsyncResult<Async<G>> {
        debug_assert!(self.stage == Stage::LockQueue);
        print_debug(self.order_lock.as_ref().unwrap().id(), "FutureGuard::poll_lock: Called");

        // Move the queue along:
        unsafe { self.order_lock.as_ref().unwrap().lock.process_queues(); }

        // Check for completion of the lock rx:
        if let Some(ref mut lock_rx) = self.lock_rx {
            match lock_rx.poll() {
                // If the poll returns `Async::Ready`, we have been popped from
                // the front of the lock queue and we now have exclusive access.
                // Otherwise, return the `NotReady`. The rx (oneshot channel) will
                // arrange for this task to be awakened when it's ready.
                Ok(status) => {
                    if PRINT_DEBUG { println!("###### [{}] FutureGuard::poll_lock: status: {:?}, \
                        (thread: {}).", self.order_lock.as_ref().unwrap().id(), status,
                        ::std::thread::current().name().unwrap_or("<unnamed>")); }
                    match status {
                        Async::Ready(_) => {
                            if let Some(ref lock_event) = self.lock_event {
                                lock_event.set_complete()?
                            }
                            self.stage = Stage::Command;
                        },
                        Async::NotReady => return Ok(Async::NotReady),
                    }
                },
                // Err(e) => return Err(e.into()),
                Err(e) => panic!("FutureGuard::poll_lock: {:?}", e),
            }
        } else {
            unreachable!();
        }

        self.poll_command()
    }


    /// Polls the lock until we have obtained a lock then polls the command
    /// event.
    #[cfg(feature = "async_block")]
    fn poll_lock(&mut self) -> AsyncResult<Async<G>> {
        debug_assert!(self.stage == Stage::LockQueue);
        print_debug(self.order_lock.as_ref().unwrap().id(), "FutureGuard::poll_lock: Called");

        // Move the queue along:
        unsafe { self.order_lock.as_ref().unwrap().lock.process_queues(); }

        // Wait until completion of the lock rx:
        self.lock_rx.take().wait()?;

        if let Some(ref lock_event) = self.lock_event {
            lock_event.set_complete()?
        }

        self.stage = Stage::Command;
        // if PRINT_DEBUG { println!("###### [{}] FutureGuard::poll_lock: Moving to command stage.",
        //     self.order_lock.as_ref().unwrap().id()); }
        return self.poll_command();
    }

    /// Polls the command event until it is complete then returns an `OrderGuard`
    /// which can be safely accessed immediately.
    fn poll_command(&mut self) -> AsyncResult<Async<G>> {
        debug_assert!(self.stage == Stage::Command);
        print_debug(self.order_lock.as_ref().unwrap().id(), "FutureGuard::poll_command: Called");

        if let Some(ref mut command_completion) = self.command_completion {
            // if PRINT_DEBUG { println!("###### [{}] FutureGuard::poll_command: Polling command \
            //     completion event (thread: {}).", self.order_lock.as_ref().unwrap().id(), ::std::thread::current().name()
            //     .unwrap_or("<unnamed>")); }

            if let Async::NotReady = command_completion.poll()? {
                return Ok(Async::NotReady);
            }
        }

        // Set cmd event to `None` so it doesn't get waited on unnecessarily
        // when this `FutureGuard` drops.
        self.command_completion = None;

        if self.upgrade_after_command {
            self.stage = Stage::Upgrade;
            self.poll_upgrade()
        } else {
            Ok(Async::Ready(self.into_guard()))
        }
    }

    /// Polls the lock until it has been upgraded.
    ///
    /// Only used if `::upgrade_after_command` has been called.
    ///
    #[cfg(not(feature = "async_block"))]
    fn poll_upgrade(&mut self) -> AsyncResult<Async<G>> {
        debug_assert!(self.stage == Stage::Upgrade);
        debug_assert!(self.upgrade_after_command);
        print_debug(self.order_lock.as_ref().unwrap().id(), "FutureGuard::poll_upgrade: Called");

        if self.upgrade_rx.is_none() {
            match unsafe { self.order_lock.as_ref().unwrap().lock.upgrade_read_lock() } {
                Ok(_) => {
                    print_debug(self.order_lock.as_ref().unwrap().id(),
                        "FutureGuard::poll_upgrade: Write lock acquired. Upgrading immediately.");
                    Ok(Async::Ready(self.into_guard()))
                },
                Err(rx) => {
                    self.upgrade_rx = Some(rx);
                    match self.upgrade_rx.as_mut().unwrap().poll() {
                        Ok(res) => {
                            // print_debug(self.order_lock.as_ref().unwrap().id(),
                            //     "FutureGuard::poll_upgrade: Channel completed. Upgrading.");
                            // Ok(res.map(|_| self.into_guard()))
                            match res {
                                Async::Ready(_) => {
                                    print_debug(self.order_lock.as_ref().unwrap().id(),
                                        "FutureGuard::poll_upgrade: Channel completed. Upgrading.");
                                    Ok(Async::Ready(self.into_guard()))
                                },
                                Async::NotReady => {
                                    print_debug(self.order_lock.as_ref().unwrap().id(),
                                        "FutureGuard::poll_upgrade: Upgrade rx not ready.");
                                    Ok(Async::NotReady)
                                },
                            }
                        },
                        // Err(e) => Err(e.into()),
                        Err(e) => panic!("FutureGuard::poll_upgrade: {:?}", e),
                   }
                },
            }
        } else {
            // Check for completion of the upgrade rx:
            match self.upgrade_rx.as_mut().unwrap().poll() {
                Ok(status) => {
                    print_debug(self.order_lock.as_ref().unwrap().id(),
                        &format!("FutureGuard::poll_upgrade: Status: {:?}", status));
                    Ok(status.map(|_| self.into_guard()))
                },
                // Err(e) => Err(e.into()),
                Err(e) => panic!("FutureGuard::poll_upgrade: {:?}", e),
            }
        }
    }

    /// Polls the lock until it has been upgraded.
    ///
    /// Only used if `::upgrade_after_command` has been called.
    ///
    #[cfg(feature = "async_block")]
    fn poll_upgrade(&mut self) -> AsyncResult<Async<G>> {
        debug_assert!(self.stage == Stage::Upgrade);
        debug_assert!(self.upgrade_after_command);
        print_debug(self.order_lock.as_ref().unwrap().id(), "FutureGuard::poll_upgrade: Called");

        match unsafe { self.order_lock.as_ref().unwrap().lock.upgrade_read_lock() } {
            Ok(_) => Ok(Async::Ready(self.into_guard())),
            Err(rx) => {
                self.upgrade_rx = Some(rx);
                self.upgrade_rx.take().unwrap().wait()?;
                Ok(Async::Ready(self.into_guard()))
            }
        }
    }

    /// Resolves this `FutureGuard` into the appropriate result guard.
    fn into_guard(&mut self) -> G {
        print_debug(self.order_lock.as_ref().unwrap().id(), "FutureGuard::into_guard: All polling complete");
        G::new(self.order_lock.take().unwrap(), self.release_event.take())
    }
}

impl<V, G> Future for FutureGuard<V, G> where G: OrderGuard<V> {
    type Item = G;
    type Error = AsyncError;

    #[inline]
    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        if self.order_lock.is_some() {
            match self.stage {
                Stage::WaitEvents => self.poll_wait_events(),
                Stage::LockQueue => self.poll_lock(),
                Stage::Command => self.poll_command(),
                Stage::Upgrade => self.poll_upgrade(),
            }
        } else {
            Err("FutureGuard::poll: Task already completed.".into())
        }
    }
}

impl<V, G> Drop for FutureGuard<V, G> {
    /// Drops this FutureGuard.
    ///
    /// Blocks the current thread until the command associated with this
    /// `FutureGuard` (represented by the command completion event)
    /// completes. This ensures that the underlying value is not dropped
    /// before the command completes (which would cause obvious problems).
    //
    //
    //
    //  [FIXME]: Investigate what happens if we drop after `::poll_lock` has
    //  succeeded (is that possible?).
    //
    //
    //
    fn drop(&mut self) {
        if let Some(ref ccev) = self.command_completion {
            // println!("###### FutureGuard::drop: Event ({:?}) incomplete...", ccev);
            // panic!("###### FutureGuard::drop: Event ({:?}) incomplete...", ccev);
            ccev.wait_for().expect("Error waiting on command completion event \
                while dropping 'FutureGuard'");
        }
        if let Some(ref rev) = self.release_event {
            rev.set_complete().expect("Error setting release event complete \
                while dropping 'FutureGuard'");
        }
    }
}

// a.k.a. FutureRead<V>
impl<V> FutureGuard<V, ReadGuard<V>> {
    pub fn upgrade_after_command(self) -> FutureGuard<V, WriteGuard<V>> {
        use std::ptr::read;

        let future_guard = unsafe {
            FutureGuard {
                order_lock: read(&self.order_lock),
                lock_rx: read(&self.lock_rx),
                wait_list: read(&self.wait_list),
                lock_event: read(&self.lock_event),
                upgrade_after_command: true,
                upgrade_rx: None,
                command_completion: read(&self.command_completion),
                release_event: read(&self.release_event),
                stage: read(&self.stage),
                _guard: PhantomData,
            }
        };

        ::std::mem::forget(self);

        future_guard
    }
}


/// A lock with conserved global order which interoperates with OpenCL events
/// and Rust futures to provide exclusive access to data.
///
/// Calling `::read` or `::write` returns a future which will resolve into a
/// `OrderGuard`.
///
/// ## Platform Compatibility
///
/// Some CPU device/platform combinations have synchronization problems when
/// accessing an `OrderLock` from multiple threads. Known platforms with problems
/// are 2nd and 4th gen Intel Core processors (Sandy Bridge and Haswell) with
/// Intel OpenCL CPU drivers. Others may be likewise affected. Run the
/// `device_check.rs` example to determine if your device/platform is
/// affected. AMD platform drivers are known to work properly on the
/// aforementioned CPUs so use those instead if possible.
#[derive(Debug)]
pub struct OrderLock<V> {
    lock: QrwLock<V>,
}

impl<V> OrderLock<V> {
    /// Creates and returns a new `OrderLock`.
    #[inline]
    pub fn new(data: V) -> OrderLock<V> {
        OrderLock {
            lock: QrwLock::new(data)
        }
    }

    /// Returns a new `FutureGuard` which will resolve into a a `OrderGuard`.
    pub fn read(self) -> FutureGuard<V, ReadGuard<V>> {
        print_debug(self.id(), "OrderLock::read: Read lock requested");
        let (tx, rx) = oneshot::channel();
        unsafe { self.lock.enqueue_lock_request(QrwRequest::new(tx, RequestKind::Read)); }
        FutureGuard::new(self.into(), rx)
    }

    /// Returns a new `FutureGuard` which will resolve into a a `OrderGuard`.
    pub fn write(self) -> FutureGuard<V, WriteGuard<V>> {
        print_debug(self.id(), "OrderLock::write: Write lock requested");
        let (tx, rx) = oneshot::channel();
        unsafe { self.lock.enqueue_lock_request(QrwRequest::new(tx, RequestKind::Write)); }
        FutureGuard::new(self.into(), rx)
    }

    /// Returns a reference to the inner value.
    ///
    #[inline]
    pub fn as_ptr(&self) -> *const V {
        self.lock.as_ptr()
    }

    /// Returns a mutable reference to the inner value.
    ///
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut V {
        self.lock.as_mut_ptr()
    }

    /// Returns a pointer address to the internal array, usable as a unique
    /// identifier.
    fn id(&self) -> usize {
        self.lock.as_ptr() as usize
    }
}

impl<V> From<QrwLock<V>> for OrderLock<V> {
    fn from(q: QrwLock<V>) -> OrderLock<V> {
        OrderLock { lock: q }
    }
}

impl<V> From<V> for OrderLock<V> {
    fn from(vec: V) -> OrderLock<V> {
        OrderLock { lock: QrwLock::new(vec) }
    }
}

impl<V> Clone for OrderLock<V> {
    #[inline]
    fn clone(&self) -> OrderLock<V> {
        OrderLock {
            lock: self.lock.clone(),
        }
    }
}