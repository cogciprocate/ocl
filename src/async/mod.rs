mod error;
mod future_mem_map;

use std;
// use futures::future::Result;
use futures::future;

#[cfg(feature = "experimental_async_rw")]
use futures::{task, Future, Poll, Async};
// pub use futures::future::{result, ok, err};
pub use self::error::{Error, Result};
pub use self::future_mem_map::{FutureMemMap};


pub type FutureResult<T> = future::FutureResult<T, self::Error>;



// [TODO]: Implement this:
//
// pub struct EventListTrigger {
//     wait_events: EventList,
//     completion_event: UserEvent,
//     callback_is_set: bool,
// }

// pub struct EventTrigger {
//     wait_event: Event,
//     completion_event: UserEvent,
//     callback_is_set: bool,
// }

// impl EventTrigger {
//     pub fn new(wait_event: Event, completion_event: UserEvent) -> EventTrigger {
//         EventTrigger {
//             wait_event: wait_event,
//             completion_event: completion_event ,
//             callback_is_set: false,
//         }
//     }
// }




#[allow(dead_code)]
#[cfg(feature = "experimental_async_rw")]
pub struct ReadCompletion<'d, T> where T: 'd {
    event: Event,
    data: &'d mut [T],
}

#[allow(dead_code)]
#[cfg(feature = "experimental_async_rw")]
impl<'d, T> ReadCompletion<'d, T> where T: 'd + OclPrm {
    pub fn new(event: Event, data: &'d mut [T]) -> ReadCompletion<'d, T> {
        ReadCompletion {
            event: event,
            data: data,
        }
    }
}

/// Non-blocking, proper implementation.
#[cfg(feature = "event_callbacks")]
#[cfg(feature = "experimental_async_rw")]
impl<'d, T> Future for ReadCompletion<'d, T> where T: 'd + OclPrm {
    type Item = ();
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        match self.event.is_complete() {
            Ok(true) => {
                Ok(Async::Ready(()))
            }
            Ok(false) => {
                let task_ptr = box_raw_void(task::park());
                unsafe { self.event.set_callback(Some(_unpark_task), task_ptr)?; };
                Ok(Async::NotReady)
            },
            Err(err) => Err(err),
        }
    }
}

/// Blocking implementation (yuk).
#[cfg(not(feature = "event_callbacks"))]
#[cfg(feature = "experimental_async_rw")]
impl<'d, T> Future for ReadCompletion<'d, T> {
    type Item = &'d mut [T];
    type Error = OclError;

    fn poll(&mut self) -> Poll<Self::Item, Self::Error> {
        self.event.wait_for()?;
        Ok(Async::Ready(()))
    }
}


/// Creates a new "leaf future" which will resolve with the given result.
///
/// The returned future represents a computation which is finshed immediately.
/// This can be useful with the `finished` and `failed` base future types to
/// convert an immediate value to a future to interoperate elsewhere.
///
/// # Examples
///
/// ```
/// use ocl::async::result;
///
/// let future_of_1 = result::<u32>(Ok(1));
/// let future_of_err_2 = result::<u32>(Err("2".into()));
/// ```
///
//
// Shamelessly stolen from `https://github.com/alexcrichton/futures-rs`.
//
pub fn result<T>(r: std::result::Result<T, self::Error>) -> self::FutureResult<T> {
    future::result(r)
}

/// Creates a "leaf future" from an immediate value of a finished and
/// successful computation.
///
/// The returned future is similar to `done` where it will immediately run a
/// scheduled callback with the provided value.
///
/// # Examples
///
/// ```
/// use ocl::async::ok;
///
/// let future_of_1 = ok::<u32>(1);
/// ```
//
// Shamelessly stolen from `https://github.com/alexcrichton/futures-rs`.
//
pub fn ok<T>(t: T) -> self::FutureResult<T> {
    result(Ok(t))
}

/// Creates a "leaf future" from an immediate value of a failed computation.
///
/// The returned future is similar to `done` where it will immediately run a
/// scheduled callback with the provided value.
///
/// # Examples
///
/// ```
/// use ocl::async::err;
///
/// let future_of_err_1 = err::<u32>("1".into());
/// ```
//
// Shamelessly stolen from `https://github.com/alexcrichton/futures-rs`.
//
pub fn err<T>(e: self::Error) -> self::FutureResult<T> {
    result(Err(e))
}