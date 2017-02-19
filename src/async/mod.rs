mod error;
mod mapped_mem;

use std;
// use futures::future::Result;
use futures::future;

// pub use futures::future::{result, ok, err};
pub use self::error::{Error, Result};
pub use self::mapped_mem::{FutureMemMap, MemMap};

pub type FutureResult<T> = future::FutureResult<T, self::Error>;


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