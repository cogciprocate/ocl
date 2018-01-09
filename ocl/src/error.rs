//! Standard error type for ocl futures.
//!

use std;
use failure::{Context, Fail, Backtrace};
use futures::sync::oneshot::Canceled as OneshotCanceled;
use futures::sync::mpsc::SendError;
use core::error::{Error as OclCoreError};
use ::BufferCmdError;

pub type Result<T> = std::result::Result<T, Error>;


/// An enum containing either a `String` or one of several other error types.
///
/// Implements the usual error traits.
#[derive(Debug, Fail)]
pub enum ErrorKind {
    #[fail(display = "ocl-core error: {}", _0)]
    OclCore(OclCoreError),
    #[fail(display = "mpsc send error: {}", _0)]
    MpscSendError(String),
    #[fail(display = "{}", _0)]
    OneshotCanceled(#[cause] OneshotCanceled),
    #[fail(display = "BufferCmd error: {}", _0)]
    BufferCmdError(BufferCmdError),
}


/// An Error.
pub struct Error {
    inner: Context<ErrorKind>,
}

impl Error {
    /// Returns a new `Error::String` with the given description.
    #[deprecated]
    pub fn string<S: Into<String>>(desc: S) -> Error {
        Error { inner: Context::new(ErrorKind::OclCore(OclCoreError::from(desc.into()))) }
    }

    /// Returns the error variant and contents.
    pub fn kind(&self) -> &ErrorKind {
        self.inner.get_context()
    }

    /// Returns the immediate cause of this error (e.g. the next error in the
    /// chain).
    pub fn cause(&self) -> Option<&Fail> {
        self.inner.cause()
    }
}

impl Fail for Error {
    fn cause(&self) -> Option<&Fail> {
        self.inner.cause()
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        self.inner.backtrace()
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

impl From<OclCoreError> for Error {
    fn from(err: OclCoreError) -> Error {
        Error { inner: Context::new(ErrorKind::OclCore(err)) }
    }
}

impl<T> From<SendError<T>> for Error {
    fn from(err: SendError<T>) -> Error {
        let debug = format!("{:?}", err);
        let display = format!("{}", err);
        Error { inner: Context::new(ErrorKind::MpscSendError(format!("{}: '{}'", debug, display))) }
    }
}

impl From<OneshotCanceled> for Error {
    fn from(err: OneshotCanceled) -> Error {
        Error { inner: Context::new(ErrorKind::OneshotCanceled(err)) }
    }
}

// TODO: Remove eventually
impl From<String> for Error {
    fn from(desc: String) -> Error {
        Error { inner: Context::new(ErrorKind::OclCore(desc.into())) }
    }
}

// TODO: Remove eventually
impl<'a> From<&'a str> for Error {
    fn from(desc: &'a str) -> Error {
        Error { inner: Context::new(ErrorKind::OclCore(desc.into())) }
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(err: std::ffi::NulError) -> Error {
        Error { inner: Context::new(ErrorKind::OclCore(err.into())) }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error { inner: Context::new(ErrorKind::OclCore(err.into())) }
    }
}

impl From<BufferCmdError> for Error {
    fn from(err: BufferCmdError) -> Error {
        Error { inner: Context::new(ErrorKind::BufferCmdError(err)) }
    }
}

impl From<Error> for String {
    fn from(err: Error) -> String {
        err.to_string()
    }
}


unsafe impl Send for Error {}
unsafe impl Sync for Error {}