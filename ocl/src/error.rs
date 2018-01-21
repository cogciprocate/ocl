//! Standard error type for ocl futures.
//!

use std;
// use std::sync::mpsc::{SendError as StdMpscSendError, RecvError as StdMpscRecvError};
use failure::{Context, Fail, Backtrace};
use futures::sync::oneshot::Canceled as OneshotCanceled;
use futures::sync::mpsc::SendError;
use core::error::{Error as OclCoreError};
use core::Status;
use standard::{DeviceError, PlatformError};
use ::BufferCmdError;

pub type Result<T> = std::result::Result<T, Error>;


/// An enum containing either a `String` or one of several other error types.
///
/// Implements the usual error traits.
#[derive(Debug, Fail)]
pub enum ErrorKind {
    #[fail(display = "{}", _0)]
    OclCore(OclCoreError),
    #[fail(display = "{}", _0)]
    FuturesMpscSend(String),
    // #[fail(display = "{}", _0)]
    // StdMpscSend(String),
    // #[fail(display = "{}", _0)]
    // StdMpscRecv(StdMpscRecvError),
    #[fail(display = "{}", _0)]
    OneshotCanceled(#[cause] OneshotCanceled),
    #[fail(display = "{}", _0)]
    BufferCmd(BufferCmdError),
    #[fail(display = "{}", _0)]
    Device(DeviceError),
    #[fail(display = "{}", _0)]
    Platform(PlatformError)
}


/// An Error.
pub struct Error {
    inner: Context<ErrorKind>,
}

impl Error {
    /// Returns the error status code for `OclCore` variants.
    pub fn api_status(&self) -> Option<Status> {
        match *self.kind() {
            ErrorKind::OclCore(ref err) => err.api_status(),
            _ => None,
        }
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
        Error { inner: Context::new(ErrorKind::FuturesMpscSend(
            format!("{}: '{}'", debug, display))) }
    }
}

// impl<T> From<StdMpscSendError<T>> for Error {
//     fn from(err: StdMpscSendError<T>) -> Error {
//         let debug = format!("{:?}", err);
//         let display = format!("{}", err);
//         Error { inner: Context::new(ErrorKind::StdMpscSend(
//             format!("{}: '{}'", debug, display))) }
//     }
// }

// impl From<StdMpscRecvError> for Error {
//     fn from(err: StdMpscRecvError) -> Error {
//         Error { inner: Context::new(ErrorKind::StdMpscRecv(err)) }
//     }
// }

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
        Error { inner: Context::new(ErrorKind::BufferCmd(err)) }
    }
}

impl From<DeviceError> for Error {
    fn from(err: DeviceError) -> Error {
        Error { inner: Context::new(ErrorKind::Device(err)) }
    }
}

impl From<PlatformError> for Error {
    fn from(err: PlatformError) -> Error {
        Error { inner: Context::new(ErrorKind::Platform(err)) }
    }
}

impl From<Error> for String {
    fn from(err: Error) -> String {
        err.to_string()
    }
}


unsafe impl Send for Error {}
unsafe impl Sync for Error {}