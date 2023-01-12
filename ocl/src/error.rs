//! Standard error type for ocl futures.
//!

use std;
// use std::sync::mpsc::{SendError as StdMpscSendError, RecvError as StdMpscRecvError};
use crate::core::error::Error as OclCoreError;
use crate::core::Status;
use crate::standard::{DeviceError, KernelError, PlatformError};
use futures::sync::mpsc::SendError;
use futures::sync::oneshot::Canceled as OneshotCanceled;

use crate::BufferCmdError;

pub type Result<T> = std::result::Result<T, Error>;

/// An enum containing either a `String` or one of several other error types.
///
/// Implements the usual error traits.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    OclCore(OclCoreError),
    #[error("{0}")]
    FuturesMpscSend(String),
    // #[error("{0}")]
    // StdMpscSend(String),
    // #[error("{0}")]
    // StdMpscRecv(StdMpscRecvError),
    #[error("{0}")]
    OneshotCanceled(#[from] OneshotCanceled),
    #[error("{0}")]
    BufferCmd(BufferCmdError),
    #[error("{0}")]
    Device(DeviceError),
    #[error("{0}")]
    Platform(PlatformError),
    #[error("{0}")]
    Kernel(KernelError),
}

impl Error {
    /// Returns the error status code for `OclCore` variants.
    pub fn api_status(&self) -> Option<Status> {
        match *self {
            Error::OclCore(ref err) => err.api_status(),
            _ => None,
        }
    }
}

impl From<OclCoreError> for Error {
    fn from(err: OclCoreError) -> Error {
        Error::OclCore(err)
    }
}

impl<T> From<SendError<T>> for Error {
    fn from(err: SendError<T>) -> Error {
        let debug = format!("{:?}", err);
        let display = format!("{}", err);
        Error::FuturesMpscSend(format!("{}: '{}'", debug, display))
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

// TODO: Remove eventually
impl From<String> for Error {
    fn from(desc: String) -> Error {
        Error::OclCore(desc.into())
    }
}

// TODO: Remove eventually
impl<'a> From<&'a str> for Error {
    fn from(desc: &'a str) -> Error {
        Error::OclCore(desc.into())
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(err: std::ffi::NulError) -> Error {
        Error::OclCore(err.into())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::OclCore(err.into())
    }
}

impl From<BufferCmdError> for Error {
    fn from(err: BufferCmdError) -> Error {
        Error::BufferCmd(err)
    }
}

impl From<DeviceError> for Error {
    fn from(err: DeviceError) -> Error {
        Error::Device(err)
    }
}

impl From<PlatformError> for Error {
    fn from(err: PlatformError) -> Error {
        Error::Platform(err)
    }
}

impl From<KernelError> for Error {
    fn from(err: KernelError) -> Error {
        Error::Kernel(err)
    }
}

impl From<Error> for String {
    fn from(err: Error) -> String {
        err.to_string()
    }
}

unsafe impl Send for Error {}
unsafe impl Sync for Error {}
