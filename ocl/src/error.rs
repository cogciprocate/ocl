//! Standard error type for ocl futures.
//!

use std;
use futures::sync::oneshot::Canceled as OneshotCanceled;
use futures::sync::mpsc::SendError;
use core::error::{Error as OclCoreError};
use ::BufferCmdError;

pub type Result<T> = std::result::Result<T, self::Error>;



/// An enum containing either a `String` or one of several other error types.
///
/// Implements the usual error traits.
#[derive(Fail)]
pub enum Error {
    #[fail(display = "ocl-core error: {}", _0)]
    Ocl(OclCoreError),
    #[fail(display = "mpsc send error: {}", _0)]
    MpscSendError(String),
    #[fail(display = "{}", _0)]
    OneshotCanceled(OneshotCanceled),
    #[fail(display = "BufferCmd error: {}", _0)]
    BufferCmdError(BufferCmdError),
    #[fail(display = "other error: {}", _0)]
    Other(Box<std::error::Error>),
}

impl self::Error {
    /// Returns a new `Error::String` with the given description.
    pub fn string<S: Into<String>>(desc: S) -> self::Error {
        self::Error::Ocl(OclCoreError::from(desc.into()))
    }
}

impl From<OclCoreError> for self::Error {
    fn from(err: OclCoreError) -> self::Error {
        Error::Ocl(err)
    }
}

impl From<self::Error> for OclCoreError {
    fn from(err: self::Error) -> OclCoreError {
        match err {
            Error::Ocl(err) => err,
            _ => format!("{}", err).into(),
        }
    }
}

impl<T> From<SendError<T>> for self::Error {
    fn from(err: SendError<T>) -> self::Error {
        let debug = format!("{:?}", err);
        let display = format!("{}", err);
        Error::MpscSendError(format!("{}: '{}'", debug, display))
    }
}

impl From<OneshotCanceled> for self::Error {
    fn from(err: OneshotCanceled) -> self::Error {
        Error::OneshotCanceled(err)
    }
}

impl From<Box<std::error::Error>> for self::Error {
    fn from(err: Box<std::error::Error>) -> self::Error {
        self::Error::Other(err)
    }
}

impl From<String> for self::Error {
    fn from(desc: String) -> self::Error {
        self::Error::string(desc)
    }
}

impl<'a> From<&'a str> for self::Error {
    fn from(desc: &'a str) -> self::Error {
        self::Error::string(String::from(desc))
    }
}

impl From<std::ffi::NulError> for self::Error {
    fn from(err: std::ffi::NulError) -> self::Error {
        self::Error::Ocl(err.into())
    }
}

impl From<std::io::Error> for self::Error {
    fn from(err: std::io::Error) -> self::Error {
        self::Error::Ocl(err.into())
    }
}

impl From<BufferCmdError> for self::Error {
    fn from(err: BufferCmdError) -> self::Error {
        self::Error::BufferCmdError(err)
    }
}

impl From<self::Error> for String {
    fn from(err: self::Error) -> String {
        err.to_string()
    }
}

// impl std::fmt::Display for self::Error {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         use std::error::Error;
//         f.write_str(self.description())
//     }
// }

impl std::fmt::Debug for self::Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // use std::error::Error;
        // f.write_str(self)
        write!(f, "{}", self)
    }
}

unsafe impl std::marker::Send for self::Error {}
unsafe impl std::marker::Sync for self::Error {}