//! Standard error type for ocl futures.
//!

use std;
// use futures::Canceled as FuturesCanceled;
use futures::sync::oneshot::Canceled as OneshotCanceled;
use futures::sync::mpsc::SendError;
use core::error::{Error as OclError, ErrorKind as OclErrorKind};
use ::BufferCmdError;

pub type Result<T> = std::result::Result<T, self::Error>;

/// An enum containing either a `String` or one of several other error types.
///
/// Implements the usual error traits.
pub enum Error {
    Ocl(OclError),
    MpscSendError(String),
    OneshotCanceled(OneshotCanceled),
    // FuturesCanceled(FuturesCanceled),
    BufferCmdError(BufferCmdError),
    Other(Box<std::error::Error>),
}

impl self::Error {
    /// Returns a new `Error::String` with the given description.
    pub fn string<S: Into<String>>(desc: S) -> self::Error {
        self::Error::Ocl(OclError::from(desc.into()))
    }

    /// If this is a `String` variant, concatenate `txt` to the front of the
    /// contained string. Otherwise, do nothing at all.
    pub fn prepend<'s, S: AsRef<&'s str>>(&'s mut self, txt: S) {
        if let &mut Error::Ocl(OclError { kind: OclErrorKind::String(ref mut string), ..}) = self {
            string.reserve_exact(txt.as_ref().len());
            let old_string_copy = string.clone();
            string.clear();
            string.push_str(txt.as_ref());
            string.push_str(&old_string_copy);
        }
    }
}

impl std::error::Error for self::Error {
    fn description(&self) -> &str {
        match *self {
            Error::Ocl(ref err) => err.description(),
            Error::MpscSendError(ref err) => err,
            Error::OneshotCanceled(ref err) => err.description(),
            // Error::FuturesCanceled(ref err) => err.description(),
            Error::BufferCmdError(ref err) => err.description(),
            Error::Other(ref err) => err.description(),
        }
    }
}

impl From<OclError> for self::Error {
    fn from(err: OclError) -> self::Error {
        Error::Ocl(err)
    }
}

impl From<self::Error> for OclError {
    fn from(err: self::Error) -> OclError {
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

// impl From<FuturesCanceled> for self::Error {
//     fn from(err: FuturesCanceled) -> self::Error {
//         Error::FuturesCanceled(err)
//     }
// }

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

impl From<()> for self::Error {
    fn from(_: ()) -> self::Error {
        self::Error::Ocl(().into())
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

impl std::fmt::Display for self::Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use std::error::Error;
        f.write_str(self.description())
    }
}

impl std::fmt::Debug for self::Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use std::error::Error;
        f.write_str(self.description())
    }
}

unsafe impl std::marker::Send for self::Error {}
unsafe impl std::marker::Sync for self::Error {}