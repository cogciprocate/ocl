//! Standard error type for ocl futures.
//!

use std;
use futures::sync::mpsc::SendError;
use core::error::Error as OclError;

pub type Result<T> = std::result::Result<T, self::Error>;

/// An enum containing either a `String` or one of several other error types.
///
/// Implements the usual error traits.
pub enum Error {
    Ocl(OclError),
    FuturesSendError(String),
}

impl self::Error {
    /// Returns a new `Error::String` with the given description.
    pub fn string<S: Into<String>>(desc: S) -> self::Error {
        self::Error::Ocl(OclError::String(desc.into()))
    }

    /// If this is a `String` variant, concatenate `txt` to the front of the
    /// contained string. Otherwise, do nothing at all.
    pub fn prepend<'s, S: AsRef<&'s str>>(&'s mut self, txt: S) {
        if let &mut Error::Ocl(OclError::String(ref mut string)) = self {
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
            Error::FuturesSendError(ref err) => err,

        }
    }
}

impl From<OclError> for self::Error {
    fn from(err: OclError) -> self::Error {
        Error::Ocl(err)
    }
}

impl<T> From<SendError<T>> for self::Error where T: std::fmt::Debug {
    fn from(err: SendError<T>) -> self::Error {
        let debug = format!("{:?}", err);
        let display = format!("{}", err);
        let msg = err.into_inner();
        Error::FuturesSendError(format!("{:?}: '{}' (msg: '{:?}')", debug, display, msg))
    }
}

impl From<()> for self::Error {
    fn from(_: ()) -> self::Error {
        self::Error::Ocl(OclError::Void)
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
        self::Error::Ocl(OclError::Nul(err))
    }
}

impl From<std::io::Error> for self::Error {
    fn from(err: std::io::Error) -> self::Error {
        self::Error::Ocl(OclError::Io(err))
    }
}

impl Into<String> for self::Error {
    fn into(self) -> String {
        use std::error::Error;
        self.description().to_string()
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
