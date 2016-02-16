//! Standard error type for ocl.

use std;
// use std::error::Error;
// use std::io;
// use std::fmt;
use std::convert::Into;
// use std::ffi;

/// `ocl::Error` result type.
pub type Result<T> = std::result::Result<T, Error>;


/// An enum containing either a `String` or one of several other standard 
/// error types.
///
/// Implements the usual error traits.
///
pub enum Error {
    // description: String,
    ErrCode(i32, String),
    String(String),
    Nul(std::ffi::NulError),
    Io(std::io::Error),
}

impl Error {
    /// Returns a new `Error` with the description string: `desc`.
    pub fn new<S: Into<String>>(desc: S) -> Error {
        Error::String(desc.into())
    }

    /// Returns a new `ocl::Result::Err` containing an `ocl::Error` with the 
    /// given description.
    pub fn err<T, S: Into<String>>(desc: S) -> self::Result<T> {
        Err(Error::String(desc.into()))
    }

    /// Returns a new `ocl::Result::Err` containing an `ocl::Error` with the 
    /// given error code and description.
    pub fn errcode<T, S: Into<String>>(code: i32, desc: S) -> self::Result<T> {
        Err(Error::ErrCode(code, desc.into()))
    }
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        match self {
            &Error::String(ref desc) => &desc,
            &Error::Nul(ref err) => err.description(),
            &Error::Io(ref err) => err.description(),
            &Error::ErrCode(_, ref desc) => &desc,
        }
    }
}

impl From<String> for Error {
    fn from(desc: String) -> Error {
        Error::new(desc)
    }
}

impl<'a> From<&'a str> for Error {
    fn from(desc: &'a str) -> Error {
        Error::new(String::from(desc))
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(err: std::ffi::NulError) -> Error {
        Error::Nul(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Error {
        Error::Io(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use std::error::Error;
        f.write_str(&self.description())
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use std::error::Error;
        f.write_str(&self.description())
    }
}
