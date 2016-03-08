//! Standard error type for ocl.

use std;
// use std::error::Error;
// use std::io;
// use std::fmt;
use std::convert::Into;
// use std::collections::str::FromUtf8Error;
// use std::ffi;

use cl_h::Status;
/// `ocl::Error` result type.
pub type Result<T> = std::result::Result<T, self::Error>;


/// An enum containing either a `String` or one of several other standard 
/// error types.
///
/// Implements the usual error traits.
///
pub enum Error {
    // description: String,
    Status(Status, String),
    String(String),
    Nul(std::ffi::NulError),
    Io(std::io::Error),
    FromUtf8Error(std::string::FromUtf8Error),
    UnspecifiedDimensions,
}

impl self::Error {
    /// Returns a new `Error` with the description string: `desc`.
    pub fn new<S: Into<String>>(desc: S) -> self::Error {
        self::Error::String(desc.into())
    }

    /// Returns a new `ocl::Result::Err` containing an `ocl::Error` with the 
    /// given description.
    pub fn err<T, S: Into<String>>(desc: S) -> self::Result<T> {
        Err(Error::String(desc.into()))
    }

    /// Returns a new `ocl::Result::Err` containing an `ocl::Error` with the 
    /// given error code and description.
    pub fn status<T, S: Into<String>>(status: Status, desc: S) -> self::Result<T> {
        Err(Error::Status(status, desc.into()))
    }

    /// If this is a `String` variant, concatenate `txt` to the front of the
    /// contained string. Otherwise, do nothing at all.
    pub fn prepend<'s, S: AsRef<&'s str>>(&'s mut self, txt: S) {
        match self {
            &mut Error::String(ref mut string) => {
                string.reserve_exact(txt.as_ref().len());
                let old_string_copy = string.clone();
                string.clear();
                string.push_str(txt.as_ref());
                string.push_str(&old_string_copy);
            },
            _ => (),
        }
    }

    
}

impl std::error::Error for self::Error {
    fn description(&self) -> &str {
        match self {            
            &Error::Nul(ref err) => err.description(),
            &Error::Io(ref err) => err.description(),
            &Error::FromUtf8Error(ref err) => err.description(),
            &Error::Status(_, ref desc) => &desc,
            &Error::String(ref desc) => &desc,
            &Error::UnspecifiedDimensions => "Cannot convert to a valid set of dimensions. \
                Please specify some dimensions.",
            // _ => panic!("OclError::description()"),
        }
    }
}

impl Into<String> for self::Error {
    fn into(self) -> String {
        use std::error::Error;
        self.description().to_string()
    }
}

impl From<String> for self::Error {
    fn from(desc: String) -> self::Error {
        self::Error::new(desc)
    }
}

impl<'a> From<&'a str> for self::Error {
    fn from(desc: &'a str) -> self::Error {
        self::Error::new(String::from(desc))
    }
}

impl From<std::ffi::NulError> for self::Error {
    fn from(err: std::ffi::NulError) -> self::Error {
        self::Error::Nul(err)
    }
}

impl From<std::io::Error> for self::Error {
    fn from(err: std::io::Error) -> self::Error {
        self::Error::Io(err)
    }
}

impl From<std::string::FromUtf8Error> for self::Error {
    fn from(err: std::string::FromUtf8Error) -> self::Error {
        self::Error::FromUtf8Error(err)
    }
}

impl std::fmt::Display for self::Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use std::error::Error;
        f.write_str(&self.description())
    }
}

impl std::fmt::Debug for self::Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use std::error::Error;
        f.write_str(&self.description())
    }
}
