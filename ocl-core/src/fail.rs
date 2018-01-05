#![allow(dead_code, unused_variables, unused_imports)]

use std::fmt;
use std::error::Error as StdError;
use failure::{Context, Fail, Backtrace};
use num::FromPrimitive;
use ::{Status, EmptyInfoResult, OpenclVersion};

#[derive(Debug, Fail)]
enum ErrorKind {
    // Void: An error with no description:
    #[fail(display = "OpenCL Error (void)",)]
    Void,
    // Conversion:
    #[fail(display = "Conversion failure")]
    Conversion,
    // Status: OpenCL Function Call Error:
    #[fail(display = "{}", _0)]
    Status {
        status: Status, status_string: String, fn_name: &'static str, fn_info: String, desc: String
    },
    // String: An arbitrary error:
    // TODO: Remove eventually.
    #[fail(display = "{}", _0)]
    String(String),
    // FfiNul: Ffi string conversion error:
    #[fail(display = "{}", _0)]
    FfiNul(#[cause] ::std::ffi::NulError),
    // Io: std::io error:
    #[fail(display = "{}", _0)]
    Io(#[cause] ::std::io::Error),
    // FromUtf8Error: String conversion error:
    #[fail(display = "{}", _0)]
    FromUtf8Error(#[cause] ::std::string::FromUtf8Error),
    // UnspecifiedDimensions:
    #[fail(display = "Cannot convert to a valid set of dimensions. \
        Please specify some dimensions.")]
    UnspecifiedDimensions,
    // IntoStringError: Ffi string conversion error:
    #[fail(display = "{}", _0)]
    IntoStringError(#[cause] ::std::ffi::IntoStringError),
    // EmptyInfoResult:
    #[fail(display = "{}", _0)]
    EmptyInfoResult(EmptyInfoResult),
    // VersionLow:
    #[fail(display = "OpenCL version too low to use this feature \
        (detected: {}, required: {}).", detected, required)]
    VersionLow { detected: OpenclVersion, required: OpenclVersion },
    // #[fail(display = "{}", _0)]
    // Other(Box<StdError>),
}

// #[derive(Debug)]
// struct Error {
//     inner: Context<ErrorKind>,
// }

// impl Error {
//     pub fn kind(&self) -> ErrorKind {
//         *self.inner.get_context()
//     }
// }

// impl Fail for Error {
//     fn cause(&self) -> Option<&Fail> {
//         self.inner.cause()
//     }

//     fn backtrace(&self) -> Option<&Backtrace> {
//         self.inner.backtrace()
//     }
// }

// impl fmt::Display for Error {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         fmt::Display::fmt(&self.inner, f)
//     }
// }

// impl From<ErrorKind> for Error {
//     fn from(kind: ErrorKind) -> Error {
//         Error { inner: Context::new(kind) }
//     }
// }

// impl From<Context<ErrorKind>> for Error {
//     fn from(inner: Context<ErrorKind>) -> Error {
//         Error { inner: inner }
//     }
// }