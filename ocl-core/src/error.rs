//! Standard error type for ocl.
//!

use std::fmt;
use failure::{Context, Fail, Backtrace};
use util::UtilError;
use functions::{ApiError, VersionLowError};
use ::{Status, EmptyInfoResult, OpenclVersion};


/// Ocl error result type.
pub type Result<T> = ::std::result::Result<T, Error>;


/// An enum one of several error types.
#[derive(Debug, Fail)]
pub enum ErrorKind {
    // Void: An error with no description:
    #[fail(display = "OpenCL Error (void)",)]
    Void,
    // String: An arbitrary error:
    // TODO: Remove eventually.
    #[fail(display = "{}", _0)]
    String(String),
    // FfiNul: Ffi string conversion error:
    #[fail(display = "{}", _0)]
    FfiNul(#[cause] ::std::ffi::NulError),
    // Io: std::io error:
    #[fail(display = "{}", _0)]
    Io(::std::io::Error),
    // FromUtf8Error: String conversion error:
    #[fail(display = "{}", _0)]
    FromUtf8Error(::std::string::FromUtf8Error),
    // IntoStringError: Ffi string conversion error:
    #[fail(display = "{}", _0)]
    IntoStringError(::std::ffi::IntoStringError),
    // EmptyInfoResult:
    #[fail(display = "{}", _0)]
    EmptyInfoResult(EmptyInfoResult),
    // VersionLow:
    // TODO: Move into its own error type.
    #[fail(display = "OpenCL version too low to use this feature \
        (detected: {}, required: {}).", detected, required)]
    VersionLow { detected: OpenclVersion, required: OpenclVersion },
    #[fail(display = "{}", _0)]
    UtilError(UtilError),
    #[fail(display = "{}", _0)]
    ApiError(ApiError),
    #[fail(display = "{}", _0)]
    VersionLowError(VersionLowError),
    // Other(Box<StdError>),
}


/// An Error.
pub struct Error {
    inner: Context<ErrorKind>,
}

impl Error {
    /// Returns a new `Err(ocl_core::ErrorKind::String(...))` variant with the
    /// given description.
    // #[deprecated(since="0.4.0", note="Use `Err(\"...\".into())` instead.")]
    pub fn err_string<T, S: Into<String>>(desc: S) -> self::Result<T> {
        Err(Error { inner: Context::new(ErrorKind::String(desc.into())) })
    }

    // TODO: REMOVE ME
    pub fn string_temporary<S: Into<String>>(desc: S) -> Self {
        Error { inner: Context::new(ErrorKind::String(desc.into())) }
    }

    /// Returns the error status code for `Status` variants.
    pub fn api_status(&self) -> Option<Status> {
        match *self.kind() {
            ErrorKind::ApiError(ref err) => Some(err.status()),
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

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self, f)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
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

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Error {
        Error { inner: Context::new(kind) }
    }
}

impl From<Context<ErrorKind>> for Error {
    fn from(inner: Context<ErrorKind>) -> Error {
        Error { inner: inner }
    }
}

impl From<()> for Error {
    fn from(_: ()) -> Self {
        Error { inner: Context::new(ErrorKind::Void) }
    }
}

impl From<EmptyInfoResult> for Error {
    fn from(err: EmptyInfoResult) -> Self {
        Error { inner: Context::new(ErrorKind::EmptyInfoResult(err)) }
    }
}

impl From<String> for Error {
    fn from(desc: String) -> Self {
        Error { inner: Context::new(ErrorKind::String(desc)) }
    }
}

impl From<Error> for String {
    fn from(err: Error) -> String {
        format!("{}", err)
    }
}

impl<'a> From<&'a str> for Error {
    fn from(desc: &'a str) -> Self {
        Error { inner: Context::new(ErrorKind::String(String::from(desc))) }
    }
}

impl From<::std::ffi::NulError> for Error {
    fn from(err: ::std::ffi::NulError) -> Self {
        Error { inner: Context::new(ErrorKind::FfiNul(err)) }
    }
}

impl From<::std::io::Error> for Error {
    fn from(err: ::std::io::Error) -> Self {
        Error { inner: Context::new(ErrorKind::Io(err)) }
    }
}

impl From<::std::string::FromUtf8Error> for Error {
    fn from(err: ::std::string::FromUtf8Error) -> Self {
        Error { inner: Context::new(ErrorKind::FromUtf8Error(err)) }
    }
}

impl From<::std::ffi::IntoStringError> for Error {
    fn from(err: ::std::ffi::IntoStringError) -> Self {
        Error { inner: Context::new(ErrorKind::IntoStringError(err)) }
    }
}

impl From<UtilError> for Error {
    fn from(err: UtilError) -> Self {
        Error { inner: Context::new(ErrorKind::UtilError(err)) }
    }
}

impl From<ApiError> for Error {
    fn from(err: ApiError) -> Self {
        Error { inner: Context::new(ErrorKind::ApiError(err)) }
    }
}

impl From<VersionLowError> for Error {
    fn from(err: VersionLowError) -> Self {
        Error { inner: Context::new(ErrorKind::VersionLowError(err)) }
    }
}
