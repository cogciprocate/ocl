//! Standard error type for ocl.
//!

use std::fmt;
use failure::{Context, Fail, Backtrace};
use util::UtilError;
use functions::{ApiError, VersionLowError, ProgramBuildError};
use ::{Status, EmptyInfoResultError};


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
    FromUtf8(::std::string::FromUtf8Error),
    // IntoStringError: Ffi string conversion error:
    #[fail(display = "{}", _0)]
    IntoString(::std::ffi::IntoStringError),
    // EmptyInfoResultError:
    #[fail(display = "{}", _0)]
    EmptyInfoResult(EmptyInfoResultError),
    // UtilError:
    #[fail(display = "{}", _0)]
    Util(UtilError),
    // ApiError:
    #[fail(display = "{}", _0)]
    Api(ApiError),
    // VersionLow:
    #[fail(display = "{}", _0)]
    VersionLow(VersionLowError),
    // ProgramBuild:
    #[fail(display = "{}", _0)]
    ProgramBuild(ProgramBuildError),
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
            ErrorKind::Api(ref err) => Some(err.status()),
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

impl From<EmptyInfoResultError> for Error {
    fn from(err: EmptyInfoResultError) -> Self {
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
        Error { inner: Context::new(ErrorKind::FromUtf8(err)) }
    }
}

impl From<::std::ffi::IntoStringError> for Error {
    fn from(err: ::std::ffi::IntoStringError) -> Self {
        Error { inner: Context::new(ErrorKind::IntoString(err)) }
    }
}

impl From<UtilError> for Error {
    fn from(err: UtilError) -> Self {
        Error { inner: Context::new(ErrorKind::Util(err)) }
    }
}

impl From<ApiError> for Error {
    fn from(err: ApiError) -> Self {
        Error { inner: Context::new(ErrorKind::Api(err)) }
    }
}

impl From<VersionLowError> for Error {
    fn from(err: VersionLowError) -> Self {
        Error { inner: Context::new(ErrorKind::VersionLow(err)) }
    }
}

impl From<ProgramBuildError> for Error {
    fn from(err: ProgramBuildError) -> Self {
        Error { inner: Context::new(ErrorKind::ProgramBuild(err)) }
    }
}
