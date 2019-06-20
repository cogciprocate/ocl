//! Standard error type for ocl.
//!

use std::fmt;
use failure::{Context, Fail, Backtrace};
use crate::util::UtilError;
use crate::functions::{ApiError, VersionLowError, ProgramBuildError, ApiWrapperError};
use crate::{Status, EmptyInfoResultError};


/// Ocl error result type.
pub type Result<T> = ::std::result::Result<T, Error>;


/// An enum one of several error types.
#[derive(Debug, Fail)]
pub enum ErrorKind {
    // String: An arbitrary error:
    //
    // TODO: Remove this eventually. We need to replace every usage
    // (conversion from String/str) with a dedicated error type/variant for
    // each. In the meanwhile, refrain from creating new instances of this by
    // converting strings to `Error`!
    #[fail(display = "{}", _0)]
    String(String),
    // FfiNul: Ffi string conversion error:
    #[fail(display = "{}", _0)]
    FfiNul(#[cause] ::std::ffi::NulError),
    // Io: std::io error:
    #[fail(display = "{}", _0)]
    Io(#[cause] ::std::io::Error),
    // FromUtf8: String conversion error:
    #[fail(display = "{}", _0)]
    FromUtf8(#[cause] ::std::string::FromUtf8Error),
    // IntoString: Ffi string conversion error:
    #[fail(display = "{}", _0)]
    IntoString(#[cause] ::std::ffi::IntoStringError),
    // EmptyInfoResult:
    #[fail(display = "{}", _0)]
    EmptyInfoResult(EmptyInfoResultError),
    // Util:
    #[fail(display = "{}", _0)]
    Util(UtilError),
    // Api:
    #[fail(display = "{}", _0)]
    Api(ApiError),
    // VersionLow:
    #[fail(display = "{}", _0)]
    VersionLow(VersionLowError),
    // ProgramBuild:
    #[fail(display = "{}", _0)]
    ProgramBuild(ProgramBuildError),
    // ApiWrapper:
    #[fail(display = "{}", _0)]
    ApiWrapper(ApiWrapperError),
}


/// An Error.
pub struct Error {
    inner: Context<ErrorKind>,
}

impl Error {
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
    pub fn cause(&self) -> Option<&dyn Fail> {
        self.inner.cause()
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl Fail for Error {
    fn cause(&self) -> Option<&dyn Fail> {
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
        Error { inner }
    }
}

impl From<EmptyInfoResultError> for Error {
    fn from(err: EmptyInfoResultError) -> Self {
        Error { inner: Context::new(ErrorKind::EmptyInfoResult(err)) }
    }
}

// TODO: Remove eventually
impl<'a> From<&'a str> for Error {
    fn from(desc: &'a str) -> Self {
        Error { inner: Context::new(ErrorKind::String(String::from(desc))) }
    }
}

// TODO: Remove eventually
impl From<String> for Error {
    fn from(desc: String) -> Self {
        Error { inner: Context::new(ErrorKind::String(desc)) }
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

impl From<ApiWrapperError> for Error {
    fn from(err: ApiWrapperError) -> Self {
        Error { inner: Context::new(ErrorKind::ApiWrapper(err)) }
    }
}
