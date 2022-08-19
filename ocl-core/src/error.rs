//! Standard error type for ocl.
//!

use crate::util::UtilError;
use crate::functions::{ApiError, VersionLowError, ProgramBuildError, ApiWrapperError};
use crate::{Status, EmptyInfoResultError};


/// Ocl error result type.
pub type Result<T> = ::std::result::Result<T, Error>;


/// An enum one of several error types.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // String: An arbitrary error:
    //
    // TODO: Remove this eventually. We need to replace every usage
    // (conversion from String/str) with a dedicated error type/variant for
    // each. In the meanwhile, refrain from creating new instances of this by
    // converting strings to `Error`!
    #[error("{0}")]
    String(String),
    // FfiNul: Ffi string conversion error:
    #[error("{0}")]
    FfiNul(#[from] ::std::ffi::NulError),
    // Io: std::io error:
    #[error("{0}")]
    Io(#[from] ::std::io::Error),
    // FromUtf8: String conversion error:
    #[error("{0}")]
    FromUtf8(#[from] ::std::string::FromUtf8Error),
    // IntoString: Ffi string conversion error:
    #[error("{0}")]
    IntoString(#[from] ::std::ffi::IntoStringError),
    // EmptyInfoResult:
    #[error("{0}")]
    EmptyInfoResult(EmptyInfoResultError),
    // Util:
    #[error("{0}")]
    Util(UtilError),
    // Api:
    #[error("{0}")]
    Api(ApiError),
    // VersionLow:
    #[error("{0}")]
    VersionLow(VersionLowError),
    // ProgramBuild:
    #[error("{0}")]
    ProgramBuild(ProgramBuildError),
    // ApiWrapper:
    #[error("{0}")]
    ApiWrapper(ApiWrapperError),
}

impl Error {
    /// Returns the error status code for `Status` variants.
    pub fn api_status(&self) -> Option<Status> {
        match *self {
            Error::Api(ref err) => Some(err.status()),
            _ => None,
        }
    }
}

// TODO: Remove eventually
impl<'a> From<&'a str> for Error {
    fn from(desc: &'a str) -> Self {
        Error::String(String::from(desc))
    }
}

// TODO: Remove eventually
impl From<String> for Error {
    fn from(desc: String) -> Self {
        Error::String(desc)
    }
}

impl From<UtilError> for Error {
    fn from(err: UtilError) -> Self {
        Error::Util(err)
    }
}

impl From<ApiError> for Error {
    fn from(err: ApiError) -> Self {
        Error::Api(err)
    }
}

impl From<VersionLowError> for Error {
    fn from(err: VersionLowError) -> Self {
        Error::VersionLow(err)
    }
}

impl From<ProgramBuildError> for Error {
    fn from(err: ProgramBuildError) -> Self {
        Error::ProgramBuild(err)
    }
}

impl From<ApiWrapperError> for Error {
    fn from(err: ApiWrapperError) -> Self {
        Error::ApiWrapper(err)
    }
}
