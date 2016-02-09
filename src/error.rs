//! Standard error type for ocl.

use std::error::{ self, Error };
use std::io;
use std::fmt;
use std::convert::Into;
use std::ffi;

/// `OclError` result type.
pub type OclResult<T> = Result<T, OclError>;


/// Error type containing a string.
pub enum OclError {
	// description: String,
	String(String),
	Nul(ffi::NulError),
	Io(io::Error),
}

impl OclError {
	/// Returns a new `OclError` with the description string: `desc`.
	pub fn new<S: Into<String>>(desc: S) -> OclError {
		OclError::String(desc.into())
	}

	/// Returns a new `OclResult::Err` containing an `OclResult` with the given 
	/// description.
	pub fn err<T, S: Into<String>>(desc: S) -> OclResult<T> {
		Err(OclError::new(desc))
	}
}

impl error::Error for OclError {
	fn description(&self) -> &str {
		match self {
			&OclError::String(ref desc) => &desc,
			&OclError::Nul(ref err) => err.description(),
			&OclError::Io(ref err) => err.description(),
		}
	}
}

impl From<String> for OclError {
	fn from(desc: String) -> OclError {
		OclError::new(desc)
	}
}

impl<'a> From<&'a str> for OclError {
	fn from(desc: &'a str) -> OclError {
		OclError::new(String::from(desc))
	}
}

impl From<ffi::NulError> for OclError {
	fn from(err: ffi::NulError) -> OclError {
		OclError::Nul(err)
	}
}

impl From<io::Error> for OclError {
	fn from(err: io::Error) -> OclError {
		OclError::Io(err)
	}
}

impl fmt::Display for OclError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str(&self.description())
	}
}

impl fmt::Debug for OclError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str(&self.description())
	}
}
