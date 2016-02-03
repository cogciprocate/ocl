//! Standard error type for ocl.

use std::error;
use std::fmt;
use std::convert::Into;

/// `OclError` result type.
pub type OclResult<T> = Result<T, OclError>;

// #[derive(Debug)]
// pub struct DimOclError {
// 	description: &'static str,
// }

// impl DimOclError {
// 	pub fn new(description: &'static str) -> DimOclError {
// 		DimOclError { description: description }
// 	}
// }


// impl fmt::Display for DimOclError {
// 	fn fmt(&self, fmtr: &mut fmt::Formatter) -> fmt::Result {
// 		write!(fmtr, "Ocl dimensional error: {}", self.description)
// 	}
// }

// impl error::OclError for DimOclError {
// 	fn description(&self) -> &str {
// 		self.description
// 	}

// 	fn cause(&self) -> Option<&error::OclError> {
// 		None
// 	}
// }


/// Error type containing a string.
pub struct OclError {
	description: String,
}

impl OclError {
    /// Returns a new `OclError` with the description string: `desc`.
	pub fn new<S: Into<String>>(desc: S) -> OclError {
		OclError { description: desc.into() }
	}
}

impl error::Error for OclError {
    fn description(&self) -> &str {
        &self.description
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

impl fmt::Display for OclError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.description)
    }
}

impl fmt::Debug for OclError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.description)
    }
}
