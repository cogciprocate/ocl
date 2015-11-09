use std::error;
use std::fmt;

#[derive(Debug)]
pub struct DimError {
	description: &'static str,
}

impl DimError {
	pub fn new(description: &'static str) -> DimError {
		DimError { description: description }
	}
}


impl fmt::Display for DimError {
	fn fmt(&self, fmtr: &mut fmt::Formatter) -> fmt::Result {
		write!(fmtr, "Ocl dimensional error: {}", self.description)
	}
}

impl error::Error for DimError {
	fn description(&self) -> &str {
		self.description
	}

	fn cause(&self) -> Option<&error::Error> {
		None
	}
}
