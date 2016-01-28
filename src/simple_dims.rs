use super::{EnvoyDims, OclError, WorkSize};

pub enum SimpleDims {
	Unspecified,
	OneDim		(usize),
	TwoDims		(usize, usize),
	ThreeDims 	(usize, usize, usize),
}

impl SimpleDims {
	pub fn new(d0: Option<usize>, d1: Option<usize>, d2: Option<usize>) -> Result<SimpleDims, OclError> {
		let std_err_msg = "Dimensions must be defined from left to right. If you define the 2nd \
			dimension, you must also define the 1st, etc.";

		if d2.is_some() { 
			if d1.is_some() && d0.is_some() {
				Ok(SimpleDims::ThreeDims(d0.unwrap(), d1.unwrap(), d2.unwrap()))
			} else {
				Err(OclError::new(std_err_msg))
			}
		} else if d1.is_some() {
			if d0.is_some() {
				Ok(SimpleDims::TwoDims(d1.unwrap(), d0.unwrap()))
			} else {
				Err(OclError::new(std_err_msg))
			}
		} else if d0.is_some() {
			Ok(SimpleDims::OneDim(d0.unwrap()))
		} else {
			Ok(SimpleDims::Unspecified)
		}
	}

	pub fn work_size(&self) -> WorkSize {
		match self {
			&SimpleDims::ThreeDims(d0, d1, d2) => WorkSize::ThreeDims(d0, d1, d2),
			&SimpleDims::TwoDims(d0, d1) => WorkSize::TwoDims(d0, d1),
			&SimpleDims::OneDim(d0) => WorkSize::OneDim(d0),
			_ => WorkSize::Unspecified,
		}
	}
}

impl EnvoyDims for SimpleDims {
	fn padded_envoy_len(&self, incr: usize) -> usize {
		let simple_len = match self {
			&SimpleDims::ThreeDims(d0, d1, d2) => d0 * d1 * d2,
			&SimpleDims::TwoDims(d0, d1) => d0 * d1,
			&SimpleDims::OneDim(d0) => d0,
			_ => 0,
		};

		super::padded_len(simple_len, incr)
	}
}
