//! A simple way to specify the sizes of up to three dimensions.
use super::{BufferDims, OclError, WorkSize};

/// A simple implementation of a type specifying the sizes of up to three
/// dimensions. 
///
/// Custom types implementing `BufferDims` can and should be created
/// to express more complex relationships between buffer and work size.
///
///	[FIXME] TODO: Much more explaination needed as soon as conventions solidify.
/// [UNSTABLE]: MAY BE CONSOLIDATED WITH `WorkSize`.
pub enum SimpleDims {
	Unspecified,
	OneDim		(usize),
	TwoDims		(usize, usize),
	ThreeDims 	(usize, usize, usize),
}

impl SimpleDims {
	/// Returns a new `SimpleDims`.
	///
	/// Dimensions must be specified in order from d0 -> d1 -> d2; i.e. `d1` 
	/// cannot be `Some(x)` if `d0` is `None`.
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

	/// Returns a `WorkSize` corresponding to the dimensions of this `SimpleDims`.
	pub fn work_size(&self) -> WorkSize {
		match self {
			&SimpleDims::ThreeDims(d0, d1, d2) => WorkSize::ThreeDims(d0, d1, d2),
			&SimpleDims::TwoDims(d0, d1) => WorkSize::TwoDims(d0, d1),
			&SimpleDims::OneDim(d0) => WorkSize::OneDim(d0),
			_ => WorkSize::Unspecified,
		}
	}
}

impl BufferDims for SimpleDims {
	fn padded_buffer_len(&self, incr: usize) -> usize {
		let simple_len = match self {
			&SimpleDims::ThreeDims(d0, d1, d2) => d0 * d1 * d2,
			&SimpleDims::TwoDims(d0, d1) => d0 * d1,
			&SimpleDims::OneDim(d0) => d0,
			_ => 0,
		};

		super::padded_len(simple_len, incr)
	}
}
