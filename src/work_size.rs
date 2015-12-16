use std::ptr;
use libc::size_t;

use cl_h::cl_uint;

/// Defines the amount of work to be done by a kernel for each of up to three 
/// dimensions.
#[derive(PartialEq, Debug, Clone)]
pub enum WorkSize {
	Unspecified,
	OneDim		(usize),
	TwoDims		(usize, usize),
	ThreeDims 	(usize, usize, usize),
}

impl WorkSize {
	/// Returns the number of dimensions defined by this `WorkSize`.
	pub fn dim_count(&self) -> cl_uint {
		use self::WorkSize::*;
		match self {
			&ThreeDims(..) 		=> 3,
			&TwoDims(..) 		=> 2,
			&OneDim(..) 		=> 1,
			&Unspecified 		=> 0,
		}

	}

	/// Returns the amount work to be done in three dimensional terms.
	pub fn complete_worksize(&self) -> (usize, usize, usize) {
		match self {
			&WorkSize::OneDim(x) => {
				(x, 1, 1)
			},
			&WorkSize::TwoDims(x, y) => {
				(x, y, 1)
			},
			&WorkSize::ThreeDims(x, y, z) => {
				(x, y, z)
			},
			_ => (1, 1, 1)
		}
	}

	/// Returns a raw pointer to the enum.
	pub fn as_ptr(&self) -> *const size_t {
		match self {
			&WorkSize::OneDim(x) => {
				&x as *const usize as *const size_t
			},
			&WorkSize::TwoDims(x, _) => {
				&x as *const usize as *const size_t
			},
			&WorkSize::ThreeDims(x, _, _) => {
				&x as *const usize as *const size_t
			},
			_ => ptr::null(),
		}
	}
}
