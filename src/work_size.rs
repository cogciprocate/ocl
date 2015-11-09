use std::ptr;
use libc;


#[derive(PartialEq, Debug, Clone)]
pub enum WorkSize {
	Unspecified,
	OneDim		(usize),
	TwoDims		(usize, usize),
	ThreeDims 	(usize, usize, usize),
}

impl WorkSize {
	pub fn dim_count(&self) -> super::cl_uint {
		use self::WorkSize::*;
		match self {
			&ThreeDims(..) 		=> 3,
			&TwoDims(..) 		=> 2,
			&OneDim(..) 		=> 1,
			&Unspecified 		=> 0,
		}

	}

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

	pub fn as_ptr(&self) -> *const libc::size_t {

		match self {
			&WorkSize::OneDim(x) => {
				let s: (usize, usize, usize) = (x, 1, 1);
				(&s as *const (usize, usize, usize)) as *const libc::size_t
			},
			&WorkSize::TwoDims(x, y) => {
				let s: (usize, usize, usize) = (x, y, 1);
				(&s as *const (usize, usize, usize)) as *const libc::size_t
			},
			&WorkSize::ThreeDims(x, y, z) => {
				let s: (usize, usize, usize) = (x, y, z);
				(&s as *const (usize, usize, usize)) as *const libc::size_t
			},
			_ => ptr::null(),
		}
	}
}
