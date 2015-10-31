// use std;
use std::ptr;
// use std::mem;
// use std::io::{ Read };
// use std::fs::{ File };
// use std::ffi;
// use std::iter;
// use std::collections::{ HashMap };
// use num::{ self, Integer, FromPrimitive };
use libc;


#[derive(PartialEq, Debug, Clone, Eq, Hash)]
pub enum WorkSize {
	Unspecified,
	OneDim		(usize),
	TwoDim		(usize, usize),
	ThreeDim 	(usize, usize, usize),
}

impl WorkSize {
	pub fn dims(&self) -> super::cl_uint {
		use self::WorkSize::*;
		match self {
			&ThreeDim(..) 		=> 3,
			&TwoDim(..) 		=> 2,
			&OneDim(..) 		=> 1,
			&Unspecified 		=> 0,
		}

	}

	pub fn complete_worksize(&self) -> (usize, usize, usize) {
		match self {
			&WorkSize::OneDim(x) => {
				(x, 1, 1)
			},
			&WorkSize::TwoDim(x, y) => {
				(x, y, 1)
			},
			&WorkSize::ThreeDim(x, y, z) => {
				(x, y, z)
			},
			_ => (1, 1, 1)
		}
	}

	/* AS_PTR():
		THIS COULD BE BUGGY SINCE THE POINTER TO 's' IS LEFT DANGLING
		TODO: CHECK INTO IT
	*/
	pub fn as_ptr(&self) -> *const libc::size_t {

		match self {
			&WorkSize::OneDim(x) => {
				let s: (usize, usize, usize) = (x, 1, 1);
				(&s as *const (usize, usize, usize)) as *const libc::size_t
			},
			&WorkSize::TwoDim(x, y) => {
				let s: (usize, usize, usize) = (x, y, 1);
				(&s as *const (usize, usize, usize)) as *const libc::size_t
			},
			&WorkSize::ThreeDim(x, y, z) => {
				let s: (usize, usize, usize) = (x, y, z);
				(&s as *const (usize, usize, usize)) as *const libc::size_t
			},
			_ => ptr::null(),
		}
	}
}
