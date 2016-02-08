//! Rust implementation of OpenCL.
//!
//!
//! This documentation is very much a work in progress and is roughly 60% complete.
//! Please help by filing an [issue](https://github.com/cogciprocate/ocl/issues) about 
//! unclear and/or incomplete documentation and it will be addressed (hopefully) 
//! quickly.
//!
//! An explanation of how dimensions and sizes of buffers and work queues are handled
//! and coordinated will be coming as soon as a few more things are ironed out.
//!
//! ## Links
//!
//! **GitHub:** [https://github.com/cogciprocate/ocl](https://github.com/cogciprocate/ocl)
//!
//! **crates.io:** [![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl)

// #![warn(missing_docs)]
#![feature(zero_one)]

pub use self::cl_h::{cl_platform_id, cl_device_id, cl_device_type, cl_device_info, cl_context, 
	cl_program, cl_program_build_info, cl_command_queue, cl_mem, cl_event, cl_bool,
	cl_float, cl_char, cl_uchar, cl_short, cl_ushort, cl_int, cl_uint, cl_long, 
	cl_bitfield, CLStatus, CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, 
	CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL};

pub use self::formatting as fmt;
pub use self::context::Context;
pub use self::program::Program;
pub use self::queue::Queue;
pub use self::kernel::Kernel;
pub use self::buffer::Buffer;
pub use self::image::Image;
pub use self::pro_que::ProQue;
pub use self::simple_dims::SimpleDims;
pub use self::work_size::WorkSize;
pub use self::build_config::{BuildConfig, BuildOpt};
pub use self::error::{OclError, OclResult};
pub use self::event_list::EventList;
// [FIXME]: TODO: Create an additional crate build configuration for tests
pub use self::buffer::tests::BufferTest;

#[macro_use] 
extern crate enum_primitive;
extern crate libc;
extern crate num;
extern crate rand;

mod context;
mod program;
mod queue;
pub mod cl_h;
mod buffer;
mod image;
mod pro_que;
mod simple_dims;
mod kernel;
mod work_size;
mod build_config;
mod error;
pub mod formatting;
mod event_list;
mod wrapper;
#[cfg(test)]
mod tests;


//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// pub static CL_DEVICE_TYPE_DEFAULT:                       cl_device_type = 1 << 0;
// 		CL_DEVICE_TYPE_DEFAULT:	The default OpenCL device in the system.
// pub static CL_DEVICE_TYPE_CPU:                           cl_device_type = 1 << 1;
// 		CL_DEVICE_TYPE_CPU:	An OpenCL device that is the host processor. The host processor runs the OpenCL implementations and is a single or multi-core CPU.
// pub static CL_DEVICE_TYPE_GPU:                           cl_device_type = 1 << 2;
// 		CL_DEVICE_TYPE_GPU:	An OpenCL device that is a GPU. By this we mean that the device can also be used to accelerate a 3D API such as OpenGL or DirectX.
// pub static CL_DEVICE_TYPE_ACCELERATOR:                   cl_device_type = 1 << 3;
// 		CL_DEVICE_TYPE_ACCELERATOR:	Dedicated OpenCL accelerators (for example the IBM CELL Blade). These devices communicate with the host processor using a peripheral interconnect such as PCIe.
// pub static CL_DEVICE_TYPE_ALL:                           cl_device_type = 0xFFFFFFFF;
// 		CL_DEVICE_TYPE_ALL
const DEFAULT_DEVICE_TYPE: cl_device_type = 1 << 0; // CL_DEVICE_TYPE_DEFAULT

const DEVICES_MAX: u32 = 16;
const DEFAULT_PLATFORM: usize = 0;
const DEFAULT_DEVICE: usize = 0;

//=============================================================================
//================================= TRAITS ====================================
//=============================================================================

use std::fmt::{Display, Debug};
use std::num::Zero;
use num::{NumCast, FromPrimitive, ToPrimitive};
use rand::distributions::range::SampleRange;

/// A number compatible with OpenCL.
pub trait OclNum: Copy + Clone + PartialOrd  + NumCast + Default + Zero + Display + Debug
	+ FromPrimitive + ToPrimitive + SampleRange {}

impl<T> OclNum for T where T: Copy + Clone + PartialOrd + NumCast + Default + Zero + Display + Debug
	+ FromPrimitive + ToPrimitive + SampleRange {}

/// A type which has dimensional properties allowing it to be used to define the size
/// of buffers and work sizes.
pub trait BufferDims {
	fn padded_buffer_len(&self, usize) -> usize;
}

impl<'a, T> BufferDims for &'a T where T: BufferDims {
    fn padded_buffer_len(&self, incr: usize) -> usize { (*self).padded_buffer_len(incr) }
}

//=============================================================================
//=========================== UTILITY FUNCTIONS ===============================
//=============================================================================

/// Pads `len` to make it evenly divisible by `incr`.
pub fn padded_len(len: usize, incr: usize) -> usize {
	let len_mod = len % incr;

	if len_mod == 0 {
		len
	} else {
		let pad = incr - len_mod;
		let padded_len = len + pad;
		debug_assert_eq!(padded_len % incr, 0);
		padded_len
	}
}
