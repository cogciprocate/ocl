//! Rust implementation of OpenCL.
//!
//!
//! This documentation is very much a work in progress and is roughly 60% complete.
//! Please help by filing an [issue](https://github.com/cogciprocate/ocl/issues) about 
//! unclear and/or incomplete documentation and it will be addressed (hopefully) 
//! quickly.
//!
//! An explanation of how dimensions and sizes of buffers and work queues are intended 
//! to be used will be coming as soon as a few more things are ironed out.
//!
//! # Library Wide Panics
//!
//! All operations will panic upon any OpenCL error. Some work needs to be done
//! evaluating which errors are easily uncovered during development
//! and are therefore better off 
//! continuing to panic such as invalid kernel code, and which errors are 
//! more run-timeish and should be returned in a `Result`.
//!
//! ## Links
//!
//! **GitHub:** [https://github.com/cogciprocate/ocl](https://github.com/cogciprocate/ocl)
//!
//! **crates.io:** [![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl)
//!

// #![warn(missing_docs)]
#![feature(zero_one)]

#[macro_use] extern crate enum_primitive;
#[macro_use] extern crate bitflags;
extern crate libc;
extern crate num;
extern crate rand;

mod context;
mod program_builder;
mod program;
mod queue;
mod buffer;
mod image;
mod image_format;
mod image_descriptor;
mod pro_que_builder;
mod pro_que;
mod simple_dims;
mod kernel;
mod work_size;
mod error;
mod event_list;
mod wrapper;
#[cfg(test)] mod tests;
pub mod formatting;
pub mod cl_h;

pub use self::cl_h::{cl_platform_id, cl_device_id, cl_device_type, cl_device_info, cl_context, 
    cl_program, cl_program_build_info, cl_command_queue, cl_mem, cl_event, 
    cl_bool, cl_float, cl_char, cl_uchar, cl_short, cl_ushort, cl_int, cl_uint, cl_long, 
    cl_bitfield, ClStatus, CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, 
    CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_CUSTOM, CL_DEVICE_TYPE_ALL};

pub use self::formatting as fmt;
pub use self::context::Context;
pub use self::program_builder::{ProgramBuilder, BuildOpt};
pub use self::program::Program;
pub use self::queue::Queue;
pub use self::kernel::Kernel;
pub use self::buffer::Buffer;
pub use self::image::Image;
pub use self::image_format::{ImageFormat, ChannelOrder, ChannelType};
pub use self::image_descriptor::{ImageDescriptor, MemObjectType};
pub use self::pro_que_builder::ProQueBuilder;
pub use self::pro_que::ProQue;
pub use self::simple_dims::SimpleDims;
pub use self::work_size::WorkSize;
pub use self::error::{Error, Result};
pub use self::event_list::EventList;
pub use self::buffer::tests::BufferTest;


//=============================================================================
//============================== INTERNAL USE =================================
//=============================================================================



//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

const DEFAULT_DEVICE_TYPE: cl_device_type = CL_DEVICE_TYPE_DEFAULT;

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
