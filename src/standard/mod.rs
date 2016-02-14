//! ocl standard types.

mod context;
mod program_builder;
mod program;
mod kernel;
mod queue;
mod buffer;
mod image;
mod pro_que_builder;
mod pro_que;
mod simple_dims;
mod work_dims;
mod event_list;


pub use self::context::Context;
pub use self::program_builder::{ProgramBuilder, BuildOpt};
pub use self::program::Program;
pub use self::queue::Queue;
pub use self::kernel::Kernel;
pub use self::buffer::Buffer;
pub use self::image::Image;
pub use self::pro_que_builder::ProQueBuilder;
pub use self::pro_que::ProQue;
pub use self::simple_dims::SimpleDims;
pub use self::work_dims::WorkDims;
pub use self::event_list::EventList;
#[cfg(not(release))]pub use self::buffer::tests::BufferTest;


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
