//! `ocl` standard types.

mod platform;
mod device;
mod device_specifier;
mod context;
mod program_builder;
mod program;
mod kernel;
mod queue;
mod buffer;
mod image_builder;
mod image;
mod pro_que_builder;
mod pro_que;
mod event;
mod event_list;
mod simple_dims;
// mod work_dims;


#[cfg(not(release))] pub use self::buffer::tests::BufferTest;
pub use self::platform::Platform;
pub use self::device::Device;
pub use self::device_specifier::DeviceSpecifier;
pub use self::context::Context;
pub use self::program_builder::{ProgramBuilder, BuildOpt};
pub use self::program::Program;
pub use self::queue::Queue;
pub use self::kernel::Kernel;
pub use self::buffer::Buffer;
pub use self::image_builder::ImageBuilder;
pub use self::image::Image;
pub use self::pro_que_builder::ProQueBuilder;
pub use self::pro_que::ProQue;
pub use self::event::Event;
pub use self::event_list::EventList;
pub use self::simple_dims::SimpleDims;
// pub use self::work_dims::WorkDims;


//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

pub const INFO_FORMAT_MULTILINE: bool = false;

//=============================================================================
//================================= TRAITS ====================================
//=============================================================================

/// A type which has dimensional properties allowing it to be used to define the size
/// of buffers and work sizes.
pub trait BufferDims {
    fn padded_buffer_len(&self, usize) -> usize;
}

impl<'a, T> BufferDims for &'a T where T: BufferDims {
    fn padded_buffer_len(&self, incr: usize) -> usize { (*self).padded_buffer_len(incr) }
}


pub trait WorkDims {
    /// Returns the number of dimensions defined by this `SimpleDims`.
    fn dim_count(&self) -> u32;
    fn work_dims(&self) -> Option<[usize; 3]>;
}
