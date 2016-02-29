//! `ocl` standard types.

mod platform;
mod device;
mod device_specifier;
mod context_builder;
mod context;
mod program_builder;
mod program;
mod kernel;
mod queue;
mod buffer;
mod image_builder;
mod image;
mod sampler;
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
pub use self::context_builder::ContextBuilder;
pub use self::context::Context;
pub use self::program_builder::{ProgramBuilder, BuildOpt};
pub use self::program::Program;
pub use self::queue::Queue;
pub use self::kernel::Kernel;
pub use self::buffer::Buffer;
pub use self::image_builder::ImageBuilder;
pub use self::image::Image;
pub use self::sampler::Sampler;
pub use self::pro_que_builder::ProQueBuilder;
pub use self::pro_que::ProQue;
pub use self::event::Event;
pub use self::event_list::EventList;
pub use self::simple_dims::SimpleDims;
pub use self::traits::BufferDims;
pub use self::traits::WorkDims;


//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// pub const INFO_FORMAT_MULTILINE: bool = false;

//=============================================================================
//================================= TRAITS ====================================
//=============================================================================

mod traits {
	use std::fmt::Debug;
	// use std::convert::Into;
	use num::{Num, ToPrimitive};
	use super::{SimpleDims};
	use super::simple_dims::to_usize;


	/// Types which have properties describing the amount of work to be done
	/// in multiple dimensions.
	pub trait WorkDims {
	    /// Returns the number of dimensions defined by this `SimpleDims`.
	    fn dim_count(&self) -> u32;
	    fn to_work_size(&self) -> Option<[usize; 3]>;
	    fn to_work_offset(&self) -> Option<[usize; 3]>;
	}

	/// Types which have properties allowing them to be used to define the size
	/// of buffers.
	pub trait BufferDims {
	    fn padded_buffer_len(&self, usize) -> usize;
	}

	impl<'a, T> BufferDims for &'a T where T: BufferDims {
	    fn padded_buffer_len(&self, incr: usize) -> usize { (*self).padded_buffer_len(incr) }
	}


	impl<'a, T> BufferDims for &'a (T, ) where T: Num + ToPrimitive + Debug + Copy {
	    fn padded_buffer_len(&self, incr: usize) -> usize {
	        SimpleDims::One(to_usize(self.0)).padded_buffer_len(incr)
	    }
	}

	impl<'a, T> BufferDims for &'a [T; 1] where T: Num + ToPrimitive + Debug + Copy {
		fn padded_buffer_len(&self, incr: usize) -> usize {
	        SimpleDims::One(to_usize(self[0])).padded_buffer_len(incr)
	    }
	}

	impl<'a, T> BufferDims for &'a (T, T) where T: Num + ToPrimitive + Debug + Copy {
	    fn padded_buffer_len(&self, incr: usize) -> usize {
	    	SimpleDims::Two(to_usize(self.0), to_usize(self.1)).padded_buffer_len(incr)
	    }
	}

	impl<'a, T> BufferDims for &'a [T; 2] where T: Num + ToPrimitive + Debug + Copy {
		fn padded_buffer_len(&self, incr: usize) -> usize {
	        SimpleDims::Two(to_usize(self[0]), to_usize(self[1])).padded_buffer_len(incr)
	    }
	}

	impl<'a, T> BufferDims for &'a (T, T, T) where T: Num + ToPrimitive + Debug + Copy {
	    fn padded_buffer_len(&self, incr: usize) -> usize {
	        SimpleDims::Three(to_usize(self.0), to_usize(self.1), to_usize(self.2))
	        	.padded_buffer_len(incr)
	    }
	}

	impl<'a, T> BufferDims for &'a [T; 3] where T: Num + ToPrimitive + Debug + Copy {
		fn padded_buffer_len(&self, incr: usize) -> usize {
	        SimpleDims::Three(to_usize(self[0]), to_usize(self[1]), to_usize(self[2]))
	        	.padded_buffer_len(incr)
	    }
	}	
}


