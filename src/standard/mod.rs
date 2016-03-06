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
mod buffer_cmd;
mod image_builder;
mod image;
mod image_cmd;
mod sampler;
mod pro_que_builder;
mod pro_que;
mod event;
mod event_list;
mod spatial_dims;
// mod work_dims;

// #[cfg(not(release))] pub use self::buffer::tests::BufferTest;
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
pub use self::buffer_cmd::{BufferCmd, BufferCmdKind, BufferCmdDataShape};
pub use self::image_builder::ImageBuilder;
pub use self::image::Image;
pub use self::image_cmd::{ImageCmd, ImageCmdKind};
pub use self::sampler::Sampler;
pub use self::pro_que_builder::ProQueBuilder;
pub use self::pro_que::ProQue;
pub use self::event::Event;
pub use self::event_list::EventList;
pub use self::spatial_dims::SpatialDims;
pub use self::traits::MemDims;
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
    use error::{Result as OclResult};
    use super::{SpatialDims};
    use super::spatial_dims::to_usize;


    /// Types which have properties describing the amount of work to be done
    /// in multiple dimensions.
    pub trait WorkDims {
        /// Returns the number of dimensions defined by this `SpatialDims`.
        fn dim_count(&self) -> u32;
        fn to_work_size(&self) -> Option<[usize; 3]>;
        fn to_work_offset(&self) -> Option<[usize; 3]>;
    }

    /// Types which have properties allowing them to be used to define the size
    /// of buffers.
    pub trait MemDims {
        fn padded_buffer_len(&self, usize) -> OclResult<usize>;
        fn to_size(&self) -> [usize; 3];
    }

    impl<'a, D> MemDims for &'a D where D: MemDims {
        fn padded_buffer_len(&self, incr: usize) -> OclResult<usize> { (*self).padded_buffer_len(incr) }
        fn to_size(&self) -> [usize; 3] { (*self).to_size() }
    }

    impl<D> MemDims for (D, ) where D: Num + ToPrimitive + Debug + Copy {
        fn padded_buffer_len(&self, incr: usize) -> OclResult<usize> {
            SpatialDims::One(to_usize(self.0)).padded_buffer_len(incr)
        }
        fn to_size(&self) -> [usize; 3] { [to_usize(self.0), 1, 1] }
    }

    impl<D> MemDims for [D; 1] where D: Num + ToPrimitive + Debug + Copy {
        fn padded_buffer_len(&self, incr: usize) -> OclResult<usize> {
            SpatialDims::One(to_usize(self[0])).padded_buffer_len(incr)
        }
        fn to_size(&self) -> [usize; 3] { [to_usize(self[0]), 1, 1] }
    }

    impl<D> MemDims for (D, D) where D: Num + ToPrimitive + Debug + Copy {
        fn padded_buffer_len(&self, incr: usize) -> OclResult<usize> {
            SpatialDims::Two(to_usize(self.0), to_usize(self.1)).padded_buffer_len(incr)
        }
        fn to_size(&self) -> [usize; 3] { [to_usize(self.0), to_usize(self.1), 1] }
    }

    impl<D> MemDims for [D; 2] where D: Num + ToPrimitive + Debug + Copy {
        fn padded_buffer_len(&self, incr: usize) -> OclResult<usize> {
            SpatialDims::Two(to_usize(self[0]), to_usize(self[1])).padded_buffer_len(incr)
        }
        fn to_size(&self) -> [usize; 3] { [to_usize(self[0]), to_usize(self[1]), 1] }
    }

    impl<'a, D> MemDims for (D, D, D) where D: Num + ToPrimitive + Debug + Copy {
        fn padded_buffer_len(&self, incr: usize) -> OclResult<usize> {
            SpatialDims::Three(to_usize(self.0), to_usize(self.1), to_usize(self.2))
                .padded_buffer_len(incr)
        }
        fn to_size(&self) -> [usize; 3] { [to_usize(self.0), to_usize(self.1), to_usize(self.2)] }
    }

    impl<'a, D> MemDims for [D; 3] where D: Num + ToPrimitive + Debug + Copy {
        fn padded_buffer_len(&self, incr: usize) -> OclResult<usize> {
            SpatialDims::Three(to_usize(self[0]), to_usize(self[1]), to_usize(self[2]))
                .padded_buffer_len(incr)
        }
        fn to_size(&self) -> [usize; 3] { [to_usize(self[0]), to_usize(self[1]), to_usize(self[2])] }
    }   
}


