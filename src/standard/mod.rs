//! `ocl` standard types.
//!
//! [TODO]: This module needs a rename.

mod platform;
mod device;
// mod device_specifier;
// mod context_builder;
mod context;
// mod program_builder;
mod program;
mod kernel;
mod queue;
mod buffer;
// mod buffer_cmd;
// mod image_builder;
mod image;
// mod image_cmd;
mod sampler;
// mod pro_que_builder;
mod pro_que;
mod event;
// mod event_list;
mod spatial_dims;
// mod work_dims;

// #[cfg(not(release))] pub use self::buffer::tests::BufferTest;
pub use self::platform::Platform;
pub use self::device::{Device, DeviceSpecifier};
// pub use self::device_specifier::DeviceSpecifier;
// pub use self::context_builder::ContextBuilder;
pub use self::context::{Context, ContextBuilder};
// pub use self::program_builder::{ProgramBuilder, BuildOpt};
pub use self::program::{Program, ProgramBuilder, BuildOpt};
pub use self::queue::Queue;
pub use self::kernel::{Kernel, KernelCmd};
pub use self::buffer::{MappedMem, BufferCmdKind, BufferCmdDataShape, BufferCmd, Buffer, SubBuffer};
// pub use self::buffer_cmd::{BufferCmd, BufferCmdKind, BufferCmdDataShape};
// pub use self::image_builder::ImageBuilder;
pub use self::image::{Image, ImageCmd, ImageCmdKind, ImageBuilder};
// pub use self::image_cmd::{ImageCmd, ImageCmdKind};
pub use self::sampler::Sampler;
// pub use self::pro_que_builder::ProQueBuilder;
pub use self::pro_que::{ProQue, ProQueBuilder};
pub use self::event::{Event, EventList};
// pub use self::event_list::EventList;
pub use self::spatial_dims::SpatialDims;
pub use self::traits::{MemLen, WorkDims, AsMemRef, AsMemMut};
pub use self::types::{ClEventPtrNewEnum};


//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// pub const INFO_FORMAT_MULTILINE: bool = false;

//=============================================================================
//================================== TYPES ====================================
//=============================================================================

mod types {
    use ::{Event, EventList};
    use core::ffi::cl_event;
    use core::{Result as OclResult, NullEvent as NullEventCore, EventList as EventListCore,
        ClEventPtrNew, };

    #[derive(Debug)]
    pub enum ClEventPtrNewEnum<'a> {
        NullEventCore(&'a mut NullEventCore),
        EventListCore(&'a mut EventListCore),
        Event(&'a mut Event),
        EventList(&'a mut EventList),
    }

    unsafe impl<'a> ClEventPtrNew for ClEventPtrNewEnum<'a> {
        fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_event> {
            match *self {
                ClEventPtrNewEnum::NullEventCore(ref mut e) => e.ptr_mut_ptr_new(),
                ClEventPtrNewEnum::EventListCore(ref mut e) => e.ptr_mut_ptr_new(),
                ClEventPtrNewEnum::Event(ref mut e) => e.ptr_mut_ptr_new(),
                ClEventPtrNewEnum::EventList(ref mut e) => e.ptr_mut_ptr_new(),
            }
        }
    }

    impl<'a> From<&'a mut NullEventCore> for ClEventPtrNewEnum<'a> {
        fn from(e: &'a mut NullEventCore) -> ClEventPtrNewEnum<'a> {
            ClEventPtrNewEnum::NullEventCore(e)
        }
    }

    impl<'a> From<&'a mut EventListCore> for ClEventPtrNewEnum<'a> {
        fn from(e: &'a mut EventListCore) -> ClEventPtrNewEnum<'a> {
            ClEventPtrNewEnum::EventListCore(e)
        }
    }

    impl<'a> From<&'a mut Event> for ClEventPtrNewEnum<'a> {
        fn from(e: &'a mut Event) -> ClEventPtrNewEnum<'a> {
            ClEventPtrNewEnum::Event(e)
        }
    }

    impl<'a> From<&'a mut EventList> for ClEventPtrNewEnum<'a> {
        fn from(el: &'a mut EventList) -> ClEventPtrNewEnum<'a> {
            ClEventPtrNewEnum::EventList(el)
        }
    }


}

//=============================================================================
//================================== TRAITS ===================================
//=============================================================================

mod traits {
    use std::fmt::Debug;
    // use std::convert::Into;
    use num::{Num, ToPrimitive};
    // use core::error::{Result as OclResult};
    use core::Mem as MemCore;
    use ::{SpatialDims, OclPrm};
    use super::spatial_dims::to_usize;


    /// `AsRef` with a type being carried along for convenience.
    pub trait AsMemRef<T: OclPrm> {
        fn as_mem_ref(&self) -> &MemCore;
    }


    /// `AsMut` with a type being carried along for convenience.
    pub trait AsMemMut<T: OclPrm> {
        fn as_mem_mut(&mut self) -> &mut MemCore;
    }


    /// Types which have properties describing the amount of work to be done
    /// in multiple dimensions.
    ///
    pub trait WorkDims {
        /// Returns the number of dimensions defined.
        fn dim_count(&self) -> u32;
        /// Returns an array representing the amount of work to be done by a kernel.
        ///
        /// Unspecified dimensions (for example, the 3rd dimension in a
        /// 1-dimensional work size) are set equal to `1`.
        ///
        fn to_work_size(&self) -> Option<[usize; 3]>;
        /// Returns an array representing the offset of a work item or memory
        /// location.
        ///
        /// Unspecified dimensions (for example, the 3rd dimension in a
        /// 1-dimensional work size) are set equal to `0`.
        ///
        fn to_work_offset(&self) -> Option<[usize; 3]>;
    }


    /// Types which have properties allowing them to be used to define the size
    /// of a volume of memory.
    ///
    /// Units are expressed in `bytes / size_of(T)` just like `Vec::len()`.
    ///
    pub trait MemLen {
        /// Returns the exact number of elements of a volume of memory
        /// (equivalent to `Vec::len()`).
        fn to_len(&self) -> usize;
        /// Returns the length of a volume of memory padded to the next
        /// multiple of `incr`.
        fn to_len_padded(&self, incr: usize) -> usize;
        /// Returns the exact lengths of each dimension of a volume of memory.
        fn to_lens(&self) -> [usize; 3];
    }

    impl<'a, D> MemLen for &'a D where D: MemLen {
        fn to_len(&self) -> usize { (*self).to_len() }
        fn to_len_padded(&self, incr: usize) -> usize { (*self).to_len_padded(incr) }
        fn to_lens(&self) -> [usize; 3] { (*self).to_lens() }
    }

    impl<D> MemLen for (D, ) where D: Num + ToPrimitive + Debug + Copy {
        fn to_len(&self) -> usize {
            SpatialDims::One(to_usize(self.0)).to_len()
        }
        fn to_len_padded(&self, incr: usize) -> usize {
            SpatialDims::One(to_usize(self.0)).to_len_padded(incr)
        }
        fn to_lens(&self) -> [usize; 3] { [to_usize(self.0), 1, 1] }
    }

    impl<D> MemLen for [D; 1] where D: Num + ToPrimitive + Debug + Copy {
        fn to_len(&self) -> usize {
            SpatialDims::One(to_usize(self[0])).to_len()
        }
        fn to_len_padded(&self, incr: usize) -> usize {
            SpatialDims::One(to_usize(self[0])).to_len_padded(incr)
        }
        fn to_lens(&self) -> [usize; 3] { [to_usize(self[0]), 1, 1] }
    }

    impl<D> MemLen for (D, D) where D: Num + ToPrimitive + Debug + Copy {
        fn to_len(&self) -> usize {
            SpatialDims::Two(to_usize(self.0), to_usize(self.1)).to_len()
        }
        fn to_len_padded(&self, incr: usize) -> usize {
            SpatialDims::Two(to_usize(self.0), to_usize(self.1)).to_len_padded(incr)
        }
        fn to_lens(&self) -> [usize; 3] { [to_usize(self.0), to_usize(self.1), 1] }
    }

    impl<D> MemLen for [D; 2] where D: Num + ToPrimitive + Debug + Copy {
        fn to_len(&self) -> usize {
            SpatialDims::Two(to_usize(self[0]), to_usize(self[1])).to_len()
        }
        fn to_len_padded(&self, incr: usize) -> usize {
            SpatialDims::Two(to_usize(self[0]), to_usize(self[1])).to_len_padded(incr)
        }
        fn to_lens(&self) -> [usize; 3] { [to_usize(self[0]), to_usize(self[1]), 1] }
    }

    impl<'a, D> MemLen for (D, D, D) where D: Num + ToPrimitive + Debug + Copy {
        fn to_len(&self) -> usize {
            SpatialDims::Three(to_usize(self.0), to_usize(self.1), to_usize(self.2))
                .to_len()
        }
        fn to_len_padded(&self, incr: usize) -> usize {
            SpatialDims::Three(to_usize(self.0), to_usize(self.1), to_usize(self.2))
                .to_len_padded(incr)
        }
        fn to_lens(&self) -> [usize; 3] { [to_usize(self.0), to_usize(self.1), to_usize(self.2)] }
    }

    impl<'a, D> MemLen for [D; 3] where D: Num + ToPrimitive + Debug + Copy {
        fn to_len(&self) -> usize {
            SpatialDims::Three(to_usize(self[0]), to_usize(self[1]), to_usize(self[2]))
                .to_len()
        }
        fn to_len_padded(&self, incr: usize) -> usize {
            SpatialDims::Three(to_usize(self[0]), to_usize(self[1]), to_usize(self[2]))
                .to_len_padded(incr)
        }
        fn to_lens(&self) -> [usize; 3] { [to_usize(self[0]), to_usize(self[1]), to_usize(self[2])] }
    }
}


