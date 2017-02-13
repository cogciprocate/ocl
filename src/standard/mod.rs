//! `ocl` standard types.
//!
//! [TODO]: This module needs a rename.

mod platform;
mod device;
mod context;
mod program;
mod kernel;
mod queue;
mod buffer;
mod mapped_mem;
mod image;
mod sampler;
mod pro_que;
mod event;
mod spatial_dims;

pub use self::platform::Platform;
pub use self::device::{Device, DeviceSpecifier};
pub use self::context::{Context, ContextBuilder};
pub use self::program::{Program, ProgramBuilder, BuildOpt};
pub use self::queue::Queue;
pub use self::kernel::{Kernel, KernelCmd};
pub use self::buffer::{BufferCmdKind, BufferCmdDataShape, BufferCmd, Buffer, SubBuffer};
pub use self::image::{Image, ImageCmd, ImageCmdKind, ImageBuilder};
pub use self::mapped_mem::{FutureMappedMem, MappedMem};
pub use self::sampler::Sampler;
pub use self::pro_que::{ProQue, ProQueBuilder};
pub use self::event::{Event, EventList};
pub use self::spatial_dims::SpatialDims;
pub use self::traits::{MemLen, WorkDims};
pub use self::types::{ClNullEventPtrEnum, ClWaitListPtrEnum};


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
    use core::{Event as EventCore,
        UserEvent as UserEventCore, EventList as EventListCore, ClNullEventPtr, ClWaitListPtr};


    #[derive(Debug)]
    pub enum ClWaitListPtrEnum<'a> {
        EventCore(&'a EventCore),
        UserEventCore(&'a UserEventCore),
        EventListCore(&'a EventListCore),
        Event(&'a Event),
        EventList(&'a EventList),
        EventPtrSlice(&'a [cl_event]),
    }

    unsafe impl<'a> ClWaitListPtr for ClWaitListPtrEnum<'a> {
        unsafe fn as_ptr_ptr(&self) -> *const cl_event {
            match *self {
                ClWaitListPtrEnum::EventCore(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::UserEventCore(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::EventListCore(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::Event(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::EventList(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::EventPtrSlice(ref e) => e.as_ptr_ptr(),
            }
        }

        fn count(&self) -> u32 {
            match *self {
                ClWaitListPtrEnum::EventCore(ref e) => e.count(),
                ClWaitListPtrEnum::UserEventCore(ref e) => e.count(),
                ClWaitListPtrEnum::EventListCore(ref e) => e.count(),
                ClWaitListPtrEnum::Event(ref e) => e.count(),
                ClWaitListPtrEnum::EventList(ref e) => e.count(),
                ClWaitListPtrEnum::EventPtrSlice(ref e) => e.count(),
            }
        }
    }

    impl<'a> From<&'a EventCore> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a EventCore) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventCore(e)
        }
    }

    impl<'a> From<&'a UserEventCore> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a UserEventCore) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::UserEventCore(e)
        }
    }

    impl<'a> From<&'a EventListCore> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a EventListCore) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventListCore(e)
        }
    }

    impl<'a> From<&'a Event> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a Event) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::Event(e)
        }
    }

    impl<'a> From<&'a EventList> for ClWaitListPtrEnum<'a> {
        fn from(el: &'a EventList) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventList(el)
        }
    }

    impl<'a> From<&'a [cl_event]> for ClWaitListPtrEnum<'a> {
        fn from(el: &'a [cl_event]) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventPtrSlice(el)
        }
    }


    #[derive(Debug)]
    pub enum ClNullEventPtrEnum<'a> {
        // NullEventCore(&'a mut NullEventCore),
        EventListCore(&'a mut EventListCore),
        Event(&'a mut Event),
        EventList(&'a mut EventList),
    }

    unsafe impl<'a> ClNullEventPtr for ClNullEventPtrEnum<'a> {
        fn alloc_new(&mut self) -> *mut cl_event {
            match *self {
                // ClNullEventPtrEnum::NullEventCore(ref mut e) => e.alloc_new(),
                ClNullEventPtrEnum::EventListCore(ref mut e) => e.alloc_new(),
                ClNullEventPtrEnum::Event(ref mut e) => e.alloc_new(),
                ClNullEventPtrEnum::EventList(ref mut e) => e.alloc_new(),
            }
        }
    }

    // impl<'a> From<&'a mut NullEventCore> for ClNullEventPtrEnum<'a> {
    //     fn from(e: &'a mut NullEventCore) -> ClNullEventPtrEnum<'a> {
    //         ClNullEventPtrEnum::NullEventCore(e)
    //     }
    // }

    impl<'a> From<&'a mut EventListCore> for ClNullEventPtrEnum<'a> {
        fn from(e: &'a mut EventListCore) -> ClNullEventPtrEnum<'a> {
            ClNullEventPtrEnum::EventListCore(e)
        }
    }

    impl<'a> From<&'a mut Event> for ClNullEventPtrEnum<'a> {
        fn from(e: &'a mut Event) -> ClNullEventPtrEnum<'a> {
            ClNullEventPtrEnum::Event(e)
        }
    }

    impl<'a> From<&'a mut EventList> for ClNullEventPtrEnum<'a> {
        fn from(el: &'a mut EventList) -> ClNullEventPtrEnum<'a> {
            ClNullEventPtrEnum::EventList(el)
        }
    }


}

//=============================================================================
//================================== TRAITS ===================================
//=============================================================================

mod traits {
    use std::fmt::Debug;
    use num::{Num, ToPrimitive};
    use ::SpatialDims;
    use super::spatial_dims::to_usize;

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
        fn to_work_size(&self) -> Option<[usize; 3]>;

        /// Returns an array representing the offset of a work item or memory
        /// location.
        ///
        /// Unspecified dimensions (for example, the 3rd dimension in a
        /// 1-dimensional work size) are set equal to `0`.
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


