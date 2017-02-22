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
mod image;
mod sampler;
mod pro_que;
mod event;
mod spatial_dims;
mod mem_map;

pub use self::platform::Platform;
pub use self::device::{Device, DeviceSpecifier};
pub use self::context::{Context, ContextBuilder};
pub use self::program::{Program, ProgramBuilder, BuildOpt};
pub use self::queue::Queue;
pub use self::kernel::{Kernel, KernelCmd};
pub use self::mem_map::{MemMap};
pub use self::buffer::{BufferCmdKind, BufferCmdDataShape, BufferCmd, Buffer, SubBuffer,
    QueCtx};
pub use self::image::{Image, ImageCmd, ImageCmdKind, ImageBuilder};
pub use self::sampler::Sampler;
pub use self::pro_que::{ProQue, ProQueBuilder};
pub use self::event::{Event, EventList};
pub use self::spatial_dims::SpatialDims;
pub use self::cb::{_unpark_task, box_raw_void};
pub use self::traits::{MemLen, WorkDims};
pub use self::types::{ClNullEventPtrEnum, ClWaitListPtrEnum};


//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// pub const INFO_FORMAT_MULTILINE: bool = false;

//=============================================================================
//================================ FUNCTIONS ==================================
//=============================================================================

#[cfg(feature = "event_callbacks")]
mod cb {
    use libc::c_void;
    use futures::task::Task;
    use ffi::cl_event;
    use core::CommandExecutionStatus;

    pub fn box_raw_void<T>(item: T) -> *mut c_void {
        let item_box = Box::new(item);
        Box::into_raw(item_box) as *mut _ as *mut c_void
    }

    pub extern "C" fn _unpark_task(event_ptr: cl_event, event_status: i32, user_data: *mut c_void) {
        let _ = event_ptr;
        // println!("'_unpark_task' has been called.");

        if event_status == CommandExecutionStatus::Complete as i32 && !user_data.is_null() {
            unsafe {
                // println!("Unparking task via callback...");

                let task_ptr = user_data as *mut _ as *mut Task;
                let task = Box::from_raw(task_ptr);
                (*task).unpark();
            }
        } else {
            panic!("Wake up user data is null or event is not complete.");
        }
    }
}



//=============================================================================
//================================== TYPES ====================================
//=============================================================================

mod types {
    use std::ptr;
    use std::cell::Ref;
    use ::{Event, EventList};
    use core::ffi::cl_event;
    use core::{Event as EventCore, ClNullEventPtr, ClWaitListPtr};


    #[derive(Debug)]
    pub enum ClWaitListPtrEnum<'a> {
        Null,
        EventCore(&'a EventCore),
        // UserEventCore(&'a UserEventCore),
        // EventListCore(&'a EventListCore),
        Event(&'a Event),
        EventList(&'a EventList),
        EventSlice(&'a [Event]),
        EventPtrSlice(&'a [cl_event]),
        RefEventList(Ref<'a, EventList>),
        RefTraitObj(Ref<'a, ClWaitListPtr>),
        BoxTraitObj(Box<ClWaitListPtr>),
    }

    unsafe impl<'a> ClWaitListPtr for ClWaitListPtrEnum<'a> {
        unsafe fn as_ptr_ptr(&self) -> *const cl_event {
            match *self {
                ClWaitListPtrEnum::Null => ptr::null() as *const _ as *const cl_event,
                ClWaitListPtrEnum::EventCore(ref e) => e.as_ptr_ptr(),
                // ClWaitListPtrEnum::UserEventCore(ref e) => e.as_ptr_ptr(),
                // ClWaitListPtrEnum::EventListCore(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::Event(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::EventList(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::EventSlice(ref e) => e.as_ptr() as *const _ as *const cl_event,
                ClWaitListPtrEnum::EventPtrSlice(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::RefEventList(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::RefTraitObj(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::BoxTraitObj(ref e) => e.as_ptr_ptr(),

            }
        }

        fn count(&self) -> u32 {
            match *self {
                ClWaitListPtrEnum::Null => 0,
                ClWaitListPtrEnum::EventCore(ref e) => e.count(),
                // ClWaitListPtrEnum::UserEventCore(ref e) => e.count(),
                // ClWaitListPtrEnum::EventListCore(ref e) => e.count(),
                ClWaitListPtrEnum::Event(ref e) => e.count(),
                ClWaitListPtrEnum::EventList(ref e) => e.count(),
                ClWaitListPtrEnum::EventSlice(ref e) => e.len() as u32,
                ClWaitListPtrEnum::EventPtrSlice(ref e) => e.count(),
                ClWaitListPtrEnum::RefEventList(ref e) => e.count(),
                ClWaitListPtrEnum::RefTraitObj(ref e) => e.count(),
                ClWaitListPtrEnum::BoxTraitObj(ref e) => e.count(),
            }
        }
    }

    impl<'a> From<Ref<'a, EventList>> for ClWaitListPtrEnum<'a> {
        fn from(e: Ref<'a, EventList>) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::RefTraitObj(e)
        }
    }

    impl<'a> From<Ref<'a, ClWaitListPtr>> for ClWaitListPtrEnum<'a> {
        fn from(e: Ref<'a, ClWaitListPtr>) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::RefTraitObj(e)
        }
    }

    impl<'a> From<Box<ClWaitListPtr>> for ClWaitListPtrEnum<'a> {
        fn from(e: Box<ClWaitListPtr>) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::BoxTraitObj(e)
        }
    }

    impl<'a> From<&'a EventCore> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a EventCore) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventCore(e)
        }
    }

    // impl<'a> From<&'a UserEventCore> for ClWaitListPtrEnum<'a> {
    //     fn from(e: &'a UserEventCore) -> ClWaitListPtrEnum<'a> {
    //         ClWaitListPtrEnum::UserEventCore(e)
    //     }
    // }

    // impl<'a> From<&'a EventListCore> for ClWaitListPtrEnum<'a> {
    //     fn from(e: &'a EventListCore) -> ClWaitListPtrEnum<'a> {
    //         ClWaitListPtrEnum::EventListCore(e)
    //     }
    // }

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

    impl<'a> From<&'a [Event]> for ClWaitListPtrEnum<'a> {
        fn from(es: &'a [Event]) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventSlice(es)
        }
    }

    impl<'a> From<&'a [cl_event]> for ClWaitListPtrEnum<'a> {
        fn from(el: &'a [cl_event]) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventPtrSlice(el)
        }
    }

    impl<'a> From<()> for ClWaitListPtrEnum<'a> {
        fn from(_: ()) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::Null
        }
    }


    #[derive(Debug)]
    pub enum ClNullEventPtrEnum<'a> {
        Null,
        Event(&'a mut Event),
        EventList(&'a mut EventList),
    }

    unsafe impl<'a> ClNullEventPtr for ClNullEventPtrEnum<'a> {
        fn alloc_new(&mut self) -> *mut cl_event {
            match *self {
                ClNullEventPtrEnum::Null => ptr::null_mut() as *mut _ as *mut cl_event,
                ClNullEventPtrEnum::Event(ref mut e) => e.alloc_new(),
                ClNullEventPtrEnum::EventList(ref mut e) => e.alloc_new(),
            }
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

    impl<'a> From<()> for ClNullEventPtrEnum<'a> {
        fn from(_: ()) -> ClNullEventPtrEnum<'a> {
            ClNullEventPtrEnum::Null
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


