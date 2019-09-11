//! `ocl` standard types.
//!
//! * TODO: This module needs a rename.

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

pub use self::platform::{PlatformError, Extensions, Platform};
pub use self::device::{DeviceError, Device, DeviceSpecifier};
pub use self::context::{Context, ContextBuilder};
pub use self::program::{Program, ProgramBuilder, BuildOpt};
pub use self::queue::Queue;
pub use self::kernel::{KernelError, KernelCmd, Kernel, KernelBuilder};
pub use self::buffer::{BufferCmdKind, BufferCmdDataShape, BufferCmd, Buffer, QueCtx,
    BufferBuilder, BufferReadCmd, BufferWriteCmd, BufferMapCmd, BufferCmdError, WriteSrc};
pub use self::image::{ImageCmdKind, ImageCmd, Image, ImageBuilder};
pub use self::sampler::Sampler;
pub use self::pro_que::{ProQue, ProQueBuilder};
pub use self::event::{Event, EventArray, EventList, IntoMarker, RawEventArray, IntoRawEventArray};
pub use self::spatial_dims::SpatialDims;
#[cfg(not(feature = "async_block"))]
pub use self::cb::{_unpark_task, box_raw_void};
pub use self::traits::{MemLen, WorkDims};
pub use self::types::{ClNullEventPtrEnum, ClWaitListPtrEnum};


// use ocl_core::OclPrm;

#[derive(Debug)]
enum HostSlice<'a, T> where T: 'a {
    None,
    Use(&'a [T]),
    Copy(&'a [T]),
}

impl<'a, T> HostSlice<'a, T> where T: 'a {
    fn is_none(&self) -> bool {
        match *self {
            HostSlice::None => true,
            _ => false,
        }
    }
}

//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// pub const INFO_FORMAT_MULTILINE: bool = false;

//=============================================================================
//================================ FUNCTIONS ==================================
//=============================================================================

#[cfg(not(feature = "async_block"))]
mod cb {
    use crate::ocl_core::ffi::c_void;
    use num_traits::FromPrimitive;
    use futures::task::Task;
    use crate::ocl_core::ffi::cl_event;
    use crate::ocl_core::{CommandExecutionStatus, Status};

    pub fn box_raw_void<T>(item: T) -> *mut c_void {
        let item_box = Box::new(item);
        Box::into_raw(item_box) as *mut _ as *mut c_void
    }

    pub extern "C" fn _unpark_task(event_ptr: cl_event, event_status: i32, user_data: *mut c_void) {

        let _ = event_ptr;
        // println!("'_unpark_task' has been called.");
        if event_status == CommandExecutionStatus::Complete as i32 && !user_data.is_null() {
            unsafe {
                let task_ptr = user_data as *mut _ as *mut Task;
                let task = Box::from_raw(task_ptr);
                (*task).notify();
            }
        } else {
            let status = if event_status < 0 {
                format!("{:?}", Status::from_i32(event_status))
            } else {
                format!("{:?}", CommandExecutionStatus::from_i32(event_status))
            };

            panic!("ocl::standard::_unpark_task: \n\nWake up user data is null or event is not \
                complete: {{ status: {:?}, user_data: {:?} }}. If you are getting \
                `DEVICE_NOT_AVAILABLE` and you are using Intel drivers, switch to AMD OpenCL \
                drivers instead (will work with Intel CPUs).\n\n", status, user_data);
        }
    }
}




//=============================================================================
//================================== TYPES ====================================
//=============================================================================

mod types {
    use std::ptr;
    use std::cell::Ref;
    use crate::standard::{Event, EventList, RawEventArray, Queue};
    use crate::ocl_core::ffi::cl_event;
    use crate::ocl_core::{Event as EventCore, ClNullEventPtr, ClWaitListPtr};
    use crate::error::Result as OclResult;

    /// An enum which can represent several different ways of representing a
    /// event wait list.
    ///
    //
    // * TODO:
    //   - Figure out a way to abstract over `&` and `&mut` versions of
    //   things in `From` impls without messing up the `Ref` versions.
    //   - Possibly rename this to something friendlier.
    #[derive(Debug)]
    pub enum ClWaitListPtrEnum<'a> {
        Null,
        RawEventArray(&'a RawEventArray),
        EventCoreOwned(EventCore),
        EventOwned(Event),
        EventCore(&'a EventCore),
        Event(&'a Event),
        EventList(&'a EventList),
        EventSlice(&'a [Event]),
        EventPtrSlice(&'a [cl_event]),
        RefEventList(Ref<'a, EventList>),
        RefTraitObj(Ref<'a, dyn ClWaitListPtr>),
        BoxTraitObj(Box<dyn ClWaitListPtr>),
    }

    impl<'a> ClWaitListPtrEnum<'a> {
        /// Converts this `ClWaitListPtrEnum` into a single marker event.
        pub fn into_marker(self, queue: &Queue) -> OclResult<Event> {
            queue.enqueue_marker(Some(self))
        }

        /// Returns an `EventList` containing owned copies of each element in
        /// this `ClWaitListPtrEnum`.
        pub fn to_list(&self) -> EventList {
            match *self {
                ClWaitListPtrEnum::Null => EventList::with_capacity(0),
                ClWaitListPtrEnum::RawEventArray(ref e) => e.as_slice().into(),
                ClWaitListPtrEnum::EventCoreOwned(ref e) => EventList::from(vec![e.clone().into()]),
                ClWaitListPtrEnum::EventOwned(ref e) => EventList::from(vec![e.clone()]),
                ClWaitListPtrEnum::EventCore(e) => EventList::from(vec![e.clone().into()]),
                ClWaitListPtrEnum::Event(e) => EventList::from(vec![e.clone()]),
                ClWaitListPtrEnum::EventList(e) => e.clone(),
                ClWaitListPtrEnum::EventSlice(e) => EventList::from(e),
                ClWaitListPtrEnum::EventPtrSlice(e) => EventList::from(e),
                ClWaitListPtrEnum::RefEventList(ref e) => (*e).clone(),
                ClWaitListPtrEnum::RefTraitObj(ref e) => Ref::clone(e).into(),
                ClWaitListPtrEnum::BoxTraitObj(ref e) => e.into(),
            }
        }
    }

    unsafe impl<'a> ClWaitListPtr for ClWaitListPtrEnum<'a> {
        unsafe fn as_ptr_ptr(&self) -> *const cl_event {
            match *self {
                ClWaitListPtrEnum::Null => ptr::null() as *const _ as *const cl_event,
                ClWaitListPtrEnum::RawEventArray(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::EventCoreOwned(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::EventCore(ref e) => e.as_ptr_ptr(),
                ClWaitListPtrEnum::EventOwned(ref e) => e.as_ptr_ptr(),
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
                ClWaitListPtrEnum::RawEventArray(ref e) => e.count(),
                ClWaitListPtrEnum::EventCoreOwned(ref e) => e.count(),
                ClWaitListPtrEnum::EventCore(ref e) => e.count(),
                ClWaitListPtrEnum::EventOwned(ref e) => e.count(),
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

    impl<'a> From<&'a RawEventArray> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a RawEventArray) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::RawEventArray(e)
        }
    }

    impl<'a> From<EventCore> for ClWaitListPtrEnum<'a> {
        fn from(e: EventCore) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventCoreOwned(e)
        }
    }

    impl<'a> From<&'a EventCore> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a EventCore) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventCore(e)
        }
    }

    impl<'a> From<&'a mut EventCore> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a mut EventCore) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventCore(e)
        }
    }

    impl<'a> From<Event> for ClWaitListPtrEnum<'a> {
        fn from(e: Event) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventOwned(e)
        }
    }

    impl<'a> From<&'a Event> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a Event) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::Event(e)
        }
    }

    impl<'a> From<&'a mut Event> for ClWaitListPtrEnum<'a> {
        fn from(e: &'a mut Event) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::Event(e)
        }
    }

    impl<'a> From<&'a EventList> for ClWaitListPtrEnum<'a> {
        fn from(el: &'a EventList) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventList(el)
        }
    }

    impl<'a> From<&'a mut EventList> for ClWaitListPtrEnum<'a> {
        fn from(el: &'a mut EventList) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventList(el)
        }
    }

    impl<'a> From<&'a [Event]> for ClWaitListPtrEnum<'a> {
        fn from(es: &'a [Event]) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventSlice(es)
        }
    }

    impl<'a> From<&'a mut [Event]> for ClWaitListPtrEnum<'a> {
        fn from(es: &'a mut [Event]) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventSlice(es)
        }
    }

    impl<'a> From<&'a [cl_event]> for ClWaitListPtrEnum<'a> {
        fn from(el: &'a [cl_event]) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventPtrSlice(el)
        }
    }

    impl<'a> From<&'a mut [cl_event]> for ClWaitListPtrEnum<'a> {
        fn from(el: &'a mut [cl_event]) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::EventPtrSlice(el)
        }
    }

    impl<'a> From<()> for ClWaitListPtrEnum<'a> {
        fn from(_: ()) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::Null
        }
    }

    impl<'a> From<Ref<'a, EventList>> for ClWaitListPtrEnum<'a> {
        fn from(e: Ref<'a, EventList>) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::RefTraitObj(e)
        }
    }

    impl<'a> From<Ref<'a, dyn ClWaitListPtr>> for ClWaitListPtrEnum<'a> {
        fn from(e: Ref<'a, dyn ClWaitListPtr>) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::RefTraitObj(e)
        }
    }

    impl<'a> From<Box<dyn ClWaitListPtr>> for ClWaitListPtrEnum<'a> {
        fn from(e: Box<dyn ClWaitListPtr>) -> ClWaitListPtrEnum<'a> {
            ClWaitListPtrEnum::BoxTraitObj(e)
        }
    }

    impl<'a, Ewl> From<Option<Ewl>> for ClWaitListPtrEnum<'a>
            where Ewl: Into<ClWaitListPtrEnum<'a>> {
        fn from(e: Option<Ewl>) -> ClWaitListPtrEnum<'a> {
            match e {
                Some(e) => e.into(),
                None => ClWaitListPtrEnum::Null,
            }
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
                ClNullEventPtrEnum::Null => panic!("Void events cannot be used."),
                ClNullEventPtrEnum::Event(ref mut e) => e.alloc_new(),
                ClNullEventPtrEnum::EventList(ref mut e) => e.alloc_new(),
            }
        }

        #[inline] unsafe fn clone_from<E: AsRef<EventCore>>(&mut self, ev: E) {
            match *self {
                ClNullEventPtrEnum::Null => panic!("Void events cannot be used."),
                ClNullEventPtrEnum::Event(ref mut e) => e.clone_from(ev),
                ClNullEventPtrEnum::EventList(ref mut e) => e.clone_from(ev),
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

    impl<'a, E> From<Option<E>> for ClNullEventPtrEnum<'a>
            where E: Into<ClNullEventPtrEnum<'a>> {
        fn from(e: Option<E>) -> ClNullEventPtrEnum<'a> {
            match e {
                Some(e) => e.into(),
                None => ClNullEventPtrEnum::Null,
            }
        }
    }
}

//=============================================================================
//================================== TRAITS ===================================
//=============================================================================

mod traits {
    use std::fmt::Debug;
    use num_traits::{Num, ToPrimitive};
    use crate::SpatialDims;
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


