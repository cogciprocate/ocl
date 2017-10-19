#![allow(unused_imports, dead_code)]

use std::ops::{Deref, DerefMut};
use core::{self, Result as OclResult, OclPrm, MemMap as MemMapCore, Mem as MemCore, AsMem,
    ClWaitListPtr, ClNullEventPtr, MemFlags};
use standard::{ClWaitListPtrEnum, ClNullEventPtrEnum, Event, EventList, Queue, Buffer};
use async::{Result as AsyncResult};



/// Represents mapped memory and allows frames of data to be 'flushed'
/// (written) from host-accessible mapped memory region into its associated
/// device-visible buffer in a repeated fashion.
///
/// This represents the fastest possible method for continuously writing
/// buffer-sized frames of data to a device.
pub struct BufferSink<T: OclPrm> {
    buffer: Buffer<T>,
    // memory: MemMapCore<T>,
}

impl<T: OclPrm> BufferSink<T> {
    pub unsafe fn new(buffer: Buffer<T>) -> OclResult<BufferSink<T>> {
        // TODO: Ensure that these checks are complete enough.
        assert!(buffer.flags().contains(MemFlags::new().alloc_host_ptr()) ||
            buffer.flags().contains(MemFlags::new().use_host_ptr()),
            "A buffer sink must be created with a buffer that has either \
            the MEM_ALLOC_HOST_PTR` or `MEM_USE_HOST_PTR flag.");



        Ok(BufferSink {
            buffer,
            // memory,
        })
    }
}