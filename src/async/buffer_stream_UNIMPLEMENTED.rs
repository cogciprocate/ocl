#![allow(unused_imports)]

use std::ops::{Deref, DerefMut};
use core::{self, OclPrm, MemMap as MemMapCore, Mem as MemCore, AsMem, ClWaitListPtr, ClNullEventPtr};
use standard::{ClWaitListPtrEnum, ClNullEventPtrEnum, Event, EventList, Queue};
use async::{Result as AsyncResult};


/// Represents mapped memory and allows frames of data to be 'flooded' (read)
/// from a device-visible `Buffer` into its associated host-accessible mapped
/// memory region in a repeated fashion.
///
/// This represents the absolute fastest method for reading data from an
/// OpenCL device.
pub struct BufferStream<T: OclPrm> {
    buffer: Buffer<T>,
    map: MemMapCore<T>,
}

impl<T: OclPrm> BufferStream<T> {
    pub fn new() -> BufferStream<T> {
        BufferStream {

        }
    }
}