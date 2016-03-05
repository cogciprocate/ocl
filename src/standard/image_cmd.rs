//! An image command builder for enqueuing reads, writes, and copies.

#![allow(dead_code, unused_variables, unused_mut)]

use core::{self, OclNum, Mem as MemCore, CommandQueue as CommandQueueCore, ClEventPtrNew};
use error::{Error as OclError, Result as OclResult};
use standard::{Queue, EventList, Image};

// fn check_size(mem_len: usize, data_len: usize, offset: usize) -> OclResult<()> {
//     if offset >= mem_len { return OclError::err(
//         "ocl::Image::enq(): Offset out of range."); }
//     if data_len > (mem_len - offset) { return OclError::err(
//         "ocl::Image::enq(): Data length exceeds image length."); }
//     Ok(())
// }

/// The type of operation to be performed by a command.
pub enum ImageCmdKind<'b, T: 'b> {
    Unspecified,
    Read { data: &'b mut [T] },
    Write { data: &'b [T] },
    Fill { color: &'b [T] },
    Copy { dst_image: &'b MemCore, dst_origin: [usize; 3] },
    CopyToBuffer { buffer: &'b MemCore, dst_offset: usize },
} 

impl<'b, T: 'b> ImageCmdKind<'b, T> {
    fn is_unspec(&'b self) -> bool {
        if let &ImageCmdKind::Unspecified = self {
            true
        } else {
            false
        }
    }
}

/// An image command builder for enqueuing reads, writes, fills, and copies.
///
/// [FIXME]: Fills not yet implemented.
pub struct ImageCmd<'b, T: 'b + OclNum> {
    queue: &'b CommandQueueCore,
    obj_core: &'b MemCore,
    block: bool,
    lock_block: bool,
    origin: [usize; 3],
    region: [usize; 3],
    // row_pitch: usize,
    // slc_pitch: usize,
    kind: ImageCmdKind<'b, T>,
    // shape: ImageCmdDataShape,    
    ewait: Option<&'b EventList>,
    enew: Option<&'b mut ClEventPtrNew>,
    mem_dims: [usize; 3],
}

impl<'b, T: 'b + OclNum> ImageCmd<'b, T> {
    /// Returns a new image command builder associated with with the
    /// memory object `obj_core` along with a default `queue` and `mem_len` 
    /// (the length of the device side image).
    pub fn new(queue: &'b CommandQueueCore, obj_core: &'b MemCore, dims: [usize; 3]) 
            -> ImageCmd<'b, T>
    {
        ImageCmd {
            queue: queue,
            obj_core: obj_core,
            block: true,
            lock_block: false,
            origin: [0, 0, 0],
            region: dims,
            kind: ImageCmdKind::Unspecified,
            // shape: ImageCmdDataShape::Lin { offset: 0 },
            ewait: None,
            enew: None,
            mem_dims: dims,
        }
    }

    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'b Queue) -> ImageCmd<'b, T> {
        self.queue = queue.core_as_ref();
        self
    }

    /// Specifies whether or not to block thread until completion.
    ///
    /// Ignored if this is a copy, fill, or copy to image operation.
    ///
    /// ## Panics
    ///
    /// Will panic if `::read` has already been called. Use `::read_async`
    /// (unsafe) for a non-blocking read operation.
    ///
    pub fn block(mut self, block: bool) -> ImageCmd<'b, T> {
        if !block && self.lock_block { 
            panic!("ocl::ImageCmd::block(): Blocking for this command has been disabled by \
                the '::read' method. For non-blocking reads use '::read_async'.");
        }
        self.block = block;
        self
    }

    /// Sets the three dimensional offset, the origin point, for an operation.
    /// 
    /// Defaults to [0, 0, 0] if not set.
    ///
    /// ## Panics
    ///
    /// The 'shape' may not have already been set to rectangular by the 
    /// `::rect` function.
    pub fn origin(mut self, origin: [usize; 3]) -> ImageCmd<'b, T> {
        self.origin = origin;
        self
    }

    /// Sets the region size for an operation.
    ///
    /// Defaults to the full region size of the image(s) as defined when first
    /// created if not set.
    ///
    /// ## Panics [TENATIVE]
    ///
    /// Panics if the region is out of range on any of the three dimensions.
    ///
    /// [FIXME]: Probably delay checking this until enq().
    ///
    pub fn region(mut self, region: [usize; 3]) -> ImageCmd<'b, T> {
        self.region = region;
        self
    }

    /// [UNSTABLE] Sets the row and slice pitch for a read or write operation.
    ///
    /// `row_pitch`: Must be greater than or equal to the region width
    /// (region[0]).
    /// 
    /// `slice_pitch: Must be greater than or equal to `row_pitch` * region
    /// height (region[1]).
    ///
    /// Only needs to be set if region has been set to something other than
    /// the (default) image buffer size.
    ///
    /// ## Stability
    ///
    /// Probably will be depricated unless I can think of a reason why you'd
    /// set the pitches to something other than the image dims.
    ///
    /// [FIXME]: Remove this or figure out if it's necessary at all.
    ///
    pub unsafe fn pitch(mut self, row_pitch: usize, slc_pitch: usize) -> ImageCmd<'b, T> {
        unimplemented!();        
    }


    /// Specifies that this command will be a blocking read operation.
    ///
    /// After calling this method, the blocking state of this command will
    /// be locked to true and a call to `::block` will cause a panic.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified.
    ///
    pub fn read(mut self, dst_data: &'b mut [T]) -> ImageCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::read(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Read { data: dst_data };
        self.block = true;
        self.lock_block = true;
        self
    }

    /// Specifies that this command will be a non-blocking, asynchronous read
    /// operation.
    ///
    /// Sets the block mode to false automatically but it may still be freely
    /// toggled back. If set back to `true` this method call becomes equivalent
    /// to calling `::read`.
    ///
    /// ## Safety
    ///
    /// Caller must ensure that the container referred to by `dst_data` lives 
    /// until the call completes.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub unsafe fn read_async(mut self, dst_data: &'b mut [T]) -> ImageCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::read(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Read { data: dst_data };
        self
    }

    /// Specifies that this command will be a write operation.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn write(mut self, src_data: &'b [T]) -> ImageCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::write(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Write { data: src_data };
        self
    }

    /// Specifies that this command will be a copy operation.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Errors
    ///
    /// If this is a rectangular copy, `dst_offset` and `len` must be zero.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy(mut self, dst_image: &'b Image, dst_origin: [usize; 3]) -> ImageCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::copy(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Copy { 
            dst_image: dst_image.core_as_ref(),
            dst_origin: dst_origin,
        }; 
        self
    }

    /// Specifies that this command will be a copy to image.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy_to_buffer(mut self, buffer: &'b MemCore, dst_offset: usize) -> ImageCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::copy_to_buffer(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::CopyToBuffer { buffer: buffer, dst_offset: dst_offset }; 
        self
    }

    /// Specifies that this command will be a fill.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn fill(mut self, color: &'b [T]) -> ImageCmd<'b, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::fill(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Fill { color: color }; 
        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait(mut self, ewait: &'b EventList) -> ImageCmd<'b, T> {
        self.ewait = Some(ewait);
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt(mut self, ewait: Option<&'b EventList>) -> ImageCmd<'b, T> {
        self.ewait = ewait;
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew(mut self, enew: &'b mut ClEventPtrNew) -> ImageCmd<'b, T> {
        self.enew = Some(enew);
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt(mut self, enew: Option<&'b mut ClEventPtrNew>) -> ImageCmd<'b, T> {
        self.enew = enew;
        self
    }

    /// Enqueues this command.
    ///
    /// TODO: FOR COPY, FILL, AND COPYTOBUFFER -- ENSURE PITCHES ARE BOTH UNSET.
    pub fn enq(self) -> OclResult<()> {
        match self.kind {
            ImageCmdKind::Read { data } => { 
                // try!(check_len(self.mem_len, data.len(), offset));

                let row_pitch = self.mem_dims[0];
                let slc_pitch = self.mem_dims[0] * self.mem_dims[1];

                unsafe { core::enqueue_read_image(self.queue, self.obj_core, self.block, 
                    self.origin, self.region, row_pitch, slc_pitch, data, self.ewait, self.enew) }
            },
            // ImageCmdKind::Write { data } => {
            //     match self.shape {
            //         ImageCmdDataShape::Lin { offset } => {
            //             try!(check_len(self.mem_len, data.len(), offset));
            //             core::enqueue_write_image(self.queue, self.obj_core, self.block, 
            //                 offset, data, self.ewait, self.enew)
            //         },
            //         ImageCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
            //                 dst_row_pitch, dst_slc_pitch } => 
            //         {
            //             // Verify dims given.
            //             // try!(Ok(()));

            //             core::enqueue_write_image_rect(self.queue, self.obj_core, 
            //                 self.block, src_origin, dst_origin, region, src_row_pitch, 
            //                 src_slc_pitch, dst_row_pitch, dst_slc_pitch, data, 
            //                 self.ewait, self.enew)
            //         }
            //     }
            // },
            // ImageCmdKind::Copy { dst_image, dst_offset, len } => {
            //     match self.shape {
            //         ImageCmdDataShape::Lin { offset } => {
            //             try!(check_len(self.mem_len, len, offset));
            //             core::enqueue_copy_image::<T, _>(self.queue, 
            //                 self.obj_core, dst_image, offset, dst_offset, len, 
            //                 self.ewait, self.enew)
            //         },
            //         ImageCmdDataShape::Rect { src_origin, dst_origin, region, src_row_pitch, src_slc_pitch,
            //                 dst_row_pitch, dst_slc_pitch } => 
            //         {
            //             // Verify dims given.
            //             // try!(Ok(()));
                        
            //             if dst_offset != 0 || len != 0 { return OclError::err(
            //                 "ocl::ImageCmd::enq(): For 'rect' shaped copies, destination \
            //                 offset and length must be zero. Ex.: \
            //                 'cmd().copy(&{{buf_name}}, 0, 0)..'.");
            //             }
            //             core::enqueue_copy_image_rect::<T, _>(self.queue, self.obj_core, dst_image,
            //             src_origin, dst_origin, region, src_row_pitch, src_slc_pitch, 
            //             dst_row_pitch, dst_slc_pitch, self.ewait, self.enew)
            //         },
            //     }
            // },
            ImageCmdKind::Unspecified => return OclError::err("ocl::ImageCmd::enq(): No operation \
                specified. Use '.read(...)', 'write(...)', etc. before calling '.enq()'."),
            _ => unimplemented!(),
        }
    }
}