//! An OpenCL Image.
//!
//! TODO: Implement types for each pixel format.

#![allow(dead_code, unused_imports)]

use std;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use error::{Error as OclError, Result as OclResult};
use standard::{self, Context, Queue, ImageBuilder, EventList, SpatialDims};
use core::{self, OclPrm, Mem as MemCore, MemFlags, MemObjectType, ImageFormat, ImageDescriptor, 
    ImageInfo, ImageInfoResult, MemInfo, MemInfoResult, CommandQueue as CommandQueueCore,
    ClEventPtrNew};



/// The type of operation to be performed by a command.
pub enum ImageCmdKind<'b, E: 'b> {
    Unspecified,
    Read { data: &'b mut [E] },
    Write { data: &'b [E] },
    Fill { color: &'b [E] },
    Copy { dst_image: &'b MemCore, dst_origin: [usize; 3] },
    CopyToBuffer { buffer: &'b MemCore, dst_origin: usize },
} 

impl<'b, E: 'b> ImageCmdKind<'b, E> {
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
pub struct ImageCmd<'b, E: 'b + OclPrm> {
    queue: &'b Queue,
    obj_core: &'b MemCore,
    block: bool,
    lock_block: bool,
    origin: [usize; 3],
    region: [usize; 3],
    // row_pitch: usize,
    // slc_pitch: usize,
    kind: ImageCmdKind<'b, E>,
    ewait: Option<&'b EventList>,
    enew: Option<&'b mut ClEventPtrNew>,
    mem_dims: [usize; 3],
}

impl<'b, E: 'b + OclPrm> ImageCmd<'b, E> {
    /// Returns a new image command builder associated with with the
    /// memory object `obj_core` along with a default `queue` and `to_len` 
    /// (the length of the device side image).
    pub fn new(queue: &'b Queue, obj_core: &'b MemCore, dims: [usize; 3]) 
            -> ImageCmd<'b, E>
    {
        ImageCmd {
            queue: queue,
            obj_core: obj_core,
            block: true,
            lock_block: false,
            origin: [0, 0, 0],
            region: dims,
            kind: ImageCmdKind::Unspecified,
            ewait: None,
            enew: None,
            mem_dims: dims,
        }
    }

    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'b Queue) -> ImageCmd<'b, E> {
        self.queue = queue;
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
    pub fn block(mut self, block: bool) -> ImageCmd<'b, E> {
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
    pub fn origin(mut self, origin: [usize; 3]) -> ImageCmd<'b, E> {
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
    pub fn region(mut self, region: [usize; 3]) -> ImageCmd<'b, E> {
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
    #[allow(unused_variables, unused_mut)]
    pub unsafe fn pitch(mut self, row_pitch: usize, slc_pitch: usize) -> ImageCmd<'b, E> {
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
    pub fn read(mut self, dst_data: &'b mut [E]) -> ImageCmd<'b, E> {
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
    pub unsafe fn read_async(mut self, dst_data: &'b mut [E]) -> ImageCmd<'b, E> {
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
    pub fn write(mut self, src_data: &'b [E]) -> ImageCmd<'b, E> {
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
    /// If this is a rectangular copy, `dst_origin` and `len` must be zero.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy(mut self, dst_image: &'b Image<E>, dst_origin: [usize; 3]) -> ImageCmd<'b, E> {
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
    pub fn copy_to_buffer(mut self, buffer: &'b MemCore, dst_origin: usize) -> ImageCmd<'b, E> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::copy_to_buffer(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::CopyToBuffer { buffer: buffer, dst_origin: dst_origin }; 
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
    pub fn fill(mut self, color: &'b [E]) -> ImageCmd<'b, E> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::fill(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Fill { color: color }; 
        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait(mut self, ewait: &'b EventList) -> ImageCmd<'b, E> {
        self.ewait = Some(ewait);
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt(mut self, ewait: Option<&'b EventList>) -> ImageCmd<'b, E> {
        self.ewait = ewait;
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew(mut self, enew: &'b mut ClEventPtrNew) -> ImageCmd<'b, E> {
        self.enew = Some(enew);
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt(mut self, enew: Option<&'b mut ClEventPtrNew>) -> ImageCmd<'b, E> {
        self.enew = enew;
        self
    }

    /// Enqueues this command.
    ///
    /// TODO: FOR COPY, FILL, AND COPYTOBUFFER -- ENSURE PITCHES ARE BOTH UNSET.
    pub fn enq(self) -> OclResult<()> {
        match self.kind {
            ImageCmdKind::Read { data } => { 
                // try!(check_len(self.to_len, data.len(), offset));

                let row_pitch = self.mem_dims[0];
                let slc_pitch = self.mem_dims[0] * self.mem_dims[1];

                unsafe { core::enqueue_read_image(self.queue, self.obj_core, self.block, 
                    self.origin, self.region, row_pitch, slc_pitch, data, self.ewait, self.enew) }
            },
            ImageCmdKind::Write { data } => {
                let row_pitch = self.mem_dims[0];
                let slc_pitch = self.mem_dims[0] * self.mem_dims[1];

                core::enqueue_write_image(self.queue, self.obj_core, self.block, 
                    self.origin, self.region, row_pitch, slc_pitch, data, self.ewait, self.enew)
            },
            ImageCmdKind::Copy { dst_image, dst_origin } => {
                core::enqueue_copy_image::<E, _>(self.queue, self.obj_core, dst_image, self.origin,
                    dst_origin, self.region, self.ewait, self.enew)
            },
            ImageCmdKind::Unspecified => return OclError::err("ocl::ImageCmd::enq(): No operation \
                specified. Use '.read(...)', 'write(...)', etc. before calling '.enq()'."),
            _ => unimplemented!(),
        }
    }
}


/// A section of device memory which represents one or many images.
///
/// Use `::builder` for an easy way to create. [UNIMPLEMENTED]
///
#[derive(Clone, Debug)]
pub struct Image<E: OclPrm> {
    obj_core: MemCore,
    queue: Queue,
    dims: SpatialDims,
    pixel_element_len: usize,
    _pixel: PhantomData<E>
}

impl<E: OclPrm> Image<E> {
    /// Returns an `ImageBuilder`. This is the recommended method to create
    /// a new `Image`.
    pub fn builder() -> ImageBuilder<E> {
        ImageBuilder::new()
    }

    /// Returns a list of supported image formats.
    pub fn supported_formats(context: &Context, flags: MemFlags, mem_obj_type: MemObjectType,
                ) -> OclResult<Vec<ImageFormat>> {
        core::get_supported_image_formats(context, flags, mem_obj_type)
    }

    /// Returns a new `Image`.
    pub fn new(queue: &Queue, flags: MemFlags, image_format: ImageFormat,
            image_desc: ImageDescriptor, image_data: Option<&[E]>) -> OclResult<Image<E>>
    {
        let obj_core = unsafe { try!(core::create_image(
            queue.context_core_as_ref(),
            flags,
            &image_format,
            &image_desc,
            image_data,
        )) };

        let pixel_element_len = match try!(core::get_image_info(&obj_core, ImageInfo::ElementSize)) {
            ImageInfoResult::ElementSize(s) => s / mem::size_of::<E>(),
            ImageInfoResult::Error(err) => return Err(*err),
            _ => return OclError::err("ocl::Image::element_len(): \
                Unexpected 'ImageInfoResult' variant."),
        };

        let dims = [image_desc.image_width, image_desc.image_height, image_desc.image_depth].into(); 

        let new_img = Image {
            obj_core: obj_core,
            queue: queue.clone(),
            dims: dims,
            pixel_element_len: pixel_element_len,
            _pixel: PhantomData,
        };

        Ok(new_img)
    }

    /// Returns an image command builder used to read, write, copy, etc.
    ///
    /// Run `.enq()` to enqueue the command.
    ///
    pub fn cmd<'b>(&'b self) -> ImageCmd<'b, E> {
        ImageCmd::new(&self.queue, &self.obj_core, 
            self.dims.to_lens().expect("ocl::Image::cmd"))
    }

    /// Returns an image command builder set to read.
    ///
    /// Run `.enq()` to enqueue the command.
    ///
    pub fn read<'b>(&'b self, data: &'b mut [E]) -> ImageCmd<'b, E> {
        self.cmd().read(data)
    }

    /// Returns an image command builder set to write.
    ///
    /// Run `.enq()` to enqueue the command.
    ///
    pub fn write<'b>(&'b self, data: &'b [E]) -> ImageCmd<'b, E> {
        self.cmd().write(data)
    }

    /// Changes the default queue.
    ///
    /// Returns a ref for chaining i.e.:
    ///
    /// `image.set_default_queue(queue).write(....);`
    ///
    /// [NOTE]: Even when used as above, the queue is changed permanently,
    /// not just for the one call. Changing the queue is cheap so feel free
    /// to change as often as needed.
    ///
    /// The new queue must be associated with a valid device.
    ///
    pub fn set_default_queue<'a>(&'a mut self, queue: &Queue) -> &'a mut Image<E> {
        // self.command_queue_obj_core = queue.core_as_ref().clone();
        self.queue = queue.clone();
        self
    }

    /// Returns a reference to the default queue.
    pub fn default_queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns this image's dimensions.
    pub fn dims(&self) -> &SpatialDims {
        &self.dims
    }

    /// Returns the total number of pixels in this image.
    pub fn pixel_count(&self) -> usize {
        self.dims.to_len()
    }

    /// Returns the length of each pixel element.
    pub fn pixel_element_len(&self) -> usize {
        self.pixel_element_len
    }

    /// Returns the total number of pixel elements in this image. Equivalent to its length.
    pub fn element_count(&self) -> usize {
        self.pixel_count() * self.pixel_element_len()
    }

    /// Get information about this image.
    pub fn info(&self, info_kind: ImageInfo) -> ImageInfoResult {
        match core::get_image_info(&self.obj_core, info_kind) {
            Ok(res) => res,
            Err(err) => ImageInfoResult::Error(Box::new(err)),
        }   
    }

    /// Returns info about this image's memory.
    pub fn mem_info(&self, info_kind: MemInfo) -> MemInfoResult {
        match core::get_mem_object_info(&self.obj_core, info_kind) {
            Ok(res) => res,
            Err(err) => MemInfoResult::Error(Box::new(err)),
        }        
    }

    /// Returns the core image object reference.
    pub fn core_as_ref(&self) -> &MemCore {
        &self.obj_core
    }

    /// Format image info.
    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("ElementSize", &self.info(ImageInfo::ElementSize))
            .field("RowPitch", &self.info(ImageInfo::RowPitch))
            .field("SlicePitch", &self.info(ImageInfo::SlicePitch))
            .field("Width", &self.info(ImageInfo::Width))
            .field("Height", &self.info(ImageInfo::Height))
            .field("Depth", &self.info(ImageInfo::Depth))
            .field("ArraySize", &self.info(ImageInfo::ArraySize))
            .field("Buffer", &self.info(ImageInfo::Buffer))
            .field("NumMipLevels", &self.info(ImageInfo::NumMipLevels))
            .field("NumSamples", &self.info(ImageInfo::NumSamples))
            .finish()
    }

    /// Format image mem info.
    fn fmt_mem_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Mem")
            .field("Type", &self.mem_info(MemInfo::Type))
            .field("Flags", &self.mem_info(MemInfo::Flags))
            .field("Size", &self.mem_info(MemInfo::Size))
            .field("HostPtr", &self.mem_info(MemInfo::HostPtr))
            .field("MapCount", &self.mem_info(MemInfo::MapCount))
            .field("ReferenceCount", &self.mem_info(MemInfo::ReferenceCount))
            .field("Context", &self.mem_info(MemInfo::Context))
            .field("AssociatedMemobject", &self.mem_info(MemInfo::AssociatedMemobject))
            .field("Offset", &self.mem_info(MemInfo::Offset))
            .finish()
    }
}

impl<E: OclPrm> std::fmt::Display for Image<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        try!(self.fmt_info(f));
        try!(write!(f, " ")); 
        self.fmt_mem_info(f)
    }
}

impl<E: OclPrm> Deref for Image<E> {
    type Target = MemCore;

    fn deref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<E: OclPrm> DerefMut for Image<E> {
    fn deref_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}



    // /// Reads from the device image buffer into `data`.
    // ///
    // /// Setting `queue` to `None` will use the default queue set during creation.
    // /// Otherwise, the queue passed will be used for this call only.
    // ///
    // /// ## Safety
    // ///
    // /// Caller must ensure that `data` lives until the read is complete. Use
    // /// the new event in `dest_list` to monitor it (use [`EventList::last_clone`]).
    // ///
    // ///
    // /// See the [SDK docs](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueReadImage.html)
    // /// for more detailed information.
    // /// [`EventList::get_clone`]: http://doc.cogciprocate.com/ocl/struct.EventList.html#method.last_clone
    // ///
    // pub unsafe fn enqueue_read(&self, queue: Option<&Queue>, block: bool, origin: [usize; 3], 
    //             region: [usize; 3], row_pitch: usize, slc_pitch: usize, data: &mut [E],
    //             wait_list: Option<&EventList>, dest_list: Option<&mut ClEventPtrNew>) -> OclResult<()>
    // {
    //     let command_queue = match queue {
    //         Some(q) => q,
    //         None => &self.queue,
    //     };

    //     core::enqueue_read_image(command_queue, &self.obj_core, block, origin, region,
    //         row_pitch, slc_pitch, data, wait_list, dest_list)
    // }

    // /// Writes from `data` to the device image buffer.
    // ///
    // /// Setting `queue` to `None` will use the default queue set during creation.
    // /// Otherwise, the queue passed will be used for this call only.
    // ///
    // /// See the [SDK docs](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueWriteImage.html)
    // /// for more detailed information.
    // pub fn enqueue_write(&self, queue: Option<&Queue>, block: bool, origin: [usize; 3], 
    //             region: [usize; 3], row_pitch: usize, slc_pitch: usize, data: &[E], 
    //             wait_list: Option<&EventList>, dest_list: Option<&mut ClEventPtrNew>) -> OclResult<()>
    // {
    //     let command_queue = match queue {
    //         Some(q) => q,
    //         None => &self.queue,
    //     };

    //     core::enqueue_write_image(command_queue, &self.obj_core, block, origin, region,
    //         row_pitch, slc_pitch, data, wait_list, dest_list)
    // }

    // /// Reads the entire device image buffer into `data`, blocking until complete.
    // ///
    // /// `data` must be equal to the size of the device image buffer and must be
    // /// alligned without pitch or offset of any kind.
    // ///
    // /// Use `::enqueue_read` for the complete range of options.
    // pub fn read_old(&self, data: &mut [E]) -> OclResult<()> {
    //     // Safe because `block = true`:
    //     unsafe { self.enqueue_read(None, true, [0, 0, 0], try!(self.dims.to_lens()), 0, 0,  data, None, None) }
    // }

    // /// Writes from `data` to the device image buffer, blocking until complete.
    // ///
    // /// `data` must be equal to the size of the device image buffer and must be
    // /// alligned without pitch or offset of any kind.
    // ///
    // /// Use `::enqueue_write` for the complete range of options.
    // pub fn write_old(&self, data: &[E]) -> OclResult<()> {
    //     self.enqueue_write(None, true, [0, 0, 0], try!(self.dims.to_lens()), 0, 0,  data, None, None)
    // }
