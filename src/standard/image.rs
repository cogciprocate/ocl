//! An OpenCL Image.
//!
//! TODO: Implement types for each pixel format.

#![allow(dead_code, unused_imports)]

use std;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use error::{Error as OclError, Result as OclResult};
use standard::{self, Context, Queue, ImageBuilder, EventList, SpatialDims, ImageCmd};
use core::{self, OclPrm, Mem as MemCore, MemFlags, MemObjectType, ImageFormat, ImageDescriptor, 
    ImageInfo, ImageInfoResult, MemInfo, MemInfoResult, CommandQueue as CommandQueueCore,
    ClEventPtrNew};


/// A section of device memory which represents one or many images.
///
/// Use `::builder` for an easy way to create. [UNIMPLEMENTED]
///
#[derive(Clone, Debug)]
pub struct Image<S: OclPrm> {
    obj_core: MemCore,
    command_queue_obj_core: CommandQueueCore,
    dims: SpatialDims,
    pixel_elements: usize,
    _pixel: PhantomData<S>
}

impl<S: OclPrm> Image<S> {
    /// Returns an `ImageBuilder`. This is the recommended method to create
    // a new `Image`.
    pub fn builder() -> ImageBuilder<S> {
        ImageBuilder::new()
    }

    /// Returns a list of supported image formats.
    pub fn supported_formats(context: &Context, flags: MemFlags, mem_obj_type: MemObjectType,
                ) -> OclResult<Vec<ImageFormat>> {
        core::get_supported_image_formats(context, flags, mem_obj_type)
    }

    /// Returns a new `Image`.
    pub fn new(queue: &Queue, flags: MemFlags, image_format: ImageFormat,
            image_desc: ImageDescriptor, image_data: Option<&[S]>) -> OclResult<Image<S>>
    {
        let obj_core = unsafe { try!(core::create_image(
            queue.context_core_as_ref(),
            flags,
            &image_format,
            &image_desc,
            image_data,
        )) };

        let pixel_elements = match try!(core::get_image_info(&obj_core, ImageInfo::ElementSize)) {
            ImageInfoResult::ElementSize(s) => s / mem::size_of::<S>(),
            ImageInfoResult::Error(err) => return Err(*err),
            _ => return OclError::err("ocl::Image::element_len(): \
                Unexpected 'ImageInfoResult' variant."),
        };

        let dims = [image_desc.image_width, image_desc.image_height, image_desc.image_depth].into(); 

        let new_img = Image {
            obj_core: obj_core,
            command_queue_obj_core: queue.core_as_ref().clone(),
            dims: dims,
            pixel_elements: pixel_elements,
            _pixel: PhantomData,
        };

        Ok(new_img)
    }

    /// Returns an image command builder used to read, write, copy, etc.
    ///
    /// Run `.enq()` to enqueue the command.
    ///
    pub fn cmd<'b>(&'b self) -> ImageCmd<'b, S> {
        ImageCmd::new(&self.command_queue_obj_core, &self.obj_core, 
            self.dims.to_lens().expect("ocl::Image::cmd"))
    }

    /// Reads from the device image buffer into `data`.
    ///
    /// Setting `queue` to `None` will use the default queue set during creation.
    /// Otherwise, the queue passed will be used for this call only.
    ///
    /// ## Safety
    ///
    /// Caller must ensure that `data` lives until the read is complete. Use
    /// the new event in `dest_list` to monitor it (use [`EventList::last_clone`]).
    ///
    ///
    /// See the [SDK docs](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueReadImage.html)
    /// for more detailed information.
    /// [`EventList::get_clone`]: http://doc.cogciprocate.com/ocl/struct.EventList.html#method.last_clone
    ///
    pub unsafe fn enqueue_read(&self, queue: Option<&Queue>, block: bool, origin: [usize; 3], 
                region: [usize; 3], row_pitch: usize, slc_pitch: usize, data: &mut [S],
                wait_list: Option<&EventList>, dest_list: Option<&mut ClEventPtrNew>) -> OclResult<()>
    {
        let command_queue = match queue {
            Some(q) => q.core_as_ref(),
            None => &self.command_queue_obj_core,
        };

        core::enqueue_read_image(command_queue, &self.obj_core, block, origin, region,
            row_pitch, slc_pitch, data, wait_list, dest_list)
    }

    /// Writes from `data` to the device image buffer.
    ///
    /// Setting `queue` to `None` will use the default queue set during creation.
    /// Otherwise, the queue passed will be used for this call only.
    ///
    /// See the [SDK docs](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueWriteImage.html)
    /// for more detailed information.
    pub fn enqueue_write(&self, queue: Option<&Queue>, block: bool, origin: [usize; 3], 
                region: [usize; 3], row_pitch: usize, slc_pitch: usize, data: &[S], 
                wait_list: Option<&EventList>, dest_list: Option<&mut ClEventPtrNew>) -> OclResult<()>
    {
        let command_queue = match queue {
            Some(q) => q.core_as_ref(),
            None => &self.command_queue_obj_core,
        };

        core::enqueue_write_image(command_queue, &self.obj_core, block, origin, region,
            row_pitch, slc_pitch, data, wait_list, dest_list)
    }

    /// Reads the entire device image buffer into `data`, blocking until complete.
    ///
    /// `data` must be equal to the size of the device image buffer and must be
    /// alligned without pitch or offset of any kind.
    ///
    /// Use `::enqueue_read` for the complete range of options.
    pub fn read(&self, data: &mut [S]) -> OclResult<()> {
        // Safe because `block = true`:
        unsafe { self.enqueue_read(None, true, [0, 0, 0], try!(self.dims.to_lens()), 0, 0,  data, None, None) }
    }

    /// Writes from `data` to the device image buffer, blocking until complete.
    ///
    /// `data` must be equal to the size of the device image buffer and must be
    /// alligned without pitch or offset of any kind.
    ///
    /// Use `::enqueue_write` for the complete range of options.
    pub fn write(&self, data: &[S]) -> OclResult<()> {
        self.enqueue_write(None, true, [0, 0, 0], try!(self.dims.to_lens()), 0, 0,  data, None, None)
    }

    /// Returns the core image object pointer.
    pub fn core_as_ref(&self) -> &MemCore {
        &self.obj_core
    }

    /// Returns the length of an element.
    pub fn pixel_elements(&self) -> usize {
        // match self.info(ImageInfo::ElementSize) {
        //     ImageInfoResult::ElementSize(s) => Ok(s / mem::size_of::<S>()),
        //     // ImageInfoResult::Error(err) => panic!("ocl::Image::element_len: {}", err.description()),
        //     // _ => panic!("ocl::Image::element_len: Unexpected 'ImageInfoResult' variant."),
        //     ImageInfoResult::Error(err) => Err(*err),
        //     _ => OclError::err("ocl::Image::element_len(): Unexpected 'ImageInfoResult' variant."),
        // }
        self.pixel_elements
    }

    /// Changes the default queue.
    ///
    /// Returns a ref for chaining i.e.:
    ///
    /// `image.set_queue(queue).write(....);`
    ///
    /// [NOTE]: Even when used as above, the queue is changed permanently,
    /// not just for the one call. Changing the queue is cheap so feel free
    /// to change as often as needed.
    ///
    /// The new queue must be associated with a valid device.
    ///
    pub fn set_queue<'a>(&'a mut self, queue: &Queue) -> &'a mut Image<S> {
        self.command_queue_obj_core = queue.core_as_ref().clone();
        self
    }

    /// Returns this image's dimensions.
    pub fn dims(&self) -> &SpatialDims {
        &self.dims
    }

    pub fn pixel_count(&self) -> usize {
        self.dims.to_len()
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



impl<S: OclPrm> std::fmt::Display for Image<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        try!(self.fmt_info(f));
        try!(write!(f, " ")); 
        self.fmt_mem_info(f)
    }
}


impl<S: OclPrm> Deref for Image<S> {
    type Target = MemCore;

    fn deref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<S: OclPrm> DerefMut for Image<S> {
    fn deref_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}
