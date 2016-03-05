//! An OpenCL Image.
//!
//! TODO: Implement types for each pixel format.

#![allow(dead_code, unused_imports)]

use std;
use std::ops::{Deref, DerefMut};
use error::{Result as OclResult};
use standard::{self, Context, Queue, ImageBuilder, EventList};
use core::{self, OclNum, Mem as MemCore, MemFlags, MemObjectType, ImageFormat, ImageDescriptor, 
    ImageInfo, ImageInfoResult, MemInfo, MemInfoResult, CommandQueue as CommandQueueCore,
    ClEventPtrNew};


/// A section of device memory which represents one or many images.
///
/// Use `::builder` for an easy way to create. [UNIMPLEMENTED]
///
#[derive(Clone, Debug)]
pub struct Image {
    // default_val: PhantomData<T,
    obj_core: MemCore,
    command_queue_obj_core: CommandQueueCore,
    dims: [usize; 3],
    pixel_bytes: usize,
}

impl Image {
    /// Returns an `ImageBuilder`. This is the recommended method to create
    // a new `Image`.
    pub fn builder() -> ImageBuilder {
        ImageBuilder::new()
        // ImageBuilder::new()
    }

    /// Returns a list of supported image formats.
    pub fn supported_formats(context: &Context, flags: MemFlags, mem_obj_type: MemObjectType,
                ) -> OclResult<Vec<ImageFormat>> {
        core::get_supported_image_formats(context, flags, mem_obj_type)
    }

    /// Returns a new `Image`.
    pub fn new<T>(queue: &Queue, flags: MemFlags, image_format: ImageFormat,
            image_desc: ImageDescriptor, image_data: Option<&[T]>) -> OclResult<Image>
    {
        // let flags = core::flag::READ_WRITE;
        // let host_ptr: cl_mem = 0 as cl_mem;

        let obj_core = try!(core::create_image(
            queue.context_core_as_ref(),
            flags,
            &image_format,
            &image_desc,
            image_data,
        ));

        let dims = [image_desc.image_width, image_desc.image_height, image_desc.image_depth]; 

        Ok(Image {
            // default_val: T::default(),
            obj_core: obj_core,
            command_queue_obj_core: queue.core_as_ref().clone(),
            dims: dims,
            pixel_bytes: 4,
        })
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
    pub unsafe fn enqueue_read<T>(&self, queue: Option<&Queue>, block: bool, origin: [usize; 3], 
                region: [usize; 3], row_pitch: usize, slc_pitch: usize, data: &mut [T],
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
    pub fn enqueue_write<T>(&self, queue: Option<&Queue>, block: bool, origin: [usize; 3], 
                region: [usize; 3], row_pitch: usize, slc_pitch: usize, data: &[T], 
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
    pub fn read<T>(&self, data: &mut [T]) -> OclResult<()> {
        // Safe because `block = true`:
        unsafe { self.enqueue_read(None, true, [0, 0, 0], self.dims.clone(), 0, 0,  data, None, None) }
    }

    /// Writes from `data` to the device image buffer, blocking until complete.
    ///
    /// `data` must be equal to the size of the device image buffer and must be
    /// alligned without pitch or offset of any kind.
    ///
    /// Use `::enqueue_write` for the complete range of options.
    pub fn write<T>(&self, data: &[T]) -> OclResult<()> {
        self.enqueue_write(None, true, [0, 0, 0], self.dims.clone(), 0, 0,  data, None, None)
    }

    /// Returns the core image object pointer.
    pub fn core_as_ref(&self) -> &MemCore {
        &self.obj_core
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
    pub fn set_queue<'a>(&'a mut self, queue: &Queue) -> &'a mut Image {
        self.command_queue_obj_core = queue.core_as_ref().clone();
        self
    }

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



impl std::fmt::Display for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        try!(self.fmt_info(f));
        try!(write!(f, " ")); 
        self.fmt_mem_info(f)
    }
}


impl Deref for Image {
    type Target = MemCore;

    fn deref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl DerefMut for Image {
    fn deref_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}
