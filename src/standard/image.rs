//! [WORK IN PROGRESS][UNTESTED] An OpenCL Image.
//!
//! TODO: Implement types for each pixel format.

#![allow(dead_code, unused_imports)]

use std;
use error::{Result as OclResult};
use standard::{self, Context, Queue, ImageBuilder, EventList};
use core::{self, OclNum, Mem as MemCore, MemFlags, MemObjectType, ImageFormat, ImageDescriptor, 
    ImageInfo, ImageInfoResult, MemInfo, MemInfoResult, CommandQueue as CommandQueueCore};


/// [WORK IN PROGRESS][UNTESTED] An Image. 
///
/// Use `::builder` for an easy way to create. [UNIMPLEMENTED]
///
#[derive(Clone, Debug)]
pub struct Image {
    // default_val: PhantomData<T,
    obj_core: MemCore,
    queue_obj_core: CommandQueueCore,
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
        core::get_supported_image_formats(context.core_as_ref(), flags, mem_obj_type)
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

        Ok(Image {
            // default_val: T::default(),
            obj_core: obj_core,
            queue_obj_core: queue.core_as_ref().clone(),     
        })
    }

    pub fn enqueue_write<T>(&self, block: bool, origin: [usize; 3], region: [usize; 3], 
                row_pitch: usize, slc_pitch: usize, data: &[T], wait_list: Option<&EventList>,
                dest_list: Option<&mut EventList>) -> OclResult<()> {
        core::enqueue_write_image(&self.queue_obj_core, &self.obj_core, block, origin, region,
            row_pitch, slc_pitch, data, wait_list.map(|el| el.core_as_ref()), 
            dest_list.map(|el| el.core_as_mut()))
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
        f.debug_struct("Image Memory")
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
        try!(write!(f, ", ")); 
        self.fmt_mem_info(f)
    }
}


