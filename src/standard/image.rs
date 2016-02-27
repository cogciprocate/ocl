//! [WORK IN PROGRESS][UNTESTED] An OpenCL Image.
//!
//! TODO: Implement types for each pixel format.

#![allow(dead_code, unused_imports)]

use std;
use error::{Result as OclResult};
use standard::{self, Context, ImageBuilder};
use core::{self, OclNum, Mem as MemCore, MemFlags, MemObjectType, ImageFormat, ImageDescriptor, 
    ImageInfo, ImageInfoResult, MemInfo, MemInfoResult};


/// [WORK IN PROGRESS][UNTESTED] An Image. 
///
/// Use `::builder` for an easy way to create. [UNIMPLEMENTED]
///
#[derive(Clone, Debug)]
pub struct Image {
    // default_val: PhantomData<T,
    obj_core: MemCore,
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
    pub fn new<T>(context: &Context, flags: MemFlags, image_format: ImageFormat,
            image_desc: ImageDescriptor, image_data: Option<&[T]>) -> OclResult<Image>
    {
        // let flags = core::flag::READ_WRITE;
        // let host_ptr: cl_mem = 0 as cl_mem;

        let obj_core = try!(core::create_image(
            context.core_as_ref(),
            flags,
            &image_format,
            &image_desc,
            image_data,
        ));

        Ok(Image {
            // default_val: T::default(),
            obj_core: obj_core          
        })
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
        f.debug_struct(" Image Memory")
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
        self.fmt_mem_info(f)
        // write!(f, "{}", &self.to_string())
        // let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
        //     ("\n", "\n", "\n")
        // } else {
        //     ("{ ", ", ", " }")
        // };

        // Format = cl_h::CL_IMAGE_FORMAT as isize,
        // ElementSize = cl_h::CL_IMAGE_ELEMENT_SIZE as isize,
        // RowPitch = cl_h::CL_IMAGE_ROW_PITCH as isize,
        // SlicePitch = cl_h::CL_IMAGE_SLICE_PITCH as isize,
        // Width = cl_h::CL_IMAGE_WIDTH as isize,
        // Height = cl_h::CL_IMAGE_HEIGHT as isize,
        // Depth = cl_h::CL_IMAGE_DEPTH as isize,
        // ArraySize = cl_h::CL_IMAGE_ARRAY_SIZE as isize,
        // Buffer = cl_h::CL_IMAGE_BUFFER as isize,
        // NumMipLevels = cl_h::CL_IMAGE_NUM_MIP_LEVELS as isize,
        // NumSamples = cl_h::CL_IMAGE_NUM_SAMPLES as isize,

        // try!(write!(f, "[Image]: {b}\
        //         ElementSize: {}{d}\
        //         RowPitch: {}{d}\
        //         SlicePitch: {}{d}\
        //         Width: {}{d}\
        //         Height: {}{d}\
        //         Depth: {}{d}\
        //         ArraySize: {}{d}\
        //         Buffer: {}{d}\
        //         NumMipLevels: {}{d}\
        //         NumSamples: {}{e}\
        //     ",
        //     self.info(ImageInfo::ElementSize),
        //     self.info(ImageInfo::RowPitch),
        //     self.info(ImageInfo::SlicePitch),
        //     self.info(ImageInfo::Width),
        //     self.info(ImageInfo::Height),
        //     self.info(ImageInfo::Depth),
        //     self.info(ImageInfo::ArraySize),
        //     self.info(ImageInfo::Buffer),
        //     self.info(ImageInfo::NumMipLevels),
        //     self.info(ImageInfo::NumSamples),
        //     b = begin,
        //     d = delim,
        //     e = end,
        // ));

        // try!(f.debug_struct("Image")
        //     .field("ElementSize", &self.info(ImageInfo::ElementSize))
        //     .field("RowPitch", &self.info(ImageInfo::RowPitch))
        //     .field("SlicePitch", &self.info(ImageInfo::SlicePitch))
        //     .field("Width", &self.info(ImageInfo::Width))
        //     .field("Height", &self.info(ImageInfo::Height))
        //     .field("Depth", &self.info(ImageInfo::Depth))
        //     .field("ArraySize", &self.info(ImageInfo::ArraySize))
        //     .field("Buffer", &self.info(ImageInfo::Buffer))
        //     .field("NumMipLevels", &self.info(ImageInfo::NumMipLevels))
        //     .field("NumSamples", &self.info(ImageInfo::NumSamples))
        //     .finish());


        // write!(f, "  [Image Memory]: {b}\
        //         Type: {}{d}\
        //         Flags: {}{d}\
        //         Size: {}{d}\
        //         HostPtr: {}{d}\
        //         MapCount: {}{d}\
        //         ReferenceCount: {}{d}\
        //         Context: {}{d}\
        //         AssociatedMemobject: {}{d}\
        //         Offset: {}{e}\
        //     ",
        //     self.mem_info(MemInfo::Type),
        //     self.mem_info(MemInfo::Flags),
        //     self.mem_info(MemInfo::Size),
        //     self.mem_info(MemInfo::HostPtr),
        //     self.mem_info(MemInfo::MapCount),
        //     self.mem_info(MemInfo::ReferenceCount),
        //     self.mem_info(MemInfo::Context),
        //     self.mem_info(MemInfo::AssociatedMemobject),
        //     self.mem_info(MemInfo::Offset),
        //     b = begin,
        //     d = delim,
        //     e = end,
        // )

        // f.debug_struct(" Image Memory")
        //     .field("Type", &self.mem_info(MemInfo::Type))
        //     .field("Flags", &self.mem_info(MemInfo::Flags))
        //     .field("Size", &self.mem_info(MemInfo::Size))
        //     .field("HostPtr", &self.mem_info(MemInfo::HostPtr))
        //     .field("MapCount", &self.mem_info(MemInfo::MapCount))
        //     .field("ReferenceCount", &self.mem_info(MemInfo::ReferenceCount))
        //     .field("Context", &self.mem_info(MemInfo::Context))
        //     .field("AssociatedMemobject", &self.mem_info(MemInfo::AssociatedMemobject))
        //     .field("Offset", &self.mem_info(MemInfo::Offset))
        //     .finish()
    }
}


