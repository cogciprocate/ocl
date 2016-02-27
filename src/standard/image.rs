//! [WORK IN PROGRESS][UNTESTED] An OpenCL Image.
//!
//! TODO: Implement types for each pixel format.

#![allow(dead_code, unused_imports)]

// use std::default::Default;
use error::{Result as OclResult};
use standard::{Context, ImageBuilder};
use core::{self, OclNum, Mem as MemCore, MemFlags, ImageFormat, ImageDescriptor};


/// [WORK IN PROGRESS][UNTESTED] An Image. 
///
/// Use `::builder` for an easy way to create. [UNIMPLEMENTED]
///
pub struct Image {
    // default_val: PhantomData<T,
    obj_core: MemCore,
}

impl Image {    
    /// Returns a new `Image`.
    /// [FIXME]: Return result.
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

    pub fn builder() -> ImageBuilder {
        ImageBuilder::new()
        // ImageBuilder::new()
    }

    /// Returns the core image object pointer.
    pub fn core_as_ref(&self) -> &MemCore {
        &self.obj_core
    }
}

