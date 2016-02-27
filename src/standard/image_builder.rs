//! [UNTESTED][UNUSED] The builder type for `Image`.
//!
//!

#![allow(dead_code, unused_imports)]

// use std::default::Default;
// use core::OclNum;
use error::{Result as OclResult};
use standard::{Context, Image};
use core::{self, OclNum, Mem as MemCore, MemFlags, ImageFormat, ImageDescriptor, MemObjectType};



/// [WORK IN PROGRESS] A builder for `Image`. 
pub struct ImageBuilder<'a, T: 'a> {
    flags: MemFlags,
    image_format: ImageFormat,
    image_desc: ImageDescriptor, 
    image_data: Option<&'a [T]>,
}

impl<'a, T: 'a> ImageBuilder<'a, T> {
    pub fn new() -> ImageBuilder<'a, T> {
        ImageBuilder { 
        	flags: core::MEM_READ_WRITE,
        	image_format: ImageFormat::new_rgba(),
        	image_desc: ImageDescriptor::new(MemObjectType::Image2d, 200, 200, 1),
        	image_data: None,
        }
    }

    pub fn build(&self, context: &Context) -> OclResult<Image> {
        Image::new(context, self.flags, self.image_format.clone(), self.image_desc.clone(), 
        	self.image_data.clone())
    }
}
 
