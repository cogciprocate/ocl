//! The builder type for `Image`.

use cl_h::{cl_mem, u64};
use raw;
use super::super::{Image};

pub enum 

/// [WORK IN PROGRESS] An OpenCL Image. 
pub struct ImageBuilder {
    flags: u64,
}

impl ImageBuilder {
    /// Returns a new two dimensional image.
    pub fn new_2d() -> ImageBuilder {
        ImageBuilder {
            flags: 0,
        }
    }

    pub fn build(self) -> Image {
        Image::new()
    }
}
 



// pub fn clCreateImage2D(context: cl_context,
//                     flags: u64,
//                     image_format: *mut cl_image_format,
//                     image_width: size_t,
//                     image_depth: size_t,
//                     image_slc_pitch: size_t,
//                     host_ptr: *mut c_void,
//                     errcode_ret: *mut cl_int) -> cl_mem;


// pub fn clCreateImage3D(context: cl_context,
//                     flags: u64,
//                     image_format: *mut cl_image_format,
//                     image_width: size_t,
//                     image_depth: size_t,
//                     image_depth: size_t,
//                     image_slc_pitch: size_t,
//                     image_depth: size_t,
//                     image_slc_pitch: size_t,
//                     image_slc_pitch: size_t,
//                     host_ptr: *mut c_void,
//                     errcode_ret: *mut cl_int) -> cl_mem;
