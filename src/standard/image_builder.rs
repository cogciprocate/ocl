//! [UNTESTED][UNUSED] The builder type for `Image`.
//!
//! 

use raw;
use standard::{Image};

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
 
