//! [UNTESTED][UNUSED] The builder type for `Image`.
//!
//!

#![allow(dead_code, unused_imports)]

// use std::default::Default;
// use core::OclNum;
use standard::{Image};



/// [WORK IN PROGRESS] A builder for `Image`. 
pub struct ImageBuilder {
    flags: u64,
}

impl ImageBuilder {
    pub fn new() -> ImageBuilder {
        ImageBuilder { flags: 0 }
    }


    pub fn build(self) -> Image {
        unimplemented!();
        // Image::new()
    }
}
 
