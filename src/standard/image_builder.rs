//! [UNTESTED][UNUSED] The builder type for `Image`.
//!
//!

#![allow(dead_code)]

use std::default::Default;
use core::OclNum;
use standard::{Image};



/// [WORK IN PROGRESS] A builder for `Image`. 
pub struct ImageBuilder<T: OclNum> {
    stuff: T,
    flags: u64,
}

impl<T: OclNum> ImageBuilder<T> {
    pub fn new() -> ImageBuilder<T> {
        ImageBuilder { stuff: T::default(), flags: 0 }
    }


    pub fn build(self) -> Image<T> {
        unimplemented!();
        // Image::new()
    }
}
 
