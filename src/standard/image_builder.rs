//! [UNTESTED][UNUSED] The builder type for `Image`.
//!
//!

#![allow(dead_code, unused_imports)]

// use std::default::Default;
// use core::OclNum;
use error::{Result as OclResult};
use standard::{Context, Queue, Image};
use core::{self, OclNum, Mem as MemCore, MemFlags, ImageFormat, ImageDescriptor, MemObjectType};



/// [WORK IN PROGRESS] A builder for `Image`. 
pub struct ImageBuilder {
    flags: MemFlags,
    image_format: ImageFormat,
    image_desc: ImageDescriptor, 
    // image_data: Option<&'a [T]>,
}

impl ImageBuilder {
	/// Returns a new `ImageBuilder` with very basic defaults.
	///
	/// # Defaults
	///
	/// ocl::MEM_READ_WRITE
	///
	///	ocl::ImageFormat {
    ///    channel_order: ocl::ImageChannelOrder::Rgba,
    ///    channel_data_type: ocl::ImageChannelDataType::SnormInt8,
    /// }
    ///
    /// ocl::ImageDescriptor::new(ocl::MemObjectType::Image2d, 100, 100, 1)
    ///
    /// # Reference
    ///
    /// See the [official SDK docs](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateImage.html)
    /// for more information about what the parameters do.
	/// 
    pub fn new() -> ImageBuilder {
        ImageBuilder { 
        	flags: core::MEM_READ_WRITE,
        	image_format: ImageFormat::new_rgba(),
        	image_desc: ImageDescriptor::new(MemObjectType::Image2d, 200, 200, 1),
        	// image_data: None,
        }
    }

    /// Builds with no host side image data memory specified and returns a 
	/// new `Image`.
    pub fn build(&self, queue: &Queue) -> OclResult<Image> {
        Image::new::<usize>(queue, self.flags, self.image_format.clone(), self.image_desc.clone(), 
        	None)
    }

    /// Builds with the host side image data memory specified by `image_data`
    /// and returns a new `Image`.
    pub fn build_with_data<T>(&self, queue: &Queue, image_data: &[T]) -> OclResult<Image> {
        Image::new(queue, self.flags, self.image_format.clone(), self.image_desc.clone(), 
        	Some(image_data))
    }

    /// Sets the flags for the memory to be created.
    ///
    /// Setting this overwrites any previously set flags. To combine them,
    /// use the bitwise or operator (`|`), for example:
    ///
    /// ```notest
    /// ocl::Image::builder().flags(ocl::MEM_WRITE_ONLY | ocl::MEM_COPY_HOST_PTR)...
    /// ```
    ///
    /// Defaults to `core::MEM_READ_WRITE` if not set.
    pub fn flags<'a>(&'a mut self, flags: MemFlags) -> &'a mut ImageBuilder {
    	self.flags = flags;
    	self
	}

	/// Specifies the image pixel format.
	///
	/// If unspecified, defaults to: 
	///
	/// ```notest
	///	ImageFormat {
    ///    channel_order: ImageChannelOrder::Rgba,
    ///    channel_data_type: ImageChannelDataType::SnormInt8,
    /// }
    /// ```
	pub fn image_format<'a>(&'a mut self, image_format: ImageFormat) -> &'a mut ImageBuilder {
		self.image_format = image_format;
		self
	}

	/// Specifies the image descriptor containing a number of important settings.
	///
	/// If unspecified (not recommended), defaults to: 
	///
	/// ```notest
	/// ImageDescriptor {
	///     image_type: MemObjectType::Image2d,
	///     image_width: 100,
	///     image_height: 100,
	///     image_depth: 1,
	///     image_array_size: 0,
	///     image_row_pitch: 0,
	///     image_slice_pitch: 0,
	///     num_mip_levels: 0,
	///     num_samples: 0,
	///     buffer: None,
	/// }
    /// ```
    ///
    /// If you are unsure, just set the first four by using 
    /// `ImageDescriptor::new`. Ex.:
    /// 
    /// ```notest
    /// ocl::Image::builder()
    ///	   .image_desc(ocl::ImageDescriptor::new(
    ///	      ocl::MemObjectType::Image2d, 1280, 800, 1))
	///	   ...
	///	   ...
	///	   .build()
    /// ```
    /// 
    /// 
	pub fn image_desc<'a>(&'a mut self, image_desc: ImageDescriptor) -> &'a mut ImageBuilder {
		self.image_desc = image_desc;
		self
	}
}
 
