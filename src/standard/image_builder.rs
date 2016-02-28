//! The builder type for `Image`.
//!
//!

#![allow(dead_code, unused_imports)]

// use std::default::Default;
// use core::OclNum;
use error::{Result as OclResult};
use standard::{Context, Queue, Image, SimpleDims};
use core::{self, OclNum, Mem as MemCore, MemFlags, ImageFormat, ImageDescriptor, MemObjectType,
    ImageChannelOrder, ImageChannelDataType};



/// A builder for `Image`. 
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
	/// Flags: 
    /// ```notest
    /// ocl::MEM_READ_WRITE
	/// ```
    ///
    /// Image Format:
	///	```notest
    /// ocl::ImageFormat {
    ///    channel_order: ocl::ImageChannelOrder::Rgba,
    ///    channel_data_type: ocl::ImageChannelDataType::SnormInt8,
    /// }
    /// ```
    ///
    /// Descriptor (stores everything else - width, height, pitch, etc.):
    /// ```notest
    /// ImageDescriptor::new(MemObjectType::Image1d, 0, 0, 0, 0, 0, 0, None)
    /// ```
    ///
    /// # Reference
    ///
    /// See the [official SDK documentation] for more information.
    ///
    /// Some descriptions here are adapted from various SDK docs.
	/// 
    /// [official SDK docs]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateImage.html
    pub fn new() -> ImageBuilder {
        ImageBuilder { 
        	flags: core::MEM_READ_WRITE,
        	image_format: ImageFormat::new_rgba(),
        	image_desc: ImageDescriptor::new(MemObjectType::Image1d, 0, 0, 0, 0, 0, 0, None),
        	// image_data: None,
        }
    }

    /// Builds with no host side image data memory specified and returns a 
	/// new `Image`.
    pub fn build(&self, queue: &Queue) -> OclResult<Image> {
        Image::new::<usize>(queue, self.flags, self.image_format.clone(), self.image_desc.clone(), 
        	None)
    }

    /// Builds with the host side image data specified by `image_data`
    /// and returns a new `Image`.
    ///
    /// Useful with the `ocl::MEM_COPY_HOST_PTR` flag for initializing device
    /// memory by copying the contents of `image_data`.
    ///
    /// Also used with the `ocl::MEM_USE_HOST_PTR` and `ocl::ALLOC_HOST_PTR`
    /// flags. See the [official SDK docs] for more info.
    ///
    /// [official SDK docs]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateImage.html
    pub fn build_with_data<T>(&self, queue: &Queue, image_data: &[T]) -> OclResult<Image> {
        Image::new(queue, self.flags, self.image_format.clone(), self.image_desc.clone(), 
        	Some(image_data))
    }

    pub fn channel_order<'a>(&'a mut self, order: ImageChannelOrder) -> &'a mut ImageBuilder {
        self.image_format.channel_order = order;
        self
    }

    pub fn channel_data_type<'a>(&'a mut self, data_type: ImageChannelDataType) -> &'a mut ImageBuilder {
        self.image_format.channel_data_type = data_type;
        self
    }


    /// Sets the type of image (technically the type of memory buffer).
    ///
	/// Describes the image type and must be either `Image1d`, `Image1dBuffer`,
	/// `Image1dArray`, `Image2d`, `Image2dArray`, or `Image3d`.
	///
    pub fn image_type<'a>(&'a mut self, image_type: MemObjectType) -> &'a mut ImageBuilder {
    	self.image_desc.image_type = image_type;
    	self
    }

    /// The width, height, and depth of an image or image array:
    ///
    /// Some notes adapted from SDK docs:
    ///
    /// ##### Width
    ///
	/// The width of the image in pixels. For a 2D image and image array, the
	/// image width must be ≤ `DeviceInfo::Image2dMaxWidth`. For a 3D image, the
	/// image width must be ≤ `DeviceInfo::Image3dMaxWidth`. For a 1D image buffer,
	/// the image width must be ≤ `DeviceInfo::ImageMaxBufferSize`. For a 1D image
	/// and 1D image array, the image width must be ≤ `DeviceInfo::Image2dMaxWidth`.
	///
	/// ##### Height
	///
	/// The height of the image in pixels. This is only used if the
	/// image is a 2D, 3D or 2D image array. For a 2D image or image array, the
	/// image height must be ≤ `DeviceInfo::Image2dMaxHeight`. For a 3D image, the
	/// image height must be ≤ `DeviceInfo::Image3dMaxHeight`.
	///
	/// ##### Depth
	///
	/// image_depth The depth of the image in pixels. This is only used if the
	/// image is a 3D image and must be a value ≥ 1 and ≤
	/// `DeviceInfo::Image3dMaxDepth`.
	///
	/// #### Examples
	///
	/// * To set the dimensions of a 2d image use:
	///   `SimpleDims::Two(width, height)`.
	/// * To set the dimensions of a 2d image array use:
	///   `SimpleDims::Three(width, height, array_length)`.
	/// * To set the dimensions of a 3d image use:
	///   `SimpleDims::Three(width, height, depth)`.
	///
    pub fn dims<'a>(&'a mut self, dims: &SimpleDims) -> &'a mut ImageBuilder {
    	let size = dims.to_size();
    	self.image_desc.image_width = size[0];
    	self.image_desc.image_height = size[1];
    	self.image_desc.image_depth = size[2];
    	self
	}

	/// Image array size.
	///
	/// The number of images in the image array. This is only used if the image is
	/// a 1D or 2D image array. The values for image_array_size, if specified,
	/// must be a value ≥ 1 and ≤ `DeviceInfo::ImageMaxArraySize`.
	///
	/// Note that reading and writing 2D image arrays from a kernel with
	/// image_array_size = 1 may be lower performance than 2D images.
	///
    pub fn array_size<'a>(&'a mut self, array_size: usize) -> &'a mut ImageBuilder {
    	self.image_desc.image_array_size = array_size;
    	self
    }

    /// Image row pitch.
    ///
    /// The scan-line pitch in bytes. This must be 0 if host data is `None` and
    /// can be either 0 or ≥ image_width * size of element in bytes if host data
    /// is not `None`. If host data is not `None` and image_row_pitch = 0,
    /// image_row_pitch is calculated as image_width * size of element in bytes.
    /// If image_row_pitch is not 0, it must be a multiple of the image element
    /// size in bytes.
    ///
    pub fn row_pitch<'a>(&'a mut self, row_pitch: usize) -> &'a mut ImageBuilder {
    	self.image_desc.image_row_pitch = row_pitch;
    	self
    }


    /// Image slice pitch.
    ///
    /// The size in bytes of each 2D slice in the 3D image or the size in bytes of
    /// each image in a 1D or 2D image array. This must be 0 if host data is
    /// `None`. If host data is not `None`, image_slice_pitch can be either 0 or ≥
    /// image_row_pitch * image_height for a 2D image array or 3D image and can be
    /// either 0 or ≥ image_row_pitch for a 1D image array. If host data is not
    /// `None` and image_slice_pitch = 0, image_slice_pitch is calculated as
    /// image_row_pitch * image_height for a 2D image array or 3D image and
    /// image_row_pitch for a 1D image array. If image_slice_pitch is not 0, it
    /// must be a multiple of the image_row_pitch.
    ///
    pub fn slc_pitch<'a>(&'a mut self, slc_pitch: usize) -> &'a mut ImageBuilder {
    	self.image_desc.image_slice_pitch = slc_pitch;
        self
    }

    /// Buffer synchronization.
    ///
    /// Refers to a valid buffer memory object if image_type is
    /// `MemObjectType::Image1dBuffer`. Otherwise it must be `None` (default).
    /// For a 1D image buffer object, the image pixels are taken from the buffer
    /// object's data store. When the contents of a buffer object's data store are
    /// modified, those changes are reflected in the contents of the 1D image
    /// buffer object and vice-versa at corresponding sychronization points. The
    /// image_width * size of element in bytes must be ≤ size of buffer object
    /// data store.
    ///
	pub fn buffer_sync<'a>(&'a mut self, buffer: MemCore) -> &'a mut ImageBuilder {
        self.image_desc.buffer = Some(buffer);
        self
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
    /// ImageFormat {
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
    ///     image_type: MemObjectType::Image1d,
    ///     image_width: 0,
    ///     image_height: 0,
    ///     image_depth: 0,
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
    ///    .image_desc(ocl::ImageDescriptor::new(
    ///       ocl::MemObjectType::Image2d, 1280, 800, 1))
    ///    ...
    ///    ...
    ///    .build()
    /// ```
    /// 
    /// Setting this overwrites any previously set type, dimensions, array size, pitch, etc.
    /// 
    pub unsafe fn image_desc<'a>(&'a mut self, image_desc: ImageDescriptor) -> &'a mut ImageBuilder {
        self.image_desc = image_desc;
        self
    }
}
 
