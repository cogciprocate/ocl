//! Enums, structs, and bitfields related to images.
// use libc::c_void;
use cl_h::{self, cl_mem};
use raw::MemObjectType;

/// A structure that describes format properties of the image to be allocated. (from SDK)
///
/// Image format properties used by `Image`.
///
/// # Examples (from SDK)
///
/// To specify a normalized unsigned 8-bit / channel RGBA image:
///    image_channel_order = CL_RGBA
///    image_channel_data_type = CL_UNORM_INT8
///
/// image_channel_data_type values of CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 and CL_UNORM_INT_101010 are special cases of packed image formats where the channels of each element are packed into a single unsigned short or unsigned int. For these special packed image formats, the channels are normally packed with the first channel in the most significant bits of the bitfield, and successive channels occupying progressively less significant locations. For CL_UNORM_SHORT_565, R is in bits 15:11, G is in bits 10:5 and B is in bits 4:0. For CL_UNORM_SHORT_555, bit 15 is undefined, R is in bits 14:10, G in bits 9:5 and B in bits 4:0. For CL_UNORM_INT_101010, bits 31:30 are undefined, R is in bits 29:20, G in bits 19:10 and B in bits 9:0.
/// OpenCL implementations must maintain the minimum precision specified by the number of bits in image_channel_data_type. If the image format specified by image_channel_order, and image_channel_data_type cannot be supported by the OpenCL implementation, then the call to clCreateImage will return a NULL memory object.
///
pub struct ImageFormat {
    pub channel_order: ImageChannelOrder,
    pub channel_data_type: ImageChannelDataType,
}

impl ImageFormat {
    pub fn new_rgba() -> ImageFormat {
        ImageFormat {
            channel_order: ImageChannelOrder::Rgba,
            channel_data_type: ImageChannelDataType::SnormInt8,
        }
    }

    pub fn as_raw(&self) -> cl_h::cl_image_format {
        cl_h::cl_image_format {
            image_channel_order: self.channel_order as cl_h::cl_channel_order,
            image_channel_data_type: self.channel_data_type as cl_h::cl_channel_type,
        }
    }
}



/// Specifies the number of channels and the channel layout i.e. the memory layout in which channels are stored in the image. Valid values are described in the table below. (from SDK)
/// [FIXME]: Move to raw/enums
#[derive(Clone, Copy)]
pub enum ImageChannelOrder {
    R = cl_h::CL_R as isize,
    A = cl_h::CL_A as isize,
    Rg = cl_h::CL_RG as isize,
    Ra = cl_h::CL_RA as isize,
    /// This format can only be used if channel data type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010:
    Rgb = cl_h::CL_RGB as isize,
    Rgba = cl_h::CL_RGBA as isize,
    /// This format can only be used if channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8:
    Bgra = cl_h::CL_BGRA as isize,
    /// This format can only be used if channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8:
    Argb = cl_h::CL_ARGB as isize,
    /// This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT:
    Intensity = cl_h::CL_INTENSITY as isize,
    /// This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT:
    Luminance = cl_h::CL_LUMINANCE as isize,
    Rx = cl_h::CL_Rx as isize,
    Rgx = cl_h::CL_RGx as isize,
    /// This format can only be used if channel data type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010:
    Rgbx = cl_h::CL_RGBx as isize,
    Depth = cl_h::CL_DEPTH as isize,
    DepthStencil = cl_h::CL_DEPTH_STENCIL as isize,
}

/// Describes the size of the channel data type. The number of bits per element determined by the image_channel_data_type and image_channel_order must be a power of two. The list of supported values is described in the table below. (from SDK)
/// [FIXME]: Move to raw/enums
#[derive(Clone, Copy)]
pub enum ImageChannelDataType {
    /// Each channel component is a normalized signed 8-bit integer value:
    SnormInt8 = cl_h::CL_SNORM_INT8 as isize,
    /// Each channel component is a normalized signed 16-bit integer value:
    SnormInt16 = cl_h::CL_SNORM_INT16 as isize,
    /// Each channel component is a normalized unsigned 8-bit integer value:
    UnormInt8 = cl_h::CL_UNORM_INT8 as isize,
    /// Each channel component is a normalized unsigned 16-bit integer value:
    UnormInt16 = cl_h::CL_UNORM_INT16 as isize,
    /// Represents a normalized 5-6-5 3-channel RGB image. The channel order must be CL_RGB or CL_RGBx:
    UnormShort565 = cl_h::CL_UNORM_SHORT_565 as isize,
    /// Represents a normalized x-5-5-5 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx:
    UnormShort555 = cl_h::CL_UNORM_SHORT_555 as isize,
    /// Represents a normalized x-10-10-10 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx:
    UnormInt101010 = cl_h::CL_UNORM_INT_101010 as isize,
    /// Each channel component is an unnormalized signed 8-bit integer value:
    SignedInt8 = cl_h::CL_SIGNED_INT8 as isize,
    /// Each channel component is an unnormalized signed 16-bit integer value:
    SignedInt16 = cl_h::CL_SIGNED_INT16 as isize,
    /// Each channel component is an unnormalized signed 32-bit integer value:
    SignedInt32 = cl_h::CL_SIGNED_INT32 as isize,
    /// Each channel component is an unnormalized unsigned 8-bit integer value:
    UnsignedInt8 = cl_h::CL_UNSIGNED_INT8 as isize,
    /// Each channel component is an unnormalized unsigned 16-bit integer value:
    UnsignedInt16 = cl_h::CL_UNSIGNED_INT16 as isize,
    /// Each channel component is an unnormalized unsigned 32-bit integer value:
    UnsignedInt32 = cl_h::CL_UNSIGNED_INT32 as isize,
    /// Each channel component is a 16-bit half-float value:
    HalfFloat = cl_h::CL_HALF_FLOAT as isize,
    /// Each channel component is a single precision floating-point value:
    Float = cl_h::CL_FLOAT as isize,
    /// Each channel component is a normalized unsigned 24-bit integer value:
    UnormInt24 = cl_h::CL_UNORM_INT24 as isize,
}





/// image_type
/// Describes the image type and must be either CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE2D_ARRAY, or CL_MEM_OBJECT_IMAGE3D.
///
/// image_width
/// The width of the image in pixels. For a 2D image and image array, the image width must be ≤ CL_DEVICE_IMAGE2D_MAX_WIDTH. For a 3D image, the image width must be ≤ CL_DEVICE_IMAGE3D_MAX_WIDTH. For a 1D image buffer, the image width must be ≤ CL_DEVICE_IMAGE_MAX_BUFFER_SIZE. For a 1D image and 1D image array, the image width must be ≤ CL_DEVICE_IMAGE2D_MAX_WIDTH.
///
/// image_height
/// The height of the image in pixels. This is only used if the image is a 2D, 3D or 2D image array. For a 2D image or image array, the image height must be ≤ CL_DEVICE_IMAGE2D_MAX_HEIGHT. For a 3D image, the image height must be ≤ CL_DEVICE_IMAGE3D_MAX_HEIGHT.
///
/// image_depth
/// The depth of the image in pixels. This is only used if the image is a 3D image and must be a value ≥ 1 and ≤ CL_DEVICE_IMAGE3D_MAX_DEPTH.
///
/// image_array_size
/// The number of images in the image array. This is only used if the image is a 1D or 2D image array. The values for image_array_size, if specified, must be a value ≥ 1 and ≤ CL_DEVICE_IMAGE_MAX_ARRAY_SIZE.
///
/// Note that reading and writing 2D image arrays from a kernel with image_array_size = 1 may be lower performance than 2D images.
///
/// image_row_pitch
/// The scan-line pitch in bytes. This must be 0 if host_ptr is NULL and can be either 0 or ≥ image_width * size of element in bytes if host_ptr is not NULL. If host_ptr is not NULL and image_row_pitch = 0, image_row_pitch is calculated as image_width * size of element in bytes. If image_row_pitch is not 0, it must be a multiple of the image element size in bytes.
///
/// image_slice_pitch
/// The size in bytes of each 2D slice in the 3D image or the size in bytes of each image in a 1D or 2D image array. This must be 0 if host_ptr is NULL. If host_ptr is not NULL, image_slice_pitch can be either 0 or ≥ image_row_pitch * image_height for a 2D image array or 3D image and can be either 0 or ≥ image_row_pitch for a 1D image array. If host_ptr is not NULL and image_slice_pitch = 0, image_slice_pitch is calculated as image_row_pitch * image_height for a 2D image array or 3D image and image_row_pitch for a 1D image array. If image_slice_pitch is not 0, it must be a multiple of the image_row_pitch.
///
/// num_mip_level, num_samples
/// Must be 0.
///
/// buffer
/// Refers to a valid buffer memory object if image_type is CL_MEM_OBJECT_IMAGE1D_BUFFER. Otherwise it must be NULL. For a 1D image buffer object, the image pixels are taken from the buffer object's data store. When the contents of a buffer object's data store are modified, those changes are reflected in the contents of the 1D image buffer object and vice-versa at corresponding sychronization points. The image_width * size of element in bytes must be ≤ size of buffer object data store.
///
/// Note
/// Concurrent reading from, writing to and copying between both a buffer object and 1D image buffer object associated with the buffer object is undefined. Only reading from both a buffer object and 1D image buffer object associated with the buffer object is defined.
#[allow(dead_code)]
pub struct ImageDescriptor {
    pub image_type: MemObjectType,
    pub image_width: usize,
    pub image_height: usize,
    pub image_depth: usize,
    pub image_array_size: usize,
    pub image_row_pitch: usize,
    pub image_slice_pitch: usize,
    num_mip_levels: u32,
    num_samples: u32,
    pub buffer: Option<cl_mem>,
}

impl ImageDescriptor {
    pub fn new() -> ImageDescriptor {
        ImageDescriptor {
            image_type: MemObjectType::Buffer,
            image_width: 0,
            image_height: 0,
            image_depth: 0,
            image_array_size: 0,
            image_row_pitch: 0,
            image_slice_pitch: 0,
            num_mip_levels: 0,
            num_samples: 0,
            buffer: None,
        }
    }

    pub fn as_raw(&self) -> cl_h::cl_image_desc {
        cl_h::cl_image_desc {
            image_type: self.image_type as u32,
            image_width: self.image_width,
            image_height: self.image_height,
            image_depth: self.image_depth,
            image_array_size: self.image_array_size,
            image_row_pitch: self.image_row_pitch,
            image_slice_pitch: self.image_slice_pitch,
            num_mip_levels: self.num_mip_levels,
            num_samples: self.num_mip_levels,
            buffer: self.buffer.unwrap_or(0 as cl_mem),
        }
    }
}

