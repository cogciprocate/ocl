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
use cl_h;

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
    // CL_SNORM_INT8:          
    // CL_SNORM_INT16:         
    // CL_UNORM_INT8:          
    // CL_UNORM_INT16:         
    // CL_UNORM_SHORT_565:     
    // CL_UNORM_SHORT_555:     
    // CL_UNORM_INT_101010:    
    // CL_SIGNED_INT8:         
    // CL_SIGNED_INT16:        
    // CL_SIGNED_INT32:        
    // CL_UNSIGNED_INT8:       
    // CL_UNSIGNED_INT16:      
    // CL_UNSIGNED_INT32:      
    // CL_HALF_FLOAT:          
    // CL_FLOAT:               

    // Each channel component is a normalized signed 8-bit integer value:
    SnormInt8 = cl_h::CL_SNORM_INT8 as isize,
    // Each channel component is a normalized signed 16-bit integer value:
    SnormInt16 = cl_h::CL_SNORM_INT16 as isize,
    // Each channel component is a normalized unsigned 8-bit integer value:
    UnormInt8 = cl_h::CL_UNORM_INT8 as isize,
    // Each channel component is a normalized unsigned 16-bit integer value:
    UnormInt16 = cl_h::CL_UNORM_INT16 as isize,
    // Represents a normalized 5-6-5 3-channel RGB image. The channel order must be CL_RGB or CL_RGBx:
    UnormShort565 = cl_h::CL_UNORM_SHORT_565 as isize,
    // Represents a normalized x-5-5-5 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx:
    UnormShort555 = cl_h::CL_UNORM_SHORT_555 as isize,
    // Represents a normalized x-10-10-10 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx:
    UnormInt101010 = cl_h::CL_UNORM_INT_101010 as isize,
    // Each channel component is an unnormalized signed 8-bit integer value:
    SignedInt8 = cl_h::CL_SIGNED_INT8 as isize,
    // Each channel component is an unnormalized signed 16-bit integer value:
    SignedInt16 = cl_h::CL_SIGNED_INT16 as isize,
    // Each channel component is an unnormalized signed 32-bit integer value:
    SignedInt32 = cl_h::CL_SIGNED_INT32 as isize,
    // Each channel component is an unnormalized unsigned 8-bit integer value:
    UnsignedInt8 = cl_h::CL_UNSIGNED_INT8 as isize,
    // Each channel component is an unnormalized unsigned 16-bit integer value:
    UnsignedInt16 = cl_h::CL_UNSIGNED_INT16 as isize,
    // Each channel component is an unnormalized unsigned 32-bit integer value:
    UnsignedInt32 = cl_h::CL_UNSIGNED_INT32 as isize,
    // Each channel component is a 16-bit half-float value:
    HalfFloat = cl_h::CL_HALF_FLOAT as isize,
    // Each channel component is a single precision floating-point value:
    Float = cl_h::CL_FLOAT as isize,
    // Each channel component is a normalized unsigned 24-bit integer value:
    UnormInt24 = cl_h::CL_UNORM_INT24 as isize,
}

/// A structure that describes format properties of the image to be allocated. (from SDK)
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




/*
############### image_format ################
    ###### image_channel_order ######
    Specifies the number of channels and the channel layout i.e. the memory layout in which channels are stored in the image. Valid values are described in the table below.
    Format  Description
    CL_R, CL_Rx, or CL_A:
    CL_INTENSITY:           This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT.
    CL_LUMINANCE:           This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT.
    CL_RG, CL_RGx, or CL_RA: 
    CL_RGB or CL_RGBx:      This format can only be used if channel data type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010.
    CL_RGBA:    
    CL_ARGB, CL_BGRA:       This format can only be used if channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8.

    ###### image_channel_data_type ######
    Describes the size of the channel data type. The number of bits per element determined by the image_channel_data_type and image_channel_order must be a power of two. The list of supported values is described in the table below.
    Image Channel Data Type Description
    CL_SNORM_INT8:          Each channel component is a normalized signed 8-bit integer value.
    CL_SNORM_INT16:         Each channel component is a normalized signed 16-bit integer value.
    CL_UNORM_INT8:          Each channel component is a normalized unsigned 8-bit integer value.
    CL_UNORM_INT16:         Each channel component is a normalized unsigned 16-bit integer value.
    CL_UNORM_SHORT_565:     Represents a normalized 5-6-5 3-channel RGB image. The channel order must be CL_RGB or CL_RGBx.
    CL_UNORM_SHORT_555:     Represents a normalized x-5-5-5 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx.
    CL_UNORM_INT_101010:    Represents a normalized x-10-10-10 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx.
    CL_SIGNED_INT8:         Each channel component is an unnormalized signed 8-bit integer value.
    CL_SIGNED_INT16:        Each channel component is an unnormalized signed 16-bit integer value.
    CL_SIGNED_INT32:        Each channel component is an unnormalized signed 32-bit integer value.
    CL_UNSIGNED_INT8:       Each channel component is an unnormalized unsigned 8-bit integer value.
    CL_UNSIGNED_INT16:      Each channel component is an unnormalized unsigned 16-bit integer value.
    CL_UNSIGNED_INT32:      Each channel component is an unnormalized unsigned 32-bit integer value.
    CL_HALF_FLOAT:          Each channel component is a 16-bit half-float value.
    CL_FLOAT:               Each channel component is a single precision floating-point value.

    ###### Description ######
    For example, to specify a normalized unsigned 8-bit / channel RGBA image:
              image_channel_order = CL_RGBA
              image_channel_data_type = CL_UNORM_INT8
    image_channel_data_type values of CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 and CL_UNORM_INT_101010 are special cases of packed image formats where the channels of each element are packed into a single unsigned short or unsigned int. For these special packed image formats, the channels are normally packed with the first channel in the most significant bits of the bitfield, and successive channels occupying progressively less significant locations. For CL_UNORM_SHORT_565, R is in bits 15:11, G is in bits 10:5 and B is in bits 4:0. For CL_UNORM_SHORT_555, bit 15 is undefined, R is in bits 14:10, G in bits 9:5 and B in bits 4:0. For CL_UNORM_INT_101010, bits 31:30 are undefined, R is in bits 29:20, G in bits 19:10 and B in bits 9:0.

    OpenCL implementations must maintain the minimum precision specified by the number of bits in image_channel_data_type. If the image format specified by image_channel_order, and image_channel_data_type cannot be supported by the OpenCL implementation, then the call to clCreateImage will return a NULL memory object.

*/
