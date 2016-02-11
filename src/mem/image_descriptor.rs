// use libc::c_void;
use cl_h::{self, cl_mem};

#[derive(Clone, Copy)]
pub enum MemObjectType {
	Buffer = cl_h::CL_MEM_OBJECT_BUFFER as isize,
	Image2d = cl_h::CL_MEM_OBJECT_IMAGE2D as isize,
	Image3d = cl_h::CL_MEM_OBJECT_IMAGE3D as isize,
	Image2dArray = cl_h::CL_MEM_OBJECT_IMAGE2D_ARRAY as isize,
	Image1d = cl_h::CL_MEM_OBJECT_IMAGE1D as isize,
	Image1dArray = cl_h::CL_MEM_OBJECT_IMAGE1D_ARRAY as isize,
	Image1dBuffer = cl_h::CL_MEM_OBJECT_IMAGE1D_BUFFER as isize,
}

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


/* 


imageDescriptor
The image descriptor structure describes the type and dimensions of the image or image array and is defined as:
 typedef struct _cl_image_desc {
          cl_mem_object_type image_type;
          size_t image_width;
          size_t image_height;
          size_t image_depth;
          size_t image_array_size;
          size_t image_row_pitch;
          size_t image_slice_pitch;
          cl_uint num_mip_levels;
          cl_uint num_samples;
          cl_mem buffer;
} cl_image_desc;
Members
image_type
Describes the image type and must be either CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE2D_ARRAY, or CL_MEM_OBJECT_IMAGE3D.

image_width
The width of the image in pixels. For a 2D image and image array, the image width must be ≤ CL_DEVICE_IMAGE2D_MAX_WIDTH. For a 3D image, the image width must be ≤ CL_DEVICE_IMAGE3D_MAX_WIDTH. For a 1D image buffer, the image width must be ≤ CL_DEVICE_IMAGE_MAX_BUFFER_SIZE. For a 1D image and 1D image array, the image width must be ≤ CL_DEVICE_IMAGE2D_MAX_WIDTH.

image_height
The height of the image in pixels. This is only used if the image is a 2D, 3D or 2D image array. For a 2D image or image array, the image height must be ≤ CL_DEVICE_IMAGE2D_MAX_HEIGHT. For a 3D image, the image height must be ≤ CL_DEVICE_IMAGE3D_MAX_HEIGHT.

image_depth
The depth of the image in pixels. This is only used if the image is a 3D image and must be a value ≥ 1 and ≤ CL_DEVICE_IMAGE3D_MAX_DEPTH.

image_array_size
The number of images in the image array. This is only used if the image is a 1D or 2D image array. The values for image_array_size, if specified, must be a value ≥ 1 and ≤ CL_DEVICE_IMAGE_MAX_ARRAY_SIZE.

Note that reading and writing 2D image arrays from a kernel with image_array_size = 1 may be lower performance than 2D images.

image_row_pitch
The scan-line pitch in bytes. This must be 0 if host_ptr is NULL and can be either 0 or ≥ image_width * size of element in bytes if host_ptr is not NULL. If host_ptr is not NULL and image_row_pitch = 0, image_row_pitch is calculated as image_width * size of element in bytes. If image_row_pitch is not 0, it must be a multiple of the image element size in bytes.

image_slice_pitch
The size in bytes of each 2D slice in the 3D image or the size in bytes of each image in a 1D or 2D image array. This must be 0 if host_ptr is NULL. If host_ptr is not NULL, image_slice_pitch can be either 0 or ≥ image_row_pitch * image_height for a 2D image array or 3D image and can be either 0 or ≥ image_row_pitch for a 1D image array. If host_ptr is not NULL and image_slice_pitch = 0, image_slice_pitch is calculated as image_row_pitch * image_height for a 2D image array or 3D image and image_row_pitch for a 1D image array. If image_slice_pitch is not 0, it must be a multiple of the image_row_pitch.

num_mip_level, num_samples
Must be 0.

buffer
Refers to a valid buffer memory object if image_type is CL_MEM_OBJECT_IMAGE1D_BUFFER. Otherwise it must be NULL. For a 1D image buffer object, the image pixels are taken from the buffer object's data store. When the contents of a buffer object's data store are modified, those changes are reflected in the contents of the 1D image buffer object and vice-versa at corresponding sychronization points. The image_width * size of element in bytes must be ≤ size of buffer object data store.

Note
Concurrent reading from, writing to and copying between both a buffer object and 1D image buffer object associated with the buffer object is undefined. Only reading from both a buffer object and 1D image buffer object associated with the buffer object is defined.

*/
