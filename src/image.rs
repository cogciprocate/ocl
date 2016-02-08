//! [WORK IN PROGRESS] An OpenCL Image. 
use cl_h::{cl_mem};
use wrapper;

/// [WORK IN PROGRESS] An OpenCL Image. 
pub struct Image {
	image_obj: cl_mem,
}

impl Image {
	/// Returns a new two dimensional image.
	pub fn new_2d() -> Image {
		Image {
			image_obj: wrapper::create_image_2d(),
		}
	}

	/// Returns a new three dimensional image.
	pub fn new_3d() -> Image {
		Image {
			image_obj: wrapper::create_image_3d(),
		}
	}

	pub fn image_obj(&self) -> cl_mem {
		self.image_obj
	}
}

/****** FLAGS ******
CL_MEM_READ_WRITE	
	This flag specifies that the memory object will be read and written by a kernel. This is the default.

CL_MEM_WRITE_ONLY	
	This flags specifies that the memory object will be written but not read by a kernel.

	Reading from a buffer or image object created with CL_MEM_WRITE_ONLY inside a kernel is undefined.

	CL_MEM_READ_WRITE and CL_MEM_WRITE_ONLY are mutually exclusive.

CL_MEM_READ_ONLY	
	This flag specifies that the memory object is a read-only memory object when used inside a kernel.

	Writing to a buffer or image object created with CL_MEM_READ_ONLY inside a kernel is undefined.

	CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY and CL_MEM_READ_ONLY are mutually exclusive.

CL_MEM_USE_HOST_PTR	
	This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OpenCL implementation to use memory referenced by host_ptr as the storage bits for the memory object.

	OpenCL implementations are allowed to cache the buffer contents pointed to by host_ptr in device memory. This cached copy can be used when kernels are executed on a device.

	The result of OpenCL commands that operate on multiple buffer objects created with the same host_ptr or overlapping host regions is considered to be undefined.

	Refer to the description of the alignment rules for host_ptr for memory objects (buffer and images) created using CL_MEM_USE_HOST_PTR.

CL_MEM_ALLOC_HOST_PTR	
	This flag specifies that the application wants the OpenCL implementation to allocate memory from host accessible memory.

	CL_MEM_ALLOC_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually exclusive.

CL_MEM_COPY_HOST_PTR	
	This flag is valid only if host_ptr is not NULL. If specified, it indicates that the application wants the OpenCL implementation to allocate memory for the memory object and copy the data from memory referenced by host_ptr.

	CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually exclusive.

	CL_MEM_COPY_HOST_PTR can be used with CL_MEM_ALLOC_HOST_PTR to initialize the contents of the cl_mem object allocated using host-accessible (e.g. PCIe) memory.

CL_MEM_HOST_WRITE_ONLY	
	This flag specifies that the host will only write to the memory object (using OpenCL APIs that enqueue a write or a map for write). This can be used to optimize write access from the host (e.g. enable write combined allocations for memory objects for devices that communicate with the host over a system bus such as PCIe).

CL_MEM_HOST_READ_ONLY	
	This flag specifies that the host will only read the memory object (using OpenCL APIs that enqueue a read or a map for read).

	CL_MEM_HOST_WRITE_ONLY and CL_MEM_HOST_READ_ONLY are mutually exclusive.

CL_MEM_HOST_NO_ACCESS	
	This flag specifies that the host will not read or write the memory object.

	CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_READ_ONLY and CL_MEM_HOST_NO_ACCESS are mutually exclusive.


############### image_format ################
	###### image_channel_order ######
	Specifies the number of channels and the channel layout i.e. the memory layout in which channels are stored in the image. Valid values are described in the table below.
	Format	Description
	CL_R, CL_Rx, or CL_A:
	CL_INTENSITY:			This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT.
	CL_LUMINANCE:			This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT.
	CL_RG, CL_RGx, or CL_RA: 
	CL_RGB or CL_RGBx:		This format can only be used if channel data type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010.
	CL_RGBA:	
	CL_ARGB, CL_BGRA:		This format can only be used if channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8.

	###### image_channel_data_type ######
	Describes the size of the channel data type. The number of bits per element determined by the image_channel_data_type and image_channel_order must be a power of two. The list of supported values is described in the table below.
	Image Channel Data Type	Description
	CL_SNORM_INT8:			Each channel component is a normalized signed 8-bit integer value.
	CL_SNORM_INT16:			Each channel component is a normalized signed 16-bit integer value.
	CL_UNORM_INT8:			Each channel component is a normalized unsigned 8-bit integer value.
	CL_UNORM_INT16:			Each channel component is a normalized unsigned 16-bit integer value.
	CL_UNORM_SHORT_565:		Represents a normalized 5-6-5 3-channel RGB image. The channel order must be CL_RGB or CL_RGBx.
	CL_UNORM_SHORT_555:		Represents a normalized x-5-5-5 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx.
	CL_UNORM_INT_101010:	Represents a normalized x-10-10-10 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx.
	CL_SIGNED_INT8:			Each channel component is an unnormalized signed 8-bit integer value.
	CL_SIGNED_INT16:		Each channel component is an unnormalized signed 16-bit integer value.
	CL_SIGNED_INT32:		Each channel component is an unnormalized signed 32-bit integer value.
	CL_UNSIGNED_INT8:		Each channel component is an unnormalized unsigned 8-bit integer value.
	CL_UNSIGNED_INT16:		Each channel component is an unnormalized unsigned 16-bit integer value.
	CL_UNSIGNED_INT32:		Each channel component is an unnormalized unsigned 32-bit integer value.
	CL_HALF_FLOAT:			Each channel component is a 16-bit half-float value.
	CL_FLOAT:				Each channel component is a single precision floating-point value.

	###### Description ######
	For example, to specify a normalized unsigned 8-bit / channel RGBA image:
	          image_channel_order = CL_RGBA
	          image_channel_data_type = CL_UNORM_INT8
	image_channel_data_type values of CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 and CL_UNORM_INT_101010 are special cases of packed image formats where the channels of each element are packed into a single unsigned short or unsigned int. For these special packed image formats, the channels are normally packed with the first channel in the most significant bits of the bitfield, and successive channels occupying progressively less significant locations. For CL_UNORM_SHORT_565, R is in bits 15:11, G is in bits 10:5 and B is in bits 4:0. For CL_UNORM_SHORT_555, bit 15 is undefined, R is in bits 14:10, G in bits 9:5 and B in bits 4:0. For CL_UNORM_INT_101010, bits 31:30 are undefined, R is in bits 29:20, G in bits 19:10 and B in bits 9:0.

	OpenCL implementations must maintain the minimum precision specified by the number of bits in image_channel_data_type. If the image format specified by image_channel_order, and image_channel_data_type cannot be supported by the OpenCL implementation, then the call to clCreateImage will return a NULL memory object.


############ Image Descriptor #############
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
		references to a valid buffer memory object if image_type is CL_MEM_OBJECT_IMAGE1D_BUFFER. Otherwise it must be NULL. For a 1D image buffer object, the image pixels are taken from the buffer object's data store. When the contents of a buffer object's data store are modified, those changes are reflected in the contents of the 1D image buffer object and vice-versa at corresponding sychronization points. The image_width * size of element in bytes must be ≤ size of buffer object data store.

	Note:
	Concurrent reading from, writing to and copying between both a buffer object and 1D image buffer object associated with the buffer object is undefined. Only reading from both a buffer object and 1D image buffer object associated with the buffer object is defined.

********/
