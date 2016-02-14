//! [WORK IN PROGRESS] An OpenCL Image.
use std::default::Default;
use standard::Context;
use cl_h;
use raw::{self, MemRaw, ImageFormat, ImageDescriptor};


/// [WORK IN PROGRESS] An OpenCL Image. 
#[allow(dead_code)]
pub struct Image<T> {
    default_val: T,
    obj_raw: MemRaw,
}

impl<T: Default> Image<T> {    
    /// Returns a new `Image`.
    /// [FIXME]: Return result.
    pub fn new(context: &Context, image_format: &ImageFormat, 
            image_desc: &ImageDescriptor, image_data: Option<&[T]>) -> Image<T>
    {
        let flags: u64 = cl_h::CL_MEM_READ_WRITE;
        // let host_ptr: cl_mem = 0 as cl_mem;

        let obj_raw = raw::create_image(
            context.obj_raw(),
            flags,
            &image_format.as_raw(), 
            &image_desc.as_raw(),
            image_data,).expect("[FIXME: TEMPORARY]: Image::new():");

        Image {
            default_val: T::default(),
            obj_raw: obj_raw          
        }
    }   

    /// Returns the raw image object pointer.
    pub fn obj_raw(&self) -> &MemRaw {
        &self.obj_raw
    }
}

// pub struct cl_image_format {
//  image_channel_order:        cl_channel_order,
//  image_channel_data_type:    cl_channel_type,
// }

// pub struct cl_image_desc {
//  image_type: cl_mem_object_type,
//  image_width: size_t,
//  image_height: size_t,
//  image_depth: size_t,
//  image_array_size: size_t,
//  image_row_pitch: size_t,
//  image_slice_pitch: size_t,
//  num_mip_levels: cl_uint,
//  num_samples: cl_uint,
//  buffer: cl_mem,
// }

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
