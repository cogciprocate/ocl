#![allow(dead_code)]
use cl_h;

/// Context Info.
/// CL_CONTEXT_REFERENCE_COUNT	cl_uint	Return the context reference count. The reference count returned should be considered immediately stale. It is unsuitable for general use in applications. This feature is provided for identifying memory leaks.
/// CL_CONTEXT_DEVICES	cl_device_id[]	Return the list of devices in context.
/// CL_CONTEXT_PROPERTIES	cl_context_properties[]	Return the properties argument specified in clCreateContext.
pub enum ContextInfo {
	ReferenceCount = cl_h::CL_CONTEXT_REFERENCE_COUNT as isize,
	Devices = cl_h::CL_CONTEXT_DEVICES as isize,
	Properties = cl_h::CL_CONTEXT_PROPERTIES as isize,
}

/// cl_device_type - bitfield 
#[derive(Clone, Copy)]
pub enum DeviceType {
    Default = cl_h::CL_DEVICE_TYPE_DEFAULT as isize,
    Cpu = cl_h::CL_DEVICE_TYPE_CPU as isize,
    Gpu = cl_h::CL_DEVICE_TYPE_GPU as isize,
    Accelerator = cl_h::CL_DEVICE_TYPE_ACCELERATOR as isize,
    Custom = cl_h::CL_DEVICE_TYPE_CUSTOM as isize,
    All = cl_h::CL_DEVICE_TYPE_ALL as isize,
}

impl DeviceType {
    pub fn as_raw(&self) -> u64 {
        self.clone() as u64
    }
}

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

impl MemObjectType {
    pub fn as_raw(&self) -> u64 {
        self.clone() as u64
    }
}
