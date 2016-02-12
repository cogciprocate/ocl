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
