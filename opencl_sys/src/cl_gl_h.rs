//! OpenGL Extensions.

#![allow(non_camel_case_types, dead_code, unused_variables, improper_ctypes, non_upper_case_globals)]

use libc::{c_void, size_t};
use super::{cl_GLuint, cl_GLint, cl_GLenum};
use cl_h::{cl_context, cl_context_properties, cl_mem_flags, cl_command_queue,
    cl_int, cl_uint, cl_mem, cl_event};

pub type cl_gl_object_type      = cl_uint;
pub type cl_gl_texture_info     = cl_uint;
pub type cl_gl_platform_info    = cl_uint;
pub type cl_gl_context_info     = cl_uint;

// cl_gl_object_type = 0x2000 - 0x200F enum values are currently taken
pub const CL_GL_OBJECT_BUFFER:          cl_gl_object_type = 0x2000;
pub const CL_GL_OBJECT_TEXTURE2D:       cl_gl_object_type = 0x2001;
pub const CL_GL_OBJECT_TEXTURE3D:       cl_gl_object_type = 0x2002;
pub const CL_GL_OBJECT_RENDERBUFFER:    cl_gl_object_type = 0x2003;
pub const CL_GL_OBJECT_TEXTURE2D_ARRAY: cl_gl_object_type = 0x200E;
pub const CL_GL_OBJECT_TEXTURE1D:       cl_gl_object_type = 0x200F;
pub const CL_GL_OBJECT_TEXTURE1D_ARRAY: cl_gl_object_type = 0x2010;
pub const CL_GL_OBJECT_TEXTURE_BUFFER:  cl_gl_object_type = 0x2011;

// cl_gl_texture_info
pub const CL_GL_TEXTURE_TARGET: cl_gl_texture_info = 0x2004;
pub const CL_GL_MIPMAP_LEVEL:   cl_gl_texture_info = 0x2005;
pub const CL_GL_NUM_SAMPLES:    cl_gl_texture_info = 0x2012;

// cl_khr_gl_sharing extension
pub const CL_KHR_GL_SHARING: cl_int = 1;

// Additional Error Codes
pub const CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR: cl_int = -1000;

// cl_gl_context_info
pub const CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR: cl_gl_context_info = 0x2006;
pub const CL_DEVICES_FOR_GL_CONTEXT_KHR:        cl_gl_context_info = 0x2007;

// Additional cl_context_properties
pub const CL_GL_CONTEXT_KHR:        cl_context_properties = 0x2008;
pub const CL_EGL_DISPLAY_KHR:       cl_context_properties = 0x2009;
pub const CL_GLX_DISPLAY_KHR:       cl_context_properties = 0x200A;
pub const CL_WGL_HDC_KHR:           cl_context_properties = 0x200B;
pub const CL_CGL_SHAREGROUP_KHR:    cl_context_properties = 0x200C;

//#[link_args = "-L$OPENCL_LIB -lOpenCL"]
#[cfg_attr(target_os = "macos", link(name = "OpenCL", kind = "framework"))]
#[cfg_attr(target_os = "windows", link(name = "OpenCL"))]
#[cfg_attr(not(target_os = "macos"), link(name = "OpenCL"))]
extern "system" {

    pub fn clCreateFromGLBuffer(context: cl_context,
                                flags: cl_mem_flags,
                                bufobj: cl_GLuint,
                                errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clCreateFromGLTexture(context: cl_context,
                                 flags: cl_mem_flags,
                                 texture_target: cl_GLenum,
                                 miplevel: cl_GLint,
                                 texture: cl_GLuint,
                                 errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clGetGLObjectInfo(memobj: cl_mem,
                             gl_object_type: *mut cl_gl_object_type,
                             gl_object_name: *mut cl_GLuint) -> cl_int;

    pub fn clGetGLTextureInfo(memobj: cl_mem,
                              param_name: cl_gl_texture_info,
                              param_value_size: size_t,
                              param_value: *mut c_void,
                              param_value_size_ret: *mut size_t) -> cl_int;

    pub fn clCreateFromGLRenderbuffer(context: cl_context,
                                      flags: cl_mem_flags,
                                      renderbuffer: cl_GLuint,
                                      errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clEnqueueAcquireGLObjects(command_queue: cl_command_queue,
                                     num_objects: cl_uint,
                                     mem_objects: *const cl_mem,
                                     num_events_in_wait_list: cl_uint,
                                     event_wait_list: *const cl_event,
                                     event: *mut cl_event) -> cl_int;

    pub fn clEnqueueReleaseGLObjects(command_queue: cl_command_queue,
                                     num_objects: cl_uint,
                                     mem_objects: *const cl_mem,
                                     num_events_in_wait_list: cl_uint,
                                     event_wait_list: *const cl_event,
                                     event: *mut cl_event) -> cl_int;

    pub fn clGetGLContextInfoKHR(properties: *const cl_context_properties,
                                 param_name: cl_gl_context_info,
                                 param_value_size: size_t,
                                 param_value: *mut c_void,
                                 param_value_size_ret: *mut size_t) -> cl_int;

    // typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetGLContextInfoKHR_fn)(
    //     const cl_context_properties * properties,
    //     cl_gl_context_info            param_name,
    //     size_t                        param_value_size,
    //     void *                        param_value,
    //     size_t *                      param_value_size_ret);

    // Deprecated OpenCL 1.1 APIs
    pub fn clCreateFromGLTexture2D(context: cl_context,
                                   flags: cl_mem_flags,
                                   texture_target: cl_GLenum,
                                   miplevel: cl_GLint,
                                   texture: cl_GLuint,
                                   errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clCreateFromGLTexture3D(context: cl_context,
                                   flags: cl_mem_flags,
                                   texture_target: cl_GLenum,
                                   miplevel: cl_GLint,
                                   texture: cl_GLuint,
                                   errcode_ret: *mut cl_int) -> cl_mem;
}
