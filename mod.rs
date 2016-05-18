pub mod platform_h;
pub mod cl_gl_h;

pub use self::platform_h::{ClGlUint, ClGlint, ClGlEnum};
pub use self::cl_gl_h::{cl_gl_object_type, cl_gl_texture_info,
    cl_gl_platform_info, cl_gl_context_info};

pub use self::cl_gl_h::{ClGlObjectType, CL_GL_TEXTURE_TARGET, CL_GL_MIPMAP_LEVEL,
    CL_GL_NUM_SAMPLES, CL_KHR_GL_SHARING, CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR,
    CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, CL_DEVICES_FOR_GL_CONTEXT_KHR,
    CL_GL_CONTEXT_KHR, CL_EGL_DISPLAY_KHR, CL_GLX_DISPLAY_KHR, CL_WGL_HDC_KHR,
    CL_CGL_SHAREGROUP_KHR};

pub use self::cl_gl_h::{clCreateFromGLBuffer, clCreateFromGLTexture,
    clGetGLObjectInfo, clGetGLTextureInfo, clCreateFromGLRenderbuffer,
    clEnqueueAcquireGLObjects, clEnqueueReleaseGLObjects, clCreateFromGLTexture2D,
    clCreateFromGLTexture3D, clGetGLContextInfoKHR,};
