//! The bowels and fundament of the operation.

mod platform_h;
mod glcorearb_h;
mod cl_gl_h;
mod cl_egl_h;
mod cl_gl_ext_h;
mod cl_dx9_media_sharing_h;
mod cl_d3d10_h;
mod cl_d3d11_h;
pub mod cl_h;

pub use self::platform_h::{cl_GLuint, cl_GLint, cl_GLenum};

pub use self::glcorearb_h::{GL_TEXTURE_1D, GL_TEXTURE_1D_ARRAY, GL_TEXTURE_BUFFER,
    GL_TEXTURE_2D, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_3D, GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, GL_TEXTURE_RECTANGLE};

pub use self::cl_gl_h::{cl_gl_object_type, cl_gl_texture_info,
    cl_gl_platform_info, cl_gl_context_info};

pub use self::cl_gl_h::{CL_GL_OBJECT_BUFFER, CL_GL_OBJECT_TEXTURE2D, CL_GL_OBJECT_TEXTURE3D,
    CL_GL_OBJECT_RENDERBUFFER, CL_GL_OBJECT_TEXTURE2D_ARRAY, CL_GL_OBJECT_TEXTURE1D,
    CL_GL_OBJECT_TEXTURE1D_ARRAY, CL_GL_OBJECT_TEXTURE_BUFFER, CL_GL_TEXTURE_TARGET,
    CL_GL_MIPMAP_LEVEL, CL_GL_NUM_SAMPLES, CL_KHR_GL_SHARING, CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR,
    CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, CL_DEVICES_FOR_GL_CONTEXT_KHR, CL_GL_CONTEXT_KHR,
    CL_EGL_DISPLAY_KHR, CL_GLX_DISPLAY_KHR, CL_WGL_HDC_KHR, CL_CGL_SHAREGROUP_KHR};

pub use self::cl_gl_h::{clCreateFromGLBuffer, clCreateFromGLTexture,
    clGetGLObjectInfo, clGetGLTextureInfo, clCreateFromGLRenderbuffer,
    clEnqueueAcquireGLObjects, clEnqueueReleaseGLObjects, clCreateFromGLTexture2D,
    clCreateFromGLTexture3D, clGetGLContextInfoKHR,};

pub use self::cl_egl_h::{CLeglImageKHR, CLeglDisplayKHR, CLeglSyncKHR, cl_egl_image_properties_khr};

pub use self::cl_gl_ext_h::{CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE};

pub use self::cl_d3d10_h::{CL_CONTEXT_D3D10_DEVICE_KHR, cl_d3d10_device_source_khr,
    cl_d3d10_device_set_khr};

pub use self::cl_dx9_media_sharing_h::{CL_CONTEXT_ADAPTER_D3D9_KHR, CL_CONTEXT_ADAPTER_D3D9EX_KHR,
    CL_CONTEXT_ADAPTER_DXVA_KHR};

pub use self::cl_d3d11_h::CL_CONTEXT_D3D11_DEVICE_KHR;

pub use self::cl_h::{CL_CONTEXT_PLATFORM, CL_CONTEXT_INTEROP_USER_SYNC};
