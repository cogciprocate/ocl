//! Types from platform.h which are not imported elsewhere.

#![allow(non_camel_case_types)]

// From:
// https://raw.githubusercontent.com/KhronosGroup/OpenCL-Headers/opencl12/cl_platform.h
//
// typedef unsigned int cl_GLuint;
// typedef int          cl_GLint;
// typedef unsigned int cl_GLenum;

// pub type cl_GLuint = u32;
// pub type cl_GLint = i32;
// pub type cl_GLenum = u32;

pub type cl_GLuint = u32;
pub type cl_GLint = i32;
pub type cl_GLenum = u32;
