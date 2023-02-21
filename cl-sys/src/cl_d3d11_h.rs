//! OpenCL / DirectX 11 sharing.

// + NVIDIA extension https://registry.khronos.org/OpenCL/extensions/nv/cl_nv_d3d11_sharing.txt

#![allow(
    non_camel_case_types,
    dead_code,
    unused_variables,
    improper_ctypes,
    non_upper_case_globals
)]

use crate::cl_h::{
    cl_command_queue, cl_command_type, cl_context, cl_context_info, cl_device_id, cl_event,
    cl_image_info, cl_int, cl_mem, cl_mem_flags, cl_mem_info, cl_platform_id, cl_uint,
};
use libc::c_void;

pub type cl_d3d11_device_source = cl_uint;
pub type cl_d3d11_device_set = cl_uint;
pub type cl_id3d11_buffer = *mut c_void;
pub type cl_id3d11_texture2d = *mut c_void;
pub type cl_id3d11_texture3d = *mut c_void;

// Error Codes
pub const CL_INVALID_D3D11_DEVICE: cl_int = -1006;
pub const CL_INVALID_D3D11_RESOURCE: cl_int = -1007;
pub const CL_D3D11_RESOURCE_ALREADY_ACQUIRED: cl_int = -1008;
pub const CL_D3D11_RESOURCE_NOT_ACQUIRED: cl_int = -1009;

// cl_d3d11_device_source
pub const CL_D3D11_DEVICE: cl_d3d11_device_source = 0x4019;
pub const CL_D3D11_DXGI_ADAPTER: cl_d3d11_device_source = 0x401A;

// cl_d3d11_device_set
pub const CL_PREFERRED_DEVICES_FOR_D3D11: cl_d3d11_device_set = 0x401B;
pub const CL_ALL_DEVICES_FOR_D3D11: cl_d3d11_device_set = 0x401C;

// cl_context_info
pub const CL_CONTEXT_D3D11_DEVICE_KHR: cl_context_info = 0x401D;
pub const CL_CONTEXT_D3D11_PREFER_SHARED_RESOURCES: cl_context_info = 0x402D;

// cl_mem_info
pub const CL_MEM_D3D11_RESOURCE: cl_mem_info = 0x401E;

// cl_image_info
pub const CL_IMAGE_D3D11_SUBRESOURCE: cl_image_info = 0x401F;

// cl_command_type
pub const CL_COMMAND_ACQUIRE_D3D11_OBJECTS: cl_command_type = 0x4020;
pub const CL_COMMAND_RELEASE_D3D11_OBJECTS: cl_command_type = 0x4021;

pub type clGetDeviceIDsFromD3D11_fn = extern "system" fn(
    platform: cl_platform_id,
    d3d_device_source: cl_d3d11_device_source,
    d3d_object: *mut c_void,
    d3d_device_set: cl_d3d11_device_set,
    num_entries: cl_uint,
    devices: *mut cl_device_id,
    num_devices: *mut cl_uint,
) -> cl_int;

pub type clCreateFromD3D11Buffer_fn = extern "system" fn(
    context: cl_context,
    flags: cl_mem_flags,
    resource: cl_id3d11_buffer,
    errcode_ret: *mut cl_int,
) -> cl_mem;

pub type clCreateFromD3D11Texture2D_fn = extern "system" fn(
    context: cl_context,
    flags: cl_mem_flags,
    resource: cl_id3d11_texture2d,
    subresource: cl_uint,
    errcode_ret: *mut cl_int,
) -> cl_mem;

pub type clCreateFromD3D11Texture3D_fn = extern "system" fn(
    context: cl_context,
    flags: cl_mem_flags,
    resource: cl_id3d11_texture3d,
    subresource: cl_uint,
    errcode_ret: *mut cl_int,
) -> cl_mem;

pub type clEnqueueAcquireD3D11Objects_fn = extern "system" fn(
    command_queue: cl_command_queue,
    num_objects: cl_uint,
    mem_objects: *const cl_mem,
    num_events_in_wait_list: cl_uint,
    event_wait_list: *const cl_event,
    event: *mut cl_event,
) -> cl_int;

pub type clEnqueueReleaseD3D11Objects_fn = extern "system" fn(
    command_queue: cl_command_queue,
    num_objects: cl_uint,
    mem_objects: *const cl_mem,
    num_events_in_wait_list: cl_uint,
    event_wait_list: *const cl_event,
    event: *mut cl_event,
) -> cl_int;
