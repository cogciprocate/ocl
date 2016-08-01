//! Rust bindings for the OpenCL ABI.
//!
//! For low level access still within the confines of Rust's safety and convenience, use the extremely lightweight functions and types in the [`core`] module.
//!
//! This file was adapted from [https://www.khronos.org/registry/cl/api/1.2/cl.h](https://www.khronos.org/registry/cl/api/1.2/cl.h).
//!
//! Supports OpenCL version 1.2. Version 1.0 and 1.1 support is currently disabled. Eventually there will be support for every version (including 2.0+) and an auto-detecting best version available system.
//! [FIXME]: 1.2 implementation not 100% complete.
//!
//!
//! [`core`]: /ocl/ocl/core/index.html
//!

#![allow(non_camel_case_types, dead_code, unused_variables, improper_ctypes, non_upper_case_globals)]

use std::fmt::{Display, Formatter, Result};
use libc::{c_void, size_t, c_char, c_uchar, intptr_t};

pub type cl_platform_id     = *mut c_void;
pub type cl_device_id       = *mut c_void;
pub type cl_context         = *mut c_void;
pub type cl_command_queue   = *mut c_void;
pub type cl_mem             = *mut c_void;
pub type cl_program         = *mut c_void;
pub type cl_kernel          = *mut c_void;
pub type cl_event           = *mut c_void;
pub type cl_sampler         = *mut c_void;

pub type cl_char                            = i8;
pub type cl_uchar                           = u8;
pub type cl_short                           = i16;
pub type cl_ushort                          = u16;
pub type cl_int                             = i32;
pub type cl_uint                            = u32;
pub type cl_long                            = i64;
pub type cl_ulong                           = u64;
pub type cl_half                            = u16;
pub type cl_float                           = f32;
pub type cl_double                          = f64;
pub type cl_bool                            = cl_uint;
pub type cl_bitfield                        = cl_ulong;
pub type cl_device_type                     = cl_bitfield;
pub type cl_platform_info                   = cl_uint;
pub type cl_device_info                     = cl_uint;
pub type cl_device_fp_config                = cl_bitfield;
pub type cl_device_mem_cache_type           = cl_uint;
pub type cl_device_local_mem_type           = cl_uint;
pub type cl_device_exec_capabilities        = cl_bitfield;
pub type cl_command_queue_properties        = cl_bitfield;
pub type cl_device_partition_property       = intptr_t;
pub type cl_device_affinity_domain          = cl_bitfield;
pub type cl_context_properties              = intptr_t;
pub type cl_context_info                    = cl_uint;
pub type cl_command_queue_info              = cl_uint;
pub type cl_channel_order                   = cl_uint;
pub type cl_channel_type                    = cl_uint;
pub type cl_mem_flags                       = cl_bitfield;
pub type cl_mem_object_type                 = cl_uint;
pub type cl_mem_info                        = cl_uint;
pub type cl_mem_migration_flags             = cl_bitfield;
pub type cl_image_info                      = cl_uint;
pub type cl_buffer_create_type              = cl_uint;
pub type cl_addressing_mode                 = cl_uint;
pub type cl_filter_mode                     = cl_uint;
pub type cl_sampler_info                    = cl_uint;
pub type cl_map_flags                       = cl_bitfield;
pub type cl_program_info                    = cl_uint;
pub type cl_program_build_info              = cl_uint;
pub type cl_program_binary_type             = cl_uint;
pub type cl_build_status                    = cl_int;
pub type cl_kernel_info                     = cl_uint;
pub type cl_kernel_arg_info                 = cl_uint;
pub type cl_kernel_arg_address_qualifier    = cl_uint;
pub type cl_kernel_arg_access_qualifier     = cl_uint;
pub type cl_kernel_arg_type_qualifier       = cl_uint;
pub type cl_kernel_work_group_info          = cl_uint;
pub type cl_event_info                      = cl_uint;
pub type cl_command_type                    = cl_uint;
pub type cl_profiling_info                  = cl_uint;

#[repr(C)]
pub struct cl_image_format {
    pub image_channel_order:        cl_channel_order,
    pub image_channel_data_type:    cl_channel_type,
}

#[repr(C)]
pub struct cl_image_desc {
    pub image_type:         cl_mem_object_type,
    pub image_width:        size_t,
    pub image_height:       size_t,
    pub image_depth:        size_t,
    pub image_array_size:   size_t,
    pub image_row_pitch:    size_t,
    pub image_slice_pitch:  size_t,
    pub num_mip_levels:     cl_uint,
    pub num_samples:        cl_uint,
    pub buffer:             cl_mem,
}

#[repr(C)]
pub struct cl_buffer_region {
    pub origin:     size_t,
    pub size:       size_t,
}


// #[derive(PartialEq, Debug, FromPrimitive)]
enum_from_primitive! {
    /// TODO: MOVE ME AND LEAVE CONSTS AS THEY WERE.
    /// The status of an OpenCL API call. Used for returning success/error codes.
    #[repr(C)]
    #[derive(Debug, PartialEq, Clone)]
    pub enum Status {
        CL_SUCCESS                                      = 0,
        CL_DEVICE_NOT_FOUND                             = -1,
        CL_DEVICE_NOT_AVAILABLE                         = -2,
        CL_COMPILER_NOT_AVAILABLE                       = -3,
        CL_MEM_OBJECT_ALLOCATION_FAILURE                = -4,
        CL_OUT_OF_RESOURCES                             = -5,
        CL_OUT_OF_HOST_MEMORY                           = -6,
        CL_PROFILING_INFO_NOT_AVAILABLE                 = -7,
        CL_MEM_COPY_OVERLAP                             = -8,
        CL_IMAGE_FORMAT_MISMATCH                        = -9,
        CL_IMAGE_FORMAT_NOT_SUPPORTED                   = -10,
        CL_BUILD_PROGRAM_FAILURE                        = -11,
        CL_MAP_FAILURE                                  = -12,
        CL_MISALIGNED_SUB_BUFFER_OFFSET                 = -13,
        CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST    = -14,
        CL_COMPILE_PROGRAM_FAILURE                      = -15,
        CL_LINKER_NOT_AVAILABLE                         = -16,
        CL_LINK_PROGRAM_FAILURE                         = -17,
        CL_DEVICE_PARTITION_FAILED                      = -18,
        CL_KERNEL_ARG_INFO_NOT_AVAILABLE                = -19,
        CL_INVALID_VALUE                                = -30,
        CL_INVALID_DEVICE_TYPE                          = -31,
        CL_INVALID_PLATFORM                             = -32,
        CL_INVALID_DEVICE                               = -33,
        CL_INVALID_CONTEXT                              = -34,
        CL_INVALID_QUEUE_PROPERTIES                     = -35,
        CL_INVALID_COMMAND_QUEUE                        = -36,
        CL_INVALID_HOST_PTR                             = -37,
        CL_INVALID_MEM_OBJECT                           = -38,
        CL_INVALID_IMAGE_FORMAT_DESCRIPTOR              = -39,
        CL_INVALID_IMAGE_SIZE                           = -40,
        CL_INVALID_SAMPLER                              = -41,
        CL_INVALID_BINARY                               = -42,
        CL_INVALID_BUILD_OPTIONS                        = -43,
        CL_INVALID_PROGRAM                              = -44,
        CL_INVALID_PROGRAM_EXECUTABLE                   = -45,
        CL_INVALID_KERNEL_NAME                          = -46,
        CL_INVALID_KERNEL_DEFINITION                    = -47,
        CL_INVALID_KERNEL                               = -48,
        CL_INVALID_ARG_INDEX                            = -49,
        CL_INVALID_ARG_VALUE                            = -50,
        CL_INVALID_ARG_SIZE                             = -51,
        CL_INVALID_KERNEL_ARGS                          = -52,
        CL_INVALID_WORK_DIMENSION                       = -53,
        CL_INVALID_WORK_GROUP_SIZE                      = -54,
        CL_INVALID_WORK_ITEM_SIZE                       = -55,
        CL_INVALID_GLOBAL_OFFSET                        = -56,
        CL_INVALID_EVENT_WAIT_LIST                      = -57,
        CL_INVALID_EVENT                                = -58,
        CL_INVALID_OPERATION                            = -59,
        CL_INVALID_GL_OBJECT                            = -60,
        CL_INVALID_BUFFER_SIZE                          = -61,
        CL_INVALID_MIP_LEVEL                            = -62,
        CL_INVALID_GLOBAL_WORK_SIZE                     = -63,
        CL_INVALID_PROPERTY                             = -64,
        CL_INVALID_IMAGE_DESCRIPTOR                     = -65,
        CL_INVALID_COMPILER_OPTIONS                     = -66,
        CL_INVALID_LINKER_OPTIONS                       = -67,
        CL_INVALID_DEVICE_PARTITION_COUNT               = -68,
        CL_PLATFORM_NOT_FOUND_KHR                       = -1001,
    }
}

impl Display for Status {
    fn fmt(&self, fmtr: &mut Formatter) -> Result {
        write!(fmtr, "{:?}", self)
    }
}


// Version
pub const CL_VERSION_1_0:                               cl_bool = 1;
pub const CL_VERSION_1_1:                               cl_bool = 1;
pub const CL_VERSION_1_2:                               cl_bool = 1;
//pub const CL_VERSION_2_0:                               cl_bool = 1;

// cl_bool
pub const CL_FALSE:                                     cl_bool = 0;
pub const CL_TRUE:                                      cl_bool = 1;
pub const CL_BLOCKING:                                  cl_bool = CL_TRUE;
pub const CL_NON_BLOCKING:                              cl_bool = CL_FALSE;


// cl_platform_info
pub const CL_PLATFORM_PROFILE:                          cl_uint = 0x0900;
pub const CL_PLATFORM_VERSION:                          cl_uint = 0x0901;
pub const CL_PLATFORM_NAME:                             cl_uint = 0x0902;
pub const CL_PLATFORM_VENDOR:                           cl_uint = 0x0903;
pub const CL_PLATFORM_EXTENSIONS:                       cl_uint = 0x0904;

// cl_device_type - bitfield
pub const CL_DEVICE_TYPE_DEFAULT:                      cl_bitfield = 1 << 0;
pub const CL_DEVICE_TYPE_CPU:                          cl_bitfield = 1 << 1;
pub const CL_DEVICE_TYPE_GPU:                          cl_bitfield = 1 << 2;
pub const CL_DEVICE_TYPE_ACCELERATOR:                  cl_bitfield = 1 << 3;
pub const CL_DEVICE_TYPE_CUSTOM:                       cl_bitfield = 1 << 4;
pub const CL_DEVICE_TYPE_ALL:                          cl_bitfield = 0xFFFFFFFF;

// cl_device_info
pub const CL_DEVICE_TYPE:                               cl_uint = 0x1000;
pub const CL_DEVICE_VENDOR_ID:                          cl_uint = 0x1001;
pub const CL_DEVICE_MAX_COMPUTE_UNITS:                  cl_uint = 0x1002;
pub const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:           cl_uint = 0x1003;
pub const CL_DEVICE_MAX_WORK_GROUP_SIZE:                cl_uint = 0x1004;
pub const CL_DEVICE_MAX_WORK_ITEM_SIZES:                cl_uint = 0x1005;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:        cl_uint = 0x1006;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:       cl_uint = 0x1007;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:         cl_uint = 0x1008;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:        cl_uint = 0x1009;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:       cl_uint = 0x100A;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:      cl_uint = 0x100B;
pub const CL_DEVICE_MAX_CLOCK_FREQUENCY:                cl_uint = 0x100C;
pub const CL_DEVICE_ADDRESS_BITS:                       cl_uint = 0x100D;
pub const CL_DEVICE_MAX_READ_IMAGE_ARGS:                cl_uint = 0x100E;
pub const CL_DEVICE_MAX_WRITE_IMAGE_ARGS:               cl_uint = 0x100F;
pub const CL_DEVICE_MAX_MEM_ALLOC_SIZE:                 cl_uint = 0x1010;
pub const CL_DEVICE_IMAGE2D_MAX_WIDTH:                  cl_uint = 0x1011;
pub const CL_DEVICE_IMAGE2D_MAX_HEIGHT:                 cl_uint = 0x1012;
pub const CL_DEVICE_IMAGE3D_MAX_WIDTH:                  cl_uint = 0x1013;
pub const CL_DEVICE_IMAGE3D_MAX_HEIGHT:                 cl_uint = 0x1014;
pub const CL_DEVICE_IMAGE3D_MAX_DEPTH:                  cl_uint = 0x1015;
pub const CL_DEVICE_IMAGE_SUPPORT:                      cl_uint = 0x1016;
pub const CL_DEVICE_MAX_PARAMETER_SIZE:                 cl_uint = 0x1017;
pub const CL_DEVICE_MAX_SAMPLERS:                       cl_uint = 0x1018;
pub const CL_DEVICE_MEM_BASE_ADDR_ALIGN:                cl_uint = 0x1019;
pub const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:           cl_uint = 0x101A;
pub const CL_DEVICE_SINGLE_FP_CONFIG:                   cl_uint = 0x101B;
pub const CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:              cl_uint = 0x101C;
pub const CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:          cl_uint = 0x101D;
pub const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:              cl_uint = 0x101E;
pub const CL_DEVICE_GLOBAL_MEM_SIZE:                    cl_uint = 0x101F;
pub const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:           cl_uint = 0x1020;
pub const CL_DEVICE_MAX_CONSTANT_ARGS:                  cl_uint = 0x1021;
pub const CL_DEVICE_LOCAL_MEM_TYPE:                     cl_uint = 0x1022;
pub const CL_DEVICE_LOCAL_MEM_SIZE:                     cl_uint = 0x1023;
pub const CL_DEVICE_ERROR_CORRECTION_SUPPORT:           cl_uint = 0x1024;
pub const CL_DEVICE_PROFILING_TIMER_RESOLUTION:         cl_uint = 0x1025;
pub const CL_DEVICE_ENDIAN_LITTLE:                      cl_uint = 0x1026;
pub const CL_DEVICE_AVAILABLE:                          cl_uint = 0x1027;
pub const CL_DEVICE_COMPILER_AVAILABLE:                 cl_uint = 0x1028;
pub const CL_DEVICE_EXECUTION_CAPABILITIES:             cl_uint = 0x1029;
pub const CL_DEVICE_QUEUE_PROPERTIES:                   cl_uint = 0x102A;
pub const CL_DEVICE_NAME:                               cl_uint = 0x102B;
pub const CL_DEVICE_VENDOR:                             cl_uint = 0x102C;
pub const CL_DRIVER_VERSION:                            cl_uint = 0x102D;
pub const CL_DEVICE_PROFILE:                            cl_uint = 0x102E;
pub const CL_DEVICE_VERSION:                            cl_uint = 0x102F;
pub const CL_DEVICE_EXTENSIONS:                         cl_uint = 0x1030;
pub const CL_DEVICE_PLATFORM:                           cl_uint = 0x1031;
pub const CL_DEVICE_DOUBLE_FP_CONFIG:                   cl_uint = 0x1032;
pub const CL_DEVICE_HALF_FP_CONFIG:                     cl_uint = 0x1033;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:        cl_uint = 0x1034;
pub const CL_DEVICE_HOST_UNIFIED_MEMORY:                cl_uint = 0x1035;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:           cl_uint = 0x1036;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:          cl_uint = 0x1037;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:            cl_uint = 0x1038;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:           cl_uint = 0x1039;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:          cl_uint = 0x103A;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:         cl_uint = 0x103B;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:           cl_uint = 0x103C;
pub const CL_DEVICE_OPENCL_C_VERSION:                   cl_uint = 0x103D;
pub const CL_DEVICE_LINKER_AVAILABLE:                   cl_uint = 0x103E;
pub const CL_DEVICE_BUILT_IN_KERNELS:                   cl_uint = 0x103F;
pub const CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:              cl_uint = 0x1040;
pub const CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:               cl_uint = 0x1041;
pub const CL_DEVICE_PARENT_DEVICE:                      cl_uint = 0x1042;
pub const CL_DEVICE_PARTITION_MAX_SUB_DEVICES:          cl_uint = 0x1043;
pub const CL_DEVICE_PARTITION_PROPERTIES:               cl_uint = 0x1044;
pub const CL_DEVICE_PARTITION_AFFINITY_DOMAIN:          cl_uint = 0x1045;
pub const CL_DEVICE_PARTITION_TYPE:                     cl_uint = 0x1046;
pub const CL_DEVICE_REFERENCE_COUNT:                    cl_uint = 0x1047;
pub const CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:        cl_uint = 0x1048;
pub const CL_DEVICE_PRINTF_BUFFER_SIZE:                 cl_uint = 0x1049;
pub const CL_DEVICE_IMAGE_PITCH_ALIGNMENT:              cl_uint = 0x104A;
pub const CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT:       cl_uint = 0x104B;

// cl_device_fp_config - bitfield
pub const CL_FP_DENORM:                                 cl_bitfield = 1 << 0;
pub const CL_FP_INF_NAN:                                cl_bitfield = 1 << 1;
pub const CL_FP_ROUND_TO_NEAREST:                       cl_bitfield = 1 << 2;
pub const CL_FP_ROUND_TO_ZERO:                          cl_bitfield = 1 << 3;
pub const CL_FP_ROUND_TO_INF:                           cl_bitfield = 1 << 4;
pub const CL_FP_FMA:                                    cl_bitfield = 1 << 5;
pub const CL_FP_SOFT_FLOAT:                             cl_bitfield = 1 << 6;
pub const CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT:          cl_bitfield = 1 << 7;

// cl_device_mem_cache_type
pub const CL_NONE:                                      cl_uint = 0x0;
pub const CL_READ_ONLY_CACHE:                           cl_uint = 0x1;
pub const CL_READ_WRITE_CACHE:                          cl_uint = 0x2;

// cl_device_local_mem_type
pub const CL_LOCAL:                                     cl_uint = 0x1;
pub const CL_GLOBAL:                                    cl_uint = 0x2;

// cl_device_exec_capabilities - bitfield
pub const CL_EXEC_KERNEL:                               cl_bitfield = 1 << 0;
pub const CL_EXEC_NATIVE_KERNEL:                        cl_bitfield = 1 << 1;

// cl_command_queue_properties - bitfield
pub const CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE:       cl_bitfield = 1 << 0;
pub const CL_QUEUE_PROFILING_ENABLE:                    cl_bitfield = 1 << 1;

// cl_context_info
pub const CL_CONTEXT_REFERENCE_COUNT:                   cl_uint = 0x1080;
pub const CL_CONTEXT_DEVICES:                           cl_uint = 0x1081;
pub const CL_CONTEXT_PROPERTIES:                        cl_uint = 0x1082;
pub const CL_CONTEXT_NUM_DEVICES:                       cl_uint = 0x1083;

// cl_context_info + cl_context_properties
pub const CL_CONTEXT_PLATFORM:                          cl_uint = 0x1084;
pub const CL_CONTEXT_INTEROP_USER_SYNC:                 cl_uint = 0x1085;

// cl_device_partition_property
pub const CL_DEVICE_PARTITION_EQUALLY:                  cl_uint = 0x1086;
pub const CL_DEVICE_PARTITION_BY_COUNTS:                cl_uint = 0x1087;
pub const CL_DEVICE_PARTITION_BY_COUNTS_LIST_END:       cl_uint = 0x0;
pub const CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:       cl_uint = 0x1088;

// cl_device_affinity_domain
pub const CL_DEVICE_AFFINITY_DOMAIN_NUMA:               cl_bitfield = 1 << 0;
pub const CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE:           cl_bitfield = 1 << 1;
pub const CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE:           cl_bitfield = 1 << 2;
pub const CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE:           cl_bitfield = 1 << 3;
pub const CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE:           cl_bitfield = 1 << 4;
pub const CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE: cl_bitfield = 1 << 5;

// cl_command_queue_info
pub const CL_QUEUE_CONTEXT:                             cl_uint = 0x1090;
pub const CL_QUEUE_DEVICE:                              cl_uint = 0x1091;
pub const CL_QUEUE_REFERENCE_COUNT:                     cl_uint = 0x1092;
pub const CL_QUEUE_PROPERTIES:                          cl_uint = 0x1093;

// cl_mem_flags - bitfield
pub const CL_MEM_READ_WRITE:                            cl_bitfield = 1 << 0;
pub const CL_MEM_WRITE_ONLY:                            cl_bitfield = 1 << 1;
pub const CL_MEM_READ_ONLY:                             cl_bitfield = 1 << 2;
pub const CL_MEM_USE_HOST_PTR:                          cl_bitfield = 1 << 3;
pub const CL_MEM_ALLOC_HOST_PTR:                        cl_bitfield = 1 << 4;
pub const CL_MEM_COPY_HOST_PTR:                         cl_bitfield = 1 << 5;
// RESERVED                                             cl_bitfield = 1 << 6;
pub const CL_MEM_HOST_WRITE_ONLY:                       cl_bitfield = 1 << 7;
pub const CL_MEM_HOST_READ_ONLY:                        cl_bitfield = 1 << 8;
pub const CL_MEM_HOST_NO_ACCESS:                        cl_bitfield = 1 << 9;

// cl_mem_migration_flags - bitfield
pub const CL_MIGRATE_MEM_OBJECT_HOST:                   cl_bitfield = 1 << 0;
pub const CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED:      cl_bitfield = 1 << 1;

// cl_channel_order
pub const CL_R:                                         cl_uint = 0x10B0;
pub const CL_A:                                         cl_uint = 0x10B1;
pub const CL_RG:                                        cl_uint = 0x10B2;
pub const CL_RA:                                        cl_uint = 0x10B3;
pub const CL_RGB:                                       cl_uint = 0x10B4;
pub const CL_RGBA:                                      cl_uint = 0x10B5;
pub const CL_BGRA:                                      cl_uint = 0x10B6;
pub const CL_ARGB:                                      cl_uint = 0x10B7;
pub const CL_INTENSITY:                                 cl_uint = 0x10B8;
pub const CL_LUMINANCE:                                 cl_uint = 0x10B9;
pub const CL_Rx:                                        cl_uint = 0x10BA;
pub const CL_RGx:                                       cl_uint = 0x10BB;
pub const CL_RGBx:                                      cl_uint = 0x10BC;
pub const CL_DEPTH:                                     cl_uint = 0x10BD;
pub const CL_DEPTH_STENCIL:                             cl_uint = 0x10BE;

// cl_channel_type
pub const CL_SNORM_INT8:                                cl_uint = 0x10D0;
pub const CL_SNORM_INT16:                               cl_uint = 0x10D1;
pub const CL_UNORM_INT8:                                cl_uint = 0x10D2;
pub const CL_UNORM_INT16:                               cl_uint = 0x10D3;
pub const CL_UNORM_SHORT_565:                           cl_uint = 0x10D4;
pub const CL_UNORM_SHORT_555:                           cl_uint = 0x10D5;
pub const CL_UNORM_INT_101010:                          cl_uint = 0x10D6;
pub const CL_SIGNED_INT8:                               cl_uint = 0x10D7;
pub const CL_SIGNED_INT16:                              cl_uint = 0x10D8;
pub const CL_SIGNED_INT32:                              cl_uint = 0x10D9;
pub const CL_UNSIGNED_INT8:                             cl_uint = 0x10DA;
pub const CL_UNSIGNED_INT16:                            cl_uint = 0x10DB;
pub const CL_UNSIGNED_INT32:                            cl_uint = 0x10DC;
pub const CL_HALF_FLOAT:                                cl_uint = 0x10DD;
pub const CL_FLOAT:                                     cl_uint = 0x10DE;
pub const CL_UNORM_INT24:                               cl_uint = 0x10DF;

// cl_mem_object_type
pub const CL_MEM_OBJECT_BUFFER:                         cl_uint = 0x10F0;
pub const CL_MEM_OBJECT_IMAGE2D:                        cl_uint = 0x10F1;
pub const CL_MEM_OBJECT_IMAGE3D:                        cl_uint = 0x10F2;
pub const CL_MEM_OBJECT_IMAGE2D_ARRAY:                  cl_uint = 0x10F3;
pub const CL_MEM_OBJECT_IMAGE1D:                        cl_uint = 0x10F4;
pub const CL_MEM_OBJECT_IMAGE1D_ARRAY:                  cl_uint = 0x10F5;
pub const CL_MEM_OBJECT_IMAGE1D_BUFFER:                 cl_uint = 0x10F6;

// cl_mem_info
pub const CL_MEM_TYPE:                                  cl_uint = 0x1100;
pub const CL_MEM_FLAGS:                                 cl_uint = 0x1101;
pub const CL_MEM_SIZE:                                  cl_uint = 0x1102;
pub const CL_MEM_HOST_PTR:                              cl_uint = 0x1103;
pub const CL_MEM_MAP_COUNT:                             cl_uint = 0x1104;
pub const CL_MEM_REFERENCE_COUNT:                       cl_uint = 0x1105;
pub const CL_MEM_CONTEXT:                               cl_uint = 0x1106;
pub const CL_MEM_ASSOCIATED_MEMOBJECT:                  cl_uint = 0x1107;
pub const CL_MEM_OFFSET:                                cl_uint = 0x1108;

// cl_image_info
pub const CL_IMAGE_FORMAT:                              cl_uint = 0x1110;
pub const CL_IMAGE_ELEMENT_SIZE:                        cl_uint = 0x1111;
pub const CL_IMAGE_ROW_PITCH:                           cl_uint = 0x1112;
pub const CL_IMAGE_SLICE_PITCH:                         cl_uint = 0x1113;
pub const CL_IMAGE_WIDTH:                               cl_uint = 0x1114;
pub const CL_IMAGE_HEIGHT:                              cl_uint = 0x1115;
pub const CL_IMAGE_DEPTH:                               cl_uint = 0x1116;
pub const CL_IMAGE_ARRAY_SIZE:                          cl_uint = 0x1117;
pub const CL_IMAGE_BUFFER:                              cl_uint = 0x1118;
pub const CL_IMAGE_NUM_MIP_LEVELS:                      cl_uint = 0x1119;
pub const CL_IMAGE_NUM_SAMPLES:                         cl_uint = 0x111A;

// cl_addressing_mode
pub const CL_ADDRESS_NONE:                              cl_uint = 0x1130;
pub const CL_ADDRESS_CLAMP_TO_EDGE:                     cl_uint = 0x1131;
pub const CL_ADDRESS_CLAMP:                             cl_uint = 0x1132;
pub const CL_ADDRESS_REPEAT:                            cl_uint = 0x1133;
pub const CL_ADDRESS_MIRRORED_REPEAT:                   cl_uint = 0x1134;

// cl_filter_mode
pub const CL_FILTER_NEAREST:                            cl_uint = 0x1140;
pub const CL_FILTER_LINEAR:                             cl_uint = 0x1141;

// cl_sampler_info
pub const CL_SAMPLER_REFERENCE_COUNT:                   cl_uint = 0x1150;
pub const CL_SAMPLER_CONTEXT:                           cl_uint = 0x1151;
pub const CL_SAMPLER_NORMALIZED_COORDS:                 cl_uint = 0x1152;
pub const CL_SAMPLER_ADDRESSING_MODE:                   cl_uint = 0x1153;
pub const CL_SAMPLER_FILTER_MODE:                       cl_uint = 0x1154;

// cl_map_flags - bitfield
pub const CL_MAP_READ:                                  cl_bitfield = 1 << 0;
pub const CL_MAP_WRITE:                                 cl_bitfield = 1 << 1;
pub const CL_MAP_WRITE_INVALIDATE_REGION:               cl_bitfield = 1 << 2;

// cl_program_info
pub const CL_PROGRAM_REFERENCE_COUNT:                   cl_uint = 0x1160;
pub const CL_PROGRAM_CONTEXT:                           cl_uint = 0x1161;
pub const CL_PROGRAM_NUM_DEVICES:                       cl_uint = 0x1162;
pub const CL_PROGRAM_DEVICES:                           cl_uint = 0x1163;
pub const CL_PROGRAM_SOURCE:                            cl_uint = 0x1164;
pub const CL_PROGRAM_BINARY_SIZES:                      cl_uint = 0x1165;
pub const CL_PROGRAM_BINARIES:                          cl_uint = 0x1166;
pub const CL_PROGRAM_NUM_KERNELS:                       cl_uint = 0x1167;
pub const CL_PROGRAM_KERNEL_NAMES:                      cl_uint = 0x1168;

// cl_program_build_info
pub const CL_PROGRAM_BUILD_STATUS:                      cl_uint = 0x1181;
pub const CL_PROGRAM_BUILD_OPTIONS:                     cl_uint = 0x1182;
pub const CL_PROGRAM_BUILD_LOG:                         cl_uint = 0x1183;
pub const CL_PROGRAM_BINARY_TYPE:                       cl_uint = 0x1184;

// cl_program_binary_type
pub const CL_PROGRAM_BINARY_TYPE_NONE:                  cl_bitfield = 0x0;
pub const CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:       cl_bitfield = 0x1;
pub const CL_PROGRAM_BINARY_TYPE_LIBRARY:               cl_bitfield = 0x2;
pub const CL_PROGRAM_BINARY_TYPE_EXECUTABLE:            cl_bitfield = 0x4;

// cl_build_status
pub const CL_BUILD_SUCCESS:                             cl_uint = 0;
pub const CL_BUILD_NONE:                                cl_uint = !0 - 1;
pub const CL_BUILD_ERROR:                               cl_uint = !0 - 2;
pub const CL_BUILD_IN_PROGRESS:                         cl_uint = !0 - 3;

// cl_kernel_info
pub const CL_KERNEL_FUNCTION_NAME:                      cl_uint = 0x1190;
pub const CL_KERNEL_NUM_ARGS:                           cl_uint = 0x1191;
pub const CL_KERNEL_REFERENCE_COUNT:                    cl_uint = 0x1192;
pub const CL_KERNEL_CONTEXT:                            cl_uint = 0x1193;
pub const CL_KERNEL_PROGRAM:                            cl_uint = 0x1194;
pub const CL_KERNEL_ATTRIBUTES:                         cl_uint = 0x1195;

// cl_kernel_arg_info
pub const CL_KERNEL_ARG_ADDRESS_QUALIFIER:              cl_uint = 0x1196;
pub const CL_KERNEL_ARG_ACCESS_QUALIFIER:               cl_uint = 0x1197;
pub const CL_KERNEL_ARG_TYPE_NAME:                      cl_uint = 0x1198;
pub const CL_KERNEL_ARG_TYPE_QUALIFIER:                 cl_uint = 0x1199;
pub const CL_KERNEL_ARG_NAME:                           cl_uint = 0x119A;

// cl_kernel_arg_address_qualifier
pub const CL_KERNEL_ARG_ADDRESS_GLOBAL:                 cl_uint = 0x119B;
pub const CL_KERNEL_ARG_ADDRESS_LOCAL:                  cl_uint = 0x119C;
pub const CL_KERNEL_ARG_ADDRESS_CONSTANT:               cl_uint = 0x119D;
pub const CL_KERNEL_ARG_ADDRESS_PRIVATE:                cl_uint = 0x119E;

// cl_kernel_arg_access_qualifier
pub const CL_KERNEL_ARG_ACCESS_READ_ONLY:               cl_uint = 0x11A0;
pub const CL_KERNEL_ARG_ACCESS_WRITE_ONLY:              cl_uint = 0x11A1;
pub const CL_KERNEL_ARG_ACCESS_READ_WRITE:              cl_uint = 0x11A2;
pub const CL_KERNEL_ARG_ACCESS_NONE:                    cl_uint = 0x11A3;

// cl_kernel_arg_type_qualifer
pub const CL_KERNEL_ARG_TYPE_NONE:                      cl_bitfield = 0;
pub const CL_KERNEL_ARG_TYPE_CONST:                     cl_bitfield = 1 << 0;
pub const CL_KERNEL_ARG_TYPE_RESTRICT:                  cl_bitfield = 1 << 1;
pub const CL_KERNEL_ARG_TYPE_VOLATILE:                  cl_bitfield = 1 << 2;

// cl_kernel_work_group_info
pub const CL_KERNEL_WORK_GROUP_SIZE:                    cl_uint = 0x11B0;
pub const CL_KERNEL_COMPILE_WORK_GROUP_SIZE:            cl_uint = 0x11B1;
pub const CL_KERNEL_LOCAL_MEM_SIZE:                     cl_uint = 0x11B2;
pub const CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: cl_uint = 0x11B3;
pub const CL_KERNEL_PRIVATE_MEM_SIZE:                   cl_uint = 0x11B4;
pub const CL_KERNEL_GLOBAL_WORK_SIZE:                   cl_uint = 0x11B5;

// cl_event_info
pub const CL_EVENT_COMMAND_QUEUE:                       cl_uint = 0x11D0;
pub const CL_EVENT_COMMAND_TYPE:                        cl_uint = 0x11D1;
pub const CL_EVENT_REFERENCE_COUNT:                     cl_uint = 0x11D2;
pub const CL_EVENT_COMMAND_EXECUTION_STATUS:            cl_uint = 0x11D3;
pub const CL_EVENT_CONTEXT:                             cl_uint = 0x11D4;

// cl_command_type
pub const CL_COMMAND_NDRANGE_KERNEL:                    cl_uint = 0x11F0;
pub const CL_COMMAND_TASK:                              cl_uint = 0x11F1;
pub const CL_COMMAND_NATIVE_KERNEL:                     cl_uint = 0x11F2;
pub const CL_COMMAND_READ_BUFFER:                       cl_uint = 0x11F3;
pub const CL_COMMAND_WRITE_BUFFER:                      cl_uint = 0x11F4;
pub const CL_COMMAND_COPY_BUFFER:                       cl_uint = 0x11F5;
pub const CL_COMMAND_READ_IMAGE:                        cl_uint = 0x11F6;
pub const CL_COMMAND_WRITE_IMAGE:                       cl_uint = 0x11F7;
pub const CL_COMMAND_COPY_IMAGE:                        cl_uint = 0x11F8;
pub const CL_COMMAND_COPY_IMAGE_TO_BUFFER:              cl_uint = 0x11F9;
pub const CL_COMMAND_COPY_BUFFER_TO_IMAGE:              cl_uint = 0x11FA;
pub const CL_COMMAND_MAP_BUFFER:                        cl_uint = 0x11FB;
pub const CL_COMMAND_MAP_IMAGE:                         cl_uint = 0x11FC;
pub const CL_COMMAND_UNMAP_MEM_OBJECT:                  cl_uint = 0x11FD;
pub const CL_COMMAND_MARKER:                            cl_uint = 0x11FE;
pub const CL_COMMAND_ACQUIRE_GL_OBJECTS:                cl_uint = 0x11FF;
pub const CL_COMMAND_RELEASE_GL_OBJECTS:                cl_uint = 0x1200;
pub const CL_COMMAND_READ_BUFFER_RECT:                  cl_uint = 0x1201;
pub const CL_COMMAND_WRITE_BUFFER_RECT:                 cl_uint = 0x1202;
pub const CL_COMMAND_COPY_BUFFER_RECT:                  cl_uint = 0x1203;
pub const CL_COMMAND_USER:                              cl_uint = 0x1204;
pub const CL_COMMAND_BARRIER:                           cl_uint = 0x1205;
pub const CL_COMMAND_MIGRATE_MEM_OBJECTS:               cl_uint = 0x1206;
pub const CL_COMMAND_FILL_BUFFER:                       cl_uint = 0x1207;
pub const CL_COMMAND_FILL_IMAGE:                        cl_uint = 0x1208;

// command execution status
pub const CL_COMPLETE:                                  cl_int = 0x0;
pub const CL_RUNNING:                                   cl_int = 0x1;
pub const CL_SUBMITTED:                                 cl_int = 0x2;
pub const CL_QUEUED:                                    cl_int = 0x3;

// cl_buffer_create_type
pub const CL_BUFFER_CREATE_TYPE_REGION:                 cl_uint = 0x1220;

// cl_profiling_info
pub const CL_PROFILING_COMMAND_QUEUED:                  cl_uint = 0x1280;
pub const CL_PROFILING_COMMAND_SUBMIT:                  cl_uint = 0x1281;
pub const CL_PROFILING_COMMAND_START:                   cl_uint = 0x1282;
pub const CL_PROFILING_COMMAND_END:                     cl_uint = 0x1283;


//#[link_args = "-L$OPENCL_LIB -lOpenCL"]
#[cfg_attr(target_os = "macos", link(name = "OpenCL", kind = "framework"))]
#[cfg_attr(target_os = "windows", link(name = "OpenCL"))]
#[cfg_attr(not(target_os = "macos"), link(name = "OpenCL"))]
extern "system" {
    // Platform API
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetPlatformIDs(num_entries: cl_uint,
                            platforms: *mut cl_platform_id,
                            num_platforms: *mut cl_uint) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetPlatformInfo(platform: cl_platform_id,
                             param_name: cl_platform_info,
                             param_value_size: size_t,
                             param_value: *mut c_void,
                             param_value_size_ret: *mut size_t) -> cl_int;

    // Device APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetDeviceIDs(platform: cl_platform_id,
                      device_type: cl_device_type,
                      num_entries: cl_uint,
                      devices: *mut cl_device_id,
                      num_devices: *mut cl_uint) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetDeviceInfo(device: cl_device_id,
                       param_name: cl_device_info,
                       param_value_size: size_t,
                       param_value: *mut c_void,
                       param_value_size_ret: *mut size_t) -> cl_int;


    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clCreateSubDevices(cl_device_id                         /* in_device */,
    //                    const cl_device_partition_property * /* properties */,
    //                    cl_uint                              /* num_devices */,
    //                    cl_device_id *                       /* out_devices */,
    //                    cl_uint *                            /* num_devices_ret */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(feature = "opencl_1_2")]
    pub fn clCreateSubDevices(in_device: cl_device_id,
                       properties: *const cl_device_partition_property,
                       num_devices: cl_uint,
                       out_devices: *mut cl_device_id,
                       num_devices_ret: *mut cl_uint) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clRetainDevice(cl_device_id /* device */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(feature = "opencl_1_2")]
    pub fn clRetainDevice(device: cl_device_id) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clReleaseDevice(cl_device_id /* device */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(feature = "opencl_1_2")]
    pub fn clReleaseDevice(device: cl_device_id ) -> cl_int;

    // Context APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateContext(properties: *const cl_context_properties,
                       num_devices: cl_uint,
                       devices: *const cl_device_id,
                       pfn_notify: Option<extern fn (*const c_char, *const c_void, size_t, *mut c_void)>,
                       user_data: *mut c_void,
                       errcode_ret: *mut cl_int) -> cl_context;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateContextFromType(properties: *const cl_context_properties,
                               device_type: cl_device_type,
                               pfn_notify: Option<extern fn (*const c_char, *const c_void, size_t, *mut c_void)>,
                               user_data: *mut c_void,
                               errcode_ret: *mut cl_int) -> cl_context;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clRetainContext(context: cl_context) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clReleaseContext(context: cl_context) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetContextInfo(context: cl_context,
                        param_name: cl_context_info,
                        param_value_size: size_t,
                        param_value: *mut c_void,
                        param_value_size_ret: *mut size_t) -> cl_int;

    // Command Queue APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateCommandQueue(context: cl_context,
                            device: cl_device_id,
                            properties: cl_command_queue_properties,
                            errcode_ret: *mut cl_int) -> cl_command_queue;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clRetainCommandQueue(command_queue: cl_command_queue) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clReleaseCommandQueue(command_queue: cl_command_queue) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetCommandQueueInfo(command_queue: cl_command_queue,
                             param_name: cl_command_queue_info,
                             param_value_size: size_t,
                             param_value: *mut c_void,
                             param_value_size_ret: *mut size_t) -> cl_int;

    // Not Yet Included. Probably not ever included.
    //#[cfg(feature = "opencl_1_0")]
    // pub fn clSetCommandQueueProperty ...

    // Memory Object APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateBuffer(context: cl_context,
                      flags: cl_mem_flags,
                      size: size_t,
                      host_ptr: *mut c_void,
                      errcode_ret: *mut cl_int) -> cl_mem;

    #[cfg(any(feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateSubBuffer(buffer: cl_mem,
                        flags: cl_mem_flags,
                        buffer_create_type: cl_buffer_create_type,
                        buffer_create_info: *const c_void,
                        errcode_ret: *mut cl_int) -> cl_mem;

    //##### DEPRICATED 1.1 #####
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1"))]
    pub fn clCreateImage2D(context: cl_context,
                    flags: cl_mem_flags,
                    image_format: *const cl_image_format,
                    image_width: size_t,
                    image_depth: size_t,
                    image_slc_pitch: size_t,
                    host_ptr: *mut c_void,
                    errcode_ret: *mut cl_int) -> cl_mem;

    //##### DEPRICATED 1.1 #####
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1"))]
    pub fn clCreateImage3D(context: cl_context,
                    flags: cl_mem_flags,
                    image_format: *const cl_image_format,
                    image_width: size_t,
                    image_height: size_t,
                    image_depth: size_t,
                    image_row_pitch: size_t,
                    image_slice_pitch: size_t,
                    host_ptr: *mut c_void,
                    errcode_ret: *mut cl_int) -> cl_mem;

    //################## NEW 1.2 ###################
    #[cfg(feature = "opencl_1_2")]
    pub fn clCreateImage(context: cl_context,
                        flags: cl_mem_flags,
                        image_format: *const cl_image_format,
                        image_desc: *const cl_image_desc,
                        host_ptr: *mut c_void,
                        errcode_ret: *mut cl_int) -> cl_mem;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clRetainMemObject(memobj: cl_mem) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clReleaseMemObject(memobj: cl_mem) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetSupportedImageFormats(context: cl_context,
                                  flags: cl_mem_flags,
                                  image_type: cl_mem_object_type,
                                  num_entries: cl_uint,
                                  image_formats: *mut cl_image_format,
                                  num_image_formats: *mut cl_uint) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetMemObjectInfo(memobj: cl_mem,
                          param_name: cl_mem_info,
                          param_value_size: size_t,
                          param_value: *mut c_void,
                          param_value_size_ret: *mut size_t) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetImageInfo(image: cl_mem,
                      param_name: cl_image_info,
                      param_value_size: size_t,
                      param_value: *mut c_void,
                      param_value_size_ret: *mut size_t) -> cl_int;

    #[cfg(any(feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clSetMemObjectDestructorCallback(memobj: cl_mem,
                                        pfn_notify: Option<extern fn (cl_mem, *mut c_void)>,
                                        user_data: *mut c_void) -> cl_int;

    // Sampler APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateSampler(context: cl_context,
                       normalize_coords: cl_bool,
                       addressing_mode: cl_addressing_mode,
                       filter_mode: cl_filter_mode,
                       errcode_ret: *mut cl_int) -> cl_sampler;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clRetainSampler(sampler: cl_sampler) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clReleaseSampler(sampler: cl_sampler) ->cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetSamplerInfo(sampler: cl_sampler,
                        param_name: cl_sampler_info,
                        param_value_size: size_t,
                        param_value: *mut c_void,
                        param_value_size_ret: *mut size_t) -> cl_int;

    // Program Object APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateProgramWithSource(context: cl_context,
                                 count: cl_uint,
                                 strings: *const *const c_char,
                                 lengths: *const size_t,
                                 errcode_ret: *mut cl_int) -> cl_program;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateProgramWithBinary(context: cl_context,
                                 num_devices: cl_uint,
                                 device_list: *const cl_device_id,
                                 lengths: *const size_t,
                                 binaries: *const *const c_uchar,
                                 binary_status: *mut cl_int,
                                 errcode_ret: *mut cl_int) -> cl_program;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_program CL_API_CALL
    // clCreateProgramWithBuiltInKernels(cl_context            /* context */,
    //                                  cl_uint               /* num_devices */,
    //                                  const cl_device_id *  /* device_list */,
    //                                  const char *          /* kernel_names */,
    //                                  cl_int *              /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clCreateProgramWithBuiltInKernels(context: cl_context,
                                     num_devices: cl_uint,
                                     device_list: *const cl_device_id,
                                     kernel_names: *mut char,
                                     errcode_ret: *mut cl_int) -> cl_program;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clRetainProgram(program: cl_program) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clReleaseProgram(program: cl_program) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clBuildProgram(program: cl_program,
                      num_devices: cl_uint,
                      device_list: *const cl_device_id,
                      options: *const c_char,
                      pfn_notify: Option<extern fn (cl_program, *mut c_void)>,
                      user_data: *mut c_void) -> cl_int;

    //##### DEPRICATED 1.1 #####
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1"))]
    pub fn clUnloadCompiler() -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clCompileProgram(cl_program           /* program */,
    //                 cl_uint              /* num_devices */,
    //                 const cl_device_id * /* device_list */,
    //                 const char *         /* options */,
    //                 cl_uint              /* num_input_headers */,
    //                 const cl_program *   /* input_headers */,
    //                 const char **        /* header_include_names */,
    //                 void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
    //                 void *               /* user_data */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clCompileProgram(program: cl_program,
                    num_devices: cl_uint,
                    device_list: *const cl_device_id,
                    options: *const c_char,
                    num_input_headers: cl_uint,
                    input_headers: *const cl_program,
                    header_include_names: *const *const c_char,
                    pfn_notify: Option<extern fn (program: cl_program, user_data: *mut c_void)>,
                    user_data: *mut c_void) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_program CL_API_CALL
    // clLinkProgram(cl_context           /* context */,
    //               cl_uint              /* num_devices */,
    //               const cl_device_id * /* device_list */,
    //               const char *         /* options */,
    //               cl_uint              /* num_input_programs */,
    //               const cl_program *   /* input_programs */,
    //               void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
    //               void *               /* user_data */,
    //               cl_int *             /* errcode_ret */ ) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clLinkProgram(context: cl_context,
                  num_devices: cl_uint,
                  device_list: *const cl_device_id,
                  options: *const c_char,
                  num_input_programs: cl_uint,
                  input_programs: *const cl_program,
                  pfn_notify: Option<extern fn (program: cl_program, user_data: *mut c_void)>,
                  user_data: *mut c_void,
                  errcode_ret: *mut cl_int) -> cl_program;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clUnloadPlatformCompiler(cl_platform_id /* platform */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    // [DISABLED DUE TO PLATFORM INCOMPATABILITY]
    // pub fn clUnloadPlatformCompiler(platform: cl_platform_id) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetProgramInfo(program: cl_program,
                        param_name: cl_program_info,
                        param_value_size: size_t,
                        param_value: *mut c_void,
                        param_value_size_ret: *mut size_t) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetProgramBuildInfo(program: cl_program,
                             device: cl_device_id,
                             param_name: cl_program_build_info,
                             param_value_size: size_t,
                             param_value: *mut c_void,
                             param_value_size_ret: *mut size_t) -> cl_int;

    // Kernel Object APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateKernel(program: cl_program,
                      kernel_name: *const c_char,
                      errcode_ret: *mut cl_int) -> cl_kernel;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateKernelsInProgram(program: cl_program,
                                num_kernels: cl_uint,
                                kernels: *mut cl_kernel,
                                num_kernels_ret: *mut cl_uint) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clRetainKernel(kernel: cl_kernel) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clReleaseKernel(kernel: cl_kernel) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clSetKernelArg(kernel: cl_kernel,
                      arg_index: cl_uint,
                      arg_size: size_t,
                      arg_value: *const c_void) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetKernelInfo(kernel: cl_kernel,
                       param_name: cl_kernel_info,
                       param_value_size: size_t,
                       param_value: *mut c_void,
                       param_value_size_ret: *mut size_t) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clGetKernelArgInfo(cl_kernel       /* kernel */,
    //                   cl_uint         /* arg_indx */,
    //                   cl_kernel_arg_info  /* param_name */,
    //                   size_t          /* param_value_size */,
    //                   void *          /* param_value */,
    //                   size_t *        /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clGetKernelArgInfo(kernel: cl_kernel,
                      arg_indx: cl_uint,
                      param_name: cl_kernel_arg_info,
                      param_value_size: size_t,
                      param_value: *mut c_void,
                      param_value_size_ret: *mut size_t) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetKernelWorkGroupInfo(kernel: cl_kernel,
                                device: cl_device_id,
                                param_name: cl_kernel_work_group_info,
                                param_value_size: size_t,
                                param_value: *mut c_void,
                                param_value_size_ret: *mut size_t) -> cl_int;

    // Event Object APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clWaitForEvents(num_events: cl_uint,
                       event_list: *const cl_event) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetEventInfo(event: cl_event,
                      param_name: cl_event_info,
                      param_value_size: size_t,
                      param_value: *mut c_void,
                      param_value_size_ret: *mut size_t) -> cl_int;

    #[cfg(any(feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clCreateUserEvent(context: cl_context,
                         errcode_ret: *mut cl_int) -> cl_event;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clRetainEvent(event: cl_event) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clReleaseEvent(event: cl_event) -> cl_int;

    #[cfg(any(feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clSetUserEventStatus(event: cl_event,
                            execution_status: cl_int) -> cl_int;

    #[cfg(any(feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clSetEventCallback(event: cl_event,
                          command_exec_callback_type: cl_int,
                          pfn_notify: Option<extern fn (cl_event, cl_int, *mut c_void)>,
                          user_data: *mut c_void) -> cl_int;

    // Profiling APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clGetEventProfilingInfo(event: cl_event,
                               param_name: cl_profiling_info,
                               param_value_size: size_t,
                               param_value: *mut c_void,
                               param_value_size_ret: *mut size_t) -> cl_int;

    // Flush and Finish APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clFlush(command_queue: cl_command_queue) -> cl_int;

    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clFinish(command_queue: cl_command_queue) -> cl_int;

    // Enqueued Commands APIs
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clEnqueueReadBuffer(command_queue: cl_command_queue,
                           buffer: cl_mem,
                           blocking_read: cl_bool,
                           offset: size_t,
                           cb: size_t,
                           ptr: *mut c_void,
                           num_events_in_wait_list: cl_uint,
                           event_wait_list: *const cl_event,
                           event: *mut cl_event) -> cl_int;

    #[cfg(any(feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clEnqueueReadBufferRect(command_queue: cl_command_queue,
                               buffer: cl_mem,
                               blocking_read: cl_bool,
                               buffer_origin: *const size_t,
                               host_origin: *const size_t,
                               region: *const size_t,
                               buffer_row_pitch: size_t,
                               buffer_slc_pitch: size_t,
                               host_row_pitch: size_t,
                               host_slc_pitch: size_t,
                               ptr: *mut c_void,
                               num_events_in_wait_list: cl_uint,
                               event_wait_list: *const cl_event,
                               event: *mut cl_event) -> cl_int;

    pub fn clEnqueueWriteBuffer(command_queue: cl_command_queue,
                            buffer: cl_mem,
                            blocking_write: cl_bool,
                            offset: size_t,
                            cb: size_t,
                            ptr: *const c_void,
                            num_events_in_wait_list: cl_uint,
                            event_wait_list: *const cl_event,
                            event: *mut cl_event) -> cl_int;

    #[cfg(any(feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clEnqueueWriteBufferRect(command_queue: cl_command_queue,
                                buffer: cl_mem,
                                blocking_write: cl_bool,
                                buffer_origin: *const size_t,
                                host_origin: *const size_t,
                                region: *const size_t,
                                buffer_row_pitch: size_t,
                                buffer_slc_pitch: size_t,
                                host_row_pitch: size_t,
                                host_slc_pitch: size_t,
                                ptr: *const c_void,
                                num_events_in_wait_list: cl_uint,
                                event_wait_list: *const cl_event,
                                event: *mut cl_event) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueFillBuffer(cl_command_queue   /* command_queue */,
    //                 cl_mem             /* buffer */,
    //                 const void *       /* pattern */,
    //                 size_t             /* pattern_size */,
    //                 size_t             /* offset */,
    //                 size_t             /* size */,
    //                 cl_uint            /* num_events_in_wait_list */,
    //                 const cl_event *   /* event_wait_list */,
    //                 cl_event *         /* event */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clEnqueueFillBuffer(command_queue: cl_command_queue,
                    buffer: cl_mem,
                    pattern: *const c_void,
                    pattern_size: size_t,
                    offset: size_t,
                    size: size_t,
                    num_events_in_wait_list: cl_uint,
                    event_wait_list: *const cl_event,
                    event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyBuffer(command_queue: cl_command_queue,
                           src_buffer: cl_mem,
                           dst_buffer: cl_mem,
                           src_offset: size_t,
                           dst_offset: size_t,
                           cb: size_t,
                           num_events_in_wait_list: cl_uint,
                           event_wait_list: *const cl_event,
                           event: *mut cl_event) -> cl_int;

    #[cfg(any(feature = "opencl_1_1", feature = "opencl_1_2"))]
    pub fn clEnqueueCopyBufferRect(command_queue: cl_command_queue,
                               src_buffer: cl_mem,
                               dst_buffer: cl_mem,
                               src_origin: *const size_t,
                               dst_origin: *const size_t,
                               region: *const size_t,
                               src_row_pitch: size_t,
                               src_slc_pitch: size_t,
                               dst_row_pitch: size_t,
                               dst_slc_pitch: size_t,
                               num_events_in_wait_list: cl_uint,
                               event_wait_list: *const cl_event,
                               event: *mut cl_event) -> cl_int;

    pub fn clEnqueueReadImage(command_queue: cl_command_queue,
                          image: cl_mem,
                          blocking_read: cl_bool,
                          origin: *const size_t,
                          region: *const size_t,
                          row_pitch: size_t,
                          slc_pitch: size_t,
                          ptr: *mut c_void,
                          num_events_in_wait_list: cl_uint,
                          event_wait_list: *const cl_event,
                          event: *mut cl_event) -> cl_int;

    pub fn clEnqueueWriteImage(command_queue: cl_command_queue,
                           image: cl_mem,
                           blocking_write: cl_bool,
                           origin: *const size_t,
                           region: *const size_t,
                           input_row_pitch: size_t,
                           input_slc_pitch: size_t,
                           ptr: *const c_void,
                           num_events_in_wait_list: cl_uint,
                           event_wait_list: *const cl_event,
                           event: *mut cl_event) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueFillImage(cl_command_queue   /* command_queue */,
    //                   cl_mem             /* image */,
    //                   const void *       /* fill_color */,
    //                   const size_t *     /* origin[3] */,
    //                   const size_t *     /* region[3] */,
    //                   cl_uint            /* num_events_in_wait_list */,
    //                   const cl_event *   /* event_wait_list */,
    //                   cl_event *         /* event */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clEnqueueFillImage(command_queue: cl_command_queue,
                      image: cl_mem,
                      fill_color: *const c_void,
                      origin: *const size_t,
                      region: *const size_t,
                      num_events_in_wait_list: cl_uint,
                      event_wait_list: *const cl_event,
                      event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyImage(command_queue: cl_command_queue,
                          src_image: cl_mem,
                          dst_image: cl_mem,
                          src_origin: *const size_t,
                          dst_origin: *const size_t,
                          region: *const size_t,
                          num_events_in_wait_list: cl_uint,
                          event_wait_list: *const cl_event,
                          event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyImageToBuffer(command_queue: cl_command_queue,
                                  src_image: cl_mem,
                                  dst_buffer: cl_mem,
                                  src_origin: *const size_t,
                                  region: *const size_t,
                                  dst_offset: size_t,
                                  num_events_in_wait_list: cl_uint,
                                  event_wait_list: *const cl_event,
                                  event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyBufferToImage(command_queue: cl_command_queue,
                                  src_buffer: cl_mem,
                                  dst_image: cl_mem,
                                  src_offset: size_t,
                                  dst_origin: *const size_t,
                                  region: *const size_t,
                                  num_events_in_wait_list: cl_uint,
                                  event_wait_list: *const cl_event,
                                  event: *mut cl_event) -> cl_int;

    pub fn clEnqueueMapBuffer(command_queue: cl_command_queue,
                          buffer: cl_mem,
                          blocking_map: cl_bool,
                          map_flags: cl_map_flags,
                          offset: size_t,
                          size: size_t,
                          num_events_in_wait_list: cl_uint,
                          event_wait_list: *const cl_event,
                          event: *mut cl_event,
                          errorcode_ret: *mut cl_int) -> *mut c_void;

    pub fn clEnqueueMapImage(command_queue: cl_command_queue,
                         image: cl_mem,
                         blocking_map: cl_bool,
                         map_flags: cl_map_flags,
                         origin: *const size_t,
                         region: *const size_t,
                         image_row_pitch: size_t,
                         image_slc_pitch: size_t,
                         num_events_in_wait_list: cl_uint,
                         event_wait_list: *const cl_event,
                         event: *mut cl_event,
                         errorcode_ret: *mut cl_int) -> *mut c_void;

    pub fn clEnqueueUnmapMemObject(command_queue: cl_command_queue,
                               memobj: cl_mem,
                               mapped_ptr: *mut c_void,
                               num_events_in_wait_list: cl_uint,
                               event_wait_list: *const cl_event,
                               event: *mut cl_event) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueMigrateMemObjects(cl_command_queue       /* command_queue */,
    //                           cl_uint                /* num_mem_objects */,
    //                           const cl_mem *         /* mem_objects */,
    //                           cl_mem_migration_flags /* flags */,
    //                           cl_uint                /* num_events_in_wait_list */,
    //                           const cl_event *       /* event_wait_list */,
    //                           cl_event *             /* event */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clEnqueueMigrateMemObjects(command_queue: cl_command_queue,
                              num_mem_objects: cl_uint,
                              mem_objects: *const cl_mem,
                              flags: cl_mem_migration_flags,
                              num_events_in_wait_list: cl_uint,
                              event_wait_list: *const cl_event,
                              event: *mut cl_event) -> cl_int;

    pub fn clEnqueueNDRangeKernel(command_queue: cl_command_queue,
                              kernel: cl_kernel,
                              work_dim: cl_uint,
                              global_work_offset: *const size_t,
                              global_work_dims: *const size_t,
                              local_work_dims: *const size_t,
                              num_events_in_wait_list: cl_uint,
                              event_wait_list: *const cl_event,
                              event: *mut cl_event) -> cl_int;

    pub fn clEnqueueTask(command_queue: cl_command_queue,
                     kernel: cl_kernel,
                     num_events_in_wait_list: cl_uint,
                     event_wait_list: *const cl_event,
                     event: *mut cl_event) -> cl_int;

    pub fn clEnqueueNativeKernel(command_queue: cl_command_queue,
                             user_func: Option<extern fn (*mut c_void)>,
                             args: *mut c_void,
                             cb_args: size_t,
                             num_mem_objects: cl_uint,
                             mem_list: *const cl_mem,
                             args_mem_loc: *const *const c_void,
                             num_events_in_wait_list: cl_uint,
                             event_wait_list: *const cl_event,
                             event: *mut cl_event) -> cl_int;

    //##### DEPRICATED 1.1 #####
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1"))]
    pub fn clEnqueueMarker(command_queue: cl_command_queue,
                    event: *mut cl_event) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueMarkerWithWaitList(cl_command_queue /* command_queue */,
    //          cl_uint           /* num_events_in_wait_list */,
    //          const cl_event *  /* event_wait_list */,
    //          cl_event *        /* event */) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clEnqueueMarkerWithWaitList(command_queue: cl_command_queue,
             num_events_in_wait_list: cl_uint,
             event_wait_list: *const cl_event,
             event: *mut cl_event) -> cl_int;

    //##### DEPRICATED 1.1 #####
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1"))]
    pub fn clEnqueueWaitForEvents(command_queue: cl_command_queue,
                           num_events: cl_uint,
                           event_list: *mut cl_event) -> cl_int;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueBarrierWithWaitList(
    //          cl_command_queue
    //           // command_queue
    //          ,
    //          cl_uint
    //           // num_events_in_wait_list
    //          ,
    //          const cl_event *
    //           // event_wait_list
    //          ,
    //          cl_event *
    //           // event
    //      ) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clEnqueueBarrierWithWaitList(
             command_queue: cl_command_queue,
             num_events_in_wait_list: cl_uint,
             event_wait_list: *const cl_event,
             event: *mut cl_event) -> cl_int;


    //##### DEPRICATED 1.1 #####
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1"))]
    pub fn clEnqueueBarrier(command_queue: cl_command_queue) -> cl_int;

    //##### DEPRICATED 1.1 #####
    // Extension function access
    // Returns the extension function address for the given function name,
    // or NULL if a valid function can not be found. The client must
    // check to make sure the address is not NULL, before using or
    // or calling the returned function address.
    #[cfg(any(feature = "opencl_1_0", feature = "opencl_1_1"))]
    pub fn clGetExtensionFunctionAddress(func_name: *mut c_char) -> *mut c_void;

    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY void * CL_API_CALL
    // clGetExtensionFunctionAddressForPlatform(cl_platform_id /* platform */,
    //                    const char *
    //                     // func_name
    //                    ) CL_API_SUFFIX__VERSION_1_2;
    //################## NEW 1.2 ###################
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn clGetExtensionFunctionAddressForPlatform(platform: cl_platform_id,
                       func_name: *const c_char) -> *mut c_void;
}
