//! Rust bindings for the OpenCL ABI.
//!
//! Supports OpenCL versions 1.2+ (1.0 and 1.1 not supported).
//!
//! This file was adapted from
//! [KhronosGroup/OpenCL-Headers/CL/cl.h](https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h)
//! and will continue to be updated with additions to newer versions of that
//! document.
//!
//! The layout and format of this document are meant to mimic the original
//! source in order to ease maintenance (as loatheful as that style may be).
//!
//!

#![allow(non_camel_case_types, dead_code, unused_variables, improper_ctypes, non_upper_case_globals)]

// use std::fmt::{Display, Formatter, Result};
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
pub type cl_properties                      = cl_ulong;
pub type cl_device_type                     = cl_bitfield;
pub type cl_platform_info                   = cl_uint;
pub type cl_device_info                     = cl_uint;
pub type cl_device_fp_config                = cl_bitfield;
pub type cl_device_mem_cache_type           = cl_uint;
pub type cl_device_local_mem_type           = cl_uint;
pub type cl_device_exec_capabilities        = cl_bitfield;
pub type cl_device_svm_capabilities         = cl_bitfield;
pub type cl_command_queue_properties        = cl_bitfield;
pub type cl_device_partition_property       = intptr_t;
pub type cl_device_affinity_domain          = cl_bitfield;
pub type cl_context_properties              = intptr_t;
pub type cl_context_info                    = cl_uint;
pub type cl_queue_properties                = cl_bitfield;
pub type cl_command_queue_info              = cl_uint;
pub type cl_channel_order                   = cl_uint;
pub type cl_channel_type                    = cl_uint;
pub type cl_mem_flags                       = cl_bitfield;
pub type cl_svm_mem_flags                   = cl_bitfield;
pub type cl_mem_object_type                 = cl_uint;
pub type cl_mem_info                        = cl_uint;
pub type cl_mem_migration_flags             = cl_bitfield;
pub type cl_image_info                      = cl_uint;
pub type cl_buffer_create_type              = cl_uint;
pub type cl_addressing_mode                 = cl_uint;
pub type cl_filter_mode                     = cl_uint;
pub type cl_sampler_info                    = cl_uint;
pub type cl_map_flags                       = cl_bitfield;
pub type cl_pipe_properties                 = intptr_t;
pub type cl_pipe_info                       = cl_uint;
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
pub type cl_kernel_sub_group_info           = cl_uint;
pub type cl_event_info                      = cl_uint;
pub type cl_command_type                    = cl_uint;
pub type cl_profiling_info                  = cl_uint;
pub type cl_sampler_properties              = cl_bitfield;
pub type cl_kernel_exec_info                = cl_uint;
// #ifdef CL_VERSION_3_0
pub type cl_device_atomic_capabilities         = cl_bitfield;
pub type cl_device_device_enqueue_capabilities = cl_bitfield;
pub type cl_khronos_vendor_id                  = cl_uint;
pub type cl_mem_properties                     = cl_properties;
pub type cl_version                            = cl_uint;
// #endif

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
    // AKA `mem_object` in 2.0+
    pub buffer:             cl_mem,
}

#[repr(C)]
pub struct cl_buffer_region {
    pub origin:     size_t,
    pub size:       size_t,
}

// #ifdef CL_VERSION_3_0
pub const CL_NAME_VERSION_MAX_NAME_SIZE: usize = 64;
#[repr(C)]
pub struct cl_name_version {
    pub version: cl_version,
    pub name: [cl_char; CL_NAME_VERSION_MAX_NAME_SIZE],
}
// #endif

// Error Codes:
pub const CL_SUCCESS:                                      cl_int = 0;
pub const CL_DEVICE_NOT_FOUND:                             cl_int = -1;
pub const CL_DEVICE_NOT_AVAILABLE:                         cl_int = -2;
pub const CL_COMPILER_NOT_AVAILABLE:                       cl_int = -3;
pub const CL_MEM_OBJECT_ALLOCATION_FAILURE:                cl_int = -4;
pub const CL_OUT_OF_RESOURCES:                             cl_int = -5;
pub const CL_OUT_OF_HOST_MEMORY:                           cl_int = -6;
pub const CL_PROFILING_INFO_NOT_AVAILABLE:                 cl_int = -7;
pub const CL_MEM_COPY_OVERLAP:                             cl_int = -8;
pub const CL_IMAGE_FORMAT_MISMATCH:                        cl_int = -9;
pub const CL_IMAGE_FORMAT_NOT_SUPPORTED:                   cl_int = -10;
pub const CL_BUILD_PROGRAM_FAILURE:                        cl_int = -11;
pub const CL_MAP_FAILURE:                                  cl_int = -12;
pub const CL_MISALIGNED_SUB_BUFFER_OFFSET:                 cl_int = -13;
pub const CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    cl_int = -14;
pub const CL_COMPILE_PROGRAM_FAILURE:                      cl_int = -15;
pub const CL_LINKER_NOT_AVAILABLE:                         cl_int = -16;
pub const CL_LINK_PROGRAM_FAILURE:                         cl_int = -17;
pub const CL_DEVICE_PARTITION_FAILED:                      cl_int = -18;
pub const CL_KERNEL_ARG_INFO_NOT_AVAILABLE:                cl_int = -19;

pub const CL_INVALID_VALUE:                                cl_int = -30;
pub const CL_INVALID_DEVICE_TYPE:                          cl_int = -31;
pub const CL_INVALID_PLATFORM:                             cl_int = -32;
pub const CL_INVALID_DEVICE:                               cl_int = -33;
pub const CL_INVALID_CONTEXT:                              cl_int = -34;
pub const CL_INVALID_QUEUE_PROPERTIES:                     cl_int = -35;
pub const CL_INVALID_COMMAND_QUEUE:                        cl_int = -36;
pub const CL_INVALID_HOST_PTR:                             cl_int = -37;
pub const CL_INVALID_MEM_OBJECT:                           cl_int = -38;
pub const CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:              cl_int = -39;
pub const CL_INVALID_IMAGE_SIZE:                           cl_int = -40;
pub const CL_INVALID_SAMPLER:                              cl_int = -41;
pub const CL_INVALID_BINARY:                               cl_int = -42;
pub const CL_INVALID_BUILD_OPTIONS:                        cl_int = -43;
pub const CL_INVALID_PROGRAM:                              cl_int = -44;
pub const CL_INVALID_PROGRAM_EXECUTABLE:                   cl_int = -45;
pub const CL_INVALID_KERNEL_NAME:                          cl_int = -46;
pub const CL_INVALID_KERNEL_DEFINITION:                    cl_int = -47;
pub const CL_INVALID_KERNEL:                               cl_int = -48;
pub const CL_INVALID_ARG_INDEX:                            cl_int = -49;
pub const CL_INVALID_ARG_VALUE:                            cl_int = -50;
pub const CL_INVALID_ARG_SIZE:                             cl_int = -51;
pub const CL_INVALID_KERNEL_ARGS:                          cl_int = -52;
pub const CL_INVALID_WORK_DIMENSION:                       cl_int = -53;
pub const CL_INVALID_WORK_GROUP_SIZE:                      cl_int = -54;
pub const CL_INVALID_WORK_ITEM_SIZE:                       cl_int = -55;
pub const CL_INVALID_GLOBAL_OFFSET:                        cl_int = -56;
pub const CL_INVALID_EVENT_WAIT_LIST:                      cl_int = -57;
pub const CL_INVALID_EVENT:                                cl_int = -58;
pub const CL_INVALID_OPERATION:                            cl_int = -59;
pub const CL_INVALID_GL_OBJECT:                            cl_int = -60;
pub const CL_INVALID_BUFFER_SIZE:                          cl_int = -61;
pub const CL_INVALID_MIP_LEVEL:                            cl_int = -62;
pub const CL_INVALID_GLOBAL_WORK_SIZE:                     cl_int = -63;
pub const CL_INVALID_PROPERTY:                             cl_int = -64;
pub const CL_INVALID_IMAGE_DESCRIPTOR:                     cl_int = -65;
pub const CL_INVALID_COMPILER_OPTIONS:                     cl_int = -66;
pub const CL_INVALID_LINKER_OPTIONS:                       cl_int = -67;
pub const CL_INVALID_DEVICE_PARTITION_COUNT:               cl_int = -68;
pub const CL_INVALID_PIPE_SIZE:                            cl_int = -69;
pub const CL_INVALID_DEVICE_QUEUE:                         cl_int = -70;
// #ifdef CL_VERSION_2_2
pub const CL_INVALID_SPEC_ID:                              cl_int = -71;
pub const CL_MAX_SIZE_RESTRICTION_EXCEEDED:                cl_int = -72;
// #endif
pub const CL_PLATFORM_NOT_FOUND_KHR:                       cl_int = -1001;


// Version:
pub const CL_VERSION_1_0:                               cl_bool = 1;
pub const CL_VERSION_1_1:                               cl_bool = 1;
pub const CL_VERSION_1_2:                               cl_bool = 1;
pub const CL_VERSION_2_0:                               cl_bool = 1;
pub const CL_VERSION_2_1:                               cl_bool = 1;

// cl_bool:
pub const CL_FALSE:                                     cl_bool = 0;
pub const CL_TRUE:                                      cl_bool = 1;
pub const CL_BLOCKING:                                  cl_bool = CL_TRUE;
pub const CL_NON_BLOCKING:                              cl_bool = CL_FALSE;


// cl_platform_info:
pub const CL_PLATFORM_PROFILE:                          cl_uint = 0x0900;
pub const CL_PLATFORM_VERSION:                          cl_uint = 0x0901;
pub const CL_PLATFORM_NAME:                             cl_uint = 0x0902;
pub const CL_PLATFORM_VENDOR:                           cl_uint = 0x0903;
pub const CL_PLATFORM_EXTENSIONS:                       cl_uint = 0x0904;
// #ifdef CL_VERSION_2_1
pub const CL_PLATFORM_HOST_TIMER_RESOLUTION:            cl_uint = 0x0905;
// #endif
// #ifdef CL_VERSION_3_0
pub const CL_PLATFORM_NUMERIC_VERSION:                  cl_uint = 0x0906;
pub const CL_PLATFORM_EXTENSIONS_WITH_VERSION:          cl_uint = 0x0907;
// #endif

// cl_device_type - bitfield:
pub const CL_DEVICE_TYPE_DEFAULT:                      cl_bitfield = 1 << 0;
pub const CL_DEVICE_TYPE_CPU:                          cl_bitfield = 1 << 1;
pub const CL_DEVICE_TYPE_GPU:                          cl_bitfield = 1 << 2;
pub const CL_DEVICE_TYPE_ACCELERATOR:                  cl_bitfield = 1 << 3;
pub const CL_DEVICE_TYPE_CUSTOM:                       cl_bitfield = 1 << 4;
pub const CL_DEVICE_TYPE_ALL:                          cl_bitfield = 0xFFFFFFFF;

// cl_device_info:
pub const CL_DEVICE_TYPE:                                   cl_uint = 0x1000;
pub const CL_DEVICE_VENDOR_ID:                              cl_uint = 0x1001;
pub const CL_DEVICE_MAX_COMPUTE_UNITS:                      cl_uint = 0x1002;
pub const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:               cl_uint = 0x1003;
pub const CL_DEVICE_MAX_WORK_GROUP_SIZE:                    cl_uint = 0x1004;
pub const CL_DEVICE_MAX_WORK_ITEM_SIZES:                    cl_uint = 0x1005;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:            cl_uint = 0x1006;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:           cl_uint = 0x1007;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:             cl_uint = 0x1008;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:            cl_uint = 0x1009;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:           cl_uint = 0x100A;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:          cl_uint = 0x100B;
pub const CL_DEVICE_MAX_CLOCK_FREQUENCY:                    cl_uint = 0x100C;
pub const CL_DEVICE_ADDRESS_BITS:                           cl_uint = 0x100D;
pub const CL_DEVICE_MAX_READ_IMAGE_ARGS:                    cl_uint = 0x100E;
pub const CL_DEVICE_MAX_WRITE_IMAGE_ARGS:                   cl_uint = 0x100F;
pub const CL_DEVICE_MAX_MEM_ALLOC_SIZE:                     cl_uint = 0x1010;
pub const CL_DEVICE_IMAGE2D_MAX_WIDTH:                      cl_uint = 0x1011;
pub const CL_DEVICE_IMAGE2D_MAX_HEIGHT:                     cl_uint = 0x1012;
pub const CL_DEVICE_IMAGE3D_MAX_WIDTH:                      cl_uint = 0x1013;
pub const CL_DEVICE_IMAGE3D_MAX_HEIGHT:                     cl_uint = 0x1014;
pub const CL_DEVICE_IMAGE3D_MAX_DEPTH:                      cl_uint = 0x1015;
pub const CL_DEVICE_IMAGE_SUPPORT:                          cl_uint = 0x1016;
pub const CL_DEVICE_MAX_PARAMETER_SIZE:                     cl_uint = 0x1017;
pub const CL_DEVICE_MAX_SAMPLERS:                           cl_uint = 0x1018;
pub const CL_DEVICE_MEM_BASE_ADDR_ALIGN:                    cl_uint = 0x1019;
pub const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:               cl_uint = 0x101A;
pub const CL_DEVICE_SINGLE_FP_CONFIG:                       cl_uint = 0x101B;
pub const CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:                  cl_uint = 0x101C;
pub const CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:              cl_uint = 0x101D;
pub const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:                  cl_uint = 0x101E;
pub const CL_DEVICE_GLOBAL_MEM_SIZE:                        cl_uint = 0x101F;
pub const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:               cl_uint = 0x1020;
pub const CL_DEVICE_MAX_CONSTANT_ARGS:                      cl_uint = 0x1021;
pub const CL_DEVICE_LOCAL_MEM_TYPE:                         cl_uint = 0x1022;
pub const CL_DEVICE_LOCAL_MEM_SIZE:                         cl_uint = 0x1023;
pub const CL_DEVICE_ERROR_CORRECTION_SUPPORT:               cl_uint = 0x1024;
pub const CL_DEVICE_PROFILING_TIMER_RESOLUTION:             cl_uint = 0x1025;
pub const CL_DEVICE_ENDIAN_LITTLE:                          cl_uint = 0x1026;
pub const CL_DEVICE_AVAILABLE:                              cl_uint = 0x1027;
pub const CL_DEVICE_COMPILER_AVAILABLE:                     cl_uint = 0x1028;
pub const CL_DEVICE_EXECUTION_CAPABILITIES:                 cl_uint = 0x1029;
// DEPRICATED 2.0:
pub const CL_DEVICE_QUEUE_PROPERTIES:                       cl_uint = 0x102A;
pub const CL_DEVICE_QUEUE_ON_HOST_PROPERTIES:               cl_uint = 0x102A;
pub const CL_DEVICE_NAME:                                   cl_uint = 0x102B;
pub const CL_DEVICE_VENDOR:                                 cl_uint = 0x102C;
pub const CL_DRIVER_VERSION:                                cl_uint = 0x102D;
pub const CL_DEVICE_PROFILE:                                cl_uint = 0x102E;
pub const CL_DEVICE_VERSION:                                cl_uint = 0x102F;
pub const CL_DEVICE_EXTENSIONS:                             cl_uint = 0x1030;
pub const CL_DEVICE_PLATFORM:                               cl_uint = 0x1031;
pub const CL_DEVICE_DOUBLE_FP_CONFIG:                       cl_uint = 0x1032;
// 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG which is already defined in "cl_ext.h"
pub const CL_DEVICE_HALF_FP_CONFIG:                         cl_uint = 0x1033;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:            cl_uint = 0x1034;
// DEPRICATED 2.0:
pub const CL_DEVICE_HOST_UNIFIED_MEMORY:                    cl_uint = 0x1035;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:               cl_uint = 0x1036;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:              cl_uint = 0x1037;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:                cl_uint = 0x1038;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:               cl_uint = 0x1039;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:              cl_uint = 0x103A;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:             cl_uint = 0x103B;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:               cl_uint = 0x103C;
pub const CL_DEVICE_OPENCL_C_VERSION:                       cl_uint = 0x103D;
pub const CL_DEVICE_LINKER_AVAILABLE:                       cl_uint = 0x103E;
pub const CL_DEVICE_BUILT_IN_KERNELS:                       cl_uint = 0x103F;
pub const CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:                  cl_uint = 0x1040;
pub const CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:                   cl_uint = 0x1041;
pub const CL_DEVICE_PARENT_DEVICE:                          cl_uint = 0x1042;
pub const CL_DEVICE_PARTITION_MAX_SUB_DEVICES:              cl_uint = 0x1043;
pub const CL_DEVICE_PARTITION_PROPERTIES:                   cl_uint = 0x1044;
pub const CL_DEVICE_PARTITION_AFFINITY_DOMAIN:              cl_uint = 0x1045;
pub const CL_DEVICE_PARTITION_TYPE:                         cl_uint = 0x1046;
pub const CL_DEVICE_REFERENCE_COUNT:                        cl_uint = 0x1047;
pub const CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:            cl_uint = 0x1048;
pub const CL_DEVICE_PRINTF_BUFFER_SIZE:                     cl_uint = 0x1049;
pub const CL_DEVICE_IMAGE_PITCH_ALIGNMENT:                  cl_uint = 0x104A;
pub const CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT:           cl_uint = 0x104B;
// #ifdef CL_VERSION_2_0
pub const CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS:              cl_uint = 0x104C;
pub const CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE:               cl_uint = 0x104D;
pub const CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES:             cl_uint = 0x104E;
pub const CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE:         cl_uint = 0x104F;
pub const CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE:               cl_uint = 0x1050;
pub const CL_DEVICE_MAX_ON_DEVICE_QUEUES:                   cl_uint = 0x1051;
pub const CL_DEVICE_MAX_ON_DEVICE_EVENTS:                   cl_uint = 0x1052;
pub const CL_DEVICE_SVM_CAPABILITIES:                       cl_uint = 0x1053;
pub const CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE:   cl_uint = 0x1054;
pub const CL_DEVICE_MAX_PIPE_ARGS:                          cl_uint = 0x1055;
pub const CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS:           cl_uint = 0x1056;
pub const CL_DEVICE_PIPE_MAX_PACKET_SIZE:                   cl_uint = 0x1057;
pub const CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT:    cl_uint = 0x1058;
pub const CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT:      cl_uint = 0x1059;
pub const CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT:       cl_uint = 0x105A;
// #endif
// #ifdef CL_VERSION_2_1
pub const CL_DEVICE_IL_VERSION:                             cl_uint = 0x105B;
pub const CL_DEVICE_MAX_NUM_SUB_GROUPS:                     cl_uint = 0x105C;
pub const CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: cl_uint = 0x105D;
// #endif
// #ifdef CL_VERSION_3_0
pub const CL_DEVICE_NUMERIC_VERSION:                        cl_uint = 0x105E;
pub const CL_DEVICE_EXTENSIONS_WITH_VERSION:                cl_uint = 0x1060;
pub const CL_DEVICE_ILS_WITH_VERSION:                       cl_uint = 0x1061;
pub const CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION:          cl_uint = 0x1062;
pub const CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES:             cl_uint = 0x1063;
pub const CL_DEVICE_ATOMIC_FENCE_CAPABILITIES:              cl_uint = 0x1064;
pub const CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT:         cl_uint = 0x1065;
pub const CL_DEVICE_OPENCL_C_ALL_VERSIONS:                  cl_uint = 0x1066;
pub const CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:     cl_uint = 0x1067;
pub const CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: cl_uint = 0x1068;
pub const CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT:          cl_uint = 0x1069;
// 0x106A to 0x106E - Reserved for upcoming KHR extension
pub const CL_DEVICE_OPENCL_C_FEATURES:                      cl_uint = 0x106F;
pub const CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES:            cl_uint = 0x1070;
pub const CL_DEVICE_PIPE_SUPPORT:                           cl_uint = 0x1071;
pub const CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED:      cl_uint = 0x1072;
// #endif

// cl_device_fp_config - bitfield:
pub const CL_FP_DENORM:                                 cl_bitfield = 1 << 0;
pub const CL_FP_INF_NAN:                                cl_bitfield = 1 << 1;
pub const CL_FP_ROUND_TO_NEAREST:                       cl_bitfield = 1 << 2;
pub const CL_FP_ROUND_TO_ZERO:                          cl_bitfield = 1 << 3;
pub const CL_FP_ROUND_TO_INF:                           cl_bitfield = 1 << 4;
pub const CL_FP_FMA:                                    cl_bitfield = 1 << 5;
pub const CL_FP_SOFT_FLOAT:                             cl_bitfield = 1 << 6;
pub const CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT:          cl_bitfield = 1 << 7;

// cl_device_mem_cache_type:
pub const CL_NONE:                                      cl_uint = 0x0;
pub const CL_READ_ONLY_CACHE:                           cl_uint = 0x1;
pub const CL_READ_WRITE_CACHE:                          cl_uint = 0x2;

// cl_device_local_mem_type:
pub const CL_LOCAL:                                     cl_uint = 0x1;
pub const CL_GLOBAL:                                    cl_uint = 0x2;

// cl_device_exec_capabilities - bitfield:
pub const CL_EXEC_KERNEL:                               cl_bitfield = 1 << 0;
pub const CL_EXEC_NATIVE_KERNEL:                        cl_bitfield = 1 << 1;

// cl_command_queue_properties - bitfield:
pub const CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE:       cl_bitfield = 1 << 0;
pub const CL_QUEUE_PROFILING_ENABLE:                    cl_bitfield = 1 << 1;
    //###### NEW ########
    pub const CL_QUEUE_ON_DEVICE:                           cl_bitfield = 1 << 2;
    pub const CL_QUEUE_ON_DEVICE_DEFAULT:                   cl_bitfield = 1 << 3;

// cl_context_info:
pub const CL_CONTEXT_REFERENCE_COUNT:                   cl_uint = 0x1080;
pub const CL_CONTEXT_DEVICES:                           cl_uint = 0x1081;
pub const CL_CONTEXT_PROPERTIES:                        cl_uint = 0x1082;
pub const CL_CONTEXT_NUM_DEVICES:                       cl_uint = 0x1083;

// cl_context_info + cl_context_properties:
pub const CL_CONTEXT_PLATFORM:                          cl_uint = 0x1084;
pub const CL_CONTEXT_INTEROP_USER_SYNC:                 cl_uint = 0x1085;

// cl_device_partition_property:
pub const CL_DEVICE_PARTITION_EQUALLY:                  cl_uint = 0x1086;
pub const CL_DEVICE_PARTITION_BY_COUNTS:                cl_uint = 0x1087;
pub const CL_DEVICE_PARTITION_BY_COUNTS_LIST_END:       cl_uint = 0x0;
pub const CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:       cl_uint = 0x1088;

// cl_device_affinity_domain:
pub const CL_DEVICE_AFFINITY_DOMAIN_NUMA:               cl_bitfield = 1 << 0;
pub const CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE:           cl_bitfield = 1 << 1;
pub const CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE:           cl_bitfield = 1 << 2;
pub const CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE:           cl_bitfield = 1 << 3;
pub const CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE:           cl_bitfield = 1 << 4;
pub const CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE: cl_bitfield = 1 << 5;

    //###### NEW ########
    // cl_device_svm_capabilities:
    pub const CL_DEVICE_SVM_COARSE_GRAIN_BUFFER:           cl_bitfield = 1 << 0;
    pub const CL_DEVICE_SVM_FINE_GRAIN_BUFFER:             cl_bitfield = 1 << 1;
    pub const CL_DEVICE_SVM_FINE_GRAIN_SYSTEM:             cl_bitfield = 1 << 2;
    pub const CL_DEVICE_SVM_ATOMICS:                       cl_bitfield = 1 << 3;

// cl_command_queue_info:
pub const CL_QUEUE_CONTEXT:                             cl_uint = 0x1090;
pub const CL_QUEUE_DEVICE:                              cl_uint = 0x1091;
pub const CL_QUEUE_REFERENCE_COUNT:                     cl_uint = 0x1092;
pub const CL_QUEUE_PROPERTIES:                          cl_uint = 0x1093;
// #ifdef CL_VERSION_2_0
pub const CL_QUEUE_SIZE:                                cl_uint = 0x1094;
// #endif
// #ifdef CL_VERSION_2_1
pub const CL_QUEUE_DEVICE_DEFAULT:                      cl_uint = 0x1095;
// #endif
// #ifdef CL_VERSION_3_0
pub const CL_QUEUE_PROPERTIES_ARRAY:                    cl_uint = 0x1098;
// #endif

// cl_mem_flags and cl_svm_mem_flags - bitfield:
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
    //###### NEW ########
    pub const CL_MEM_SVM_FINE_GRAIN_BUFFER:                 cl_bitfield = 1 << 10;   // used by cl_svm_mem_flags only
    pub const CL_MEM_SVM_ATOMICS:                           cl_bitfield = 1 << 11;   // used by cl_svm_mem_flags only
    pub const CL_MEM_KERNEL_READ_AND_WRITE:                 cl_bitfield = 1 << 12;

// cl_mem_migration_flags - bitfield:
pub const CL_MIGRATE_MEM_OBJECT_HOST:                   cl_bitfield = 1 << 0;
pub const CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED:      cl_bitfield = 1 << 1;

// cl_channel_order:
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
    //###### NEW ########
    pub const CL_sRGB:                                      cl_uint = 0x10BF;
    pub const CL_sRGBx:                                     cl_uint = 0x10C0;
    pub const CL_sRGBA:                                     cl_uint = 0x10C1;
    pub const CL_sBGRA:                                     cl_uint = 0x10C2;
    pub const CL_ABGR:                                      cl_uint = 0x10C3;

// cl_channel_type:
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
    //###### NEW ########
    pub const CL_UNORM_INT_101010_2:                        cl_uint = 0x10E0;

// cl_mem_object_type:
pub const CL_MEM_OBJECT_BUFFER:                         cl_uint = 0x10F0;
pub const CL_MEM_OBJECT_IMAGE2D:                        cl_uint = 0x10F1;
pub const CL_MEM_OBJECT_IMAGE3D:                        cl_uint = 0x10F2;
pub const CL_MEM_OBJECT_IMAGE2D_ARRAY:                  cl_uint = 0x10F3;
pub const CL_MEM_OBJECT_IMAGE1D:                        cl_uint = 0x10F4;
pub const CL_MEM_OBJECT_IMAGE1D_ARRAY:                  cl_uint = 0x10F5;
pub const CL_MEM_OBJECT_IMAGE1D_BUFFER:                 cl_uint = 0x10F6;
    //###### NEW ########
    pub const CL_MEM_OBJECT_PIPE:                           cl_uint = 0x10F7;

// cl_mem_info:
pub const CL_MEM_TYPE:                                  cl_uint = 0x1100;
pub const CL_MEM_FLAGS:                                 cl_uint = 0x1101;
pub const CL_MEM_SIZE:                                  cl_uint = 0x1102;
pub const CL_MEM_HOST_PTR:                              cl_uint = 0x1103;
pub const CL_MEM_MAP_COUNT:                             cl_uint = 0x1104;
pub const CL_MEM_REFERENCE_COUNT:                       cl_uint = 0x1105;
pub const CL_MEM_CONTEXT:                               cl_uint = 0x1106;
pub const CL_MEM_ASSOCIATED_MEMOBJECT:                  cl_uint = 0x1107;
pub const CL_MEM_OFFSET:                                cl_uint = 0x1108;
// #ifdef CL_VERSION_2_0
pub const CL_MEM_USES_SVM_POINTER:                      cl_uint = 0x1109;
// #endif
// #ifdef CL_VERSION_3_0
pub const CL_MEM_PROPERTIES:                            cl_uint = 0x110A;
// #endif

// cl_image_info:
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

// cl_pipe_info:
// #ifdef CL_VERSION_2_0
pub const CL_PIPE_PACKET_SIZE:                         cl_uint = 0x1120;
pub const CL_PIPE_MAX_PACKETS:                         cl_uint = 0x1121;
// #endif
// #ifdef CL_VERSION_3_0
pub const CL_PIPE_PROPERTIES:                          cl_uint = 0x1122;
// #endif

// cl_addressing_mode:
pub const CL_ADDRESS_NONE:                              cl_uint = 0x1130;
pub const CL_ADDRESS_CLAMP_TO_EDGE:                     cl_uint = 0x1131;
pub const CL_ADDRESS_CLAMP:                             cl_uint = 0x1132;
pub const CL_ADDRESS_REPEAT:                            cl_uint = 0x1133;
pub const CL_ADDRESS_MIRRORED_REPEAT:                   cl_uint = 0x1134;

// cl_filter_mode:
pub const CL_FILTER_NEAREST:                            cl_uint = 0x1140;
pub const CL_FILTER_LINEAR:                             cl_uint = 0x1141;

// cl_sampler_info:
pub const CL_SAMPLER_REFERENCE_COUNT:                   cl_uint = 0x1150;
pub const CL_SAMPLER_CONTEXT:                           cl_uint = 0x1151;
pub const CL_SAMPLER_NORMALIZED_COORDS:                 cl_uint = 0x1152;
pub const CL_SAMPLER_ADDRESSING_MODE:                   cl_uint = 0x1153;
pub const CL_SAMPLER_FILTER_MODE:                       cl_uint = 0x1154;
// #ifdef CL_VERSION_2_0
// These enumerants are for the cl_khr_mipmap_image extension.
// They have since been added to cl_ext.h with an appropriate
// KHR suffix, but are left here for backwards compatibility.
pub const CL_SAMPLER_MIP_FILTER_MODE:                   cl_uint = 0x1155;
pub const CL_SAMPLER_LOD_MIN:                           cl_uint = 0x1156;
pub const CL_SAMPLER_LOD_MAX:                           cl_uint = 0x1157;
// #endif
// #ifdef CL_VERSION_3_0
pub const CL_SAMPLER_PROPERTIES:                        cl_uint = 0x1158;
// #endif

// cl_map_flags - bitfield:
pub const CL_MAP_READ:                                  cl_bitfield = 1 << 0;
pub const CL_MAP_WRITE:                                 cl_bitfield = 1 << 1;
pub const CL_MAP_WRITE_INVALIDATE_REGION:               cl_bitfield = 1 << 2;

// cl_program_info:
pub const CL_PROGRAM_REFERENCE_COUNT:                   cl_uint = 0x1160;
pub const CL_PROGRAM_CONTEXT:                           cl_uint = 0x1161;
pub const CL_PROGRAM_NUM_DEVICES:                       cl_uint = 0x1162;
pub const CL_PROGRAM_DEVICES:                           cl_uint = 0x1163;
pub const CL_PROGRAM_SOURCE:                            cl_uint = 0x1164;
pub const CL_PROGRAM_BINARY_SIZES:                      cl_uint = 0x1165;
pub const CL_PROGRAM_BINARIES:                          cl_uint = 0x1166;
pub const CL_PROGRAM_NUM_KERNELS:                       cl_uint = 0x1167;
pub const CL_PROGRAM_KERNEL_NAMES:                      cl_uint = 0x1168;
// #ifdef CL_VERSION_2_1
pub const CL_PROGRAM_IL:                                cl_uint = 0x1169;
// #endif
// #ifdef CL_VERSION_2_2
pub const CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT:        cl_uint = 0x116A;
pub const CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT:        cl_uint = 0x116B;
// #endif

// cl_program_build_info:
pub const CL_PROGRAM_BUILD_STATUS:                      cl_uint = 0x1181;
pub const CL_PROGRAM_BUILD_OPTIONS:                     cl_uint = 0x1182;
pub const CL_PROGRAM_BUILD_LOG:                         cl_uint = 0x1183;
pub const CL_PROGRAM_BINARY_TYPE:                       cl_uint = 0x1184;
    //###### NEW ########
    pub const CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE:  cl_uint = 0x1185;

// cl_program_binary_type:
pub const CL_PROGRAM_BINARY_TYPE_NONE:                  cl_bitfield = 0x0;
pub const CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:       cl_bitfield = 0x1;
pub const CL_PROGRAM_BINARY_TYPE_LIBRARY:               cl_bitfield = 0x2;
pub const CL_PROGRAM_BINARY_TYPE_EXECUTABLE:            cl_bitfield = 0x4;

// cl_build_status:
pub const CL_BUILD_SUCCESS:                             cl_int = 0;
pub const CL_BUILD_NONE:                                cl_int = -1;
pub const CL_BUILD_ERROR:                               cl_int = -2;
pub const CL_BUILD_IN_PROGRESS:                         cl_int = -3;

// cl_kernel_info:
pub const CL_KERNEL_FUNCTION_NAME:                      cl_uint = 0x1190;
pub const CL_KERNEL_NUM_ARGS:                           cl_uint = 0x1191;
pub const CL_KERNEL_REFERENCE_COUNT:                    cl_uint = 0x1192;
pub const CL_KERNEL_CONTEXT:                            cl_uint = 0x1193;
pub const CL_KERNEL_PROGRAM:                            cl_uint = 0x1194;
pub const CL_KERNEL_ATTRIBUTES:                         cl_uint = 0x1195;

// cl_kernel_arg_info:
pub const CL_KERNEL_ARG_ADDRESS_QUALIFIER:              cl_uint = 0x1196;
pub const CL_KERNEL_ARG_ACCESS_QUALIFIER:               cl_uint = 0x1197;
pub const CL_KERNEL_ARG_TYPE_NAME:                      cl_uint = 0x1198;
pub const CL_KERNEL_ARG_TYPE_QUALIFIER:                 cl_uint = 0x1199;
pub const CL_KERNEL_ARG_NAME:                           cl_uint = 0x119A;

// cl_kernel_arg_address_qualifier:
pub const CL_KERNEL_ARG_ADDRESS_GLOBAL:                 cl_uint = 0x119B;
pub const CL_KERNEL_ARG_ADDRESS_LOCAL:                  cl_uint = 0x119C;
pub const CL_KERNEL_ARG_ADDRESS_CONSTANT:               cl_uint = 0x119D;
pub const CL_KERNEL_ARG_ADDRESS_PRIVATE:                cl_uint = 0x119E;

// cl_kernel_arg_access_qualifier:
pub const CL_KERNEL_ARG_ACCESS_READ_ONLY:               cl_uint = 0x11A0;
pub const CL_KERNEL_ARG_ACCESS_WRITE_ONLY:              cl_uint = 0x11A1;
pub const CL_KERNEL_ARG_ACCESS_READ_WRITE:              cl_uint = 0x11A2;
pub const CL_KERNEL_ARG_ACCESS_NONE:                    cl_uint = 0x11A3;

// cl_kernel_arg_type_qualifer:
pub const CL_KERNEL_ARG_TYPE_NONE:                      cl_bitfield = 0;
pub const CL_KERNEL_ARG_TYPE_CONST:                     cl_bitfield = 1 << 0;
pub const CL_KERNEL_ARG_TYPE_RESTRICT:                  cl_bitfield = 1 << 1;
pub const CL_KERNEL_ARG_TYPE_VOLATILE:                  cl_bitfield = 1 << 2;
    //###### NEW ########
    pub const CL_KERNEL_ARG_TYPE_PIPE:                      cl_bitfield = 1 << 3;

// cl_kernel_work_group_info:
pub const CL_KERNEL_WORK_GROUP_SIZE:                    cl_uint = 0x11B0;
pub const CL_KERNEL_COMPILE_WORK_GROUP_SIZE:            cl_uint = 0x11B1;
pub const CL_KERNEL_LOCAL_MEM_SIZE:                     cl_uint = 0x11B2;
pub const CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: cl_uint = 0x11B3;
pub const CL_KERNEL_PRIVATE_MEM_SIZE:                   cl_uint = 0x11B4;
pub const CL_KERNEL_GLOBAL_WORK_SIZE:                   cl_uint = 0x11B5;

    //###### NEW ########
    // cl_kernel_sub_group_info:
    pub const CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE:    cl_uint = 0x2033;
    pub const CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE:       cl_uint = 0x2034;
    pub const CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT:    cl_uint = 0x11B8;
    pub const CL_KERNEL_MAX_NUM_SUB_GROUPS:                cl_uint = 0x11B9;
    pub const CL_KERNEL_COMPILE_NUM_SUB_GROUPS:            cl_uint = 0x11BA;

    //###### NEW ########
    // cl_kernel_exec_info:
    pub const CL_KERNEL_EXEC_INFO_SVM_PTRS:                cl_uint = 0x11B6;
    pub const CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM:   cl_uint = 0x11B7;

// cl_event_info:
pub const CL_EVENT_COMMAND_QUEUE:                       cl_uint = 0x11D0;
pub const CL_EVENT_COMMAND_TYPE:                        cl_uint = 0x11D1;
pub const CL_EVENT_REFERENCE_COUNT:                     cl_uint = 0x11D2;
pub const CL_EVENT_COMMAND_EXECUTION_STATUS:            cl_uint = 0x11D3;
pub const CL_EVENT_CONTEXT:                             cl_uint = 0x11D4;

// cl_command_type:
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
// #ifdef CL_VERSION_2_0
pub const CL_COMMAND_SVM_FREE:                          cl_uint = 0x1209;
pub const CL_COMMAND_SVM_MEMCPY:                        cl_uint = 0x120A;
pub const CL_COMMAND_SVM_MEMFILL:                       cl_uint = 0x120B;
pub const CL_COMMAND_SVM_MAP:                           cl_uint = 0x120C;
pub const CL_COMMAND_SVM_UNMAP:                         cl_uint = 0x120D;
// #endif
// #ifdef CL_VERSION_3_0
pub const CL_COMMAND_SVM_MIGRATE_MEM:                   cl_uint = 0x120E;
// #endif

// command execution status:
pub const CL_COMPLETE:                                  cl_int = 0x0;
pub const CL_RUNNING:                                   cl_int = 0x1;
pub const CL_SUBMITTED:                                 cl_int = 0x2;
pub const CL_QUEUED:                                    cl_int = 0x3;

// cl_buffer_create_type:
pub const CL_BUFFER_CREATE_TYPE_REGION:                 cl_uint = 0x1220;

// cl_profiling_info:
pub const CL_PROFILING_COMMAND_QUEUED:                  cl_uint = 0x1280;
pub const CL_PROFILING_COMMAND_SUBMIT:                  cl_uint = 0x1281;
pub const CL_PROFILING_COMMAND_START:                   cl_uint = 0x1282;
pub const CL_PROFILING_COMMAND_END:                     cl_uint = 0x1283;
    //###### NEW ########
    pub const CL_PROFILING_COMMAND_COMPLETE:                cl_uint = 0x1284;

// cl_device_device_enqueue_capabilities - bitfield
// #ifdef CL_VERSION_3_0
pub const CL_DEVICE_QUEUE_SUPPORTED:                cl_bitfield = 1 << 0;
pub const CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT:      cl_bitfield = 1 << 1;
// #endif

// cl_khronos_vendor_id
pub const CL_KHRONOS_VENDOR_ID_CODEPLAY:                cl_uint = 0x10004;

// #ifdef CL_VERSION_3_0

// cl_version
pub const CL_VERSION_MAJOR_BITS:                    cl_bitfield = 10;
pub const CL_VERSION_MINOR_BITS:                    cl_bitfield = 10;
pub const CL_VERSION_PATCH_BITS:                    cl_bitfield = 12;

pub const CL_VERSION_MAJOR_MASK: cl_bitfield = (1 << CL_VERSION_MAJOR_BITS) - 1;
pub const CL_VERSION_MINOR_MASK: cl_bitfield = (1 << CL_VERSION_MINOR_BITS) - 1;
pub const CL_VERSION_PATCH_MASK: cl_bitfield = (1 << CL_VERSION_PATCH_BITS) - 1;
// #endif

//#[link_args = "-L$OPENCL_LIB -lOpenCL"]
#[cfg_attr(target_os = "macos", link(name = "OpenCL", kind = "framework"))]
#[cfg_attr(target_os = "windows", link(name = "OpenCL"))]
#[cfg_attr(not(target_os = "macos"), link(name = "OpenCL"))]
extern "system" {
    // Platform API:
    pub fn clGetPlatformIDs(num_entries: cl_uint,
                            platforms: *mut cl_platform_id,
                            num_platforms: *mut cl_uint) -> cl_int;

    pub fn clGetPlatformInfo(platform: cl_platform_id,
                             param_name: cl_platform_info,
                             param_value_size: size_t,
                             param_value: *mut c_void,
                             param_value_size_ret: *mut size_t) -> cl_int;

    // Device APIs:
    pub fn clGetDeviceIDs(platform: cl_platform_id,
                          device_type: cl_device_type,
                          num_entries: cl_uint,
                          devices: *mut cl_device_id,
                          num_devices: *mut cl_uint) -> cl_int;

    pub fn clGetDeviceInfo(device: cl_device_id,
                       param_name: cl_device_info,
                       param_value_size: size_t,
                       param_value: *mut c_void,
                       param_value_size_ret: *mut size_t) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clCreateSubDevices(cl_device_id                         /* in_device */,
    //                    const cl_device_partition_property * /* properties */,
    //                    cl_uint                              /* num_devices */,
    //                    cl_device_id *                       /* out_devices */,
    //                    cl_uint *                            /* num_devices_ret */) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clCreateSubDevices(in_device: cl_device_id,
                       properties: *const cl_device_partition_property,
                       num_devices: cl_uint,
                       out_devices: *mut cl_device_id,
                       num_devices_ret: *mut cl_uint) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clRetainDevice(cl_device_id /* device */) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clRetainDevice(device: cl_device_id) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clReleaseDevice(cl_device_id /* device */) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clReleaseDevice(device: cl_device_id ) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clSetDefaultDeviceCommandQueue(cl_context           /* context */,
    //                                cl_device_id         /* device */,
    //                                cl_command_queue     /* command_queue */) CL_API_SUFFIX__VERSION_2_1;
    //############################### NEW 2.1 #################################
    #[cfg(feature = "opencl_version_2_1")]
    pub fn clSetDefaultDeviceCommandQueue(context: cl_context,
                                          device: cl_device_id,
                                          command_queue: cl_command_queue) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clGetDeviceAndHostTimer(cl_device_id    /* device */,
    //                         cl_ulong*       /* device_timestamp */,
    //                         cl_ulong*       /* host_timestamp */) CL_API_SUFFIX__VERSION_2_1;
    //############################### NEW 2.1 #################################
    #[cfg(feature = "opencl_version_2_1")]
    pub fn clGetDeviceAndHostTimer(device: cl_device_id,
                                   device_timestamp: cl_ulong,
                                   host_timestamp: cl_ulong) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clGetHostTimer(cl_device_id /* device */,
    //                cl_ulong *   /* host_timestamp */)  CL_API_SUFFIX__VERSION_2_1;
    //############################### NEW 2.1 #################################
    #[cfg(feature = "opencl_version_2_1")]
    pub fn clGetHostTimer(device: cl_device_id,
                          host_timestamp: cl_ulong) -> cl_int;

    // Context APIs:
    pub fn clCreateContext(properties: *const cl_context_properties,
                       num_devices: cl_uint,
                       devices: *const cl_device_id,
                       pfn_notify: Option<extern fn (*const c_char, *const c_void, size_t, *mut c_void)>,
                       user_data: *mut c_void,
                       errcode_ret: *mut cl_int) -> cl_context;

    pub fn clCreateContextFromType(properties: *const cl_context_properties,
                               device_type: cl_device_type,
                               pfn_notify: Option<extern fn (*const c_char, *const c_void, size_t, *mut c_void)>,
                               user_data: *mut c_void,
                               errcode_ret: *mut cl_int) -> cl_context;

    pub fn clRetainContext(context: cl_context) -> cl_int;

    pub fn clReleaseContext(context: cl_context) -> cl_int;

    pub fn clGetContextInfo(context: cl_context,
                        param_name: cl_context_info,
                        param_value_size: size_t,
                        param_value: *mut c_void,
                        param_value_size_ret: *mut size_t) -> cl_int;


    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clSetContextDestructorCallback(cl_context         context,
    //                                void (CL_CALLBACK* pfn_notify)(cl_context context,
    //                                                               void* user_data),
    //                                void*              user_data) CL_API_SUFFIX__VERSION_3_0;
    //############################### NEW 3.0 #################################
    #[cfg(feature = "opencl_version_3_0")]
    pub fn clSetContextDestructorCallback(context: cl_context,
        pfn_notify: Option<extern fn (context: cl_context, user_data: *mut c_void)>,
        user_data: *mut c_void) -> cl_int;

    // Command Queue APIs
    //########################## DEPRICATED 1.2 ##############################
    pub fn clCreateCommandQueue(context: cl_context,
                            device: cl_device_id,
                            properties: cl_command_queue_properties,
                            errcode_ret: *mut cl_int) -> cl_command_queue;

    // extern CL_API_ENTRY cl_command_queue CL_API_CALL
    // clCreateCommandQueueWithProperties(cl_context               /* context */,
    //                                    cl_device_id             /* device */,
    //                                    const cl_queue_properties *    /* properties */,
    //                                    cl_int *                 /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clCreateCommandQueueWithProperties(context: cl_context,
                                              device: cl_device_id,
                                              properties: *const cl_queue_properties,
                                              errcode_ret: *mut cl_int) -> cl_command_queue;

    pub fn clRetainCommandQueue(command_queue: cl_command_queue) -> cl_int;

    pub fn clReleaseCommandQueue(command_queue: cl_command_queue) -> cl_int;

    pub fn clGetCommandQueueInfo(command_queue: cl_command_queue,
                             param_name: cl_command_queue_info,
                             param_value_size: size_t,
                             param_value: *mut c_void,
                             param_value_size_ret: *mut size_t) -> cl_int;

    // Memory Object APIs:
    pub fn clCreateBuffer(context: cl_context,
                      flags: cl_mem_flags,
                      size: size_t,
                      host_ptr: *mut c_void,
                      errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clCreateSubBuffer(buffer: cl_mem,
                        flags: cl_mem_flags,
                        buffer_create_type: cl_buffer_create_type,
                        buffer_create_info: *const c_void,
                        errcode_ret: *mut cl_int) -> cl_mem;

    //########################## DEPRICATED 1.1 ##############################
    pub fn clCreateImage2D(context: cl_context,
                    flags: cl_mem_flags,
                    image_format: *mut cl_image_format,
                    image_width: size_t,
                    image_depth: size_t,
                    image_slc_pitch: size_t,
                    host_ptr: *mut c_void,
                    errcode_ret: *mut cl_int) -> cl_mem;

    //########################## DEPRICATED 1.1 ##############################
    pub fn clCreateImage3D(context: cl_context,
                    flags: cl_mem_flags,
                    image_format: *mut cl_image_format,
                    image_width: size_t,
                    image_height: size_t,
                    image_depth: size_t,
                    image_row_pitch: size_t,
                    image_slc_pitch: size_t,
                    host_ptr: *mut c_void,
                    errcode_ret: *mut cl_int) -> cl_mem;

    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clCreateImage(context: cl_context,
                        flags: cl_mem_flags,
                        image_format: *const cl_image_format,
                        image_desc: *const cl_image_desc,
                        host_ptr: *mut c_void,
                        errcode_ret: *mut cl_int) -> cl_mem;

    // extern CL_API_ENTRY cl_mem CL_API_CALL
    // clCreatePipe(cl_context                 /* context */,
    //              cl_mem_flags               /* flags */,
    //              cl_uint                    /* pipe_packet_size */,
    //              cl_uint                    /* pipe_max_packets */,
    //              const cl_pipe_properties * /* properties */,
    //              cl_int *                   /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clCreatePipe(context: cl_context,
                        flags: cl_mem_flags,
                        pipe_packet_size: cl_uint,
                        pipe_max_packets: cl_uint,
                        properties: *const cl_pipe_properties,
                        errcode_ret: *mut cl_int) -> cl_mem;

    //############################### NEW 3.0 #################################
    // extern CL_API_ENTRY cl_mem CL_API_CALL
    // clCreateBufferWithProperties(cl_context                context,
    //                              const cl_mem_properties * properties,
    //                              cl_mem_flags              flags,
    //                              size_t                    size,
    //                              void *                    host_ptr,
    //                              cl_int *                  errcode_ret) CL_API_SUFFIX__VERSION_3_0;
    #[cfg(feature = "opencl_version_3_0")]
    pub fn clCreateBufferWithProperties(context: cl_context,
                                        properties: *const cl_mem_properties,
                                        flags: cl_mem_flags,
                                        size: size_t,
                                        host_ptr: *mut c_void,
                                        errcode_ret: *mut cl_int) -> cl_mem;

    // extern CL_API_ENTRY cl_mem CL_API_CALL
    // clCreateImageWithProperties(cl_context                context,
    //                             const cl_mem_properties * properties,
    //                             cl_mem_flags              flags,
    //                             const cl_image_format *   image_format,
    //                             const cl_image_desc *     image_desc,
    //                             void *                    host_ptr,
    //                             cl_int *                  errcode_ret) CL_API_SUFFIX__VERSION_3_0;
    #[cfg(feature = "opencl_version_3_0")]
    pub fn clCreateImageWithProperties(context: cl_context,
                                       properties: *const cl_mem_properties,
                                       flags: cl_mem_flags,
                                       image_format: *const cl_image_format,
                                       image_desc: *const cl_image_desc,
                                       host_ptr: *mut c_void,
                                       errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clRetainMemObject(memobj: cl_mem) -> cl_int;

    pub fn clReleaseMemObject(memobj: cl_mem) -> cl_int;

    pub fn clGetSupportedImageFormats(context: cl_context,
                                  flags: cl_mem_flags,
                                  image_type: cl_mem_object_type,
                                  num_entries: cl_uint,
                                  image_formats: *mut cl_image_format,
                                  num_image_formats: *mut cl_uint) -> cl_int;

    pub fn clGetMemObjectInfo(memobj: cl_mem,
                          param_name: cl_mem_info,
                          param_value_size: size_t,
                          param_value: *mut c_void,
                          param_value_size_ret: *mut size_t) -> cl_int;

    pub fn clGetImageInfo(image: cl_mem,
                      param_name: cl_image_info,
                      param_value_size: size_t,
                      param_value: *mut c_void,
                      param_value_size_ret: *mut size_t) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clGetPipeInfo(cl_mem           /* pipe */,
    //               cl_pipe_info     /* param_name */,
    //               size_t           /* param_value_size */,
    //               void *           /* param_value */,
    //               size_t *         /* param_value_size_ret */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clGetPipeInfo(pipe: cl_mem,
                         param_name: cl_pipe_info,
                         param_value_size: size_t,
                         param_value: *mut c_void,
                         param_value_size_ret: *mut size_t) -> cl_int;

    pub fn clSetMemObjectDestructorCallback(memobj: cl_mem,
                                            pfn_notify: Option<extern fn (cl_mem, *mut c_void)>,
                                            user_data: *mut c_void) -> cl_int;

    // SVM Allocation APIs
    // extern CL_API_ENTRY void * CL_API_CALL
    // clSVMAlloc(cl_context       /* context */,
    //            cl_svm_mem_flags /* flags */,
    //            size_t           /* size */,
    //            cl_uint          /* alignment */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clSVMAlloc(context: cl_context,
                      flags: cl_svm_mem_flags,
                      size: size_t,
                      alignment: cl_uint) -> *mut c_void;

    // extern CL_API_ENTRY void CL_API_CALL
    // clSVMFree(cl_context        /* context */,
    //           void *            /* svm_pointer */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clSVMFree(context: cl_context,
                     svm_pointer: *mut c_void);

    // Sampler APIs:
    pub fn clCreateSampler(context: cl_context,
                       normalize_coords: cl_bool,
                       addressing_mode: cl_addressing_mode,
                       filter_mode: cl_filter_mode,
                       errcode_ret: *mut cl_int) -> cl_sampler;

    // extern CL_API_ENTRY cl_sampler CL_API_CALL
    // clCreateSamplerWithProperties(cl_context                     /* context */,
    //                               const cl_sampler_properties *  /* normalized_coords */,
    //                               cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clCreateSamplerWithProperties(context: cl_context,
                                         normalized_coords: *const cl_sampler_properties,
                                         errcode_ret: *mut cl_int) -> cl_sampler;

    pub fn clRetainSampler(sampler: cl_sampler) -> cl_int;

    pub fn clReleaseSampler(sampler: cl_sampler) ->cl_int;

    pub fn clGetSamplerInfo(sampler: cl_sampler,
                        param_name: cl_sampler_info,
                        param_value_size: size_t,
                        param_value: *mut c_void,
                        param_value_size_ret: *mut size_t) -> cl_int;

    // Program Object APIs:
    pub fn clCreateProgramWithSource(context: cl_context,
                                 count: cl_uint,
                                 strings: *const *const c_char,
                                 lengths: *const size_t,
                                 errcode_ret: *mut cl_int) -> cl_program;

    pub fn clCreateProgramWithBinary(context: cl_context,
                                 num_devices: cl_uint,
                                 device_list: *const cl_device_id,
                                 lengths: *const size_t,
                                 binaries: *const *const c_uchar,
                                 binary_status: *mut cl_int,
                                 errcode_ret: *mut cl_int) -> cl_program;

    // extern CL_API_ENTRY cl_program CL_API_CALL
    // clCreateProgramWithBuiltInKernels(cl_context            /* context */,
    //                                  cl_uint               /* num_devices */,
    //                                  const cl_device_id *  /* device_list */,
    //                                  const char *          /* kernel_names */,
    //                                  cl_int *              /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clCreateProgramWithBuiltInKernels(context: cl_context,
                                     num_devices: cl_uint,
                                     device_list: *const cl_device_id,
                                     kernel_names: *mut char,
                                     errcode_ret: *mut cl_int) -> cl_program;

    // extern CL_API_ENTRY cl_program CL_API_CALL
    // clCreateProgramWithIL(cl_context    /* context */,
    //                      const void*    /* il */,
    //                      size_t         /* length */,
    //                      cl_int*        /* errcode_ret */) CL_API_SUFFIX__VERSION_2_1;
    //############################### NEW 2.1 #################################
    #[cfg(feature = "opencl_version_2_1")]
    pub fn clCreateProgramWithIL(context: cl_context,
                                 il: *const c_void,
                                 length: size_t,
                                 errcode_ret: *mut cl_int) -> cl_program;

    pub fn clRetainProgram(program: cl_program) -> cl_int;

    pub fn clReleaseProgram(program: cl_program) -> cl_int;

    pub fn clBuildProgram(program: cl_program,
                      num_devices: cl_uint,
                      device_list: *const cl_device_id,
                      options: *const c_char,
                      pfn_notify: Option<extern fn (cl_program, *mut c_void)>,
                      user_data: *mut c_void) -> cl_int;

    //########################## DEPRICATED 1.1 ##############################
    pub fn clUnloadCompiler() -> cl_int;

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
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clCompileProgram(program: cl_program,
                    num_devices: cl_uint,
                    device_list: *const cl_device_id,
                    options: *const c_char,
                    num_input_headers: cl_uint,
                    input_headers: *const cl_program,
                    header_include_names: *const *const c_char,
                    pfn_notify: Option<extern fn (program: cl_program, user_data: *mut c_void)>,
                    user_data: *mut c_void) -> cl_int;

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
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clLinkProgram(context: cl_context,
                  num_devices: cl_uint,
                  device_list: *const cl_device_id,
                  options: *const c_char,
                  num_input_programs: cl_uint,
                  input_programs: *const cl_program,
                  pfn_notify: Option<extern fn (program: cl_program, user_data: *mut c_void)>,
                  user_data: *mut c_void,
                  errcode_ret: *mut cl_int) -> cl_program;

    //############################### NEW 2.2 #################################
    // extern CL_API_ENTRY CL_EXT_PREFIX__VERSION_2_2_DEPRECATED cl_int CL_API_CALL
    // clSetProgramReleaseCallback(cl_program          program,
    //                             void (CL_CALLBACK * pfn_notify)(cl_program program,
    //                                                             void * user_data),
    //                             void *              user_data) CL_EXT_SUFFIX__VERSION_2_2_DEPRECATED;
    #[cfg(feature = "opencl_version_2_2")]
    pub fn clSetProgramReleaseCallback(program: cl_program,
        pfn_notify: Option<extern fn (program: cl_program, user_data: *mut c_void)>,
                  user_data: *mut c_void) -> cl_int;
    
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clSetProgramSpecializationConstant(cl_program  program,
    //                                     cl_uint     spec_id,
    //                                     size_t      spec_size,
    //                                     const void* spec_value) CL_API_SUFFIX__VERSION_2_2;
    #[cfg(feature = "opencl_version_2_2")]
    pub fn clSetProgramSpecializationConstant(program: cl_program,
        spec_id: cl_uint,
        spec_size: size_t,
        spec_value: *const c_void) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clUnloadPlatformCompiler(cl_platform_id /* platform */) CL_API_SUFFIX__VERSION_1_2;
    // //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    // [DISABLED DUE TO PLATFORM INCOMPATABILITY]
    // pub fn clUnloadPlatformCompiler(platform: cl_platform_id) -> cl_int;

    pub fn clGetProgramInfo(program: cl_program,
                        param_name: cl_program_info,
                        param_value_size: size_t,
                        param_value: *mut c_void,
                        param_value_size_ret: *mut size_t) -> cl_int;

    pub fn clGetProgramBuildInfo(program: cl_program,
                             device: cl_device_id,
                             param_name: cl_program_build_info,
                             param_value_size: size_t,
                             param_value: *mut c_void,
                             param_value_size_ret: *mut size_t) -> cl_int;

    // Kernel Object APIs:
    pub fn clCreateKernel(program: cl_program,
                      kernel_name: *const c_char,
                      errcode_ret: *mut cl_int) -> cl_kernel;

    pub fn clCreateKernelsInProgram(program: cl_program,
                                num_kernels: cl_uint,
                                kernels: *mut cl_kernel,
                                num_kernels_ret: *mut cl_uint) -> cl_int;

    // extern CL_API_ENTRY cl_kernel CL_API_CALL
    // clCloneKernel(cl_kernel     /* source_kernel */,
    //               cl_int*       /* errcode_ret */) CL_API_SUFFIX__VERSION_2_1;
    //############################### NEW 2.1 #################################
    #[cfg(feature = "opencl_version_2_1")]
    pub fn clCloneKernel(source_kernel: cl_kernel,
                         errcode_ret: *mut cl_int) -> cl_kernel;


    pub fn clRetainKernel(kernel: cl_kernel) -> cl_int;

    pub fn clReleaseKernel(kernel: cl_kernel) -> cl_int;

    pub fn clSetKernelArg(kernel: cl_kernel,
                      arg_index: cl_uint,
                      arg_size: size_t,
                      arg_value: *const c_void) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clSetKernelArgSVMPointer(cl_kernel    /* kernel */,
    //                          cl_uint      /* arg_index */,
    //                          const void * /* arg_value */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clSetKernelArgSVMPointer(kernel: cl_kernel,
                                    arg_index: cl_uint,
                                    arg_value: *const c_void) -> cl_int;


    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clSetKernelExecInfo(cl_kernel            /* kernel */,
    //                     cl_kernel_exec_info  /* param_name */,
    //                     size_t               /* param_value_size */,
    //                     const void *         /* param_value */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clSetKernelExecInfo(kernel: cl_kernel,
                               param_name: cl_kernel_exec_info,
                               param_value_size: size_t,
                               param_value: *const c_void) -> cl_int;


    pub fn clGetKernelInfo(kernel: cl_kernel,
                       param_name: cl_kernel_info,
                       param_value_size: size_t,
                       param_value: *mut c_void,
                       param_value_size_ret: *mut size_t) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clGetKernelArgInfo(cl_kernel       /* kernel */,
    //                   cl_uint         /* arg_indx */,
    //                   cl_kernel_arg_info  /* param_name */,
    //                   size_t          /* param_value_size */,
    //                   void *          /* param_value */,
    //                   size_t *        /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clGetKernelArgInfo(kernel: cl_kernel,
                      arg_indx: cl_uint,
                      param_name: cl_kernel_arg_info,
                      param_value_size: size_t,
                      param_value: *mut c_void,
                      param_value_size_ret: *mut size_t) -> cl_int;

    pub fn clGetKernelWorkGroupInfo(kernel: cl_kernel,
                                device: cl_device_id,
                                param_name: cl_kernel_work_group_info,
                                param_value_size: size_t,
                                param_value: *mut c_void,
                                param_value_size_ret: *mut size_t) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clGetKernelSubGroupInfo(cl_kernel                   /* kernel */,
    //                         cl_device_id                /* device */,
    //                         cl_kernel_sub_group_info    /* param_name */,
    //                         size_t                      /* input_value_size */,
    //                         const void*                 /*input_value */,
    //                         size_t                      /* param_value_size */,
    //                         void*                       /* param_value */,
    //                         size_t*                     /* param_value_size_ret */ ) CL_API_SUFFIX__VERSION_2_1;
    //############################### NEW 2.1 #################################
    #[cfg(feature = "opencl_version_2_1")]
    pub fn clGetKernelSubGroupInfo(kernel: cl_kernel,
                                   device: cl_device_id,
                                   param_name: cl_kernel_sub_group_info,
                                   input_value_size: size_t,
                                   input_value: *const c_void,
                                   param_value_size: size_t,
                                   param_value: *mut c_void,
                                   param_value_size_ret: *mut size_t) -> cl_int;


    // Event Object APIs:
    pub fn clWaitForEvents(num_events: cl_uint,
                       event_list: *const cl_event) -> cl_int;

    pub fn clGetEventInfo(event: cl_event,
                      param_name: cl_event_info,
                      param_value_size: size_t,
                      param_value: *mut c_void,
                      param_value_size_ret: *mut size_t) -> cl_int;

    pub fn clCreateUserEvent(context: cl_context,
                         errcode_ret: *mut cl_int) -> cl_event;

    pub fn clRetainEvent(event: cl_event) -> cl_int;

    pub fn clReleaseEvent(event: cl_event) -> cl_int;

    pub fn clSetUserEventStatus(event: cl_event,
                            execution_status: cl_int) -> cl_int;

    pub fn clSetEventCallback(event: cl_event,
                          command_exec_callback_type: cl_int,
                          pfn_notify: Option<extern fn (cl_event, cl_int, *mut c_void)>,
                          user_data: *mut c_void) -> cl_int;

    // Profiling APIs:
    pub fn clGetEventProfilingInfo(event: cl_event,
                               param_name: cl_profiling_info,
                               param_value_size: size_t,
                               param_value: *mut c_void,
                               param_value_size_ret: *mut size_t) -> cl_int;

    // Flush and Finish APIs:
    pub fn clFlush(command_queue: cl_command_queue) -> cl_int;

    pub fn clFinish(command_queue: cl_command_queue) -> cl_int;

    // Enqueued Commands APIs:
    pub fn clEnqueueReadBuffer(command_queue: cl_command_queue,
                           buffer: cl_mem,
                           blocking_read: cl_bool,
                           offset: size_t,
                           cb: size_t,
                           ptr: *mut c_void,
                           num_events_in_wait_list: cl_uint,
                           event_wait_list: *const cl_event,
                           event: *mut cl_event) -> cl_int;

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
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
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

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueFillImage(cl_command_queue   /* command_queue */,
    //                   cl_mem             /* image */,
    //                   const void *       /* fill_color */,
    //                   const size_t *     /* origin[3] */,
    //                   const size_t *     /* region[3] */,
    //                   cl_uint            /* num_events_in_wait_list */,
    //                   const cl_event *   /* event_wait_list */,
    //                   cl_event *         /* event */) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
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
                         image_row_pitch: *mut size_t,
                         image_slc_pitch: *mut size_t,
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

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueMigrateMemObjects(cl_command_queue       /* command_queue */,
    //                           cl_uint                /* num_mem_objects */,
    //                           const cl_mem *         /* mem_objects */,
    //                           cl_mem_migration_flags /* flags */,
    //                           cl_uint                /* num_events_in_wait_list */,
    //                           const cl_event *       /* event_wait_list */,
    //                           cl_event *             /* event */) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
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

    //########################## DEPRICATED 1.2 ##############################
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

    //########################## DEPRICATED 1.1 ##############################
    pub fn clEnqueueMarker(command_queue: cl_command_queue,
                    event: *mut cl_event) -> cl_int;

    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueMarkerWithWaitList(cl_command_queue /* command_queue */,
    //          cl_uint           /* num_events_in_wait_list */,
    //          const cl_event *  /* event_wait_list */,
    //          cl_event *        /* event */) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clEnqueueMarkerWithWaitList(command_queue: cl_command_queue,
             num_events_in_wait_list: cl_uint,
             event_wait_list: *const cl_event,
             event: *mut cl_event) -> cl_int;

    //########################## DEPRICATED 1.1 ##############################
    pub fn clEnqueueWaitForEvents(command_queue: cl_command_queue,
                           num_events: cl_uint,
                           event_list: *mut cl_event) -> cl_int;

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
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clEnqueueBarrierWithWaitList(
             command_queue: cl_command_queue,
             num_events_in_wait_list: cl_uint,
             event_wait_list: *const cl_event,
             event: *mut cl_event) -> cl_int;


    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueSVMFree(cl_command_queue  /* command_queue */,
    //                  cl_uint           /* num_svm_pointers */,
    //                  void *[]          /* svm_pointers[] */,
    //                  void (CL_CALLBACK * /*pfn_free_func*/)(cl_command_queue /* queue */,
    //                                                         cl_uint          /* num_svm_pointers */,
    //                                                         void *[]         /* svm_pointers[] */,
    //                                                         void *           /* user_data */),
    //                  void *            /* user_data */,
    //                  cl_uint           /* num_events_in_wait_list */,
    //                  const cl_event *  /* event_wait_list */,
    //                  cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clEnqueueSVMFree(command_queue: cl_command_queue,
                            num_svm_pointers: cl_uint,
                            svm_pointers: *const *const c_void,
                            pfn_free_func: Option<extern fn (
                                        queue: cl_command_queue,
                                        num_svm_pointers: cl_uint,
                                        svm_pointers: *const *const c_void,
                                        user_data: *mut c_void)>,
                            user_data: *mut c_void,
                            num_events_in_wait_list: cl_uint,
                            event_wait_list: *const cl_event,
                            event: *mut cl_event) -> cl_int;


    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueSVMMemcpy(cl_command_queue  /* command_queue */,
    //                    cl_bool           /* blocking_copy */,
    //                    void *            /* dst_ptr */,
    //                    const void *      /* src_ptr */,
    //                    size_t            /* size */,
    //                    cl_uint           /* num_events_in_wait_list */,
    //                    const cl_event *  /* event_wait_list */,
    //                    cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clEnqueueSVMMemcpy(command_queue: cl_command_queue,
                              blocking_copy: cl_bool,
                              dst_ptr: *mut c_void,
                              src_ptr: *const c_void,
                              size: size_t,
                              num_events_in_wait_list: cl_uint,
                              event_wait_list: *const cl_event,
                              event: *mut cl_event) -> cl_int;


    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueSVMMemFill(cl_command_queue  /* command_queue */,
    //                     void *            /* svm_ptr */,
    //                     const void *      /* pattern */,
    //                     size_t            /* pattern_size */,
    //                     size_t            /* size */,
    //                     cl_uint           /* num_events_in_wait_list */,
    //                     const cl_event *  /* event_wait_list */,
    //                     cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clEnqueueSVMMemFill(command_queue: cl_command_queue,
                               svm_ptr: *mut c_void,
                               pattern: *const c_void,
                               pattern_size: size_t,
                               size: size_t,
                               num_events_in_wait_list: cl_uint,
                               event_wait_list: *const cl_event,
                               event: *mut cl_event) -> cl_int;


    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueSVMMap(cl_command_queue  /* command_queue */,
    //                 cl_bool           /* blocking_map */,
    //                 cl_map_flags      /* flags */,
    //                 void *            /* svm_ptr */,
    //                 size_t            /* size */,
    //                 cl_uint           /* num_events_in_wait_list */,
    //                 const cl_event *  /* event_wait_list */,
    //                 cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clEnqueueSVMMap(command_queue: cl_command_queue,
                           blocking_map: cl_bool,
                           flags: cl_map_flags,
                           svm_ptr: *mut c_void,
                           size: size_t,
                           num_events_in_wait_list: cl_uint,
                           event_wait_list: *const cl_event,
                           event: *mut cl_event) -> cl_int;


    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueSVMUnmap(cl_command_queue  /* command_queue */,
    //                   void *            /* svm_ptr */,
    //                   cl_uint           /* num_events_in_wait_list */,
    //                   const cl_event *  /* event_wait_list */,
    //                   cl_event *        /* event */) CL_API_SUFFIX__VERSION_2_0;
    //############################### NEW 2.0 #################################
    #[cfg(feature = "opencl_version_2_0")]
    pub fn clEnqueueSVMUnmap(command_queue: cl_command_queue,
                             svm_ptr: *mut c_void,
                             num_events_in_wait_list: cl_uint,
                             event_wait_list: *const cl_event,
                             event: *mut cl_event) -> cl_int;


    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueSVMMigrateMem(cl_command_queue         /* command_queue */,
    //                        cl_uint                  /* num_svm_pointers */,
    //                        const void **            /* svm_pointers */,
    //                        const size_t *           /* sizes */,
    //                        cl_mem_migration_flags   /* flags */,
    //                        cl_uint                  /* num_events_in_wait_list */,
    //                        const cl_event *         /* event_wait_list */,
    //                        cl_event *               /* event */) CL_API_SUFFIX__VERSION_2_1;
    //############################### NEW 2.1 #################################
    #[cfg(feature = "opencl_version_2_1")]
    pub fn clEnqueueSVMMigrateMem(command_queue: cl_command_queue,
                                  num_svm_pointers: cl_uint,
                                  svm_pointers: *const *const c_void,
                                  sizes: *const size_t,
                                  flags: cl_mem_migration_flags,
                                  num_events_in_wait_list: cl_uint,
                                  event_wait_list: *const cl_event,
                                  event: *mut cl_event) -> cl_int;


    //########################## DEPRICATED 1.1 ##############################
    pub fn clEnqueueBarrier(command_queue: cl_command_queue) -> cl_int;

    //########################## DEPRICATED 1.1 ##############################
    // Extension function access
    // Returns the extension function address for the given function name,
    // or NULL if a valid function can not be found. The client must
    // check to make sure the address is not NULL, before using or
    // or calling the returned function address.
    pub fn clGetExtensionFunctionAddress(func_name: *mut c_char);

    // extern CL_API_ENTRY void * CL_API_CALL
    // clGetExtensionFunctionAddressForPlatform(cl_platform_id /* platform */,
    //                    const char *
    //                     // func_name
    //                    ) CL_API_SUFFIX__VERSION_1_2;
    //############################### NEW 1.2 #################################
    #[cfg(feature = "opencl_version_1_2")]
    pub fn clGetExtensionFunctionAddressForPlatform(platform: cl_platform_id,
                       func_name: *const c_char) -> *mut c_void;
}
