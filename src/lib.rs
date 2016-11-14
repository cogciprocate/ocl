//! # [![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl) | [GitHub](https://github.com/cogciprocate/ocl)
//!
//! Rust implementation of `OpenCL`&trade;.
//!
//! This documentation is occasionally built from development branches and may
//! differ slightly from what is on crates.io and the master branch.
//!
//! The pages within are very much a work in progress, as is the library
//! itself. Please help by filing an
//! [issue](https://github.com/cogciprocate/ocl/issues) about unclear and/or
//! incomplete documentation and it will be addressed.
//!
//! An explanation/tutorial of how dimensions and sizes of buffers and work
//! queues are used will be coming as soon. Until then please see the
//! [examples].
//!
//!
//! ## Low Level Interfaces
//!
//! For lower level interfaces and to use `OpenCL` features that have not yet
//! been implemented on the `standard` (high-level) interface types, see the
//! [`ocl-core`] and [`cl-sys`] crates.
//!
//!
//! ## Help Wanted
//!
//! Please help complete any functionality you may need by filing an
//! [issue] or creating a [pull request](https://github.com/cogciprocate/ocl/pulls).
//!
//!
//! ## Feedback appreciated
//!
//! Suggestions and nitpicks are most welcome. This isn't ever going to be a
//! busy repo so don't hesitate to file an [issue] to offer constructive criticism.
//!
//!
//! <br/>
//! *“`OpenCL` and the `OpenCL` logo are trademarks of Apple Inc. used by permission by Khronos.”*

//!
//! [issue]: https://github.com/cogciprocate/ocl/issues
//! [`ocl-core`]: https://github.com/cogciprocate/ocl-core
//! [`cl-sys`]: https://github.com/cogciprocate/cl-sys
//! [`Result`]: /ocl/ocl/type.Result.html
//! [examples]: https://github.com/cogciprocate/ocl/tree/master/examples

// #![warn(missing_docs)]
// #![feature(zero_one)]
// #![feature(question_mark)]
// #![feature(stmt_expr_attributes)]

#[macro_use] extern crate enum_primitive;
#[macro_use] extern crate bitflags;
extern crate libc;
extern crate num;
#[cfg(test)] extern crate rand;
pub extern crate ocl_core as core;

#[cfg(test)] mod tests;
mod standard;

pub use core::ffi;
pub use standard::{Platform, Device, Context, Program, Queue, Kernel, Buffer, Image, Event,
    EventList, Sampler, SpatialDims, ProQue};
pub use core::error::{Error, Result};
pub use core::util;

pub mod aliases {
    //! Type aliases and structs meant to mirror those available within a
    //! kernel.
    //!
    //! Vector type fields can be accessed using .0, .1, .2 ... etc.
    //!
    //! [NOTE]: This module may be renamed.

    pub use ffi::{cl_char, cl_uchar, cl_short, cl_ushort, cl_int, cl_uint, cl_long, cl_ulong,
        cl_half, cl_float, cl_double, cl_bool, cl_bitfield};

    pub use core::{ClChar2, ClChar3, ClChar4, ClChar8, ClChar16,
        ClUchar2, ClUchar3, ClUchar4, ClUchar8, ClUchar16,
        ClShort2, ClShort3, ClShort4, ClShort8, ClShort16,
        ClUshort2, ClUshort3, ClUshort4, ClUshort8, ClUshort16,
        ClInt2, ClInt3, ClInt4, ClInt8, ClInt16,
        ClUint2, ClUint3, ClUint4, ClUint8, ClUint16,
        ClLong1, ClLong2, ClLong3, ClLong4, ClLong8, ClLong16,
        ClUlong1, ClUlong2, ClUlong3, ClUlong4, ClUlong8, ClUlong16,
        ClFloat2, ClFloat3, ClFloat4, ClFloat8, ClFloat16,
        ClDouble2, ClDouble3, ClDouble4, ClDouble8, ClDouble16};

    pub use ffi::{ cl_GLuint, cl_GLint, cl_GLenum };
}

pub mod traits {
    //! Commonly used traits.

    pub use standard::{WorkDims, MemLen};
    pub use core::{OclPrm, OclScl, OclVec};
}

pub mod builders {
    //! Builders and associated settings-related types.

    pub use standard::{ContextBuilder, BuildOpt, ProgramBuilder, ImageBuilder, ProQueBuilder,
        DeviceSpecifier, BufferCmd, BufferCmdKind, BufferCmdDataShape,
        ImageCmd, ImageCmdKind, KernelCmd};
    pub use core::{ImageFormat, ImageDescriptor, ContextProperties};
    // #[cfg(not(release))] pub use standard::BufferTest;
}

pub mod flags {
    //! Bitflags for various parameter types.

    pub use core::{
        // cl_device_type - bitfield
        DeviceType, DEVICE_TYPE_DEFAULT, DEVICE_TYPE_CPU, DEVICE_TYPE_GPU, DEVICE_TYPE_ACCELERATOR,
            DEVICE_TYPE_CUSTOM, DEVICE_TYPE_ALL,
        // cl_device_fp_config - bitfield
        DeviceFpConfig, FP_DENORM, FP_INF_NAN, FP_ROUND_TO_NEAREST, FP_ROUND_TO_ZERO,
            FP_ROUND_TO_INF, FP_FMA, FP_SOFT_FLOAT, FP_CORRECTLY_ROUNDED_DIVIDE_SQRT,
        // cl_device_exec_capabilities - bitfield
        DeviceExecCapabilities, EXEC_KERNEL, EXEC_NATIVE_KERNEL,
        // cl_command_queue_properties - bitfield
        CommandQueueProperties, QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, QUEUE_PROFILING_ENABLE,
        // cl_device_affinity_domain
        DeviceAffinityDomain, DEVICE_AFFINITY_DOMAIN_NUMA, DEVICE_AFFINITY_DOMAIN_L4_CACHE,
            DEVICE_AFFINITY_DOMAIN_L3_CACHE, DEVICE_AFFINITY_DOMAIN_L2_CACHE,
            DEVICE_AFFINITY_DOMAIN_L1_CACHE, DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
        // cl_mem_flags - bitfield
        MemFlags, MEM_READ_WRITE, MEM_WRITE_ONLY, MEM_READ_ONLY, MEM_USE_HOST_PTR,
            MEM_ALLOC_HOST_PTR, MEM_COPY_HOST_PTR, MEM_HOST_WRITE_ONLY, MEM_HOST_READ_ONLY,
            MEM_HOST_NO_ACCESS,
        // cl_mem_migration_flags - bitfield
        MemMigrationFlags, MIGRATE_MEM_OBJECT_HOST, MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
        // cl_map_flags - bitfield
        MapFlags, MAP_READ, MAP_WRITE, MAP_WRITE_INVALIDATE_REGION,
        // cl_program_binary_type
        ProgramBinaryType, PROGRAM_BINARY_TYPE_NONE, PROGRAM_BINARY_TYPE_COMPILED_OBJECT,
            PROGRAM_BINARY_TYPE_LIBRARY, PROGRAM_BINARY_TYPE_EXECUTABLE,
        // cl_kernel_arg_type_qualifer
        KernelArgTypeQualifier, KERNEL_ARG_TYPE_NONE, KERNEL_ARG_TYPE_CONST,
            KERNEL_ARG_TYPE_RESTRICT, KERNEL_ARG_TYPE_VOLATILE,
    };
}

pub mod enums {
    //! Enumerators for settings and information requests.

    pub use standard::{DeviceSpecifier, BufferCmdKind, BufferCmdDataShape};

    // API enums.
    pub use core::{ImageChannelOrder, ImageChannelDataType, Cbool, Polling, PlatformInfo,
        DeviceInfo, DeviceMemCacheType, DeviceLocalMemType, ContextInfo, ContextProperty,
        ContextInfoOrPropertiesPointerType, DevicePartitionProperty, CommandQueueInfo, ChannelType,
        MemObjectType, MemInfo, ImageInfo, AddressingMode, FilterMode, SamplerInfo, ProgramInfo,
        ProgramBuildInfo, ProgramBuildStatus, KernelInfo, KernelArgInfo, KernelArgAddressQualifier,
        KernelArgAccessQualifier, KernelWorkGroupInfo, EventInfo, CommandType,
        CommandExecutionStatus, BufferCreateType, ProfilingInfo};

    // Custom enums.
    pub use core::{KernelArg, ContextPropertyValue, PlatformInfoResult, DeviceInfoResult,
        ContextInfoResult, CommandQueueInfoResult, MemInfoResult, ImageInfoResult,
        SamplerInfoResult, ProgramInfoResult, ProgramBuildInfoResult, KernelInfoResult,
        KernelArgInfoResult, KernelWorkGroupInfoResult, EventInfoResult, ProfilingInfoResult};

    // Error status.
    pub use core::Status;
}
