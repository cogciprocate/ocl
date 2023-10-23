//! # [![](https://img.shields.io/crates/v/ocl.svg)](https://crates.io/crates/ocl) | [GitHub](https://github.com/cogciprocate/ocl)
//!
//! Rust implementation of the [OpenCL&trade; API].
//!
//! Some versions of this documentation are built from development branches
//! and may differ slightly between what is on crates.io and the master
//! branch. See the [repo page](https://github.com/cogciprocate/ocl) for links
//! to both versions.
//!
//! Please report unclear documentation or places where examples would be
//! appreciated by filing an [issue].
//!
//!
//! ## Foundations
//!
//! For lower level interfaces and to use OpenCL features that have not yet
//! been implemented on the `standard` (high-level) interface types, see the
//! [`ocl-core`] and [`cl-sys`] crates.
//!
//!
//! ## Help Wanted
//!
//! Please request or help complete any functionality you may need by filing
//! an [issue] or creating a [pull
//! request](https://github.com/cogciprocate/ocl/pulls).
//!
//! Keep an eye out for places where examples would be useful and let us know!
//!
//!
//! ## Feedback appreciated
//!
//! Suggestions and nitpicks are most welcome. Don't hesitate to file an
//! [issue] just to offer constructive criticism.
//!
//!
//! <br/>
//! *“`OpenCL` and the `OpenCL` logo are trademarks of Apple Inc. used by permission by Khronos.”*
//!
//!
//! [OpenCL&trade; API]: https://www.khronos.org/registry/OpenCL/
//! [issue]: https://github.com/cogciprocate/ocl/issues
//! [`ocl-core`]: https://github.com/cogciprocate/ocl-core
//! [`cl-sys`]: https://github.com/cogciprocate/cl-sys
//! [`Result`]: /ocl/ocl/type.Result.html
//! [examples]: https://github.com/cogciprocate/ocl/tree/master/examples

#![doc(html_root_url = "https://docs.rs/ocl/0.19.5")]

// #![warn(missing_docs)]

extern crate futures;
extern crate num_traits;
pub extern crate ocl_core;

pub use ocl_core as core;

pub mod r#async;
pub mod error;
mod standard;
#[cfg(test)]
mod tests;

pub use self::r#async::{
    FutureMemMap, FutureReadGuard, FutureWriteGuard, MemMap, ReadGuard, RwVec, WriteGuard,
};
pub use self::standard::{
    Buffer, BufferCmdError, Context, Device, Event, EventArray, EventList, Extensions, Image,
    Kernel, Platform, ProQue, Program, Queue, Sampler, SpatialDims,
};
#[doc(no_inline)]
pub use crate::core::ffi;
#[doc(no_inline)]
pub use crate::core::util;
pub use crate::core::Error as OclCoreError;
#[doc(no_inline)]
pub use crate::core::{
    CommandQueueProperties, DeviceType, MapFlags, MemFlags, OclPrm, OclScl, OclVec,
};
pub use crate::error::{Error, Result};

pub mod prm {
    //! OpenCL scalar and vector primitive types.
    //!
    //! Rust primitives may have subtly different behaviour than OpenCL
    //! primitives within kernels. Wrapping is one example of this. Scalar
    //! integers within Rust may do overflow checks where in the kernel they
    //! do not. Therefore OpenCL-compatible implementations of each of the
    //! types are provided so that host and device side operations can be
    //! perfectly consistent.
    //!
    //! The `cl_...` (`cl_uchar`, `cl_int`, `cl_float`, etc.) types are simple
    //! aliases of the Rust built-in primitive types and do **not** behave the
    //! same way that the kernel-side equivalents do. The uppercase-named
    //! types, on the other hand, (`Uchar`, `Int`, `Float`, etc.) are designed
    //! to behave identically to their corresponding types within kernels.
    //!
    //! Please file an issue if any of the uppercase-named kernel-mimicking
    //! types deviate from what they should (as they are reasonably new this
    //! is definitely something to watch out for).
    //!
    //! Vector type fields can be accessed using index operations i.e. [0],
    //! [1], [2] ... etc. Plans for other ways of accessing fields (such as
    //! `.x()`, `.y()`, `.s0()`, `.s15()`, etc.) will be considered in the
    //! future (pending a number of additions/stabilizations to the Rust
    //! language). Create an issue if you have an opinion on the matter.
    //!
    //! [NOTE]: This module may be renamed.

    pub use crate::ffi::{
        cl_bitfield, cl_bool, cl_char, cl_double, cl_float, cl_half, cl_int, cl_long, cl_short,
        cl_uchar, cl_uint, cl_ulong, cl_ushort,
    };

    pub use crate::ffi::{cl_GLenum, cl_GLint, cl_GLuint};

    // Wrapping types. Use these to mimic in-kernel behaviour:
    pub use crate::core::{
        Char, Char16, Char2, Char3, Char4, Char8, Double, Double16, Double2, Double3, Double4,
        Double8, Float, Float16, Float2, Float3, Float4, Float8, Int, Int16, Int2, Int3, Int4,
        Int8, Long, Long16, Long2, Long3, Long4, Long8, Short, Short16, Short2, Short3, Short4,
        Short8, Uchar, Uchar16, Uchar2, Uchar3, Uchar4, Uchar8, Uint, Uint16, Uint2, Uint3, Uint4,
        Uint8, Ulong, Ulong16, Ulong2, Ulong3, Ulong4, Ulong8, Ushort, Ushort16, Ushort2, Ushort3,
        Ushort4, Ushort8,
    };
}

pub mod traits {
    //! Commonly used traits.

    pub use crate::core::{OclPrm, OclScl, OclVec};
    pub use crate::standard::{IntoMarker, IntoRawEventArray, MemLen, WorkDims};
}

pub mod builders {
    //! Builders and associated settings-related types.

    pub use crate::core::{ContextProperties, ImageDescriptor, ImageFormat};
    pub use crate::standard::{
        BufferBuilder, BufferCmd, BufferCmdDataShape, BufferCmdKind, BufferMapCmd, BufferReadCmd,
        BufferWriteCmd, BuildOpt, ContextBuilder, DeviceSpecifier, ImageBuilder, ImageCmd,
        ImageCmdKind, KernelBuilder, KernelCmd, ProQueBuilder, ProgramBuilder,
    };
    pub use crate::standard::{ClNullEventPtrEnum, ClWaitListPtrEnum};
    // #[cfg(not(release))] pub use standard::BufferTest;
}

pub mod flags {
    //! Bitflags for various parameter types.

    pub use crate::core::{
        // cl_command_queue_properties - bitfield
        CommandQueueProperties,
        // cl_device_affinity_domain
        DeviceAffinityDomain,
        // cl_device_exec_capabilities - bitfield
        DeviceExecCapabilities,
        // cl_device_fp_config - bitfield
        DeviceFpConfig,
        // cl_device_type - bitfield
        DeviceType,
        // cl_kernel_arg_type_qualifer
        KernelArgTypeQualifier,
        // cl_map_flags - bitfield
        MapFlags,
        // cl_mem_flags - bitfield
        MemFlags,
        // cl_mem_migration_flags - bitfield
        MemMigrationFlags,
        // cl_program_binary_type
        ProgramBinaryType,
        DEVICE_AFFINITY_DOMAIN_L1_CACHE,
        DEVICE_AFFINITY_DOMAIN_L2_CACHE,
        DEVICE_AFFINITY_DOMAIN_L3_CACHE,
        DEVICE_AFFINITY_DOMAIN_L4_CACHE,
        DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
        DEVICE_AFFINITY_DOMAIN_NUMA,
        DEVICE_TYPE_ACCELERATOR,
        DEVICE_TYPE_ALL,
        DEVICE_TYPE_CPU,
        DEVICE_TYPE_CUSTOM,
        DEVICE_TYPE_DEFAULT,
        DEVICE_TYPE_GPU,
        EXEC_KERNEL,
        EXEC_NATIVE_KERNEL,
        FP_CORRECTLY_ROUNDED_DIVIDE_SQRT,
        FP_DENORM,
        FP_FMA,
        FP_INF_NAN,
        FP_ROUND_TO_INF,
        FP_ROUND_TO_NEAREST,
        FP_ROUND_TO_ZERO,
        FP_SOFT_FLOAT,
        KERNEL_ARG_TYPE_CONST,
        KERNEL_ARG_TYPE_NONE,
        KERNEL_ARG_TYPE_RESTRICT,
        KERNEL_ARG_TYPE_VOLATILE,
        MAP_READ,
        MAP_WRITE,
        MAP_WRITE_INVALIDATE_REGION,
        MEM_ALLOC_HOST_PTR,
        MEM_COPY_HOST_PTR,
        MEM_HOST_NO_ACCESS,
        MEM_HOST_READ_ONLY,
        MEM_HOST_WRITE_ONLY,
        MEM_READ_ONLY,
        MEM_READ_WRITE,
        MEM_USE_HOST_PTR,
        MEM_WRITE_ONLY,
        MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
        MIGRATE_MEM_OBJECT_HOST,
        PROGRAM_BINARY_TYPE_COMPILED_OBJECT,
        PROGRAM_BINARY_TYPE_EXECUTABLE,
        PROGRAM_BINARY_TYPE_LIBRARY,
        PROGRAM_BINARY_TYPE_NONE,
        QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        QUEUE_PROFILING_ENABLE,
    };
}

pub mod enums {
    //! Enumerators for settings and information requests.

    pub use crate::standard::{BufferCmdDataShape, BufferCmdKind, DeviceSpecifier, WriteSrc};

    // API enums.
    pub use crate::core::{
        AddressingMode, BufferCreateType, Cbool, ChannelType, CommandExecutionStatus,
        CommandQueueInfo, CommandType, ContextInfo, ContextInfoOrPropertiesPointerType,
        ContextProperty, DeviceInfo, DeviceLocalMemType, DeviceMemCacheType,
        DevicePartitionProperty, EventInfo, FilterMode, ImageChannelDataType, ImageChannelOrder,
        ImageInfo, KernelArgAccessQualifier, KernelArgAddressQualifier, KernelArgInfo, KernelInfo,
        KernelWorkGroupInfo, MemInfo, MemObjectType, PlatformInfo, Polling, ProfilingInfo,
        ProgramBuildInfo, ProgramBuildStatus, ProgramInfo, SamplerInfo,
    };

    // Custom enums.
    pub use crate::core::{
        ArgVal, CommandQueueInfoResult, ContextInfoResult, ContextPropertyValue, DeviceInfoResult,
        EventInfoResult, ImageInfoResult, KernelArgInfoResult, KernelInfoResult,
        KernelWorkGroupInfoResult, MemInfoResult, PlatformInfoResult, ProfilingInfoResult,
        ProgramBuildInfoResult, ProgramInfoResult, SamplerInfoResult,
    };

    // Error status.
    pub use crate::core::Status;
}
