//! # [![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl) | [GitHub](https://github.com/cogciprocate/ocl)
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

#![doc(html_root_url = "https://docs.rs/ocl/0.19.4")]

// #![warn(missing_docs)]

extern crate num_traits;
extern crate futures;
pub extern crate ocl_core as core;


#[cfg(test)]
mod tests;
mod standard;
pub mod error;
pub mod r#async;

pub use self::standard::{Platform, Extensions, Device, Context, Program, Queue, Kernel, Buffer, Image, Event,
    EventList, EventArray, Sampler, SpatialDims, ProQue, BufferCmdError};
pub use self::r#async::{MemMap, FutureMemMap, RwVec, ReadGuard, WriteGuard,
    FutureReadGuard, FutureWriteGuard};
pub use crate::error::{Error, Result};
pub use crate::{core::Error as OclCoreError};
#[doc(no_inline)]
pub use crate::core::ffi;
#[doc(no_inline)]
pub use crate::core::util;
#[doc(no_inline)]
pub use crate::core::{OclPrm, OclScl, OclVec, DeviceType, CommandQueueProperties, MemFlags, MapFlags};


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

    pub use crate::ffi::{cl_char, cl_uchar, cl_short, cl_ushort, cl_int, cl_uint, cl_long, cl_ulong,
        cl_half, cl_float, cl_double, cl_bool, cl_bitfield};

    pub use crate::ffi::{ cl_GLuint, cl_GLint, cl_GLenum };

    // Wrapping types. Use these to mimic in-kernel behaviour:
    pub use crate::core::{
        Char, Char2, Char3, Char4, Char8, Char16,
        Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
        Short, Short2, Short3, Short4, Short8, Short16,
        Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
        Int, Int2, Int3, Int4, Int8, Int16,
        Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
        Long, Long2, Long3, Long4, Long8, Long16,
        Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
        Float, Float2, Float3, Float4, Float8, Float16,
        Double, Double2, Double3, Double4, Double8, Double16};
}

pub mod traits {
    //! Commonly used traits.

    pub use crate::standard::{WorkDims, MemLen, IntoMarker, IntoRawEventArray};
    pub use crate::core::{OclPrm, OclScl, OclVec};
}

pub mod builders {
    //! Builders and associated settings-related types.

    pub use crate::standard::{ContextBuilder, BuildOpt, ProgramBuilder, ImageBuilder, ProQueBuilder,
        DeviceSpecifier, BufferCmdKind, BufferCmdDataShape, BufferCmd, BufferReadCmd,
        BufferWriteCmd, BufferMapCmd, ImageCmdKind, ImageCmd, KernelCmd, BufferBuilder,
        KernelBuilder};
    pub use crate::standard::{ClNullEventPtrEnum, ClWaitListPtrEnum};
    pub use crate::core::{ImageFormat, ImageDescriptor, ContextProperties};
    // #[cfg(not(release))] pub use standard::BufferTest;
}

pub mod flags {
    //! Bitflags for various parameter types.

    pub use crate::core::{
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

    pub use crate::standard::{DeviceSpecifier, BufferCmdKind, BufferCmdDataShape, WriteSrc};

    // API enums.
    pub use crate::core::{ImageChannelOrder, ImageChannelDataType, Cbool, Polling, PlatformInfo,
        DeviceInfo, DeviceMemCacheType, DeviceLocalMemType, ContextInfo, ContextProperty,
        ContextInfoOrPropertiesPointerType, DevicePartitionProperty, CommandQueueInfo, ChannelType,
        MemObjectType, MemInfo, ImageInfo, AddressingMode, FilterMode, SamplerInfo, ProgramInfo,
        ProgramBuildInfo, ProgramBuildStatus, KernelInfo, KernelArgInfo, KernelArgAddressQualifier,
        KernelArgAccessQualifier, KernelWorkGroupInfo, EventInfo, CommandType,
        CommandExecutionStatus, BufferCreateType, ProfilingInfo};

    // Custom enums.
    pub use crate::core::{ArgVal, ContextPropertyValue, PlatformInfoResult, DeviceInfoResult,
        ContextInfoResult, CommandQueueInfoResult, MemInfoResult, ImageInfoResult,
        SamplerInfoResult, ProgramInfoResult, ProgramBuildInfoResult, KernelInfoResult,
        KernelArgInfoResult, KernelWorkGroupInfoResult, EventInfoResult, ProfilingInfoResult};

    // Error status.
    pub use crate::core::Status;
}
