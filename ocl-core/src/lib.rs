// ocl::core::

//! Thin wrappers for the `OpenCL` FFI functions and types.
//!
//! Allows access to `OpenCL` FFI functions with a minimal layer of zero-cost
//! abstraction, providing, safety, performance, and convenience. The [`ocl`]
//! crate contains higher level and easier to use interfaces to the
//! functionality contained within.
//!
//!
//! ## Even Lower Level: [`cl-sys`]
//!
//! If there's still something missing or for some reason you need direct FFI
//! access, use the functions in the [`cl-sys`] module. The pointers used by
//! [`cl-sys`] functions can be wrapped in [`ocl-core`] wrappers
//! (`ocl_core::PlatformId`, `ocl_core::Context`, etc.) and passed to
//! [`ocl-core`] module functions. Likewise the other way around (using, for
//! example: [`EventRaw::as_ptr`]).
//!
//!
//! ## Performance
//!
//! Performance between all three interface layers, [`cl-sys`], [`ocl-core`],
//! and the 'standard' ([`ocl`]) types, is identical or virtually identical
//! (if not, please file an issue).
//!
//!
//! ## Safety
//!
//! Running any kernel at all is inherently unsafe. The guarantee we aim to
//! make in this library is that if there is a segfault or invalid memory
//! access, the fault lies within the kernel. No effort is or will be made to
//! verify or sanitize kernels. Consider all kernels within user-supplied
//! programs just as if you would an `unsafe fn`.
//!
//!
//! ## Length vs Size
//!
//! No, not that...
//!
//! Quantifiers passed to functions in the `OpenCL` API are generally
//! expressed in bytes. Units passed to functions in *this* library are
//! expected to be `bytes / sizeof(T)` (corresponding with units returned by
//! the ubiquitous `.len()` method). The suffix '_size' or '_bytes' is
//! generally used when a parameter deviates from this convention.
//!
//!
//! ## Version Control
//!
//! The version control system is in place to ensure that you don't call
//! functions that your hardware/driver does not support.
//!
//! Functions in this crate with the `[Version Controlled: OpenCL {...}+]` tag
//! in the description require an additional parameter, `device_version` or
//! `device_versions`: a parsed result (or slice of results) of
//! `DeviceInfo::Version`. This is a runtime check to ensure that the device
//! supports the function being called. Calling a function which a particular
//! device does not support will likely cause a segmentation fault and
//! possibly data corruption.
//!
//! Saving the `OpenclVersion` returned from `device_version()` for your
//! device(s) at the start of your program and passing it each time you call
//! a version controlled function is the fastest and safest method (see the
//! `ocl` library for an example). The cost of this check is little more than
//! a single `if` statement.
//!
//! Passing `None` for `device_version` will cause an automated version check
//! which has a small cost (calling info function, parsing the version number
//! etc.) but is a safe option if you are not sure what to do.
//!
//! Passing the result of a call to `OpenclVersion::max()` or passing a fake
//! version will bypass any safety checks and has all of the risks described
//! above. Only do this if you're absolutely sure you know what you're doing
//! and are not concerned about segfaults and data integrity.
//!

//!
//!
//! ## More Documentation
//!
//! As most of the functions here are minimally documented, please refer to
//! the official `OpenCL` documentation linked below. Although there isn't a
//! precise 1:1 parameter mapping between the `core` and original functions,
//! it's close enough (modulo the size/len difference discussed above) to help
//! sort out any questions you may have until a more thorough documentation
//! pass can be made. View the source code in [`src/types/functions.rs`] for
//! more mapping details.
//!
//! ['OpenCL' 1.2 SDK Reference: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/]
//!
//!
//!
//!
//!
//!
//! [`ocl`]: https://github.com/cogciprocate/ocl
//! [`ocl-core`]: https://github.com/cogciprocate/ocl-core
//! [`cl-sys`]: https://github.com/cogciprocate/cl-sys
//! [issue]: https://github.com/cogciprocate/ocl-core/issues
//! ['OpenCL' 1.2 SDK Reference: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/
//! [`src/types/functions.rs`]: /ocl-core/src/ocl-core/src/types/functions.rs.html

// #![doc(html_root_url="https://docs.rs/ocl-core/0.3/")]

#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate enum_primitive;
extern crate num_traits;
extern crate num_complex;
#[cfg(feature = "ocl-core-vector")]
extern crate ocl_core_vector as vector;
pub extern crate cl_sys as ffi;

#[cfg(test)] mod tests;
mod functions;
pub mod types;
pub mod error;
pub mod util;

pub use self::error::{Error, Result};

pub use self::types::abs::{ClWaitListPtr, ClNullEventPtr, ClEventPtrRef, ClPlatformIdPtr,
    ClDeviceIdPtr, ClContextPtr, EventRefWrapper, PlatformId, DeviceId, Context, CommandQueue, Mem,
    Program, Kernel, Event, Sampler, ClVersions, AsMem, MemCmdRw, MemCmdAll, MemMap};

pub use self::types::structs::{self, OpenclVersion, ContextProperties, ImageFormatParseError,
    ImageFormatParseResult, ImageFormat, ImageDescriptor, BufferRegion, ContextPropertyValue,
    ArgVal};

pub use self::types::enums::{EmptyInfoResultError, PlatformInfoResult, DeviceInfoResult,
    ContextInfoResult, GlContextInfoResult, CommandQueueInfoResult, MemInfoResult, ImageInfoResult,
    SamplerInfoResult, ProgramInfoResult, ProgramBuildInfoResult, KernelInfoResult,
    KernelArgInfoResult, KernelWorkGroupInfoResult, EventInfoResult, ProfilingInfoResult};

pub use self::functions::{get_platform_ids, get_platform_info, get_device_ids, get_device_info,
    create_sub_devices, retain_device, release_device, create_context, create_context_from_type,
    retain_context, release_context, get_context_info,
    create_command_queue, retain_command_queue, release_command_queue, get_command_queue_info,
    create_buffer, create_sub_buffer, create_image, retain_mem_object, release_mem_object,
    get_supported_image_formats, get_mem_object_info, get_image_info,
    set_mem_object_destructor_callback, create_sampler, retain_sampler, release_sampler,
    get_sampler_info, create_program_with_source, create_program_with_binary,
    create_program_with_built_in_kernels, retain_program, release_program, build_program,
    compile_program, link_program, create_build_program, get_program_info, get_program_build_info,
    create_kernel, create_kernels_in_program, retain_kernel, release_kernel, set_kernel_arg,
    get_kernel_info, get_kernel_arg_info, get_kernel_work_group_info, wait_for_events,
    get_event_info, create_user_event, retain_event, release_event, set_user_event_status,
    set_event_callback, get_event_profiling_info, flush, finish, enqueue_read_buffer,
    enqueue_read_buffer_rect, enqueue_write_buffer, enqueue_write_buffer_rect, enqueue_copy_buffer,
    enqueue_copy_buffer_rect,
    enqueue_read_image, enqueue_write_image, enqueue_fill_image, enqueue_copy_image,
    enqueue_copy_image_to_buffer, enqueue_copy_buffer_to_image, enqueue_map_buffer,
    enqueue_map_image, enqueue_unmap_mem_object, enqueue_migrate_mem_objects, enqueue_kernel,
    enqueue_task, enqueue_native_kernel, enqueue_marker_with_wait_list,
    enqueue_barrier_with_wait_list, get_extension_function_address_for_platform, wait_for_event,
    event_status, default_platform_idx, program_build_err, verify_context, default_platform,
    default_device_type, device_versions, event_is_complete, _dummy_event_callback,
    _complete_user_event, get_context_platform, get_device_info_raw};

#[cfg(not(feature="opencl_vendor_mesa"))]
pub use self::functions::{
    get_gl_context_info_khr,
    create_from_gl_buffer, create_from_gl_renderbuffer, create_from_gl_texture,
    create_from_gl_texture_2d, create_from_gl_texture_3d, enqueue_fill_buffer,
    enqueue_acquire_gl_objects, enqueue_release_gl_objects};

pub use crate::traits::{OclPrm, OclNum, OclScl};

#[cfg(feature = "ocl-core-vector")]
pub use self::vector::{
// pub use self::types::vector::{
    Char, Char2, Char3, Char4, Char8, Char16,
    Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
    Short, Short2, Short3, Short4, Short8, Short16,
    Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
    Int, Int2, Int3, Int4, Int8, Int16,
    Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
    Long, Long2, Long3, Long4, Long8, Long16,
    Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
    Float, Float2, Float3, Float4, Float8, Float16,
    Double, Double2, Double3, Double4, Double8, Double16,
};

#[cfg(feature = "ocl-core-vector")]
pub use crate::traits::OclVec;

#[cfg(feature = "opencl_version_2_1")]
pub use self::functions::{create_program_with_il};




//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

pub const DEVICES_MAX: u32 = 64;

//=============================================================================
//================================= TYPEDEFS ==================================
//=============================================================================

pub type EventCallbackFn = extern "C" fn (ffi::cl_event, i32, *mut ffi::c_void);
pub type CreateContextCallbackFn = extern "C" fn (*const ffi::c_char, *const ffi::c_void,
    ffi::size_t, *mut ffi::c_void);
pub type BuildProgramCallbackFn = extern "C" fn (*mut ffi::c_void, *mut ffi::c_void);
pub type UserDataPtr = *mut ffi::c_void;

//=============================================================================
//================================== TRAITS ===================================
//=============================================================================

mod traits {
    use std::fmt::{Display, Debug};
    use std::ops::*;
    use std::iter::{Sum, Product};
    use num_traits::{NumCast, FromPrimitive, ToPrimitive, Zero, One};
    use num_complex::{Complex32, Complex64};

    // Implements an unsafe trait for a list of types.
    macro_rules! impl_unsafe {
        ($trt:ident: $( $ty:ident ),+) => {
            $( unsafe impl $trt for $ty {} )+
        }
    }

    #[cfg(feature = "ocl-core-vector")]
    pub use self::ocl_vec::OclVec;

    /// A primitive type usable within `OpenCL` kernels.
    ///
    /// Includes all of the signed, unsigned, and floating point 8 bit - 64 bit
    /// scalar primitives (ex.: cl_char, cl_uint, cl_double) (exception: cl_half)
    /// and their vector counterparts (ex.: cl_int4, cl_float3, cl_short16);
    ///
    /// Can also be implemented for custom types as long as layout and
    /// alignment are conserved between Rust and OpenCL (repr "C").
    pub unsafe trait OclPrm: Debug + Clone + Copy + Default + PartialEq + Send + Sync + 'static {}

    impl_unsafe!(OclPrm: u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64,
        Complex32, Complex64);

    /// A set of traits common to numeric built-in OpenCL scalar and vector
    /// primitives.
    ///
    /// To describe the contents of buffers, etc., prefer using the more general
    /// `OclPrm` trait unless numeric operations are required.
    pub unsafe trait OclNum: Debug + Display + Clone + Copy + Default + PartialOrd +
        Zero<Output=Self> + One<Output=Self> + Add<Self, Output=Self> + Sub<Self, Output=Self> +
        Mul<Self, Output=Self> + Div<Self, Output=Self> + Rem<Self, Output=Self> + PartialEq<Self>
        + AddAssign<Self> + SubAssign<Self> + MulAssign<Self> + DivAssign<Self> + RemAssign<Self> +
        Sum<Self> + Product<Self> + Send + Sync + 'static {}

    impl_unsafe!(OclNum: u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);

    /// A scalar type usable within OpenCL kernels.
    ///
    /// To describe the contents of buffers, etc., prefer using the more general
    /// `OclPrm` trait unless numeric operations are required.
    pub unsafe trait OclScl: OclPrm + OclNum + NumCast + FromPrimitive + ToPrimitive {}

    impl_unsafe!(OclScl: u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);

    #[cfg(feature = "ocl-core-vector")]
    mod ocl_vec {
        use crate::traits::{OclPrm, OclNum};

        /// A vector type usable within `OpenCL` kernels.
        ///
        /// To describe the contents of buffers, etc., prefer using the more general
        /// `OclPrm` trait unless numeric operations are required.
        pub unsafe trait OclVec: OclPrm + OclNum {}

        use crate::vector::{
            Char, Char2, Char3, Char4, Char8, Char16,
            Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
            Short, Short2, Short3, Short4, Short8, Short16,
            Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
            Int, Int2, Int3, Int4, Int8, Int16,
            Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
            Long, Long2, Long3, Long4, Long8, Long16,
            Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
            Float, Float2, Float3, Float4, Float8, Float16,
            Double, Double2, Double3, Double4, Double8, Double16
        };

        impl_unsafe!(OclNum:
            Char, Char2, Char3, Char4, Char8, Char16,
            Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
            Short, Short2, Short3, Short4, Short8, Short16,
            Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
            Int, Int2, Int3, Int4, Int8, Int16,
            Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
            Long, Long2, Long3, Long4, Long8, Long16,
            Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
            Float, Float2, Float3, Float4, Float8, Float16,
            Double, Double2, Double3, Double4, Double8, Double16
        );

        impl_unsafe!(OclPrm:
            Char, Char2, Char3, Char4, Char8, Char16,
            Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
            Short, Short2, Short3, Short4, Short8, Short16,
            Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
            Int, Int2, Int3, Int4, Int8, Int16,
            Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
            Long, Long2, Long3, Long4, Long8, Long16,
            Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
            Float, Float2, Float3, Float4, Float8, Float16,
            Double, Double2, Double3, Double4, Double8, Double16
        );

        impl_unsafe!(OclVec:
            Char, Char2, Char3, Char4, Char8, Char16,
            Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
            Short, Short2, Short3, Short4, Short8, Short16,
            Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
            Int, Int2, Int3, Int4, Int8, Int16,
            Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
            Long, Long2, Long3, Long4, Long8, Long16,
            Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
            Float, Float2, Float3, Float4, Float8, Float16,
            Double, Double2, Double3, Double4, Double8, Double16
        );
    }
}

//=============================================================================
//================================ BITFIELDS ==================================
//=============================================================================

bitflags! {
    /// cl_device_type - bitfield
    ///
    /// * `CL_DEVICE_TYPE_DEFAULT`: The default `OpenCL` device in the system.
    /// * `CL_DEVICE_TYPE_CPU`: An `OpenCL` device that is the host processor.
    ///   The host processor runs the `OpenCL` implementations and is a single
    ///   or multi-core CPU.
    /// * `CL_DEVICE_TYPE_GPU`: An `OpenCL` device that is a GPU. By this we
    ///   mean that the device can also be used to accelerate a 3D API such as
    ///   OpenGL or DirectX.
    /// * `CL_DEVICE_TYPE_ACCELERATOR`: Dedicated `OpenCL` accelerators (for
    ///   example the IBM CELL Blade). These devices communicate with the host
    ///   processor using a peripheral interconnect such as PCIe.
    /// * `CL_DEVICE_TYPE_ALL`: A union of all flags.
    ///
    pub struct DeviceType: u64 {
        const DEFAULT = 1;
        const CPU = 1 << 1;
        const GPU = 1 << 2;
        const ACCELERATOR = 1 << 3;
        const CUSTOM = 1 << 4;
        const ALL = 0xFFFF_FFFF;
    }
}

impl DeviceType {
    #[inline] pub fn new() -> DeviceType { DeviceType::empty() }
    #[inline] pub fn system_default(self) -> DeviceType { self | DeviceType::DEFAULT }
    #[inline] pub fn cpu(self) -> DeviceType { self | DeviceType::CPU }
    #[inline] pub fn gpu(self) -> DeviceType { self | DeviceType::GPU }
    #[inline] pub fn accelerator(self) -> DeviceType { self | DeviceType::ACCELERATOR }
    #[inline] pub fn custom(self) -> DeviceType { self | DeviceType::CUSTOM }
    // #[inline] pub fn all() -> DeviceType { DeviceType::ALL }
}

impl Default for DeviceType {
    #[inline] fn default() -> DeviceType { DeviceType::ALL }
}

pub const DEVICE_TYPE_DEFAULT: DeviceType = DeviceType::DEFAULT;
pub const DEVICE_TYPE_CPU: DeviceType = DeviceType::CPU;
pub const DEVICE_TYPE_GPU: DeviceType = DeviceType::GPU;
pub const DEVICE_TYPE_ACCELERATOR: DeviceType = DeviceType::ACCELERATOR;
pub const DEVICE_TYPE_CUSTOM: DeviceType = DeviceType::CUSTOM;
pub const DEVICE_TYPE_ALL: DeviceType = DeviceType::ALL;


bitflags! {
    /// cl_device_fp_config - bitfield
    pub struct DeviceFpConfig: u64 {
        const DENORM = 1;
        const INF_NAN = 1 << 1;
        const ROUND_TO_NEAREST = 1 << 2;
        const ROUND_TO_ZERO = 1 << 3;
        const ROUND_TO_INF = 1 << 4;
        const FMA = 1 << 5;
        const SOFT_FLOAT = 1 << 6;
        const CORRECTLY_ROUNDED_DIVIDE_SQRT = 1 << 7;
    }
}

pub const FP_DENORM: DeviceFpConfig = DeviceFpConfig::DENORM;
pub const FP_INF_NAN: DeviceFpConfig = DeviceFpConfig::INF_NAN;
pub const FP_ROUND_TO_NEAREST: DeviceFpConfig = DeviceFpConfig::ROUND_TO_NEAREST;
pub const FP_ROUND_TO_ZERO: DeviceFpConfig = DeviceFpConfig::ROUND_TO_ZERO;
pub const FP_ROUND_TO_INF: DeviceFpConfig = DeviceFpConfig::ROUND_TO_INF;
pub const FP_FMA: DeviceFpConfig = DeviceFpConfig::FMA;
pub const FP_SOFT_FLOAT: DeviceFpConfig = DeviceFpConfig::SOFT_FLOAT;
pub const FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: DeviceFpConfig = DeviceFpConfig::CORRECTLY_ROUNDED_DIVIDE_SQRT;


bitflags! {
    /// cl_device_exec_capabilities - bitfield
    pub struct DeviceExecCapabilities: u64 {
        const KERNEL = 1;
        const NATIVE_KERNEL = 1 << 1;
    }
}

pub const EXEC_KERNEL: DeviceExecCapabilities = DeviceExecCapabilities::KERNEL;
pub const EXEC_NATIVE_KERNEL: DeviceExecCapabilities = DeviceExecCapabilities::NATIVE_KERNEL;


bitflags! {
    /// cl_command_queue_properties - bitfield
    pub struct CommandQueueProperties: u64 {
        const OUT_OF_ORDER_EXEC_MODE_ENABLE = 1;
        const PROFILING_ENABLE = 1 << 1;
        const ON_DEVICE = 1 << 2;
        const ON_DEVICE_DEFAULT = 1 << 3;
    }
}

impl CommandQueueProperties {
    #[inline] pub fn new() -> CommandQueueProperties { CommandQueueProperties::empty() }
    #[inline] pub fn out_of_order(self) -> CommandQueueProperties { self |
        CommandQueueProperties::OUT_OF_ORDER_EXEC_MODE_ENABLE }
    #[inline] pub fn profiling(self) -> CommandQueueProperties { self |
        CommandQueueProperties::PROFILING_ENABLE }
}

impl Default for CommandQueueProperties {
    #[inline] fn default() -> CommandQueueProperties { CommandQueueProperties::empty() }
}

pub const QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE: CommandQueueProperties =
    CommandQueueProperties::OUT_OF_ORDER_EXEC_MODE_ENABLE;
pub const QUEUE_PROFILING_ENABLE: CommandQueueProperties =
    CommandQueueProperties::PROFILING_ENABLE;
pub const QUEUE_ON_DEVICE: CommandQueueProperties =
    CommandQueueProperties::ON_DEVICE;
pub const QUEUE_ON_DEVICE_DEFAULT: CommandQueueProperties =
    CommandQueueProperties::ON_DEVICE_DEFAULT;

bitflags! {
    /// cl_device_affinity_domain
    pub struct DeviceAffinityDomain: u64 {
        const NUMA = 1;
        const L4_CACHE = 1 << 1;
        const L3_CACHE = 1 << 2;
        const L2_CACHE = 1 << 3;
        const L1_CACHE = 1 << 4;
        const NEXT_PARTITIONABLE = 1 << 5;
    }
}

pub const DEVICE_AFFINITY_DOMAIN_NUMA: DeviceAffinityDomain = DeviceAffinityDomain::NUMA;
pub const DEVICE_AFFINITY_DOMAIN_L4_CACHE: DeviceAffinityDomain = DeviceAffinityDomain::L4_CACHE;
pub const DEVICE_AFFINITY_DOMAIN_L3_CACHE: DeviceAffinityDomain = DeviceAffinityDomain::L3_CACHE;
pub const DEVICE_AFFINITY_DOMAIN_L2_CACHE: DeviceAffinityDomain = DeviceAffinityDomain::L2_CACHE;
pub const DEVICE_AFFINITY_DOMAIN_L1_CACHE: DeviceAffinityDomain = DeviceAffinityDomain::L1_CACHE;
pub const DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE: DeviceAffinityDomain = DeviceAffinityDomain::NEXT_PARTITIONABLE;


bitflags! {
    /// cl_mem_flags - bitfield
    pub struct MemFlags: u64 {
        const READ_WRITE = 1;
        const WRITE_ONLY = 1 << 1;
        const READ_ONLY = 1 << 2;
        const USE_HOST_PTR = 1 << 3;
        const ALLOC_HOST_PTR = 1 << 4;
        const COPY_HOST_PTR = 1 << 5;
        // RESERVED            1<< 6;
        const HOST_WRITE_ONLY = 1 << 7;
        const HOST_READ_ONLY = 1 << 8;
        const HOST_NO_ACCESS = 1 << 9;
    }
}

impl MemFlags {
    #[inline] pub fn new() -> MemFlags { MemFlags::empty() }
    #[inline] pub fn read_write(self) -> MemFlags { self | MemFlags::READ_WRITE }
    #[inline] pub fn write_only(self) -> MemFlags { self | MemFlags::WRITE_ONLY }
    #[inline] pub fn read_only(self) -> MemFlags { self | MemFlags::READ_ONLY }
    #[inline] pub fn use_host_ptr(self) -> MemFlags { self | MemFlags::USE_HOST_PTR }
    #[inline] pub fn alloc_host_ptr(self) -> MemFlags { self | MemFlags::ALLOC_HOST_PTR }
    #[inline] pub fn copy_host_ptr(self) -> MemFlags { self | MemFlags::COPY_HOST_PTR }
    #[inline] pub fn host_write_only(self) -> MemFlags { self | MemFlags::HOST_WRITE_ONLY }
    #[inline] pub fn host_read_only(self) -> MemFlags { self | MemFlags::HOST_READ_ONLY }
    #[inline] pub fn host_no_access(self) -> MemFlags { self | MemFlags::HOST_NO_ACCESS }
}

impl Default for MemFlags {
    #[inline] fn default() -> MemFlags { MemFlags::READ_WRITE }
}

pub const MEM_READ_WRITE: MemFlags = MemFlags::READ_WRITE;
pub const MEM_WRITE_ONLY: MemFlags = MemFlags::WRITE_ONLY;
pub const MEM_READ_ONLY: MemFlags = MemFlags::READ_ONLY;
pub const MEM_USE_HOST_PTR: MemFlags = MemFlags::USE_HOST_PTR;
pub const MEM_ALLOC_HOST_PTR: MemFlags = MemFlags::ALLOC_HOST_PTR;
pub const MEM_COPY_HOST_PTR: MemFlags = MemFlags::COPY_HOST_PTR;
// RESERVED            1<< 6;
pub const MEM_HOST_WRITE_ONLY: MemFlags = MemFlags::HOST_WRITE_ONLY;
pub const MEM_HOST_READ_ONLY: MemFlags = MemFlags::HOST_READ_ONLY;
pub const MEM_HOST_NO_ACCESS: MemFlags = MemFlags::HOST_NO_ACCESS;


bitflags! {
    /// cl_mem_migration_flags - bitfield
    pub struct MemMigrationFlags: u64 {
        const OBJECT_HOST = 1;
        const OBJECT_CONTENT_UNDEFINED = 1 << 1;
    }
}

pub const MIGRATE_MEM_OBJECT_HOST: MemMigrationFlags = MemMigrationFlags::OBJECT_HOST;
pub const MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED: MemMigrationFlags = MemMigrationFlags::OBJECT_CONTENT_UNDEFINED;


bitflags! {
    /// cl_map_flags - bitfield
    pub struct MapFlags: u64 {
        const READ = 1;
        const WRITE = 1 << 1;
        const WRITE_INVALIDATE_REGION = 1 << 2;
    }
}

impl MapFlags {
    #[inline] pub fn new() -> MapFlags { MapFlags::empty() }
    #[inline] pub fn read(self) -> MapFlags { self | MapFlags::READ }
    #[inline] pub fn write(self) -> MapFlags { self | MapFlags::WRITE }
    #[inline] pub fn write_invalidate_region(self) -> MapFlags { self |
        MapFlags::WRITE_INVALIDATE_REGION }
}

impl Default for MapFlags {
    #[inline] fn default() -> MapFlags { MapFlags::empty() }
}

pub const MAP_READ: MapFlags = MapFlags::READ;
pub const MAP_WRITE: MapFlags = MapFlags::WRITE;
pub const MAP_WRITE_INVALIDATE_REGION: MapFlags = MapFlags::WRITE_INVALIDATE_REGION;


bitflags! {
    /// cl_program_binary_type
    pub struct ProgramBinaryType: u32 {
        const NONE = 0x0;
        const COMPILED_OBJECT = 0x1;
        const LIBRARY = 0x2;
        const EXECUTABLE = 0x4;
    }
}

pub const PROGRAM_BINARY_TYPE_NONE: ProgramBinaryType = ProgramBinaryType::NONE;
pub const PROGRAM_BINARY_TYPE_COMPILED_OBJECT: ProgramBinaryType = ProgramBinaryType::COMPILED_OBJECT;
pub const PROGRAM_BINARY_TYPE_LIBRARY: ProgramBinaryType = ProgramBinaryType::LIBRARY;
pub const PROGRAM_BINARY_TYPE_EXECUTABLE: ProgramBinaryType = ProgramBinaryType::EXECUTABLE;


bitflags! {
    /// cl_kernel_arg_type_qualifer
    pub struct KernelArgTypeQualifier: u64 {
        const NONE = 0;
        const CONST = 1;
        const RESTRICT = 1 << 1;
        const VOLATILE = 1 << 2;
    }
}

pub const KERNEL_ARG_TYPE_NONE: KernelArgTypeQualifier = KernelArgTypeQualifier::NONE;
pub const KERNEL_ARG_TYPE_CONST: KernelArgTypeQualifier = KernelArgTypeQualifier::CONST;
pub const KERNEL_ARG_TYPE_RESTRICT: KernelArgTypeQualifier = KernelArgTypeQualifier::RESTRICT;
pub const KERNEL_ARG_TYPE_VOLATILE: KernelArgTypeQualifier = KernelArgTypeQualifier::VOLATILE;

//=============================================================================
//=============================== ENUMERATORS =================================
//=============================================================================


// #[derive(PartialEq, Debug, FromPrimitive)]
enum_from_primitive! {
    /// The status of an OpenCL API call. Used for returning success/error codes.
    #[allow(non_camel_case_types)]
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
        CL_INVALID_PIPE_SIZE                            = -69,
        CL_INVALID_DEVICE_QUEUE                         = -70,
        CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR          = -1000,
        CL_PLATFORM_NOT_FOUND_KHR                       = -1001,
        CL_NV_INVALID_MEM_ACCESS                        = -9999,
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, fmtr: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmtr, "{:?}", self)
    }
}


enum_from_primitive! {
    /// specify the texture target type
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum GlTextureTarget {
        GlTexture1d = ffi::GL_TEXTURE_1D as isize,
        GlTexture1dArray = ffi::GL_TEXTURE_1D_ARRAY as isize,
        GlTextureBuffer = ffi::GL_TEXTURE_BUFFER as isize,
        GlTexture2d = ffi::GL_TEXTURE_2D as isize,
        GlTexture2dArray = ffi::GL_TEXTURE_2D_ARRAY as isize,
        GlTexture3d = ffi::GL_TEXTURE_3D as isize,
        GlTextureCubeMapPositiveX = ffi::GL_TEXTURE_CUBE_MAP_POSITIVE_X as isize,
        GlTextureCubeMapPositiveY = ffi::GL_TEXTURE_CUBE_MAP_POSITIVE_Y as isize,
        GlTextureCubeMapPositiveZ = ffi::GL_TEXTURE_CUBE_MAP_POSITIVE_Z as isize,
        GlTextureCubeMapNegativeX = ffi::GL_TEXTURE_CUBE_MAP_NEGATIVE_X as isize,
        GlTextureCubeMapNegativeY = ffi::GL_TEXTURE_CUBE_MAP_NEGATIVE_Y as isize,
        GlTextureCubeMapNegativeZ = ffi::GL_TEXTURE_CUBE_MAP_NEGATIVE_Z as isize,
        GlTextureRectangle = ffi::GL_TEXTURE_RECTANGLE as isize,
    }
}

enum_from_primitive! {
    // cl_gl_object_type = 0x2000 - 0x200F enum values are currently taken
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ClGlObjectType {
        ClGlObjectBuffer = ffi::CL_GL_OBJECT_BUFFER as isize,
        ClGlObjectTexture2D = ffi::CL_GL_OBJECT_TEXTURE2D as isize,
        ClGlObjectTexture3D = ffi::CL_GL_OBJECT_TEXTURE3D as isize,
        ClGlObjectRenderbuffer = ffi::CL_GL_OBJECT_RENDERBUFFER as isize,
        ClGlObjectTexture2DArray = ffi::CL_GL_OBJECT_TEXTURE2D_ARRAY as isize,
        ClGlObjectTexture1D = ffi::CL_GL_OBJECT_TEXTURE1D as isize,
        ClGlObjectTexture1DArray = ffi::CL_GL_OBJECT_TEXTURE1D_ARRAY as isize,
        ClGlObjectTextureBuffer = ffi::CL_GL_OBJECT_TEXTURE_BUFFER as isize,
    }
}

enum_from_primitive! {
    /// Specifies the number of channels and the channel layout i.e. the memory layout in which channels are stored in the image. Valid values are described in the table below. (from SDK)
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ImageChannelOrder {
        R = ffi::CL_R as isize,
        A = ffi::CL_A as isize,
        Rg = ffi::CL_RG as isize,
        Ra = ffi::CL_RA as isize,
        // This format can only be used if channel data type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010:
        Rgb = ffi::CL_RGB as isize,
        Rgba = ffi::CL_RGBA as isize,
        // This format can only be used if channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8:
        Bgra = ffi::CL_BGRA as isize,
        // This format can only be used if channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8:
        Argb = ffi::CL_ARGB as isize,
        // This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT:
        Intensity = ffi::CL_INTENSITY as isize,
        // This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT:
        Luminance = ffi::CL_LUMINANCE as isize,
        Rx = ffi::CL_Rx as isize,
        Rgx = ffi::CL_RGx as isize,
        // This format can only be used if channel data type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010:
        Rgbx = ffi::CL_RGBx as isize,
        Depth = ffi::CL_DEPTH as isize,
        DepthStencil = ffi::CL_DEPTH_STENCIL as isize,
    }
}

enum_from_primitive! {
    /// Describes the size of the channel data type. The number of bits per element determined by the image_channel_data_type and image_channel_order must be a power of two. The list of supported values is described in the table below. (from SDK)
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ImageChannelDataType {
        // Each channel component is a normalized signed 8-bit integer value:
        SnormInt8 = ffi::CL_SNORM_INT8 as isize,
        // Each channel component is a normalized signed 16-bit integer value:
        SnormInt16 = ffi::CL_SNORM_INT16 as isize,
        // Each channel component is a normalized unsigned 8-bit integer value:
        UnormInt8 = ffi::CL_UNORM_INT8 as isize,
        // Each channel component is a normalized unsigned 16-bit integer value:
        UnormInt16 = ffi::CL_UNORM_INT16 as isize,
        // Represents a normalized 5-6-5 3-channel RGB image. The channel order must be CL_RGB or CL_RGBx:
        UnormShort565 = ffi::CL_UNORM_SHORT_565 as isize,
        // Represents a normalized x-5-5-5 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx:
        UnormShort555 = ffi::CL_UNORM_SHORT_555 as isize,
        // Represents a normalized x-10-10-10 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx:
        UnormInt101010 = ffi::CL_UNORM_INT_101010 as isize,
        // Each channel component is an unnormalized signed 8-bit integer value:
        SignedInt8 = ffi::CL_SIGNED_INT8 as isize,
        // Each channel component is an unnormalized signed 16-bit integer value:
        SignedInt16 = ffi::CL_SIGNED_INT16 as isize,
        // Each channel component is an unnormalized signed 32-bit integer value:
        SignedInt32 = ffi::CL_SIGNED_INT32 as isize,
        // Each channel component is an unnormalized unsigned 8-bit integer value:
        UnsignedInt8 = ffi::CL_UNSIGNED_INT8 as isize,
        // Each channel component is an unnormalized unsigned 16-bit integer value:
        UnsignedInt16 = ffi::CL_UNSIGNED_INT16 as isize,
        // Each channel component is an unnormalized unsigned 32-bit integer value:
        UnsignedInt32 = ffi::CL_UNSIGNED_INT32 as isize,
        // Each channel component is a 16-bit half-float value:
        HalfFloat = ffi::CL_HALF_FLOAT as isize,
        // Each channel component is a single precision floating-point value:
        Float = ffi::CL_FLOAT as isize,
        // Each channel component is a normalized unsigned 24-bit integer value:
        UnormInt24 = ffi::CL_UNORM_INT24 as isize,
    }
}


enum_from_primitive! {
    /// cl_bool
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Cbool {
        False = ffi::CL_FALSE as isize,
        True = ffi::CL_TRUE as isize,
    }
}


enum_from_primitive! {
    /// cl_bool: Polling
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Polling {
        Blocking = ffi::CL_BLOCKING as isize,
        NonBlocking = ffi::CL_NON_BLOCKING as isize,
    }
}


enum_from_primitive! {
    /// cl_platform_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum PlatformInfo {
        Profile = ffi::CL_PLATFORM_PROFILE as isize,
        Version = ffi::CL_PLATFORM_VERSION as isize,
        Name = ffi::CL_PLATFORM_NAME as isize,
        Vendor = ffi::CL_PLATFORM_VENDOR as isize,
        Extensions = ffi::CL_PLATFORM_EXTENSIONS as isize,
    }
}


enum_from_primitive! {
    /// cl_device_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum DeviceInfo {
        Type = ffi::CL_DEVICE_TYPE as isize,
        VendorId = ffi::CL_DEVICE_VENDOR_ID as isize,
        MaxComputeUnits = ffi::CL_DEVICE_MAX_COMPUTE_UNITS as isize,
        MaxWorkItemDimensions = ffi::CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS as isize,
        MaxWorkGroupSize = ffi::CL_DEVICE_MAX_WORK_GROUP_SIZE as isize,
        MaxWorkItemSizes = ffi::CL_DEVICE_MAX_WORK_ITEM_SIZES as isize,
        PreferredVectorWidthChar = ffi::CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR as isize,
        PreferredVectorWidthShort = ffi::CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT as isize,
        PreferredVectorWidthInt = ffi::CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT as isize,
        PreferredVectorWidthLong = ffi::CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG as isize,
        PreferredVectorWidthFloat = ffi::CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT as isize,
        PreferredVectorWidthDouble = ffi::CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE as isize,
        MaxClockFrequency = ffi::CL_DEVICE_MAX_CLOCK_FREQUENCY as isize,
        AddressBits = ffi::CL_DEVICE_ADDRESS_BITS as isize,
        MaxReadImageArgs = ffi::CL_DEVICE_MAX_READ_IMAGE_ARGS as isize,
        MaxWriteImageArgs = ffi::CL_DEVICE_MAX_WRITE_IMAGE_ARGS as isize,
        MaxMemAllocSize = ffi::CL_DEVICE_MAX_MEM_ALLOC_SIZE as isize,
        Image2dMaxWidth = ffi::CL_DEVICE_IMAGE2D_MAX_WIDTH as isize,
        Image2dMaxHeight = ffi::CL_DEVICE_IMAGE2D_MAX_HEIGHT as isize,
        Image3dMaxWidth = ffi::CL_DEVICE_IMAGE3D_MAX_WIDTH as isize,
        Image3dMaxHeight = ffi::CL_DEVICE_IMAGE3D_MAX_HEIGHT as isize,
        Image3dMaxDepth = ffi::CL_DEVICE_IMAGE3D_MAX_DEPTH as isize,
        ImageSupport = ffi::CL_DEVICE_IMAGE_SUPPORT as isize,
        MaxParameterSize = ffi::CL_DEVICE_MAX_PARAMETER_SIZE as isize,
        MaxSamplers = ffi::CL_DEVICE_MAX_SAMPLERS as isize,
        MemBaseAddrAlign = ffi::CL_DEVICE_MEM_BASE_ADDR_ALIGN as isize,
        MinDataTypeAlignSize = ffi::CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE as isize,
        SingleFpConfig = ffi::CL_DEVICE_SINGLE_FP_CONFIG as isize,
        GlobalMemCacheType = ffi::CL_DEVICE_GLOBAL_MEM_CACHE_TYPE as isize,
        GlobalMemCachelineSize = ffi::CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE as isize,
        GlobalMemCacheSize = ffi::CL_DEVICE_GLOBAL_MEM_CACHE_SIZE as isize,
        GlobalMemSize = ffi::CL_DEVICE_GLOBAL_MEM_SIZE as isize,
        MaxConstantBufferSize = ffi::CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE as isize,
        MaxConstantArgs = ffi::CL_DEVICE_MAX_CONSTANT_ARGS as isize,
        LocalMemType = ffi::CL_DEVICE_LOCAL_MEM_TYPE as isize,
        LocalMemSize = ffi::CL_DEVICE_LOCAL_MEM_SIZE as isize,
        ErrorCorrectionSupport = ffi::CL_DEVICE_ERROR_CORRECTION_SUPPORT as isize,
        ProfilingTimerResolution = ffi::CL_DEVICE_PROFILING_TIMER_RESOLUTION as isize,
        EndianLittle = ffi::CL_DEVICE_ENDIAN_LITTLE as isize,
        Available = ffi::CL_DEVICE_AVAILABLE as isize,
        CompilerAvailable = ffi::CL_DEVICE_COMPILER_AVAILABLE as isize,
        ExecutionCapabilities = ffi::CL_DEVICE_EXECUTION_CAPABILITIES as isize,
        QueueProperties = ffi::CL_DEVICE_QUEUE_PROPERTIES as isize,
        Name = ffi::CL_DEVICE_NAME as isize,
        Vendor = ffi::CL_DEVICE_VENDOR as isize,
        DriverVersion = ffi::CL_DRIVER_VERSION as isize,
        Profile = ffi::CL_DEVICE_PROFILE as isize,
        Version = ffi::CL_DEVICE_VERSION as isize,
        Extensions = ffi::CL_DEVICE_EXTENSIONS as isize,
        Platform = ffi::CL_DEVICE_PLATFORM as isize,
        DoubleFpConfig = ffi::CL_DEVICE_DOUBLE_FP_CONFIG as isize,
        HalfFpConfig = ffi::CL_DEVICE_HALF_FP_CONFIG as isize,
        PreferredVectorWidthHalf = ffi::CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF as isize,
        HostUnifiedMemory = ffi::CL_DEVICE_HOST_UNIFIED_MEMORY as isize,
        NativeVectorWidthChar = ffi::CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR as isize,
        NativeVectorWidthShort = ffi::CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT as isize,
        NativeVectorWidthInt = ffi::CL_DEVICE_NATIVE_VECTOR_WIDTH_INT as isize,
        NativeVectorWidthLong = ffi::CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG as isize,
        NativeVectorWidthFloat = ffi::CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT as isize,
        NativeVectorWidthDouble = ffi::CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE as isize,
        NativeVectorWidthHalf = ffi::CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF as isize,
        OpenclCVersion = ffi::CL_DEVICE_OPENCL_C_VERSION as isize,
        LinkerAvailable = ffi::CL_DEVICE_LINKER_AVAILABLE as isize,
        BuiltInKernels = ffi::CL_DEVICE_BUILT_IN_KERNELS as isize,
        ImageMaxBufferSize = ffi::CL_DEVICE_IMAGE_MAX_BUFFER_SIZE as isize,
        ImageMaxArraySize = ffi::CL_DEVICE_IMAGE_MAX_ARRAY_SIZE as isize,
        ParentDevice = ffi::CL_DEVICE_PARENT_DEVICE as isize,
        PartitionMaxSubDevices = ffi::CL_DEVICE_PARTITION_MAX_SUB_DEVICES as isize,
        PartitionProperties = ffi::CL_DEVICE_PARTITION_PROPERTIES as isize,
        PartitionAffinityDomain = ffi::CL_DEVICE_PARTITION_AFFINITY_DOMAIN as isize,
        PartitionType = ffi::CL_DEVICE_PARTITION_TYPE as isize,
        ReferenceCount = ffi::CL_DEVICE_REFERENCE_COUNT as isize,
        PreferredInteropUserSync = ffi::CL_DEVICE_PREFERRED_INTEROP_USER_SYNC as isize,
        PrintfBufferSize = ffi::CL_DEVICE_PRINTF_BUFFER_SIZE as isize,
        ImagePitchAlignment = ffi::CL_DEVICE_IMAGE_PITCH_ALIGNMENT as isize,
        ImageBaseAddressAlignment = ffi::CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT as isize,
    }
}


enum_from_primitive! {
    /// cl_mem_cache_type
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum DeviceMemCacheType {
        None = ffi::CL_NONE as isize,
        ReadOnlyCache = ffi::CL_READ_ONLY_CACHE as isize,
        ReadWriteCache = ffi::CL_READ_WRITE_CACHE as isize,
    }
}


enum_from_primitive! {
    /// cl_device_local_mem_type
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum DeviceLocalMemType {
        None = ffi::CL_NONE as isize,
        Local = ffi::CL_LOCAL as isize,
        Global = ffi::CL_GLOBAL as isize,
    }
}


enum_from_primitive! {
    /// cl_context_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ContextInfo {
        ReferenceCount = ffi::CL_CONTEXT_REFERENCE_COUNT as isize,
        Devices = ffi::CL_CONTEXT_DEVICES as isize,
        Properties = ffi::CL_CONTEXT_PROPERTIES as isize,
        NumDevices = ffi::CL_CONTEXT_NUM_DEVICES as isize,
    }
}


enum_from_primitive! {
    /// cl_gl_context_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum GlContextInfo {
        // Return the CL device currently associated with the specified OpenGL context.
        CurrentDevice = ffi::CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR as isize,
        // List of all CL devices which may be associated with the specified OpenGL context.
        Devices = ffi::CL_DEVICES_FOR_GL_CONTEXT_KHR as isize,
    }
}


// [TODO]: Do proper auto-detection of available OpenGL context type.
#[cfg(target_os = "macos")]
pub const CL_CGL_SHAREGROUP_KHR_OS_SPECIFIC: isize = ffi::CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE;
#[cfg(not(target_os = "macos"))]
pub const CL_CGL_SHAREGROUP_KHR_OS_SPECIFIC: isize = ffi::CL_CGL_SHAREGROUP_KHR;

enum_from_primitive! {
    /// cl_context_info + cl_context_properties
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ContextProperty {
        Platform = ffi::CL_CONTEXT_PLATFORM as isize,
        InteropUserSync = ffi::CL_CONTEXT_INTEROP_USER_SYNC as isize,
        D3d10DeviceKhr = ffi::CL_CONTEXT_D3D10_DEVICE_KHR as isize,
        GlContextKhr = ffi::CL_GL_CONTEXT_KHR as isize,
        EglDisplayKhr = ffi::CL_EGL_DISPLAY_KHR as isize,
        GlxDisplayKhr = ffi::CL_GLX_DISPLAY_KHR as isize,
        CglSharegroupKhr = CL_CGL_SHAREGROUP_KHR_OS_SPECIFIC,
        WglHdcKhr = ffi::CL_WGL_HDC_KHR as isize,
        AdapterD3d9Khr = ffi::CL_CONTEXT_ADAPTER_D3D9_KHR as isize,
        AdapterD3d9exKhr = ffi::CL_CONTEXT_ADAPTER_D3D9EX_KHR as isize,
        AdapterDxvaKhr = ffi::CL_CONTEXT_ADAPTER_DXVA_KHR as isize,
        D3d11DeviceKhr = ffi::CL_CONTEXT_D3D11_DEVICE_KHR as isize,
    }
}


enum_from_primitive! {
    /// cl_context_info + cl_context_properties
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ContextInfoOrPropertiesPointerType {
        Platform = ffi::CL_CONTEXT_PLATFORM as isize,
        InteropUserSync = ffi::CL_CONTEXT_INTEROP_USER_SYNC as isize,
    }
}


enum_from_primitive! {
    /// [INCOMPLETE] cl_device_partition_property
    ///
    /// [FIXME]: This types variants should also contain data described in:
    /// [https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateSubDevices.html]
    /// (https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateSubDevices.html)
    ///
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum DevicePartitionProperty {
        Equally = ffi::CL_DEVICE_PARTITION_EQUALLY as isize,
        ByCounts = ffi::CL_DEVICE_PARTITION_BY_COUNTS as isize,
        ByCountsListEnd = ffi::CL_DEVICE_PARTITION_BY_COUNTS_LIST_END as isize,
        ByAffinityDomain = ffi::CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN as isize,
    }
}


enum_from_primitive! {
    /// cl_command_queue_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum CommandQueueInfo {
        Context = ffi::CL_QUEUE_CONTEXT as isize,
        Device = ffi::CL_QUEUE_DEVICE as isize,
        ReferenceCount = ffi::CL_QUEUE_REFERENCE_COUNT as isize,
        Properties = ffi::CL_QUEUE_PROPERTIES as isize,
    }
}


enum_from_primitive! {
    /// cl_channel_type
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ChannelType {
        SnormInt8 = ffi::CL_SNORM_INT8 as isize,
        SnormInt16 = ffi::CL_SNORM_INT16 as isize,
        UnormInt8 = ffi::CL_UNORM_INT8 as isize,
        UnormInt16 = ffi::CL_UNORM_INT16 as isize,
        UnormShort565 = ffi::CL_UNORM_SHORT_565 as isize,
        UnormShort555 = ffi::CL_UNORM_SHORT_555 as isize,
        UnormInt101010 = ffi::CL_UNORM_INT_101010 as isize,
        SignedInt8 = ffi::CL_SIGNED_INT8 as isize,
        SignedInt16 = ffi::CL_SIGNED_INT16 as isize,
        SignedInt32 = ffi::CL_SIGNED_INT32 as isize,
        UnsignedInt8 = ffi::CL_UNSIGNED_INT8 as isize,
        UnsignedInt16 = ffi::CL_UNSIGNED_INT16 as isize,
        UnsignedInt32 = ffi::CL_UNSIGNED_INT32 as isize,
        HalfFloat = ffi::CL_HALF_FLOAT as isize,
        Float = ffi::CL_FLOAT as isize,
        UnormInt24 = ffi::CL_UNORM_INT24 as isize,
    }
}


enum_from_primitive! {
    /// cl_mem_object_type
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum MemObjectType {
        Buffer = ffi::CL_MEM_OBJECT_BUFFER as isize,
        Image2d = ffi::CL_MEM_OBJECT_IMAGE2D as isize,
        Image3d = ffi::CL_MEM_OBJECT_IMAGE3D as isize,
        Image2dArray = ffi::CL_MEM_OBJECT_IMAGE2D_ARRAY as isize,
        Image1d = ffi::CL_MEM_OBJECT_IMAGE1D as isize,
        Image1dArray = ffi::CL_MEM_OBJECT_IMAGE1D_ARRAY as isize,
        Image1dBuffer = ffi::CL_MEM_OBJECT_IMAGE1D_BUFFER as isize,
    }
}


enum_from_primitive! {
    /// cl_mem_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum MemInfo {
        Type = ffi::CL_MEM_TYPE as isize,
        Flags = ffi::CL_MEM_FLAGS as isize,
        Size = ffi::CL_MEM_SIZE as isize,
        HostPtr = ffi::CL_MEM_HOST_PTR as isize,
        MapCount = ffi::CL_MEM_MAP_COUNT as isize,
        ReferenceCount = ffi::CL_MEM_REFERENCE_COUNT as isize,
        Context = ffi::CL_MEM_CONTEXT as isize,
        AssociatedMemobject = ffi::CL_MEM_ASSOCIATED_MEMOBJECT as isize,
        Offset = ffi::CL_MEM_OFFSET as isize,
    }
}


enum_from_primitive! {
    /// cl_image_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ImageInfo {
        Format = ffi::CL_IMAGE_FORMAT as isize,
        ElementSize = ffi::CL_IMAGE_ELEMENT_SIZE as isize,
        RowPitch = ffi::CL_IMAGE_ROW_PITCH as isize,
        SlicePitch = ffi::CL_IMAGE_SLICE_PITCH as isize,
        Width = ffi::CL_IMAGE_WIDTH as isize,
        Height = ffi::CL_IMAGE_HEIGHT as isize,
        Depth = ffi::CL_IMAGE_DEPTH as isize,
        ArraySize = ffi::CL_IMAGE_ARRAY_SIZE as isize,
        Buffer = ffi::CL_IMAGE_BUFFER as isize,
        NumMipLevels = ffi::CL_IMAGE_NUM_MIP_LEVELS as isize,
        NumSamples = ffi::CL_IMAGE_NUM_SAMPLES as isize,
    }
}


enum_from_primitive! {
    /// cl_addressing_mode
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum AddressingMode {
        None = ffi::CL_ADDRESS_NONE as isize,
        ClampToEdge = ffi::CL_ADDRESS_CLAMP_TO_EDGE as isize,
        Clamp = ffi::CL_ADDRESS_CLAMP as isize,
        Repeat = ffi::CL_ADDRESS_REPEAT as isize,
        MirroredRepeat = ffi::CL_ADDRESS_MIRRORED_REPEAT as isize,
    }
}


enum_from_primitive! {
    /// cl_filter_mode
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum FilterMode {
        Nearest = ffi::CL_FILTER_NEAREST as isize,
        Linear = ffi::CL_FILTER_LINEAR as isize,
    }
}


enum_from_primitive! {
    /// cl_sampler_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum SamplerInfo {
        ReferenceCount = ffi::CL_SAMPLER_REFERENCE_COUNT as isize,
        Context = ffi::CL_SAMPLER_CONTEXT as isize,
        NormalizedCoords = ffi::CL_SAMPLER_NORMALIZED_COORDS as isize,
        AddressingMode = ffi::CL_SAMPLER_ADDRESSING_MODE as isize,
        FilterMode = ffi::CL_SAMPLER_FILTER_MODE as isize,
    }
}


enum_from_primitive! {
    /// cl_program_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ProgramInfo {
        ReferenceCount = ffi::CL_PROGRAM_REFERENCE_COUNT as isize,
        Context = ffi::CL_PROGRAM_CONTEXT as isize,
        NumDevices = ffi::CL_PROGRAM_NUM_DEVICES as isize,
        Devices = ffi::CL_PROGRAM_DEVICES as isize,
        Source = ffi::CL_PROGRAM_SOURCE as isize,
        BinarySizes = ffi::CL_PROGRAM_BINARY_SIZES as isize,
        Binaries = ffi::CL_PROGRAM_BINARIES as isize,
        NumKernels = ffi::CL_PROGRAM_NUM_KERNELS as isize,
        KernelNames = ffi::CL_PROGRAM_KERNEL_NAMES as isize,
    }
}


enum_from_primitive! {
    /// cl_program_build_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ProgramBuildInfo {
        BuildStatus = ffi::CL_PROGRAM_BUILD_STATUS as isize,
        BuildOptions = ffi::CL_PROGRAM_BUILD_OPTIONS as isize,
        BuildLog = ffi::CL_PROGRAM_BUILD_LOG as isize,
        BinaryType = ffi::CL_PROGRAM_BINARY_TYPE as isize,
    }
}



enum_from_primitive! {
    /// cl_build_status
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ProgramBuildStatus {
        Success = ffi::CL_BUILD_SUCCESS as isize,
        None = ffi::CL_BUILD_NONE as isize,
        Error = ffi::CL_BUILD_ERROR as isize,
        InProgress = ffi::CL_BUILD_IN_PROGRESS as isize,
    }
}


enum_from_primitive! {
    /// cl_kernel_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum KernelInfo {
        FunctionName = ffi::CL_KERNEL_FUNCTION_NAME as isize,
        NumArgs = ffi::CL_KERNEL_NUM_ARGS as isize,
        ReferenceCount = ffi::CL_KERNEL_REFERENCE_COUNT as isize,
        Context = ffi::CL_KERNEL_CONTEXT as isize,
        Program = ffi::CL_KERNEL_PROGRAM as isize,
        Attributes = ffi::CL_KERNEL_ATTRIBUTES as isize,
    }
}


enum_from_primitive! {
    /// cl_kernel_arg_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum KernelArgInfo {
        AddressQualifier = ffi::CL_KERNEL_ARG_ADDRESS_QUALIFIER as isize,
        AccessQualifier = ffi::CL_KERNEL_ARG_ACCESS_QUALIFIER as isize,
        TypeName = ffi::CL_KERNEL_ARG_TYPE_NAME as isize,
        TypeQualifier = ffi::CL_KERNEL_ARG_TYPE_QUALIFIER as isize,
        Name = ffi::CL_KERNEL_ARG_NAME as isize,
    }
}


enum_from_primitive! {
    /// cl_kernel_arg_address_qualifier
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum KernelArgAddressQualifier {
        Global = ffi::CL_KERNEL_ARG_ADDRESS_GLOBAL as isize,
        Local = ffi::CL_KERNEL_ARG_ADDRESS_LOCAL as isize,
        Constant = ffi::CL_KERNEL_ARG_ADDRESS_CONSTANT as isize,
        Private = ffi::CL_KERNEL_ARG_ADDRESS_PRIVATE as isize,
    }
}


enum_from_primitive! {
    /// cl_kernel_arg_access_qualifier
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum KernelArgAccessQualifier {
        ReadOnly = ffi::CL_KERNEL_ARG_ACCESS_READ_ONLY as isize,
        WriteOnly = ffi::CL_KERNEL_ARG_ACCESS_WRITE_ONLY as isize,
        ReadWrite = ffi::CL_KERNEL_ARG_ACCESS_READ_WRITE as isize,
        None = ffi::CL_KERNEL_ARG_ACCESS_NONE as isize,
     }
}


enum_from_primitive! {
    /// cl_kernel_work_group_info
    ///
    /// [NOTE] PrivateMemSize: If device is not a custom device or kernel is not a built-in
    /// kernel, clGetKernelArgInfo returns the error CL_INVALID_VALUE:
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum KernelWorkGroupInfo {
        WorkGroupSize = ffi::CL_KERNEL_WORK_GROUP_SIZE as isize,
        CompileWorkGroupSize = ffi::CL_KERNEL_COMPILE_WORK_GROUP_SIZE as isize,
        LocalMemSize = ffi::CL_KERNEL_LOCAL_MEM_SIZE as isize,
        PreferredWorkGroupSizeMultiple = ffi::CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE as isize,
        PrivateMemSize = ffi::CL_KERNEL_PRIVATE_MEM_SIZE as isize,
        GlobalWorkSize = ffi::CL_KERNEL_GLOBAL_WORK_SIZE as isize,
    }
}


enum_from_primitive! {
    /// cl_event_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum EventInfo {
        CommandQueue = ffi::CL_EVENT_COMMAND_QUEUE as isize,
        CommandType = ffi::CL_EVENT_COMMAND_TYPE as isize,
        ReferenceCount = ffi::CL_EVENT_REFERENCE_COUNT as isize,
        CommandExecutionStatus = ffi::CL_EVENT_COMMAND_EXECUTION_STATUS as isize,
        Context = ffi::CL_EVENT_CONTEXT as isize,
    }
}


enum_from_primitive! {
    /// cl_command_type
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum CommandType {
        NdrangeKernel = ffi::CL_COMMAND_NDRANGE_KERNEL as isize,
        Task = ffi::CL_COMMAND_TASK as isize,
        NativeKernel = ffi::CL_COMMAND_NATIVE_KERNEL as isize,
        ReadBuffer = ffi::CL_COMMAND_READ_BUFFER as isize,
        WriteBuffer = ffi::CL_COMMAND_WRITE_BUFFER as isize,
        CopyBuffer = ffi::CL_COMMAND_COPY_BUFFER as isize,
        ReadImage = ffi::CL_COMMAND_READ_IMAGE as isize,
        WriteImage = ffi::CL_COMMAND_WRITE_IMAGE as isize,
        CopyImage = ffi::CL_COMMAND_COPY_IMAGE as isize,
        CopyImageToBuffer = ffi::CL_COMMAND_COPY_IMAGE_TO_BUFFER as isize,
        CopyBufferToImage = ffi::CL_COMMAND_COPY_BUFFER_TO_IMAGE as isize,
        MapBuffer = ffi::CL_COMMAND_MAP_BUFFER as isize,
        MapImage = ffi::CL_COMMAND_MAP_IMAGE as isize,
        UnmapMemObject = ffi::CL_COMMAND_UNMAP_MEM_OBJECT as isize,
        Marker = ffi::CL_COMMAND_MARKER as isize,
        AcquireGlObjects = ffi::CL_COMMAND_ACQUIRE_GL_OBJECTS as isize,
        ReleaseGlObjects = ffi::CL_COMMAND_RELEASE_GL_OBJECTS as isize,
        ReadBufferRect = ffi::CL_COMMAND_READ_BUFFER_RECT as isize,
        WriteBufferRect = ffi::CL_COMMAND_WRITE_BUFFER_RECT as isize,
        CopyBufferRect = ffi::CL_COMMAND_COPY_BUFFER_RECT as isize,
        User = ffi::CL_COMMAND_USER as isize,
        Barrier = ffi::CL_COMMAND_BARRIER as isize,
        MigrateMemObjects = ffi::CL_COMMAND_MIGRATE_MEM_OBJECTS as isize,
        FillBuffer = ffi::CL_COMMAND_FILL_BUFFER as isize,
        FillImage = ffi::CL_COMMAND_FILL_IMAGE as isize,
    }
}


enum_from_primitive! {
    /// command execution status
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum CommandExecutionStatus {
        Complete = ffi::CL_COMPLETE as isize,
        Running = ffi::CL_RUNNING as isize,
        Submitted = ffi::CL_SUBMITTED as isize,
        Queued = ffi::CL_QUEUED as isize,
    }
}


enum_from_primitive! {
    /// cl_buffer_create_type
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum BufferCreateType {
        Region = ffi::CL_BUFFER_CREATE_TYPE_REGION as isize,
        __DUMMY,
    }
}


enum_from_primitive! {
    /// cl_profiling_info
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ProfilingInfo {
        Queued = ffi::CL_PROFILING_COMMAND_QUEUED as isize,
        Submit = ffi::CL_PROFILING_COMMAND_SUBMIT as isize,
        Start = ffi::CL_PROFILING_COMMAND_START as isize,
        End = ffi::CL_PROFILING_COMMAND_END as isize,
    }
}
