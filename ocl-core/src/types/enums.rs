//! Custom enumerators not specifically based on OpenCL C-style enums.
//!
//!
//!
//
// [TODO]: Evaluate usefulness of `Error` impls and potentially remove.
// [TODO]: Possibly remove custom implementation of `Debug` and derive instead.
//

#![allow(dead_code)]

use std::fmt;
use failure::Fail;
use num_traits::FromPrimitive;
use util;
use ffi::{cl_image_format, cl_context_properties, size_t, c_void};

use ::{OclPrm, CommandQueueProperties, PlatformId, PlatformInfo, DeviceId, DeviceInfo, ContextInfo,
    GlContextInfo, Context, CommandQueue, CommandQueueInfo, CommandType, CommandExecutionStatus,
    Mem, MemInfo, MemObjectType, MemFlags, Sampler, SamplerInfo, AddressingMode, FilterMode,
    ProgramInfo, ProgramBuildInfo, Program, ProgramBuildStatus, ProgramBinaryType, KernelInfo,
    KernelArgInfo, KernelWorkGroupInfo, KernelArgAddressQualifier, KernelArgAccessQualifier,
    KernelArgTypeQualifier, ImageInfo, ImageFormat, EventInfo, ProfilingInfo, DeviceType,
    DeviceFpConfig, DeviceMemCacheType, DeviceLocalMemType, DeviceExecCapabilities,
    DevicePartitionProperty, DeviceAffinityDomain, OpenclVersion, ContextProperties,
    ImageFormatParseResult, Status};

use error::{Result as OclCoreResult, Error as OclCoreError};


#[derive(Fail)]
pub enum EmptyInfoResultError {
    #[fail(display = "Platform info unavailable")]
    Platform,
    #[fail(display = "Device info unavailable")]
    Device,
    #[fail(display = "Context info unavailable")]
    Context,
    #[fail(display = "OpenGL info unavailable")]
    GlContext,
    #[fail(display = "Command queue info unavailable")]
    CommandQueue,
    #[fail(display = "Mem object info unavailable")]
    Mem,
    #[fail(display = "Image info unavailable")]
    Image,
    #[fail(display = "Sampler info unavailable")]
    Sampler,
    #[fail(display = "Program info unavailable")]
    Program,
    #[fail(display = "Program build info unavailable")]
    ProgramBuild,
    #[fail(display = "Kernel info unavailable")]
    Kernel,
    #[fail(display = "Kernel argument info unavailable")]
    KernelArg,
    #[fail(display = "Kernel work-group info unavailable")]
    KernelWorkGroup,
    #[fail(display = "Event info unavailable")]
    Event,
    #[fail(display = "Event profiling info unavailable")]
    Profiling,
}

impl fmt::Debug for EmptyInfoResultError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}



/// [UNSAFE] Kernel argument option type.
///
/// The type argument `T` is ignored for `Mem`, `Sampler`, and `UnsafePointer`
/// (just put `usize` or anything).
///
/// ## Safety
///
/// If there was some way for this enum to be marked unsafe it would be.
///
/// The `Mem`, `Sampler`, `Scalar`, and `Local` variants are tested and will
/// work perfectly well.
///
/// * `Vector`: The `Vector` variant is poorly tested and probably a bit
///   platform dependent. Use at your own risk.
/// * `UnsafePointer`: Really know what you're doing when using the
///   `UnsafePointer` variant. Setting its properties, `size` and `value`,
///   incorrectly can cause bugs, crashes, and data integrity issues that are
///   very hard to track down. This is due to the fact that the pointer value
///   is intended to be a pointer to a memory structure in YOUR programs
///   memory, NOT a copy of an OpenCL object pointer (such as a `cl_h::cl_mem`
///   for example, which is itself a `*mut libc::c_void`). This is made more
///   complicated by the fact that the pointer can also be a pointer to a
///   scalar (ex: `*const u32`, etc.). See the [SDK docs] for more details.
///
/// [SDK docs]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clSetKernelArg.html
#[derive(Debug)]
pub enum KernelArg<'a, T: 'a + OclPrm> {
    /// Type `T` is ignored.
    Mem(&'a Mem),
    /// Type `T` is ignored.
    MemNull,
    /// Type `T` is ignored.
    Sampler(&'a Sampler),
    /// Type `T` is ignored.
    SamplerNull,
    Scalar(T),
    Vector(T),
    /// Length in multiples of T (not bytes).
    Local(&'a usize),
    /// `size`: size in bytes. Type `T` is ignored.
    UnsafePointer { size: size_t, value: *const c_void },
}



/// Platform info result.
///
// #[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlatformInfoResult {
    Profile(String),
    Version(String),
    Name(String),
    Vendor(String),
    Extensions(String),
}

impl PlatformInfoResult {
    pub fn from_bytes(request: PlatformInfo, result: Vec<u8>)
            -> OclCoreResult<PlatformInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(EmptyInfoResultError::Platform));
        }

        let string = util::bytes_into_string(result)?;

        Ok(match request {
            PlatformInfo::Profile => PlatformInfoResult::Profile(string),
            PlatformInfo::Version => PlatformInfoResult::Version(string),
            PlatformInfo::Name => PlatformInfoResult::Name(string),
            PlatformInfo::Vendor => PlatformInfoResult::Vendor(string),
            PlatformInfo::Extensions => PlatformInfoResult::Extensions(string),
        })
    }

    /// Parse the `Version` string and get a numeric result as `OpenclVersion`.
    pub fn as_opencl_version(&self) -> OclCoreResult<OpenclVersion> {
        if let PlatformInfoResult::Version(ref ver) = *self {
            OpenclVersion::from_info_str(ver)
        } else {
            Err(format!("PlatformInfoResult::as_opencl_version(): Invalid platform info \
                result variant: ({:?}). This function can only be called on a \
                'PlatformInfoResult::Version' variant.", self).into())
        }
    }
}

impl fmt::Debug for PlatformInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for PlatformInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            PlatformInfoResult::Profile(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Version(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Name(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Vendor(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Extensions(ref s) => write!(f, "{}", s),
        }
    }
}

impl From<PlatformInfoResult> for String {
    fn from(ir: PlatformInfoResult) -> String {
        match ir {
            PlatformInfoResult::Profile(string)
            | PlatformInfoResult::Version(string)
            | PlatformInfoResult::Name(string)
            | PlatformInfoResult::Vendor(string)
            | PlatformInfoResult::Extensions(string) => string,
        }
    }
}


/// A device info result.
///
// #[derive(Debug)]
pub enum DeviceInfoResult {
    Type(DeviceType),                    // cl_device_type      FLAGS u64
    VendorId(u32),                 // cl_uint
    MaxComputeUnits(u32),          // cl_uint
    MaxWorkItemDimensions(u32),    // cl_uint
    MaxWorkGroupSize(usize),                            // usize
    MaxWorkItemSizes(Vec<usize>),         // [usize; CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS]
    PreferredVectorWidthChar(u32), // cl_uint
    PreferredVectorWidthShort(u32),// cl_uint
    PreferredVectorWidthInt(u32),  // cl_uint
    PreferredVectorWidthLong(u32), // cl_uint
    PreferredVectorWidthFloat(u32),// cl_uint
    PreferredVectorWidthDouble(u32),// cl_uint
    MaxClockFrequency(u32),        // cl_uint
    AddressBits(u32),              // cl_uint
    MaxReadImageArgs(u32),         // cl_uint
    MaxWriteImageArgs(u32),        // cl_uint
    MaxMemAllocSize(u64),          // cl_ulong
    Image2dMaxWidth(usize),          // usize
    Image2dMaxHeight(usize),         // usize
    Image3dMaxWidth(usize),          // usize
    Image3dMaxHeight(usize),         // usize
    Image3dMaxDepth(usize),          // usize
    ImageSupport(bool),             // cl_bool
    MaxParameterSize(usize),         // usize
    MaxSamplers(u32),              // cl_uint
    MemBaseAddrAlign(u32),         // cl_uint
    MinDataTypeAlignSize(u32),     // cl_uint
    SingleFpConfig(DeviceFpConfig),           // cl_device_fp_config    FLAGS u64
    GlobalMemCacheType(DeviceMemCacheType),       // cl_device_mem_cache_type   ENUM
    GlobalMemCachelineSize(u32),   // cl_uint
    GlobalMemCacheSize(u64),       // cl_ulong
    GlobalMemSize(u64),            // cl_ulong
    MaxConstantBufferSize(u64),    // cl_ulong
    MaxConstantArgs(u32),          // cl_uint
    LocalMemType(DeviceLocalMemType),             // cl_device_local_mem_type     ENUM
    LocalMemSize(u64),             // cl_ulong
    ErrorCorrectionSupport(bool),   // cl_bool
    ProfilingTimerResolution(usize), // usize
    EndianLittle(bool),             // cl_bool
    Available(bool),                // cl_bool
    CompilerAvailable(bool),        // cl_bool
    ExecutionCapabilities(DeviceExecCapabilities),    // cl_device_exec_capabilities    FLAGS u64
    QueueProperties(CommandQueueProperties),          // cl_command_queue_properties    FLAGS u64
    Name(String),                     // String
    Vendor(String),                   // String
    DriverVersion(String),            // String
    Profile(String),                  // String
    Version(OpenclVersion),
    Extensions(String),               // String
    Platform(PlatformId),             // cl_platform_id
    DoubleFpConfig(DeviceFpConfig),           // cl_device_fp_config    FLAGS u64
    HalfFpConfig(DeviceFpConfig),             // cl_device_fp_config    FLAGS u64
    PreferredVectorWidthHalf(u32), // cl_uint
    HostUnifiedMemory(bool),        // cl_bool
    NativeVectorWidthChar(u32),    // cl_uint
    NativeVectorWidthShort(u32),   // cl_uint
    NativeVectorWidthInt(u32),     // cl_uint
    NativeVectorWidthLong(u32),    // cl_uint
    NativeVectorWidthFloat(u32),   // cl_uint
    NativeVectorWidthDouble(u32),  // cl_uint
    NativeVectorWidthHalf(u32),    // cl_uint
    OpenclCVersion(String),           // String
    LinkerAvailable(bool),          // cl_bool
    BuiltInKernels(String),           // String
    ImageMaxBufferSize(usize),       // usize
    ImageMaxArraySize(usize),        // usize
    ParentDevice(Option<DeviceId>),   // cl_device_id
    PartitionMaxSubDevices(u32),   // cl_uint
    PartitionProperties(Vec<DevicePartitionProperty>),      // cl_device_partition_property  ENUM
    PartitionAffinityDomain(DeviceAffinityDomain),  // cl_device_affinity_domain    FLAGS u64
    PartitionType(Vec<DevicePartitionProperty>),            // cl_device_partition_property  ENUM
    ReferenceCount(u32),           // cl_uint
    PreferredInteropUserSync(bool), // cl_bool
    PrintfBufferSize(usize),         // usize
    ImagePitchAlignment(u32),      // cl_uint
    ImageBaseAddressAlignment(u32),// cl_uint
}

impl DeviceInfoResult {
    /// Returns a new `DeviceInfoResult::MaxWorkItemSizes` variant.
    pub fn from_bytes_max_work_item_sizes(request: DeviceInfo, result: Vec<u8>,
                max_wi_dims: u32) -> OclCoreResult<DeviceInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(EmptyInfoResultError::Device));
        }
        match request {
            DeviceInfo::MaxWorkItemSizes => {
                match max_wi_dims {
                    3 => {
                       let r = unsafe { util::bytes_into::<[usize; 3]>(result)? };
                        let mut v = Vec::with_capacity(3);
                        v.extend_from_slice(&r);
                        Ok(DeviceInfoResult::MaxWorkItemSizes(v))
                    },
                    2 => {
                        let r = unsafe { util::bytes_into::<[usize; 2]>(result)? };
                        let mut v = Vec::with_capacity(2);
                        v.extend_from_slice(&r);
                        Ok(DeviceInfoResult::MaxWorkItemSizes(v))
                    },
                    1 => {
                        let r = unsafe { util::bytes_into::<[usize; 1]>(result)? };
                        let mut v = Vec::with_capacity(1);
                        v.extend_from_slice(&r);
                        Ok(DeviceInfoResult::MaxWorkItemSizes(v))
                    },
                    _ => Err(OclCoreError::from("Error \
                        determining number of dimensions for MaxWorkItemSizes.")),
                }
            },
            _ => panic!("DeviceInfoResult::from_bytes_max_work_item_sizes: Called with \
                invalid info variant ({:?}). Call '::from_bytes` instead.", request),
        }
    }

    /// Returns a new `DeviceInfoResult` for all variants except `MaxWorkItemSizes`.
    pub fn from_bytes(request: DeviceInfo, result: Vec<u8>)
            -> OclCoreResult<DeviceInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::Device));
        }

        let ir = match request {
            DeviceInfo::Type => {
                let r = unsafe { util::bytes_into::<DeviceType>(result)? };
                DeviceInfoResult::Type(r)
            },
            DeviceInfo::VendorId => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::VendorId(r)
            },
            DeviceInfo::MaxComputeUnits => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MaxComputeUnits(r)
            },
            DeviceInfo::MaxWorkItemDimensions => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MaxWorkItemDimensions(r)
            },
            DeviceInfo::MaxWorkGroupSize => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::MaxWorkGroupSize(r)
            },
            DeviceInfo::MaxWorkItemSizes => {
                panic!("DeviceInfoResult::from_bytes: Called with invalid info variant ({:?}). \
                    Call '::from_bytes_max_work_item_sizes` instead.", request);
            },
            DeviceInfo::PreferredVectorWidthChar => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PreferredVectorWidthChar(r)
            },
            DeviceInfo::PreferredVectorWidthShort => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PreferredVectorWidthShort(r)
            },
            DeviceInfo::PreferredVectorWidthInt => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PreferredVectorWidthInt(r)
            },
            DeviceInfo::PreferredVectorWidthLong => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PreferredVectorWidthLong(r)
            },
            DeviceInfo::PreferredVectorWidthFloat => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PreferredVectorWidthFloat(r)
            },
            DeviceInfo::PreferredVectorWidthDouble => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PreferredVectorWidthDouble(r)
            },
            DeviceInfo::MaxClockFrequency => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MaxClockFrequency(r)
            },
            DeviceInfo::AddressBits => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::AddressBits(r)
            },
            DeviceInfo::MaxReadImageArgs => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MaxReadImageArgs(r)
            },
            DeviceInfo::MaxWriteImageArgs => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MaxWriteImageArgs(r)
            },
            DeviceInfo::MaxMemAllocSize => {
                let r = unsafe { util::bytes_into::<u64>(result)? };
                DeviceInfoResult::MaxMemAllocSize(r)
            },
            DeviceInfo::Image2dMaxWidth => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::Image2dMaxWidth(r)
            },
            DeviceInfo::Image2dMaxHeight => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::Image2dMaxHeight(r)
            },
            DeviceInfo::Image3dMaxWidth => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::Image3dMaxWidth(r)
            },
            DeviceInfo::Image3dMaxHeight => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::Image3dMaxHeight(r)
            },
            DeviceInfo::Image3dMaxDepth => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::Image3dMaxDepth(r)
            },
            DeviceInfo::ImageSupport => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::ImageSupport(r != 0)
            },
            DeviceInfo::MaxParameterSize => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::MaxParameterSize(r)
            },
            DeviceInfo::MaxSamplers => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MaxSamplers(r)
            },
            DeviceInfo::MemBaseAddrAlign => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MemBaseAddrAlign(r)
            },
            DeviceInfo::MinDataTypeAlignSize => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MinDataTypeAlignSize(r)
            },
            DeviceInfo::SingleFpConfig => {
                let r = unsafe { util::bytes_into::<DeviceFpConfig>(result)? };
                DeviceInfoResult::SingleFpConfig(r)
            },
            DeviceInfo::GlobalMemCacheType => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                match DeviceMemCacheType::from_u32(r) {
                    Some(e) => DeviceInfoResult::GlobalMemCacheType(e),
                    None => return Err(OclCoreError::from(format!("Error converting '{:X}' to \
                            DeviceMemCacheType.", r))),
                }
            },
            DeviceInfo::GlobalMemCachelineSize => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::GlobalMemCachelineSize(r)
            },
            DeviceInfo::GlobalMemCacheSize => {
                let r = unsafe { util::bytes_into::<u64>(result)? };
                DeviceInfoResult::GlobalMemCacheSize(r)
            },
            DeviceInfo::GlobalMemSize => {
                let r = unsafe { util::bytes_into::<u64>(result)? };
                DeviceInfoResult::GlobalMemSize(r)
            },
            DeviceInfo::MaxConstantBufferSize => {
                let r = unsafe { util::bytes_into::<u64>(result)? };
                DeviceInfoResult::MaxConstantBufferSize(r)
            },
            DeviceInfo::MaxConstantArgs => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::MaxConstantArgs(r)
            },
            DeviceInfo::LocalMemType => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                match DeviceLocalMemType::from_u32(r) {
                    Some(e) => DeviceInfoResult::LocalMemType(e),
                    None => return Err(OclCoreError::from(format!("Error converting '{:X}' to \
                            DeviceLocalMemType.", r))),
                }
            },
            DeviceInfo::LocalMemSize => {
                let r = unsafe { util::bytes_into::<u64>(result)? };
                DeviceInfoResult::LocalMemSize(r)
            },
            DeviceInfo::ErrorCorrectionSupport => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::ErrorCorrectionSupport(r != 0)
            },
            DeviceInfo::ProfilingTimerResolution => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::ProfilingTimerResolution(r)
            },
            DeviceInfo::EndianLittle => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::EndianLittle(r != 0)
            },
            DeviceInfo::Available => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::Available(r != 0)
            },
            DeviceInfo::CompilerAvailable => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::CompilerAvailable(r != 0)
            },
            DeviceInfo::ExecutionCapabilities => {
                let r = unsafe { util::bytes_into::<DeviceExecCapabilities>(result)? };
                DeviceInfoResult::ExecutionCapabilities(r)
            },
            DeviceInfo::QueueProperties => {
                let r = unsafe { util::bytes_into::<CommandQueueProperties>(result)? };
                DeviceInfoResult::QueueProperties(r)
            },
            DeviceInfo::Name => {
                match util::bytes_into_string(result) {
                    Ok(s) => DeviceInfoResult::Name(s),
                    Err(err) => return Err(err.into()),
                }
            },
            DeviceInfo::Vendor => {
                match util::bytes_into_string(result) {
                    Ok(s) => DeviceInfoResult::Vendor(s),
                    Err(err) => return Err(err.into()),
                }
            },
            DeviceInfo::DriverVersion => {
                match util::bytes_into_string(result) {
                    Ok(s) => DeviceInfoResult::DriverVersion(s),
                    Err(err) => return Err(err.into()),
                }
            },
            DeviceInfo::Profile => {
                match util::bytes_into_string(result) {
                    Ok(s) => DeviceInfoResult::Profile(s),
                    Err(err) => return Err(err.into()),
                }
            },
            DeviceInfo::Version => {
                match util::bytes_into_string(result) {
                    Ok(s) => DeviceInfoResult::Version(OpenclVersion::from_info_str(&s)?),
                    Err(err) => return Err(err.into()),
                }
            },
            DeviceInfo::Extensions => {
                match util::bytes_into_string(result) {
                    Ok(s) => DeviceInfoResult::Extensions(s),
                    Err(err) => return Err(err.into()),
                }
            },
            DeviceInfo::Platform => {
                let r = unsafe { util::bytes_into::<PlatformId>(result)? };
                DeviceInfoResult::Platform(r)
            },
            DeviceInfo::DoubleFpConfig => {
                let r = unsafe { util::bytes_into::<DeviceFpConfig>(result)? };
                DeviceInfoResult::DoubleFpConfig(r)
            },
            DeviceInfo::HalfFpConfig => {
                let r = unsafe { util::bytes_into::<DeviceFpConfig>(result)? };
                DeviceInfoResult::HalfFpConfig(r)
            },
            DeviceInfo::PreferredVectorWidthHalf => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PreferredVectorWidthHalf(r)
            },
            DeviceInfo::HostUnifiedMemory => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::HostUnifiedMemory(r != 0)
            },
            DeviceInfo::NativeVectorWidthChar => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::NativeVectorWidthChar(r)
            },
            DeviceInfo::NativeVectorWidthShort => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::NativeVectorWidthShort(r)
            },
            DeviceInfo::NativeVectorWidthInt => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::NativeVectorWidthInt(r)
            },
            DeviceInfo::NativeVectorWidthLong => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::NativeVectorWidthLong(r)
            },
            DeviceInfo::NativeVectorWidthFloat => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::NativeVectorWidthFloat(r)
            },
            DeviceInfo::NativeVectorWidthDouble => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::NativeVectorWidthDouble(r)
            },
            DeviceInfo::NativeVectorWidthHalf => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::NativeVectorWidthHalf(r)
            },
            DeviceInfo::OpenclCVersion => {
                match util::bytes_into_string(result) {
                    Ok(s) => DeviceInfoResult::OpenclCVersion(s),
                    Err(err) => return Err(err.into()),
                }
            },
            DeviceInfo::LinkerAvailable => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::LinkerAvailable(r != 0)
            },
            DeviceInfo::BuiltInKernels => {
                match util::bytes_into_string(result) {
                    Ok(s) => DeviceInfoResult::BuiltInKernels(s),
                    Err(err) => return Err(err.into()),
                }
            },
            DeviceInfo::ImageMaxBufferSize => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::ImageMaxBufferSize(r)
            },
            DeviceInfo::ImageMaxArraySize => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::ImageMaxArraySize(r)
            },
            DeviceInfo::ParentDevice => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                if ptr.is_null() {
                    DeviceInfoResult::ParentDevice(None)
                } else {
                    DeviceInfoResult::ParentDevice(Some(unsafe { DeviceId::from_raw(ptr) }))
                }
            },
            DeviceInfo::PartitionMaxSubDevices => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PartitionMaxSubDevices(r)
            },
            DeviceInfo::PartitionProperties => {
                // [FIXME]: INCOMPLETE:
                //
                // let r = unsafe { util::bytes_into::<u32>(result)? };
                // match DevicePartitionProperty::from_u32(r) {
                //     Some(e) => DeviceInfoResult::PartitionProperties(e),
                //     None => DeviceInfoResult::Error(Box::new(
                //         OclCoreError::from(format!("Error converting '{:X}' to \
                //             DevicePartitionProperty.", r)))),
                // }
                DeviceInfoResult::PartitionProperties(Vec::with_capacity(0))
            },
            DeviceInfo::PartitionAffinityDomain => {
                let r = unsafe { util::bytes_into::<DeviceAffinityDomain>(result)? };
                DeviceInfoResult::PartitionAffinityDomain(r)
            },
            DeviceInfo::PartitionType => {
                // [FIXME]: INCOMPLETE:
                //
                // let r = unsafe { util::bytes_into::<u32>(result)? };
                // match DevicePartitionProperty::from_u32(r) {
                //     Some(e) => DeviceInfoResult::PartitionType(e),
                //     None => DeviceInfoResult::Error(Box::new(
                //         OclCoreError::from(format!("Error converting '{:X}' to \
                //             DevicePartitionProperty.", r)))),
                // }
                DeviceInfoResult::PartitionType(Vec::with_capacity(0))
            },
            DeviceInfo::ReferenceCount => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::ReferenceCount(r)
            },
            DeviceInfo::PreferredInteropUserSync => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::PreferredInteropUserSync(r != 0)
            },
            DeviceInfo::PrintfBufferSize => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                DeviceInfoResult::PrintfBufferSize(r)
            },
            DeviceInfo::ImagePitchAlignment => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::ImagePitchAlignment(r)
            },
            DeviceInfo::ImageBaseAddressAlignment => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                DeviceInfoResult::ImageBaseAddressAlignment(r)
            },
            // _ => DeviceInfoResult::TemporaryPlaceholderVariant(result),
        };

        Ok(ir)
    }

    /// Parse the `Version` string and get a numeric result as `OpenclVersion`.
    pub fn as_opencl_version(&self) -> OclCoreResult<OpenclVersion> {
        if let DeviceInfoResult::Version(ver) = *self {
            Ok(ver)
        } else {
            Err(format!("DeviceInfoResult::as_opencl_version(): Invalid device info \
                result variant: ({:?}). This function can only be called on a \
                'DeviceInfoResult::Version' variant.", self).into())
        }
    }
}

impl fmt::Debug for DeviceInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self)
    }
}

impl fmt::Display for DeviceInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DeviceInfoResult::Type(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::VendorId(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxComputeUnits(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxWorkItemDimensions(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxWorkGroupSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxWorkItemSizes(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::PreferredVectorWidthChar(ref s) => write!(f, "{}", s),
            DeviceInfoResult::PreferredVectorWidthShort(ref s) => write!(f, "{}", s),
            DeviceInfoResult::PreferredVectorWidthInt(ref s) => write!(f, "{}", s),
            DeviceInfoResult::PreferredVectorWidthLong(ref s) => write!(f, "{}", s),
            DeviceInfoResult::PreferredVectorWidthFloat(ref s) => write!(f, "{}", s),
            DeviceInfoResult::PreferredVectorWidthDouble(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxClockFrequency(ref s) => write!(f, "{}", s),
            DeviceInfoResult::AddressBits(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxReadImageArgs(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxWriteImageArgs(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxMemAllocSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Image2dMaxWidth(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Image2dMaxHeight(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Image3dMaxWidth(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Image3dMaxHeight(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Image3dMaxDepth(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ImageSupport(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxParameterSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxSamplers(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MemBaseAddrAlign(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MinDataTypeAlignSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::SingleFpConfig(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::GlobalMemCacheType(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::GlobalMemCachelineSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::GlobalMemCacheSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::GlobalMemSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxConstantBufferSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::MaxConstantArgs(ref s) => write!(f, "{}", s),
            DeviceInfoResult::LocalMemType(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::LocalMemSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ErrorCorrectionSupport(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ProfilingTimerResolution(ref s) => write!(f, "{}", s),
            DeviceInfoResult::EndianLittle(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Available(ref s) => write!(f, "{}", s),
            DeviceInfoResult::CompilerAvailable(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ExecutionCapabilities(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::QueueProperties(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::Name(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Vendor(ref s) => write!(f, "{}", s),
            DeviceInfoResult::DriverVersion(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Profile(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Version(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Extensions(ref s) => write!(f, "{}", s),
            DeviceInfoResult::Platform(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::DoubleFpConfig(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::HalfFpConfig(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::PreferredVectorWidthHalf(ref s) => write!(f, "{}", s),
            DeviceInfoResult::HostUnifiedMemory(ref s) => write!(f, "{}", s),
            DeviceInfoResult::NativeVectorWidthChar(ref s) => write!(f, "{}", s),
            DeviceInfoResult::NativeVectorWidthShort(ref s) => write!(f, "{}", s),
            DeviceInfoResult::NativeVectorWidthInt(ref s) => write!(f, "{}", s),
            DeviceInfoResult::NativeVectorWidthLong(ref s) => write!(f, "{}", s),
            DeviceInfoResult::NativeVectorWidthFloat(ref s) => write!(f, "{}", s),
            DeviceInfoResult::NativeVectorWidthDouble(ref s) => write!(f, "{}", s),
            DeviceInfoResult::NativeVectorWidthHalf(ref s) => write!(f, "{}", s),
            DeviceInfoResult::OpenclCVersion(ref s) => write!(f, "{}", s),
            DeviceInfoResult::LinkerAvailable(ref s) => write!(f, "{}", s),
            DeviceInfoResult::BuiltInKernels(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ImageMaxBufferSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ImageMaxArraySize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ParentDevice(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::PartitionMaxSubDevices(ref s) => write!(f, "{}", s),
            DeviceInfoResult::PartitionProperties(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::PartitionAffinityDomain(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::PartitionType(ref s) => write!(f, "{:?}", s),
            DeviceInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            DeviceInfoResult::PreferredInteropUserSync(ref s) => write!(f, "{}", s),
            DeviceInfoResult::PrintfBufferSize(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ImagePitchAlignment(ref s) => write!(f, "{}", s),
            DeviceInfoResult::ImageBaseAddressAlignment(ref s) => write!(f, "{}", s),
        }
    }
}



/// A context info result.
///
/// [INCOMPLETE][FIXME]: Figure out what to do with the properties variant.
pub enum ContextInfoResult {
    ReferenceCount(u32),
    Devices(Vec<DeviceId>),
    Properties(ContextProperties),
    NumDevices(u32),
}

impl ContextInfoResult {
    pub fn from_bytes(request: ContextInfo, result: Vec<u8>) -> OclCoreResult<ContextInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::Context));
        }
        let r = match request {
            ContextInfo::ReferenceCount => {
                ContextInfoResult::ReferenceCount(util::bytes_to_u32(&result))
            },
            ContextInfo::Devices => { unsafe {
                ContextInfoResult::Devices(util::bytes_into_vec::<DeviceId>(result)?)
            } },
            ContextInfo::Properties => { unsafe {

                let props_raw = util::bytes_into_vec::<cl_context_properties>(result)?;

                let props = ContextProperties::from_raw(props_raw.as_slice())?;
                ContextInfoResult::Properties(props)
            } },
            ContextInfo::NumDevices => ContextInfoResult::NumDevices(util::bytes_to_u32(&result)),
        };
        Ok(r)
    }

    pub fn platform(&self) -> Option<PlatformId> {
        match *self {
            ContextInfoResult::Properties(ref props) => {
                props.get_platform()
            }
            _ => panic!("ContextInfoResult::platform: Not a 'ContextInfoResult::Properties(...)'"),
        }
    }
}

impl fmt::Debug for ContextInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for ContextInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ContextInfoResult::ReferenceCount(ref count) => write!(f, "{}", count),
            ContextInfoResult::Devices(ref vec) => write!(f, "{:?}", vec),
            ContextInfoResult::Properties(ref props) => write!(f, "{:?}", props),
            ContextInfoResult::NumDevices(ref num) => write!(f, "{}", num),
        }
    }
}

impl From<ContextInfoResult> for String {
    fn from(ir: ContextInfoResult) -> String {
        ir.to_string()
    }
}



/// An OpenGL context info result.
pub enum GlContextInfoResult {
    CurrentDevice(DeviceId),
    Devices(Vec<DeviceId>),

}

impl GlContextInfoResult {
    pub fn from_bytes(request: GlContextInfo, result: Vec<u8>)
            -> OclCoreResult<GlContextInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::GlContext));
        }
        let ir = match request {
            GlContextInfo::CurrentDevice => { unsafe {
                GlContextInfoResult::CurrentDevice(util::bytes_into::<DeviceId>(result)?)
            } },
            GlContextInfo::Devices => { unsafe {
                GlContextInfoResult::Devices(util::bytes_into_vec::<DeviceId>(result)?)
            } },
        };
        Ok(ir)
    }

    /// Returns the device contained within.
    pub fn device(self) -> OclCoreResult<DeviceId> {
        match self {
            GlContextInfoResult::CurrentDevice(d) => Ok(d),
            _ => Err("GlContextInfoResult::device: Not a 'GlContextInfoResult::Device(...)'.".into()),
        }
    }
}

impl fmt::Debug for GlContextInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for GlContextInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GlContextInfoResult::CurrentDevice(ref d) => write!(f, "{:?}", d),
            GlContextInfoResult::Devices(ref vec) => write!(f, "{:?}", vec),
        }
    }
}

impl From<GlContextInfoResult> for String {
    fn from(ir: GlContextInfoResult) -> String {
        ir.to_string()
    }
}



/// A command queue info result.
pub enum CommandQueueInfoResult {
    Context(Context),
    Device(DeviceId),
    ReferenceCount(u32),
    Properties(CommandQueueProperties),
}

impl CommandQueueInfoResult {
    pub fn from_bytes(request: CommandQueueInfo, result: Vec<u8>)
            -> OclCoreResult<CommandQueueInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::CommandQueue));
        }

        let ir = match request {
            CommandQueueInfo::Context => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                CommandQueueInfoResult::Context(unsafe { Context::from_raw_copied_ptr(ptr) })
            },
            CommandQueueInfo::Device => {
                let device = unsafe { util::bytes_into::<DeviceId>(result)? };
                CommandQueueInfoResult::Device(device)
            },
            CommandQueueInfo::ReferenceCount => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                CommandQueueInfoResult::ReferenceCount(r)
            }
            CommandQueueInfo::Properties => {
                let r = unsafe { util::bytes_into::<CommandQueueProperties>(result)? };
                CommandQueueInfoResult::Properties(r)
            }
        };
        Ok(ir)
    }
}

impl fmt::Debug for CommandQueueInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for CommandQueueInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CommandQueueInfoResult::Context(ref s) => write!(f, "{:?}", s),
            CommandQueueInfoResult::Device(ref s) => write!(f, "{:?}", s),
            CommandQueueInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            CommandQueueInfoResult::Properties(ref s) => write!(f, "{:?}", s),
            // _ => panic!("CommandQueueInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl From<CommandQueueInfoResult> for String {
    fn from(ir: CommandQueueInfoResult) -> String {
        ir.to_string()
    }
}


/// A mem info result.
///
/// [UNSTABLE][INCOMPLETE]
///
/// [TODO]: Do something with `HostPtr`. It should not be be a raw pointer.
//
// ### From Docs:
//
// If memobj is created with clCreateBuffer or clCreateImage and
// CL_MEM_USE_HOST_PTR is specified in mem_flags, return the host_ptr argument
// value specified when memobj is created. Otherwise a NULL value is returned.
//
// If memobj is created with clCreateSubBuffer, return the host_ptr + origin
// value specified when memobj is created. host_ptr is the argument value
// specified to clCreateBuffer and CL_MEM_USE_HOST_PTR is specified in
// mem_flags for memory object from which memobj is created. Otherwise a NULL
// value is returned.
//
pub enum MemInfoResult {
    Type(MemObjectType),
    Flags(MemFlags),
    Size(usize),
    // Incomplete:
    HostPtr(Option<(*mut c_void, Option<usize>)>),
    MapCount(u32),
    ReferenceCount(u32),
    Context(Context),
    AssociatedMemobject(Option<Mem>),
    Offset(usize),
}


impl MemInfoResult {
    pub fn from_bytes(request: MemInfo, result: Vec<u8>) -> OclCoreResult<MemInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::Mem));
        }

        let ir = match request {
            MemInfo::Type => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                match MemObjectType::from_u32(r) {
                    Some(am) => MemInfoResult::Type(am),
                    None => return Err(OclCoreError::from(format!("Error converting '{}' to \
                            MemObjectType.", r))),
                }
            },
            MemInfo::Flags => {
                let r = unsafe { util::bytes_into::<MemFlags>(result)? };
                MemInfoResult::Flags(r)
            },
            MemInfo::Size => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                MemInfoResult::Size(r)
            },
            MemInfo::HostPtr => {
                // [FIXME]: UNTESTED, INCOMPLETE.
                if result.len() == 8 {
                    let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };

                    if ptr.is_null() {
                        MemInfoResult::HostPtr(None)
                    } else {
                        MemInfoResult::HostPtr(Some((ptr, None)))
                    }
                } else if result.len() == 16 {
                    let ptr_and_origin = unsafe {
                        util::bytes_into::<(*mut c_void, usize)>(result)?
                    };

                    if ptr_and_origin.0.is_null() {
                        MemInfoResult::HostPtr(None)
                    } else {
                        MemInfoResult::HostPtr(Some((ptr_and_origin.0,
                            Some(ptr_and_origin.1))))
                    }
                } else {
                    MemInfoResult::HostPtr(None)
                }
            },
            MemInfo::MapCount => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                MemInfoResult::MapCount(r)
            },
            MemInfo::ReferenceCount => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                MemInfoResult::ReferenceCount(r)
            },
            MemInfo::Context => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                MemInfoResult::Context(unsafe { Context::from_raw_copied_ptr(ptr) })
            },
            MemInfo::AssociatedMemobject => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                if ptr.is_null() {
                    MemInfoResult::AssociatedMemobject(None)
                } else {
                    MemInfoResult::AssociatedMemobject(Some(unsafe { Mem::from_raw_copied_ptr(ptr) }))
                }
            },
            MemInfo::Offset => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                MemInfoResult::Offset(r)
            },

        };
        Ok(ir)
    }
}

impl fmt::Debug for MemInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for MemInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            MemInfoResult::Type(ref s) => write!(f, "{:?}", s),
            MemInfoResult::Flags(ref s) => write!(f, "{:?}", s),
            MemInfoResult::Size(ref s) => write!(f, "{}", s),
            MemInfoResult::HostPtr(ref s) => write!(f, "{:?}", s),
            MemInfoResult::MapCount(ref s) => write!(f, "{}", s),
            MemInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            MemInfoResult::Context(ref s) => write!(f, "{:?}", s),
            MemInfoResult::AssociatedMemobject(ref s) => write!(f, "{:?}", s),
            MemInfoResult::Offset(ref s) => write!(f, "{}", s),
        }
    }
}

impl From<MemInfoResult> for String {
    fn from(ir: MemInfoResult) -> String {
        ir.to_string()
    }
}


/// An image info result.
pub enum ImageInfoResult {
    Format(ImageFormatParseResult),
    ElementSize(usize),
    RowPitch(usize),
    SlicePitch(usize),
    Width(usize),
    Height(usize),
    Depth(usize),
    ArraySize(usize),
    Buffer(Option<Mem>),
    NumMipLevels(u32),
    NumSamples(u32),
}

impl ImageInfoResult {
    pub fn from_bytes(request: ImageInfo, result: Vec<u8>) -> OclCoreResult<ImageInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::Image));
        }
        let ir = match request {
            ImageInfo::Format => {
                let r = unsafe { util::bytes_into::<cl_image_format>(result)? };
                // match ImageFormat::from_raw(r) {
                //     Ok(f) => ImageInfoResult::Format(f),
                //     Err(err) => ImageInfoResult::Error(Box::new(err)),
                // }
                ImageInfoResult::Format(ImageFormat::from_raw(r))
            },
            ImageInfo::ElementSize => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                ImageInfoResult::ElementSize(r)
            },
            ImageInfo::RowPitch => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                ImageInfoResult::RowPitch(r)
            },
            ImageInfo::SlicePitch => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                ImageInfoResult::SlicePitch(r)
            },
            ImageInfo::Width => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                ImageInfoResult::Width(r)
            },
            ImageInfo::Height => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                ImageInfoResult::Height(r)
            },
            ImageInfo::Depth => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                ImageInfoResult::Depth(r)
            },
            ImageInfo::ArraySize => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                ImageInfoResult::ArraySize(r)
            },
            ImageInfo::Buffer => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                if ptr.is_null() {
                    ImageInfoResult::Buffer(None)
                } else {
                    ImageInfoResult::Buffer(Some(unsafe { Mem::from_raw_copied_ptr(ptr) }))
                }
            },
            ImageInfo::NumMipLevels => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                ImageInfoResult::NumMipLevels(r)
            },
            ImageInfo::NumSamples => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                ImageInfoResult::NumSamples(r)
            },
        };
        Ok(ir)
    }
}

impl fmt::Debug for ImageInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for ImageInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ImageInfoResult::Format(ref s) => write!(f, "{:?}", s),
            ImageInfoResult::ElementSize(s) => write!(f, "{}", s),
            ImageInfoResult::RowPitch(s) => write!(f, "{}", s),
            ImageInfoResult::SlicePitch(s) => write!(f, "{}", s),
            ImageInfoResult::Width(s) => write!(f, "{}", s),
            ImageInfoResult::Height(s) => write!(f, "{}", s),
            ImageInfoResult::Depth(s) => write!(f, "{}", s),
            ImageInfoResult::ArraySize(s) => write!(f, "{}", s),
            ImageInfoResult::Buffer(ref s) => write!(f, "{:?}", s),
            ImageInfoResult::NumMipLevels(s) => write!(f, "{}", s),
            ImageInfoResult::NumSamples(s) => write!(f, "{}", s),
        }
    }
}

impl From<ImageInfoResult> for String {
    fn from(ir: ImageInfoResult) -> String {
        ir.to_string()
    }
}


/// A sampler info result.
pub enum SamplerInfoResult {
    ReferenceCount(u32),
    Context(Context),
    NormalizedCoords(bool),
    AddressingMode(AddressingMode),
    FilterMode(FilterMode),
}

impl SamplerInfoResult {
    pub fn from_bytes(request: SamplerInfo, result: Vec<u8>) -> OclCoreResult<SamplerInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::Sampler));
        }
        let ir = match request {
            SamplerInfo::ReferenceCount => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                SamplerInfoResult::ReferenceCount(r)
            },
            SamplerInfo::Context => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                SamplerInfoResult::Context(unsafe { Context::from_raw_copied_ptr(ptr) })
            },
            SamplerInfo::NormalizedCoords => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                SamplerInfoResult::NormalizedCoords(r!= 0u32)
            },
            SamplerInfo::AddressingMode => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                match AddressingMode::from_u32(r) {
                    Some(am) => SamplerInfoResult::AddressingMode(am),
                    None => return Err(OclCoreError::from(format!("Error converting '{}' to \
                        AddressingMode.", r))),
                }
            },
            SamplerInfo::FilterMode => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                match FilterMode::from_u32(r) {
                    Some(fm) => SamplerInfoResult::FilterMode(fm),
                    None => return Err(OclCoreError::from(format!("Error converting '{}' to \
                        FilterMode.", r))),
                }
            },
        };
        Ok(ir)
    }
}

impl fmt::Debug for SamplerInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for SamplerInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SamplerInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            SamplerInfoResult::Context(ref s) => write!(f, "{:?}", s),
            SamplerInfoResult::NormalizedCoords(ref s) => write!(f, "{}", s),
            SamplerInfoResult::AddressingMode(ref s) => write!(f, "{:?}", s),
            SamplerInfoResult::FilterMode(ref s) => write!(f, "{:?}", s),
        }
    }
}

impl From<SamplerInfoResult> for String {
    fn from(ir: SamplerInfoResult) -> String {
        ir.to_string()
    }
}


/// A program info result.
pub enum ProgramInfoResult {
    ReferenceCount(u32),
    Context(Context),
    NumDevices(u32),
    Devices(Vec<DeviceId>),
    Source(String),
    BinarySizes(Vec<usize>),
    Binaries(Vec<Vec<u8>>),
    NumKernels(usize),
    KernelNames(String),
}

impl ProgramInfoResult {
    pub fn from_bytes(request: ProgramInfo, result: Vec<u8>) -> OclCoreResult<ProgramInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::Program));
        }

        let ir = match request {
            ProgramInfo::ReferenceCount => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                ProgramInfoResult::ReferenceCount(r)
            },
            ProgramInfo::Context => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                ProgramInfoResult::Context(unsafe { Context::from_raw_copied_ptr(ptr) })
            },
            ProgramInfo::NumDevices => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                ProgramInfoResult::NumDevices(r)
            },
            ProgramInfo::Devices => {
                ProgramInfoResult::Devices(
                    unsafe { util::bytes_into_vec::<DeviceId>(result)? }
                )
            },
            ProgramInfo::Source => {
                match util::bytes_into_string(result) {
                    Ok(s) => ProgramInfoResult::Source(s),
                    Err(err) => return Err(err.into()),
                }
            },
            ProgramInfo::BinarySizes => { ProgramInfoResult::BinarySizes(
                    unsafe { util::bytes_into_vec::<usize>(result)? }
            ) },
            ProgramInfo::Binaries => {
                // [FIXME]: UNIMPLEMENTED
                ProgramInfoResult::Binaries(Vec::with_capacity(0))
            },
            ProgramInfo::NumKernels => {
                let r = unsafe { util::bytes_into::<usize>(result)? };
                ProgramInfoResult::NumKernels(r)
            },
            ProgramInfo::KernelNames => {
                match util::bytes_into_string(result) {
                    Ok(s) => ProgramInfoResult::KernelNames(s),
                    Err(err) => return Err(err.into()),
                }
            },
        };
        Ok(ir)

    }
}

impl fmt::Debug for ProgramInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for ProgramInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ProgramInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            ProgramInfoResult::Context(ref s) => write!(f, "{:?}", s),
            ProgramInfoResult::NumDevices(ref s) => write!(f, "{}", s),
            ProgramInfoResult::Devices(ref s) => write!(f, "{:?}", s),
            ProgramInfoResult::Source(ref s) => write!(f, "{}", s),
            ProgramInfoResult::BinarySizes(ref s) => write!(f, "{:?}", s),
            ProgramInfoResult::Binaries(_) => write!(f, "{{unprintable}}"),
            ProgramInfoResult::NumKernels(ref s) => write!(f, "{}", s),
            ProgramInfoResult::KernelNames(ref s) => write!(f, "{}", s),
        }
    }
}

impl From<ProgramInfoResult> for String {
    fn from(ir: ProgramInfoResult) -> String {
        ir.to_string()
    }
}



/// A program build info result.
pub enum ProgramBuildInfoResult {
    BuildStatus(ProgramBuildStatus),
    BuildOptions(String),
    BuildLog(String),
    BinaryType(ProgramBinaryType),
}

impl ProgramBuildInfoResult {
    pub fn from_bytes(request: ProgramBuildInfo, result: Vec<u8>) -> OclCoreResult<ProgramBuildInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::ProgramBuild));
        }
        let ir = match request {
            ProgramBuildInfo::BuildStatus => {
                let r = unsafe { util::bytes_into::<i32>(result)? };
                match ProgramBuildStatus::from_i32(r) {
                    Some(b) => ProgramBuildInfoResult::BuildStatus(b),
                    None => return Err(OclCoreError::from(format!("Error converting '{}' to \
                        ProgramBuildStatus.", r))),
                }
            },
            ProgramBuildInfo::BuildOptions => {
                match util::bytes_into_string(result) {
                    Ok(s) => ProgramBuildInfoResult::BuildOptions(s),
                    Err(err) => return Err(err.into()),
                }
            },
            ProgramBuildInfo::BuildLog => {
                match util::bytes_into_string(result) {
                    Ok(s) => ProgramBuildInfoResult::BuildLog(s),
                    Err(err) => return Err(err.into()),
                }
            },
            ProgramBuildInfo::BinaryType => {
                let r = unsafe { util::bytes_into::<ProgramBinaryType>(result)? };
                ProgramBuildInfoResult::BinaryType(r)
            },
        };
        Ok(ir)
    }
}

impl fmt::Debug for ProgramBuildInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for ProgramBuildInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ProgramBuildInfoResult::BuildStatus(ref s) => write!(f, "{:?}", s),
            ProgramBuildInfoResult::BuildOptions(ref s) => write!(f, "{}", s),
            ProgramBuildInfoResult::BuildLog(ref s) => write!(f, "{}", s),
            ProgramBuildInfoResult::BinaryType(ref s) => write!(f, "{:?}", s),
        }
    }
}

impl From<ProgramBuildInfoResult> for String {
    fn from(ir: ProgramBuildInfoResult) -> String {
        ir.to_string()
    }
}


/// A kernel info result.
pub enum KernelInfoResult {
    FunctionName(String),
    NumArgs(u32),
    ReferenceCount(u32),
    Context(Context),
    Program(Program),
    Attributes(String),
}

impl KernelInfoResult {
    pub fn from_bytes(request: KernelInfo, result: Vec<u8>) -> OclCoreResult<KernelInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::Kernel));
        }
        let ir = match request {
            KernelInfo::FunctionName => {
                match util::bytes_into_string(result) {
                    Ok(s) => KernelInfoResult::FunctionName(s),
                    Err(err) => return Err(err.into()),
                }
            },
            KernelInfo::NumArgs => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                KernelInfoResult::NumArgs(r)
            },
            KernelInfo::ReferenceCount => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                KernelInfoResult::ReferenceCount(r)
            },
            KernelInfo::Context => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                KernelInfoResult::Context(unsafe { Context::from_raw_copied_ptr(ptr) })
            },
            KernelInfo::Program => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                KernelInfoResult::Program(unsafe { Program::from_raw_copied_ptr(ptr) })
            },
            KernelInfo::Attributes => {
                match util::bytes_into_string(result) {
                    Ok(s) => KernelInfoResult::Attributes(s),
                    Err(err) => return Err(err.into()),
                }
            },
        };
        Ok(ir)
    }
}

impl fmt::Debug for KernelInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for KernelInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            KernelInfoResult::FunctionName(ref s) => write!(f, "{}", s),
            KernelInfoResult::NumArgs(s) => write!(f, "{}", s),
            KernelInfoResult::ReferenceCount(s) => write!(f, "{}", s),
            KernelInfoResult::Context(ref s) => write!(f, "{:?}", s),
            KernelInfoResult::Program(ref s) => write!(f, "{:?}", s),
            KernelInfoResult::Attributes(ref s) => write!(f, "{}", s),
        }
    }
}

impl From<KernelInfoResult> for String {
    fn from(ir: KernelInfoResult) -> String {
        ir.to_string()
    }
}


/// A kernel arg info result.
pub enum KernelArgInfoResult {
    AddressQualifier(KernelArgAddressQualifier),
    AccessQualifier(KernelArgAccessQualifier),
    TypeName(String),
    TypeQualifier(KernelArgTypeQualifier),
    Name(String),
}

impl KernelArgInfoResult {
    pub fn from_bytes(request: KernelArgInfo, result: Vec<u8>) -> OclCoreResult<KernelArgInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::KernelArg));
        }
        let ir = match request {
            KernelArgInfo::AddressQualifier => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                match KernelArgAddressQualifier::from_u32(r) {
                    Some(kaaq) => KernelArgInfoResult::AddressQualifier(kaaq),
                    None => return Err(OclCoreError::from(format!("Error converting '{}' to \
                            KernelArgAddressQualifier.", r))),
                }
            },
            KernelArgInfo::AccessQualifier => {
                let r = unsafe { util::bytes_into::<u32>(result)? };
                match KernelArgAccessQualifier::from_u32(r) {
                    Some(kaaq) => KernelArgInfoResult::AccessQualifier(kaaq),
                    None => return Err(OclCoreError::from(format!("Error converting '{}' to \
                            KernelArgAccessQualifier.", r))),
                }
            },
            KernelArgInfo::TypeName => {
                match util::bytes_into_string(result) {
                    Ok(s) => KernelArgInfoResult::TypeName(s),
                    Err(err) => return Err(err.into()),
                }
            },
            KernelArgInfo::TypeQualifier => {
                let r = unsafe { util::bytes_into::<KernelArgTypeQualifier>(result)? };
                KernelArgInfoResult::TypeQualifier(r)
            },
            KernelArgInfo::Name => {
                match util::bytes_into_string(result) {
                    Ok(s) => KernelArgInfoResult::Name(s),
                    Err(err) => return Err(err.into()),
                }
            },
        };
        Ok(ir)
    }
}

impl fmt::Debug for KernelArgInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for KernelArgInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            KernelArgInfoResult::AddressQualifier(s) => write!(f, "{:?}", s),
            KernelArgInfoResult::AccessQualifier(s) => write!(f, "{:?}", s),
            KernelArgInfoResult::TypeName(ref s) => write!(f, "{}", s),
            KernelArgInfoResult::TypeQualifier(s) => write!(f, "{:?}", s),
            KernelArgInfoResult::Name(ref s) => write!(f, "{}", s),
        }
    }
}

impl From<KernelArgInfoResult> for String {
    fn from(ir: KernelArgInfoResult) -> String {
        ir.to_string()
    }
}


impl Fail for KernelArgInfoResult {}


/// A kernel work group info result.
pub enum KernelWorkGroupInfoResult {
    WorkGroupSize(usize),
    CompileWorkGroupSize([usize; 3]),
    LocalMemSize(u64),
    PreferredWorkGroupSizeMultiple(usize),
    PrivateMemSize(u64),
    GlobalWorkSize([usize; 3]),
    Empty(EmptyInfoResultError),
    Unavailable(Status),
    CustomBuiltinOnly,
}

impl KernelWorkGroupInfoResult {
    pub fn from_bytes(request: KernelWorkGroupInfo, result: Vec<u8>)
            -> OclCoreResult<KernelWorkGroupInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::KernelWorkGroup));
        }
        let ir = match request {
            KernelWorkGroupInfo::WorkGroupSize => {
                if result.is_empty() {
                    KernelWorkGroupInfoResult::WorkGroupSize(0)
                } else {
                    let r = unsafe { util::bytes_into::<usize>(result)? };
                    KernelWorkGroupInfoResult::WorkGroupSize(r)
                }
            },
            KernelWorkGroupInfo::CompileWorkGroupSize => {
                if result.is_empty() {
                    KernelWorkGroupInfoResult::CompileWorkGroupSize([0, 0, 0])
                } else {
                    let r = unsafe { util::bytes_into::<[usize; 3]>(result)? };
                    KernelWorkGroupInfoResult::CompileWorkGroupSize(r)
                }
            }
            KernelWorkGroupInfo::LocalMemSize => {
                if result.is_empty() {
                    KernelWorkGroupInfoResult::LocalMemSize(0)
                } else {
                    let r = unsafe { util::bytes_into::<u64>(result)? };
                    KernelWorkGroupInfoResult::LocalMemSize(r)
                }
            },
            KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple => {
                if result.is_empty() {
                    KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(0)
                } else {
                    let r = unsafe { util::bytes_into::<usize>(result)? };
                    KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(r)
                }
            },
            KernelWorkGroupInfo::PrivateMemSize => {
                if result.is_empty() {
                    KernelWorkGroupInfoResult::PrivateMemSize(0)
                } else {
                    let r = unsafe { util::bytes_into::<u64>(result)? };
                    KernelWorkGroupInfoResult::PrivateMemSize(r)
                }
            },
            KernelWorkGroupInfo::GlobalWorkSize => {
                if result.is_empty() {
                    KernelWorkGroupInfoResult::GlobalWorkSize([0, 0, 0])
                } else {
                    let r = unsafe { util::bytes_into::<[usize; 3]>(result)? };
                    KernelWorkGroupInfoResult::GlobalWorkSize(r)
                }
            },
        };
        Ok(ir)
    }
}

impl fmt::Debug for KernelWorkGroupInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for KernelWorkGroupInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            KernelWorkGroupInfoResult::WorkGroupSize(s) => write!(f, "{}", s),
            KernelWorkGroupInfoResult::CompileWorkGroupSize(s) => write!(f, "{:?}", s),
            KernelWorkGroupInfoResult::LocalMemSize(s) => write!(f, "{}", s),
            KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(s) => write!(f, "{}", s),
            KernelWorkGroupInfoResult::PrivateMemSize(s) => write!(f, "{}", s),
            KernelWorkGroupInfoResult::GlobalWorkSize(s) => write!(f, "{:?}", s),
            KernelWorkGroupInfoResult::Empty(ref r) => write!(f, "{}", r),
            KernelWorkGroupInfoResult::Unavailable(ref s) => write!(f, "unavailable ({})", s),
            KernelWorkGroupInfoResult::CustomBuiltinOnly => write!(f,
                "only available for custom devices or built-in kernels"),
        }
    }
}

impl From<KernelWorkGroupInfoResult> for String {
    fn from(ir: KernelWorkGroupInfoResult) -> String {
        ir.to_string()
    }
}


/// An event info result.
pub enum EventInfoResult {
    CommandQueue(CommandQueue),
    CommandType(CommandType),
    ReferenceCount(u32),
    CommandExecutionStatus(CommandExecutionStatus),
    Context(Context),
}

impl EventInfoResult {
    pub fn from_bytes(request: EventInfo, result: Vec<u8>) -> OclCoreResult<EventInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(
                EmptyInfoResultError::Event));
        }
        let ir = match request {
            EventInfo::CommandQueue => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                EventInfoResult::CommandQueue(unsafe { CommandQueue::from_raw_copied_ptr(ptr) })
            },
            EventInfo::CommandType => {
                let code = unsafe { util::bytes_into::<u32>(result)? };
                match CommandType::from_u32(code) {
                    Some(ces) => EventInfoResult::CommandType(ces),
                    None => return Err(OclCoreError::from(format!(
                        "Error converting '{}' to CommandType.", code))),
                }
            },
            EventInfo::ReferenceCount => { EventInfoResult::ReferenceCount(
                    unsafe { util::bytes_into::<u32>(result)? }
            ) },
            EventInfo::CommandExecutionStatus => {
                let code = unsafe { util::bytes_into::<i32>(result)? };
                match CommandExecutionStatus::from_i32(code) {
                    Some(ces) => EventInfoResult::CommandExecutionStatus(ces),
                    None => return Err(OclCoreError::from(format!("Error converting '{}' to \
                            CommandExecutionStatus.", code))),
                }
            },
            EventInfo::Context => {
                let ptr = unsafe { util::bytes_into::<*mut c_void>(result)? };
                EventInfoResult::Context(unsafe { Context::from_raw_copied_ptr(ptr) })
            },
        };
        Ok(ir)
    }
}

impl fmt::Debug for EventInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for EventInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            EventInfoResult::CommandQueue(ref s) => write!(f, "{:?}", s),
            EventInfoResult::CommandType(ref s) => write!(f, "{:?}", s),
            EventInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            EventInfoResult::CommandExecutionStatus(ref s) => write!(f, "{:?}", s),
            EventInfoResult::Context(ref s) => write!(f, "{:?}", s),
        }
    }
}

impl From<EventInfoResult> for String {
    fn from(ir: EventInfoResult) -> String {
        ir.to_string()
    }
}


/// A profiling info result.
pub enum ProfilingInfoResult {
    Queued(u64),
    Submit(u64),
    Start(u64),
    End(u64),
}

impl ProfilingInfoResult {
    pub fn from_bytes(request: ProfilingInfo, result: Vec<u8>)
            -> OclCoreResult<ProfilingInfoResult> {
        if result.is_empty() {
            return Err(OclCoreError::from(EmptyInfoResultError::Profiling));
        }
        let ir = match request {
            ProfilingInfo::Queued => ProfilingInfoResult::Queued(
                    unsafe { util::bytes_into::<u64>(result)? }),
            ProfilingInfo::Submit => ProfilingInfoResult::Submit(
                    unsafe { util::bytes_into::<u64>(result)? }),
            ProfilingInfo::Start => ProfilingInfoResult::Start(
                    unsafe { util::bytes_into::<u64>(result)? }),
            ProfilingInfo::End => ProfilingInfoResult::End(
                    unsafe { util::bytes_into::<u64>(result)? }),
        };
        Ok(ir)
    }

    pub fn time(self) -> OclCoreResult<u64> {
        match self {
            ProfilingInfoResult::Queued(time_ns) => Ok(time_ns),
            ProfilingInfoResult::Submit(time_ns) => Ok(time_ns),
            ProfilingInfoResult::Start(time_ns) => Ok(time_ns),
            ProfilingInfoResult::End(time_ns) => Ok(time_ns),
        }
    }
}

impl fmt::Debug for ProfilingInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for ProfilingInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ProfilingInfoResult::Queued(ref s) => write!(f, "{}", s),
            ProfilingInfoResult::Submit(ref s) => write!(f, "{}", s),
            ProfilingInfoResult::Start(ref s) => write!(f, "{}", s),
            ProfilingInfoResult::End(ref s) => write!(f, "{}", s),
        }
    }
}

impl From<ProfilingInfoResult> for String {
    fn from(ir: ProfilingInfoResult) -> String {
        ir.to_string()
    }
}