//! Custom enumerators not specifically based on OpenCL C-style enums.
//!
//!
//!
//
// [TODO]: Evaluate usefulness of `Error` impls and potentially remove.
// [TODO]: Possibly remove custom implementation of `Debug` and derive instead.
//

#![allow(dead_code)]

use std;
// use std::mem;
// use std::error::Error;
// use std::ffi::CString;
use std::convert::Into;
use libc::{size_t, c_void};
use num::FromPrimitive;
use util;
use ffi::{cl_image_format};
use ::{OclPrm, CommandQueueProperties, PlatformId, PlatformInfo, DeviceId, DeviceInfo,
    ContextInfo, Context, CommandQueue, CommandQueueInfo, CommandType, CommandExecutionStatus,
    Mem, MemInfo, MemObjectType, MemFlags, Sampler, SamplerInfo, AddressingMode, FilterMode,
    ProgramInfo, ProgramBuildInfo, Program, ProgramBuildStatus, ProgramBinaryType, KernelInfo,
    KernelArgInfo, KernelWorkGroupInfo,
    KernelArgAddressQualifier, KernelArgAccessQualifier, KernelArgTypeQualifier, ImageInfo,
    ImageFormat, EventInfo, ProfilingInfo,
    DeviceType, DeviceFpConfig, DeviceMemCacheType, DeviceLocalMemType, DeviceExecCapabilities,
    DevicePartitionProperty, DeviceAffinityDomain, OpenclVersion};
use error::{Result as OclResult, Error as OclError};
// use cl_h;


/// `try!` for `***InfoResult` types.
macro_rules! try_ir {
    ( $ expr : expr ) => {
        match $expr {
            Ok(val) => val,
            Err(err) => return err.into(),
        }
    };
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
    Error(Box<OclError>),
}

impl PlatformInfoResult {
    pub fn from_bytes(request: PlatformInfo, result: OclResult<Vec<u8>>)
            -> PlatformInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return PlatformInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }

                let string = match util::bytes_into_string(result) {
                    Ok(s) => s,
                    Err(err) => return PlatformInfoResult::Error(Box::new(err)),
                };

                match request {
                    PlatformInfo::Profile => PlatformInfoResult::Profile(string),
                    PlatformInfo::Version => PlatformInfoResult::Version(string),
                    PlatformInfo::Name => PlatformInfoResult::Name(string),
                    PlatformInfo::Vendor => PlatformInfoResult::Vendor(string),
                    PlatformInfo::Extensions => PlatformInfoResult::Extensions(string),
                }
            }
            Err(err) => PlatformInfoResult::Error(Box::new(err)),
        }
    }

    /// Parse the `Version` string and get a numeric result as `OpenclVersion`.
    pub fn as_opencl_version(&self) -> OclResult<OpenclVersion> {
        if let PlatformInfoResult::Version(ref ver) = *self {
            OpenclVersion::from_info_str(ver)
        } else {
            OclError::err(format!("PlatformInfoResult::as_opencl_version(): Invalid platform info \
                result variant: ({:?}). This function can only be called on a \
                'PlatformInfoResult::Version' variant.", self))
        }
    }
}

impl std::fmt::Debug for PlatformInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for PlatformInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            PlatformInfoResult::Profile(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Version(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Name(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Vendor(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Extensions(ref s) => write!(f, "{}", s),
            PlatformInfoResult::Error(ref err) => write!(f, "{}", err),
        }
    }
}

impl Into<String> for PlatformInfoResult {
    fn into(self) -> String {
        match self {
            PlatformInfoResult::Profile(string)
            | PlatformInfoResult::Version(string)
            | PlatformInfoResult::Name(string)
            | PlatformInfoResult::Vendor(string)
            | PlatformInfoResult::Extensions(string) => string,
            PlatformInfoResult::Error(err) => err.to_string(),
        }
    }
}

impl From<OclError> for PlatformInfoResult {
    fn from(err: OclError) -> PlatformInfoResult {
        PlatformInfoResult::Error(Box::new(err))
    }
}

impl From<std::ffi::IntoStringError> for PlatformInfoResult {
    fn from(err: std::ffi::IntoStringError) -> PlatformInfoResult {
        PlatformInfoResult::Error(Box::new(err.into()))
    }
}

impl From<std::ffi::NulError> for PlatformInfoResult {
    fn from(err: std::ffi::NulError) -> PlatformInfoResult {
        PlatformInfoResult::Error(Box::new(err.into()))
    }
}

impl std::error::Error for PlatformInfoResult {
    fn description(&self) -> &str {
        match *self {
            PlatformInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}


/// [UNSTABLE][INCOMPLETE] A device info result.
///
/// [FIXME]: Implement the rest of this beast... eventually...
// #[derive(Debug)]
pub enum DeviceInfoResult {
    // TemporaryPlaceholderVariant(Vec<u8>),
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
    Version(String),                  // String
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
    Error(Box<OclError>),
}

impl DeviceInfoResult {
    /// Returns a new `DeviceInfoResult::MaxWorkItemSizes` variant.
    pub fn from_bytes_max_work_item_sizes(request: DeviceInfo, result: OclResult<Vec<u8>>,
                max_wi_dims: u32) -> DeviceInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return DeviceInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
            match request {
                DeviceInfo::MaxWorkItemSizes => {
                    match max_wi_dims {
                        3 => {
                            // let r = match unsafe { try_ir!(util::bytes_into::<[usize; 3]>(result)) } {
                            //     Ok(r) => r,
                            //     Err(err) => return err.into(),
                            // };

                            let r = unsafe { try_ir!(util::bytes_into::<[usize; 3]>(result)) };

                            let mut v = Vec::with_capacity(3);
                            v.extend_from_slice(&r);
                            DeviceInfoResult::MaxWorkItemSizes(v)
                        },
                        2 => {
                            let r = unsafe { try_ir!(util::bytes_into::<[usize; 2]>(result)) };
                            let mut v = Vec::with_capacity(2);
                            v.extend_from_slice(&r);
                            DeviceInfoResult::MaxWorkItemSizes(v)
                        },
                        1 => {
                            let r = unsafe { try_ir!(util::bytes_into::<[usize; 1]>(result)) };
                            let mut v = Vec::with_capacity(1);
                            v.extend_from_slice(&r);
                            DeviceInfoResult::MaxWorkItemSizes(v)
                        },
                        _ => DeviceInfoResult::Error(Box::new(OclError::string("Error \
                            determining number of dimensions for MaxWorkItemSizes."))),
                    }
                },
                _ => panic!("DeviceInfoResult::from_bytes_max_work_item_sizes: Called with \
                    invalid info variant ({:?}). Call '::from_bytes` instead.", request),
            } },
            Err(err) => DeviceInfoResult::Error(Box::new(err)),
        }
    }

    /// Returns a new `DeviceInfoResult` for all variants except `MaxWorkItemSizes`.
    pub fn from_bytes(request: DeviceInfo, result: OclResult<Vec<u8>>)
            -> DeviceInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return DeviceInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    DeviceInfo::Type => {
                        let r = unsafe { try_ir!(util::bytes_into::<DeviceType>(result)) };
                        DeviceInfoResult::Type(r)
                    },
                    DeviceInfo::VendorId => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::VendorId(r)
                    },
                    DeviceInfo::MaxComputeUnits => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MaxComputeUnits(r)
                    },
                    DeviceInfo::MaxWorkItemDimensions => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MaxWorkItemDimensions(r)
                    },
                    DeviceInfo::MaxWorkGroupSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::MaxWorkGroupSize(r)
                    },
                    DeviceInfo::MaxWorkItemSizes => {
                        panic!("DeviceInfoResult::from_bytes: Called with invalid info variant ({:?}). \
                            Call '::from_bytes_max_work_item_sizes` instead.", request);
                    },
                    DeviceInfo::PreferredVectorWidthChar => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PreferredVectorWidthChar(r)
                    },
                    DeviceInfo::PreferredVectorWidthShort => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PreferredVectorWidthShort(r)
                    },
                    DeviceInfo::PreferredVectorWidthInt => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PreferredVectorWidthInt(r)
                    },
                    DeviceInfo::PreferredVectorWidthLong => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PreferredVectorWidthLong(r)
                    },
                    DeviceInfo::PreferredVectorWidthFloat => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PreferredVectorWidthFloat(r)
                    },
                    DeviceInfo::PreferredVectorWidthDouble => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PreferredVectorWidthDouble(r)
                    },
                    DeviceInfo::MaxClockFrequency => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MaxClockFrequency(r)
                    },
                    DeviceInfo::AddressBits => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::AddressBits(r)
                    },
                    DeviceInfo::MaxReadImageArgs => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MaxReadImageArgs(r)
                    },
                    DeviceInfo::MaxWriteImageArgs => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MaxWriteImageArgs(r)
                    },
                    DeviceInfo::MaxMemAllocSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<u64>(result)) };
                        DeviceInfoResult::MaxMemAllocSize(r)
                    },
                    DeviceInfo::Image2dMaxWidth => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::Image2dMaxWidth(r)
                    },
                    DeviceInfo::Image2dMaxHeight => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::Image2dMaxHeight(r)
                    },
                    DeviceInfo::Image3dMaxWidth => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::Image3dMaxWidth(r)
                    },
                    DeviceInfo::Image3dMaxHeight => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::Image3dMaxHeight(r)
                    },
                    DeviceInfo::Image3dMaxDepth => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::Image3dMaxDepth(r)
                    },
                    DeviceInfo::ImageSupport => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::ImageSupport(r != 0)
                    },
                    DeviceInfo::MaxParameterSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::MaxParameterSize(r)
                    },
                    DeviceInfo::MaxSamplers => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MaxSamplers(r)
                    },
                    DeviceInfo::MemBaseAddrAlign => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MemBaseAddrAlign(r)
                    },
                    DeviceInfo::MinDataTypeAlignSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MinDataTypeAlignSize(r)
                    },
                    DeviceInfo::SingleFpConfig => {
                        let r = unsafe { try_ir!(util::bytes_into::<DeviceFpConfig>(result)) };
                        DeviceInfoResult::SingleFpConfig(r)
                    },
                    DeviceInfo::GlobalMemCacheType => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        match DeviceMemCacheType::from_u32(r) {
                            Some(e) => DeviceInfoResult::GlobalMemCacheType(e),
                            None => DeviceInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{:X}' to \
                                    DeviceMemCacheType.", r)))),
                        }
                    },
                    DeviceInfo::GlobalMemCachelineSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::GlobalMemCachelineSize(r)
                    },
                    DeviceInfo::GlobalMemCacheSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<u64>(result)) };
                        DeviceInfoResult::GlobalMemCacheSize(r)
                    },
                    DeviceInfo::GlobalMemSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<u64>(result)) };
                        DeviceInfoResult::GlobalMemSize(r)
                    },
                    DeviceInfo::MaxConstantBufferSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<u64>(result)) };
                        DeviceInfoResult::MaxConstantBufferSize(r)
                    },
                    DeviceInfo::MaxConstantArgs => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::MaxConstantArgs(r)
                    },
                    DeviceInfo::LocalMemType => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        match DeviceLocalMemType::from_u32(r) {
                            Some(e) => DeviceInfoResult::LocalMemType(e),
                            None => DeviceInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{:X}' to \
                                    DeviceLocalMemType.", r)))),
                        }
                    },
                    DeviceInfo::LocalMemSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<u64>(result)) };
                        DeviceInfoResult::LocalMemSize(r)
                    },
                    DeviceInfo::ErrorCorrectionSupport => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::ErrorCorrectionSupport(r != 0)
                    },
                    DeviceInfo::ProfilingTimerResolution => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::ProfilingTimerResolution(r)
                    },
                    DeviceInfo::EndianLittle => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::EndianLittle(r != 0)
                    },
                    DeviceInfo::Available => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::Available(r != 0)
                    },
                    DeviceInfo::CompilerAvailable => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::CompilerAvailable(r != 0)
                    },
                    DeviceInfo::ExecutionCapabilities => {
                        let r = unsafe { try_ir!(util::bytes_into::<DeviceExecCapabilities>(result)) };
                        DeviceInfoResult::ExecutionCapabilities(r)
                    },
                    DeviceInfo::QueueProperties => {
                        let r = unsafe { try_ir!(util::bytes_into::<CommandQueueProperties>(result)) };
                        DeviceInfoResult::QueueProperties(r)
                    },
                    DeviceInfo::Name => {
                        match util::bytes_into_string(result) {
                            Ok(s) => DeviceInfoResult::Name(s),
                            Err(err) => DeviceInfoResult::Error(Box::new(err)),
                        }
                    },
                    DeviceInfo::Vendor => {
                        match util::bytes_into_string(result) {
                            Ok(s) => DeviceInfoResult::Vendor(s),
                            Err(err) => DeviceInfoResult::Error(Box::new(err)),
                        }
                    },
                    DeviceInfo::DriverVersion => {
                        match util::bytes_into_string(result) {
                            Ok(s) => DeviceInfoResult::DriverVersion(s),
                            Err(err) => DeviceInfoResult::Error(Box::new(err)),
                        }
                    },
                    DeviceInfo::Profile => {
                        match util::bytes_into_string(result) {
                            Ok(s) => DeviceInfoResult::Profile(s),
                            Err(err) => DeviceInfoResult::Error(Box::new(err)),
                        }
                    },
                    DeviceInfo::Version => {
                        match util::bytes_into_string(result) {
                            Ok(s) => DeviceInfoResult::Version(s),
                            Err(err) => DeviceInfoResult::Error(Box::new(err)),
                        }
                    },
                    DeviceInfo::Extensions => {
                        match util::bytes_into_string(result) {
                            Ok(s) => DeviceInfoResult::Extensions(s),
                            Err(err) => DeviceInfoResult::Error(Box::new(err)),
                        }
                    },
                    DeviceInfo::Platform => {
                        let r = unsafe { try_ir!(util::bytes_into::<PlatformId>(result)) };
                        DeviceInfoResult::Platform(r)
                    },
                    DeviceInfo::DoubleFpConfig => {
                        let r = unsafe { try_ir!(util::bytes_into::<DeviceFpConfig>(result)) };
                        DeviceInfoResult::DoubleFpConfig(r)
                    },
                    DeviceInfo::HalfFpConfig => {
                        let r = unsafe { try_ir!(util::bytes_into::<DeviceFpConfig>(result)) };
                        DeviceInfoResult::HalfFpConfig(r)
                    },
                    DeviceInfo::PreferredVectorWidthHalf => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PreferredVectorWidthHalf(r)
                    },
                    DeviceInfo::HostUnifiedMemory => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::HostUnifiedMemory(r != 0)
                    },
                    DeviceInfo::NativeVectorWidthChar => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::NativeVectorWidthChar(r)
                    },
                    DeviceInfo::NativeVectorWidthShort => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::NativeVectorWidthShort(r)
                    },
                    DeviceInfo::NativeVectorWidthInt => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::NativeVectorWidthInt(r)
                    },
                    DeviceInfo::NativeVectorWidthLong => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::NativeVectorWidthLong(r)
                    },
                    DeviceInfo::NativeVectorWidthFloat => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::NativeVectorWidthFloat(r)
                    },
                    DeviceInfo::NativeVectorWidthDouble => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::NativeVectorWidthDouble(r)
                    },
                    DeviceInfo::NativeVectorWidthHalf => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::NativeVectorWidthHalf(r)
                    },
                    DeviceInfo::OpenclCVersion => {
                        match util::bytes_into_string(result) {
                            Ok(s) => DeviceInfoResult::OpenclCVersion(s),
                            Err(err) => DeviceInfoResult::Error(Box::new(err)),
                        }
                    },
                    DeviceInfo::LinkerAvailable => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::LinkerAvailable(r != 0)
                    },
                    DeviceInfo::BuiltInKernels => {
                        match util::bytes_into_string(result) {
                            Ok(s) => DeviceInfoResult::BuiltInKernels(s),
                            Err(err) => DeviceInfoResult::Error(Box::new(err)),
                        }
                    },
                    DeviceInfo::ImageMaxBufferSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::ImageMaxBufferSize(r)
                    },
                    DeviceInfo::ImageMaxArraySize => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::ImageMaxArraySize(r)
                    },
                    DeviceInfo::ParentDevice => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        if ptr.is_null() {
                            DeviceInfoResult::ParentDevice(None)
                        } else {
                            DeviceInfoResult::ParentDevice(Some(unsafe { DeviceId::from_copied_ptr(ptr) }))
                        }
                    },
                    DeviceInfo::PartitionMaxSubDevices => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PartitionMaxSubDevices(r)
                    },
                    DeviceInfo::PartitionProperties => {
                        // [FIXME]: INCOMPLETE:
                        //
                        // let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        // match DevicePartitionProperty::from_u32(r) {
                        //     Some(e) => DeviceInfoResult::PartitionProperties(e),
                        //     None => DeviceInfoResult::Error(Box::new(
                        //         OclError::string(format!("Error converting '{:X}' to \
                        //             DevicePartitionProperty.", r)))),
                        // }
                        DeviceInfoResult::PartitionProperties(Vec::with_capacity(0))
                    },
                    DeviceInfo::PartitionAffinityDomain => {
                        let r = unsafe { try_ir!(util::bytes_into::<DeviceAffinityDomain>(result)) };
                        DeviceInfoResult::PartitionAffinityDomain(r)
                    },
                    DeviceInfo::PartitionType => {
                        // [FIXME]: INCOMPLETE:
                        //
                        // let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        // match DevicePartitionProperty::from_u32(r) {
                        //     Some(e) => DeviceInfoResult::PartitionType(e),
                        //     None => DeviceInfoResult::Error(Box::new(
                        //         OclError::string(format!("Error converting '{:X}' to \
                        //             DevicePartitionProperty.", r)))),
                        // }
                        DeviceInfoResult::PartitionType(Vec::with_capacity(0))
                    },
                    DeviceInfo::ReferenceCount => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::ReferenceCount(r)
                    },
                    DeviceInfo::PreferredInteropUserSync => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::PreferredInteropUserSync(r != 0)
                    },
                    DeviceInfo::PrintfBufferSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        DeviceInfoResult::PrintfBufferSize(r)
                    },
                    DeviceInfo::ImagePitchAlignment => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::ImagePitchAlignment(r)
                    },
                    DeviceInfo::ImageBaseAddressAlignment => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        DeviceInfoResult::ImageBaseAddressAlignment(r)
                    },
                    // _ => DeviceInfoResult::TemporaryPlaceholderVariant(result),
                }
            },
            Err(err) => err.into(),
        }
    }

    /// Parse the `Version` string and get a numeric result as `OpenclVersion`.
    pub fn as_opencl_version(&self) -> OclResult<OpenclVersion> {
        if let DeviceInfoResult::Version(ref ver) = *self {
            OpenclVersion::from_info_str(ver)
        } else {
            OclError::err(format!("DeviceInfoResult::as_opencl_version(): Invalid device info \
                result variant: ({:?}). This function can only be called on a \
                'DeviceInfoResult::Version' variant.", self))
        }
    }
}

impl std::fmt::Debug for DeviceInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self)
    }
}

impl std::fmt::Display for DeviceInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            // DeviceInfoResult::TemporaryPlaceholderVariant(ref v) => {
            //     // TEMPORARY (and retarded):
            //     write!(f, "{}", to_string_retarded(v))
            // },
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
            DeviceInfoResult::Error(ref err) => write!(f, "{}", err),
            // r @ _ => panic!("DeviceInfoResult: Converting '{:?}' to string not yet implemented.", r),
        }
    }
}

impl Into<String> for DeviceInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for DeviceInfoResult {
    fn from(err: OclError) -> DeviceInfoResult {
        DeviceInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for DeviceInfoResult {
    fn description(&self) -> &str {
        match *self {
            DeviceInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



/// [UNSTABLE][INCOMPLETE] A context info result.
///
/// [FIXME]: Figure out what to do with the properties variant.
pub enum ContextInfoResult {
    ReferenceCount(u32),
    Devices(Vec<DeviceId>),
    Properties(Vec<u8>),
    NumDevices(u32),
    Error(Box<OclError>),
}

impl ContextInfoResult {
    pub fn from_bytes(request: ContextInfo, result: OclResult<Vec<u8>>) -> ContextInfoResult {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return ContextInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    ContextInfo::ReferenceCount => {
                        ContextInfoResult::ReferenceCount(util::bytes_to_u32(&result))
                    },
                    ContextInfo::Devices => {
                        ContextInfoResult::Devices(unsafe { try_ir!(util::bytes_into_vec::<DeviceId>(result)) })
                    },
                    ContextInfo::Properties => ContextInfoResult::Properties(result),
                    ContextInfo::NumDevices => ContextInfoResult::NumDevices(util::bytes_to_u32(&result)),
                }
            },
            Err(err) => ContextInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for ContextInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for ContextInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            ContextInfoResult::ReferenceCount(ref count) => write!(f, "{}", count),
            ContextInfoResult::Devices(ref vec) => write!(f, "{:?}", vec),
            ContextInfoResult::Properties(ref props) => write!(f, "{:?}", props),
            ContextInfoResult::NumDevices(ref num) => write!(f, "{}", num),
            ContextInfoResult::Error(ref err) => write!(f, "{}", err),
        }
    }
}

impl Into<String> for ContextInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for ContextInfoResult {
    fn from(err: OclError) -> ContextInfoResult {
        ContextInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for ContextInfoResult {
    fn description(&self) -> &str {
        match *self {
            ContextInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



/// A command queue info result.
pub enum CommandQueueInfoResult {
    // TemporaryPlaceholderVariant(Vec<u8>),
    Context(Context),
    Device(DeviceId),
    ReferenceCount(u32),
    Properties(CommandQueueProperties),
    Error(Box<OclError>),
}

impl CommandQueueInfoResult {
    pub fn from_bytes(request: CommandQueueInfo, result: OclResult<Vec<u8>>)
            -> CommandQueueInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return CommandQueueInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }

                match request {
                    CommandQueueInfo::Context => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        CommandQueueInfoResult::Context(unsafe { Context::from_copied_ptr(ptr) })
                    },
                    CommandQueueInfo::Device => {
                        let device = unsafe { try_ir!(util::bytes_into::<DeviceId>(result)) };
                        CommandQueueInfoResult::Device(device)
                    },
                    CommandQueueInfo::ReferenceCount => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        CommandQueueInfoResult::ReferenceCount(r)
                    }
                    CommandQueueInfo::Properties => {
                        let r = unsafe { try_ir!(util::bytes_into::<CommandQueueProperties>(result)) };
                        CommandQueueInfoResult::Properties(r)
                    }
                    // _ => CommandQueueInfoResult::TemporaryPlaceholderVariant(result),
                }
            },
            Err(err) => CommandQueueInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for CommandQueueInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for CommandQueueInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            // CommandQueueInfoResult::TemporaryPlaceholderVariant(ref v) => {
            //    write!(f, "{}", to_string_retarded(v))
            // },
            CommandQueueInfoResult::Context(ref s) => write!(f, "{:?}", s),
            CommandQueueInfoResult::Device(ref s) => write!(f, "{:?}", s),
            CommandQueueInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            CommandQueueInfoResult::Properties(ref s) => write!(f, "{:?}", s),
            CommandQueueInfoResult::Error(ref err) => write!(f, "{}", err),
            // _ => panic!("CommandQueueInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for CommandQueueInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for CommandQueueInfoResult {
    fn from(err: OclError) -> CommandQueueInfoResult {
        CommandQueueInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for CommandQueueInfoResult {
    fn description(&self) -> &str {
        match *self {
            CommandQueueInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



/// [UNSTABLE][INCOMPLETE] A mem info result. /
///
// [TODO]: Do something with `HostPtr`. It should not be be a raw pointer.
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
    // TemporaryPlaceholderVariant(Vec<u8>),
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
    Error(Box<OclError>),
}


impl MemInfoResult {
    pub fn from_bytes(request: MemInfo, result: OclResult<Vec<u8>>)
            -> MemInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return MemInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    MemInfo::Type => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        match MemObjectType::from_u32(r) {
                            Some(am) => MemInfoResult::Type(am),
                            None => MemInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{}' to \
                                    MemObjectType.", r)))),
                        }
                    },
                    MemInfo::Flags => {
                        let r = unsafe { try_ir!(util::bytes_into::<MemFlags>(result)) };
                        MemInfoResult::Flags(r)
                    },
                    MemInfo::Size => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        MemInfoResult::Size(r)
                    },
                    MemInfo::HostPtr => {
                        // [FIXME]: UNTESTED, INCOMPLETE.
                        if result.len() == 8 {
                            let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };

                            if ptr.is_null() {
                                MemInfoResult::HostPtr(None)
                            } else {
                                MemInfoResult::HostPtr(Some((ptr, None)))
                            }
                        } else if result.len() == 16 {
                            let ptr_and_origin = unsafe {
                                try_ir!(util::bytes_into::<(*mut c_void, usize)>(result)
                            )};

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
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        MemInfoResult::MapCount(r)
                    },
                    MemInfo::ReferenceCount => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        MemInfoResult::ReferenceCount(r)
                    },
                    MemInfo::Context => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        MemInfoResult::Context(unsafe { Context::from_copied_ptr(ptr) })
                    },
                    MemInfo::AssociatedMemobject => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        if ptr.is_null() {
                            MemInfoResult::AssociatedMemobject(None)
                        } else {
                            MemInfoResult::AssociatedMemobject(Some(unsafe { Mem::from_copied_ptr(ptr) }))
                        }
                    },
                    MemInfo::Offset => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        MemInfoResult::Offset(r)
                    },

                    // _ => MemInfoResult::TemporaryPlaceholderVariant(result),
                }
            },
            Err(err) => MemInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for MemInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for MemInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            // MemInfoResult::TemporaryPlaceholderVariant(ref v) => {
            //    write!(f, "{}", to_string_retarded(v))
            // },
            MemInfoResult::Type(ref s) => write!(f, "{:?}", s),
            MemInfoResult::Flags(ref s) => write!(f, "{:?}", s),
            MemInfoResult::Size(ref s) => write!(f, "{}", s),
            MemInfoResult::HostPtr(ref s) => write!(f, "{:?}", s),
            MemInfoResult::MapCount(ref s) => write!(f, "{}", s),
            MemInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            MemInfoResult::Context(ref s) => write!(f, "{:?}", s),
            MemInfoResult::AssociatedMemobject(ref s) => write!(f, "{:?}", s),
            MemInfoResult::Offset(ref s) => write!(f, "{}", s),
            MemInfoResult::Error(ref err) => write!(f, "{}", err),
            // _ => panic!("MemInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for MemInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for MemInfoResult {
    fn from(err: OclError) -> MemInfoResult {
        MemInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for MemInfoResult {
    fn description(&self) -> &str {
        match *self {
            MemInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



/// An image info result.
pub enum ImageInfoResult {
    // TemporaryPlaceholderVariant(Vec<u8>),
    Format(ImageFormat),
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
    Error(Box<OclError>),
}

impl ImageInfoResult {
    pub fn from_bytes(request: ImageInfo, result: OclResult<Vec<u8>>) -> ImageInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return ImageInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    ImageInfo::Format => {
                        let r = unsafe { try_ir!(util::bytes_into::<cl_image_format>(result)) };
                        match ImageFormat::from_raw(r) {
                            Ok(f) => ImageInfoResult::Format(f),
                            Err(err) => ImageInfoResult::Error(Box::new(err)),
                        }
                    },
                    ImageInfo::ElementSize => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        ImageInfoResult::ElementSize(r)
                    },
                    ImageInfo::RowPitch => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        ImageInfoResult::RowPitch(r)
                    },
                    ImageInfo::SlicePitch => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        ImageInfoResult::SlicePitch(r)
                    },
                    ImageInfo::Width => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        ImageInfoResult::Width(r)
                    },
                    ImageInfo::Height => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        ImageInfoResult::Height(r)
                    },
                    ImageInfo::Depth => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        ImageInfoResult::Depth(r)
                    },
                    ImageInfo::ArraySize => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        ImageInfoResult::ArraySize(r)
                    },
                    ImageInfo::Buffer => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        if ptr.is_null() {
                            ImageInfoResult::Buffer(None)
                        } else {
                            ImageInfoResult::Buffer(Some(unsafe { Mem::from_copied_ptr(ptr) }))
                        }
                    },
                    ImageInfo::NumMipLevels => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        ImageInfoResult::NumMipLevels(r)
                    },
                    ImageInfo::NumSamples => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        ImageInfoResult::NumSamples(r)
                    },
                    // _ => ImageInfoResult::TemporaryPlaceholderVariant(result),
                }
            }
            Err(err) => ImageInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for ImageInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for ImageInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            // ImageInfoResult::TemporaryPlaceholderVariant(ref v) => {
            //    write!(f, "{}", to_string_retarded(v))
            // },
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
            ImageInfoResult::Error(ref err) => write!(f, "{}", err),
            // _ => panic!("ImageInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for ImageInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for ImageInfoResult {
    fn from(err: OclError) -> ImageInfoResult {
        ImageInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for ImageInfoResult {
    fn description(&self) -> &str {
        match *self {
            ImageInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}

/// A sampler info result.
pub enum SamplerInfoResult {
    // TemporaryPlaceholderVariant(Vec<u8>),
    ReferenceCount(u32),
    Context(Context),
    NormalizedCoords(bool),
    AddressingMode(AddressingMode),
    FilterMode(FilterMode),
    Error(Box<OclError>),
}

impl SamplerInfoResult {
    pub fn from_bytes(request: SamplerInfo, result: OclResult<Vec<u8>>)
            -> SamplerInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return SamplerInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    SamplerInfo::ReferenceCount => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        SamplerInfoResult::ReferenceCount(r)
                    },
                    SamplerInfo::Context => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        SamplerInfoResult::Context(unsafe { Context::from_copied_ptr(ptr) })
                    },
                    SamplerInfo::NormalizedCoords => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        SamplerInfoResult::NormalizedCoords(r!= 0u32)
                    },
                    SamplerInfo::AddressingMode => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        match AddressingMode::from_u32(r) {
                            Some(am) => SamplerInfoResult::AddressingMode(am),
                            None => SamplerInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{}' to \
                                    AddressingMode.", r)))),
                        }
                    },
                    SamplerInfo::FilterMode => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        match FilterMode::from_u32(r) {
                            Some(fm) => SamplerInfoResult::FilterMode(fm),
                            None => SamplerInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{}' to \
                                    FilterMode.", r)))),
                        }
                    },
                    // _ => SamplerInfoResult::TemporaryPlaceholderVariant(result),
                }
            }
            Err(err) => SamplerInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for SamplerInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for SamplerInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            // SamplerInfoResult::TemporaryPlaceholderVariant(ref v) => {
            //    write!(f, "{}", to_string_retarded(v))
            // },
            SamplerInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            SamplerInfoResult::Context(ref s) => write!(f, "{:?}", s),
            SamplerInfoResult::NormalizedCoords(ref s) => write!(f, "{}", s),
            SamplerInfoResult::AddressingMode(ref s) => write!(f, "{:?}", s),
            SamplerInfoResult::FilterMode(ref s) => write!(f, "{:?}", s),
            SamplerInfoResult::Error(ref err) => write!(f, "{}", err),
            // _ => panic!("SamplerInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for SamplerInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for SamplerInfoResult {
    fn from(err: OclError) -> SamplerInfoResult {
        SamplerInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for SamplerInfoResult {
    fn description(&self) -> &str {
        match *self {
            SamplerInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



/// A program info result.
pub enum ProgramInfoResult {
    // TemporaryPlaceholderVariant(Vec<u8>),
    ReferenceCount(u32),
    Context(Context),
    NumDevices(u32),
    Devices(Vec<DeviceId>),
    Source(String),
    BinarySizes(Vec<usize>),
    Binaries(Vec<Vec<u8>>),
    NumKernels(usize),
    KernelNames(String),
    Error(Box<OclError>),
}

impl ProgramInfoResult {
    pub fn from_bytes(request: ProgramInfo, result: OclResult<Vec<u8>>)
            -> ProgramInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return ProgramInfoResult::Error(Box::new(OclError::string("[NONE]")));
                }

                match request {
                    ProgramInfo::ReferenceCount => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        ProgramInfoResult::ReferenceCount(r)
                    },
                    ProgramInfo::Context => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        ProgramInfoResult::Context(unsafe { Context::from_copied_ptr(ptr) })
                    },
                    ProgramInfo::NumDevices => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        ProgramInfoResult::NumDevices(r)
                    },
                    ProgramInfo::Devices => {
                        ProgramInfoResult::Devices(
                            unsafe { try_ir!(util::bytes_into_vec::<DeviceId>(result)) }
                        )
                    },
                    ProgramInfo::Source => {
                        match util::bytes_into_string(result) {
                            Ok(s) => ProgramInfoResult::Source(s),
                            Err(err) => ProgramInfoResult::Error(Box::new(err)),
                        }
                    },
                    ProgramInfo::BinarySizes => { ProgramInfoResult::BinarySizes(
                            unsafe { try_ir!(util::bytes_into_vec::<usize>(result)) }
                    ) },
                    ProgramInfo::Binaries => {
                        // [FIXME]: UNIMPLEMENTED
                        ProgramInfoResult::Binaries(Vec::with_capacity(0))
                    },
                    ProgramInfo::NumKernels => {
                        let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                        ProgramInfoResult::NumKernels(r)
                    },
                    ProgramInfo::KernelNames => {
                        match util::bytes_into_string(result) {
                            Ok(s) => ProgramInfoResult::KernelNames(s),
                            Err(err) => ProgramInfoResult::Error(Box::new(err)),
                        }
                    },
                    // _ => ProgramInfoResult::TemporaryPlaceholderVariant(result),
                }
            }
            Err(err) => ProgramInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for ProgramInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for ProgramInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            // ProgramInfoResult::TemporaryPlaceholderVariant(ref v) => {
            //    write!(f, "{}", to_string_retarded(v))
            // },
            ProgramInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            ProgramInfoResult::Context(ref s) => write!(f, "{:?}", s),
            ProgramInfoResult::NumDevices(ref s) => write!(f, "{}", s),
            ProgramInfoResult::Devices(ref s) => write!(f, "{:?}", s),
            ProgramInfoResult::Source(ref s) => write!(f, "{}", s),
            ProgramInfoResult::BinarySizes(ref s) => write!(f, "{:?}", s),
            ProgramInfoResult::Binaries(_) => write!(f, "[Unprintable]"),
            ProgramInfoResult::NumKernels(ref s) => write!(f, "{}", s),
            ProgramInfoResult::KernelNames(ref s) => write!(f, "{}", s),
            ProgramInfoResult::Error(ref err) => write!(f, "{}", err),
            // _ => panic!("ProgramInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for ProgramInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for ProgramInfoResult {
    fn from(err: OclError) -> ProgramInfoResult {
        ProgramInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for ProgramInfoResult {
    fn description(&self) -> &str {
        match *self {
            ProgramInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}


/// A program build info result.
pub enum ProgramBuildInfoResult {
    BuildStatus(ProgramBuildStatus),
    BuildOptions(String),
    BuildLog(String),
    BinaryType(ProgramBinaryType),
    Error(Box<OclError>),
}

impl ProgramBuildInfoResult {
    pub fn from_bytes(request: ProgramBuildInfo, result: OclResult<Vec<u8>>)
            -> ProgramBuildInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return ProgramBuildInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    ProgramBuildInfo::BuildStatus => {
                        let r = unsafe { try_ir!(util::bytes_into::<i32>(result)) };
                        match ProgramBuildStatus::from_i32(r) {
                            Some(b) => ProgramBuildInfoResult::BuildStatus(b),
                            None => ProgramBuildInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{}' to \
                                    ProgramBuildStatus.", r)))),
                        }
                    },
                    ProgramBuildInfo::BuildOptions => {
                        match util::bytes_into_string(result) {
                            Ok(s) => ProgramBuildInfoResult::BuildOptions(s),
                            Err(err) => ProgramBuildInfoResult::Error(Box::new(err)),
                        }
                    },
                    ProgramBuildInfo::BuildLog => {
                        match util::bytes_into_string(result) {
                            Ok(s) => ProgramBuildInfoResult::BuildLog(s),
                            Err(err) => ProgramBuildInfoResult::Error(Box::new(err)),
                        }
                    },
                    ProgramBuildInfo::BinaryType => {
                        let r = unsafe { try_ir!(util::bytes_into::<ProgramBinaryType>(result)) };
                        ProgramBuildInfoResult::BinaryType(r)
                    },
                }
            },
            Err(err) => ProgramBuildInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for ProgramBuildInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for ProgramBuildInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            ProgramBuildInfoResult::BuildStatus(ref s) => write!(f, "{:?}", s),
            ProgramBuildInfoResult::BuildOptions(ref s) => write!(f, "{}", s),
            ProgramBuildInfoResult::BuildLog(ref s) => write!(f, "{}", s),
            ProgramBuildInfoResult::BinaryType(ref s) => write!(f, "{:?}", s),
            ProgramBuildInfoResult::Error(ref err) => write!(f, "{}", err),
        }
    }
}

impl Into<String> for ProgramBuildInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for ProgramBuildInfoResult {
    fn from(err: OclError) -> ProgramBuildInfoResult {
        ProgramBuildInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for ProgramBuildInfoResult {
    fn description(&self) -> &str {
        match *self {
            ProgramBuildInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
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
    Error(Box<OclError>),
}

impl KernelInfoResult {
    pub fn from_bytes(request: KernelInfo, result: OclResult<Vec<u8>>)
            -> KernelInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return KernelInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    KernelInfo::FunctionName => {
                        match util::bytes_into_string(result) {
                            Ok(s) => KernelInfoResult::FunctionName(s),
                            Err(err) => KernelInfoResult::Error(Box::new(err)),
                        }
                    },
                    KernelInfo::NumArgs => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        KernelInfoResult::NumArgs(r)
                    },
                    KernelInfo::ReferenceCount => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        KernelInfoResult::ReferenceCount(r)
                    },
                    KernelInfo::Context => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        KernelInfoResult::Context(unsafe { Context::from_copied_ptr(ptr) })
                    },
                    KernelInfo::Program => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        KernelInfoResult::Program(unsafe { Program::from_copied_ptr(ptr) })
                    },
                    KernelInfo::Attributes => {
                        match util::bytes_into_string(result) {
                            Ok(s) => KernelInfoResult::Attributes(s),
                            Err(err) => KernelInfoResult::Error(Box::new(err)),
                        }
                    },
                }
            },
            Err(err) => KernelInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for KernelInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for KernelInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            KernelInfoResult::FunctionName(ref s) => write!(f, "{}", s),
            KernelInfoResult::NumArgs(s) => write!(f, "{}", s),
            KernelInfoResult::ReferenceCount(s) => write!(f, "{}", s),
            KernelInfoResult::Context(ref s) => write!(f, "{:?}", s),
            KernelInfoResult::Program(ref s) => write!(f, "{:?}", s),
            KernelInfoResult::Attributes(ref s) => write!(f, "{}", s),
            KernelInfoResult::Error(ref err) => write!(f, "{}", err),
        }
    }
}

impl Into<String> for KernelInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for KernelInfoResult {
    fn from(err: OclError) -> KernelInfoResult {
        KernelInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for KernelInfoResult {
    fn description(&self) -> &str {
        match *self {
            KernelInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



/// A kernel arg info result.
pub enum KernelArgInfoResult {
    AddressQualifier(KernelArgAddressQualifier),
    AccessQualifier(KernelArgAccessQualifier),
    TypeName(String),
    TypeQualifier(KernelArgTypeQualifier),
    Name(String),
    Error(Box<OclError>),
}

impl KernelArgInfoResult {
    pub fn from_bytes(request: KernelArgInfo, result: OclResult<Vec<u8>>)
            -> KernelArgInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return KernelArgInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    KernelArgInfo::AddressQualifier => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        match KernelArgAddressQualifier::from_u32(r) {
                            Some(kaaq) => KernelArgInfoResult::AddressQualifier(kaaq),
                            None => KernelArgInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{}' to \
                                    KernelArgAddressQualifier.", r)))),
                        }
                    },
                    KernelArgInfo::AccessQualifier => {
                        let r = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        match KernelArgAccessQualifier::from_u32(r) {
                            Some(kaaq) => KernelArgInfoResult::AccessQualifier(kaaq),
                            None => KernelArgInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{}' to \
                                    KernelArgAccessQualifier.", r)))),
                        }
                    },
                    KernelArgInfo::TypeName => {
                        match util::bytes_into_string(result) {
                            Ok(s) => KernelArgInfoResult::TypeName(s),
                            Err(err) => KernelArgInfoResult::Error(Box::new(err)),
                        }
                    },
                    KernelArgInfo::TypeQualifier => {
                        let r = unsafe { try_ir!(util::bytes_into::<KernelArgTypeQualifier>(result)) };
                        KernelArgInfoResult::TypeQualifier(r)
                    },
                    KernelArgInfo::Name => {
                        match util::bytes_into_string(result) {
                            Ok(s) => KernelArgInfoResult::Name(s),
                            Err(err) => KernelArgInfoResult::Error(Box::new(err)),
                        }
                    },
                }
            },
            Err(err) => KernelArgInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for KernelArgInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for KernelArgInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            KernelArgInfoResult::AddressQualifier(s) => write!(f, "{:?}", s),
            KernelArgInfoResult::AccessQualifier(s) => write!(f, "{:?}", s),
            KernelArgInfoResult::TypeName(ref s) => write!(f, "{}", s),
            KernelArgInfoResult::TypeQualifier(s) => write!(f, "{:?}", s),
            KernelArgInfoResult::Name(ref s) => write!(f, "{}", s),
            KernelArgInfoResult::Error(ref err) => write!(f, "{}", err),
        }
    }
}

impl Into<String> for KernelArgInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for KernelArgInfoResult {
    fn from(err: OclError) -> KernelArgInfoResult {
        KernelArgInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for KernelArgInfoResult {
    fn description(&self) -> &str {
        match *self {
            KernelArgInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}


/// A kernel work group info result.
pub enum KernelWorkGroupInfoResult {
    WorkGroupSize(usize),
    CompileWorkGroupSize([usize; 3]),
    LocalMemSize(u64),
    PreferredWorkGroupSizeMultiple(usize),
    PrivateMemSize(u64),
    GlobalWorkSize([usize; 3]),
    Error(Box<OclError>),
}

impl KernelWorkGroupInfoResult {
    pub fn from_bytes(request: KernelWorkGroupInfo, result: OclResult<Vec<u8>>)
            -> KernelWorkGroupInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return KernelWorkGroupInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    KernelWorkGroupInfo::WorkGroupSize => {
                        if result.is_empty() {
                            KernelWorkGroupInfoResult::WorkGroupSize(0)
                        } else {
                            let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                            KernelWorkGroupInfoResult::WorkGroupSize(r)
                        }
                    },
                    KernelWorkGroupInfo::CompileWorkGroupSize => {
                        if result.is_empty() {
                            KernelWorkGroupInfoResult::CompileWorkGroupSize([0, 0, 0])
                        } else {
                            let r = unsafe { try_ir!(util::bytes_into::<[usize; 3]>(result)) };
                            KernelWorkGroupInfoResult::CompileWorkGroupSize(r)
                        }
                    }
                    KernelWorkGroupInfo::LocalMemSize => {
                        if result.is_empty() {
                            KernelWorkGroupInfoResult::LocalMemSize(0)
                        } else {
                            let r = unsafe { try_ir!(util::bytes_into::<u64>(result)) };
                            KernelWorkGroupInfoResult::LocalMemSize(r)
                        }
                    },
                    KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple => {
                        if result.is_empty() {
                            KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(0)
                        } else {
                            let r = unsafe { try_ir!(util::bytes_into::<usize>(result)) };
                            KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(r)
                        }
                    },
                    KernelWorkGroupInfo::PrivateMemSize => {
                        if result.is_empty() {
                            KernelWorkGroupInfoResult::PrivateMemSize(0)
                        } else {
                            let r = unsafe { try_ir!(util::bytes_into::<u64>(result)) };
                            KernelWorkGroupInfoResult::PrivateMemSize(r)
                        }
                    },
                    KernelWorkGroupInfo::GlobalWorkSize => {
                        if result.is_empty() {
                            KernelWorkGroupInfoResult::GlobalWorkSize([0, 0, 0])
                        } else {
                            let r = unsafe { try_ir!(util::bytes_into::<[usize; 3]>(result)) };
                            KernelWorkGroupInfoResult::GlobalWorkSize(r)
                        }
                    },
                }
            },
            Err(err) => KernelWorkGroupInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for KernelWorkGroupInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for KernelWorkGroupInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            KernelWorkGroupInfoResult::WorkGroupSize(s) => write!(f, "{}", s),
            KernelWorkGroupInfoResult::CompileWorkGroupSize(s) => write!(f, "{:?}", s),
            KernelWorkGroupInfoResult::LocalMemSize(s) => write!(f, "{}", s),
            KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(s) => write!(f, "{}", s),
            KernelWorkGroupInfoResult::PrivateMemSize(s) => write!(f, "{}", s),
            KernelWorkGroupInfoResult::GlobalWorkSize(s) => write!(f, "{:?}", s),
            KernelWorkGroupInfoResult::Error(ref err) => write!(f, "{}", err),
        }
    }
}

impl Into<String> for KernelWorkGroupInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for KernelWorkGroupInfoResult {
    fn from(err: OclError) -> KernelWorkGroupInfoResult {
        KernelWorkGroupInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for KernelWorkGroupInfoResult {
    fn description(&self) -> &str {
        match *self {
            KernelWorkGroupInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



/// An event info result.
pub enum EventInfoResult {
    CommandQueue(CommandQueue),
    CommandType(CommandType),
    ReferenceCount(u32),
    CommandExecutionStatus(CommandExecutionStatus),
    Context(Context),
    Error(Box<OclError>),
}

impl EventInfoResult {
    pub fn from_bytes(request: EventInfo, result: OclResult<Vec<u8>>)
            -> EventInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return EventInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    EventInfo::CommandQueue => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        EventInfoResult::CommandQueue(unsafe { CommandQueue::from_copied_ptr(ptr) })
                    },
                    EventInfo::CommandType => {
                        let code = unsafe { try_ir!(util::bytes_into::<u32>(result)) };
                        match CommandType::from_u32(code) {
                            Some(ces) => EventInfoResult::CommandType(ces),
                            None => EventInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{}' to CommandType.", code)))),
                        }
                    },
                    EventInfo::ReferenceCount => { EventInfoResult::ReferenceCount(
                            unsafe { try_ir!(util::bytes_into::<u32>(result)) }
                    ) },
                    EventInfo::CommandExecutionStatus => {
                        let code = unsafe { try_ir!(util::bytes_into::<i32>(result)) };
                        match CommandExecutionStatus::from_i32(code) {
                            Some(ces) => EventInfoResult::CommandExecutionStatus(ces),
                            None => EventInfoResult::Error(Box::new(
                                OclError::string(format!("Error converting '{}' to \
                                    CommandExecutionStatus.", code)))),
                        }
                    },
                    EventInfo::Context => {
                        let ptr = unsafe { try_ir!(util::bytes_into::<*mut c_void>(result)) };
                        EventInfoResult::Context(unsafe { Context::from_copied_ptr(ptr) })
                    },
                }
            },
            Err(err) => EventInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for EventInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl std::fmt::Display for EventInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            EventInfoResult::CommandQueue(ref s) => write!(f, "{:?}", s),
            EventInfoResult::CommandType(ref s) => write!(f, "{:?}", s),
            EventInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            EventInfoResult::CommandExecutionStatus(ref s) => write!(f, "{:?}", s),
            EventInfoResult::Context(ref s) => write!(f, "{:?}", s),
            EventInfoResult::Error(ref err) => write!(f, "{}", err),
        }
    }
}

impl Into<String> for EventInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for EventInfoResult {
    fn from(err: OclError) -> EventInfoResult {
        EventInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for EventInfoResult {
    fn description(&self) -> &str {
        match *self {
            EventInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



/// A profiling info result.
pub enum ProfilingInfoResult {
    Queued(u64),
    Submit(u64),
    Start(u64),
    End(u64),
    Error(Box<OclError>),
}

impl ProfilingInfoResult {
    pub fn from_bytes(request: ProfilingInfo, result: OclResult<Vec<u8>>)
            -> ProfilingInfoResult
    {
        match result {
            Ok(result) => {
                if result.is_empty() {
                    return ProfilingInfoResult::Error(Box::new(OclError::string(
                        "[NONE]")));
                }
                match request {
                    ProfilingInfo::Queued => ProfilingInfoResult::Queued(
                            unsafe { try_ir!(util::bytes_into::<u64>(result)) }),
                    ProfilingInfo::Submit => ProfilingInfoResult::Submit(
                            unsafe { try_ir!(util::bytes_into::<u64>(result)) }),
                    ProfilingInfo::Start => ProfilingInfoResult::Start(
                            unsafe { try_ir!(util::bytes_into::<u64>(result)) }),
                    ProfilingInfo::End => ProfilingInfoResult::End(
                            unsafe { try_ir!(util::bytes_into::<u64>(result)) }),
                }
            },
            Err(err) => ProfilingInfoResult::Error(Box::new(err)),
        }
    }
}

impl std::fmt::Debug for ProfilingInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for ProfilingInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            ProfilingInfoResult::Queued(ref s) => write!(f, "{}", s),
            ProfilingInfoResult::Submit(ref s) => write!(f, "{}", s),
            ProfilingInfoResult::Start(ref s) => write!(f, "{}", s),
            ProfilingInfoResult::End(ref s) => write!(f, "{}", s),
            ProfilingInfoResult::Error(ref err) => write!(f, "{}", err),
        }
    }
}

impl Into<String> for ProfilingInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}

impl From<OclError> for ProfilingInfoResult {
    fn from(err: OclError) -> ProfilingInfoResult {
        ProfilingInfoResult::Error(Box::new(err))
    }
}

impl std::error::Error for ProfilingInfoResult {
    fn description(&self) -> &str {
        match *self {
            ProfilingInfoResult::Error(ref err) => err.description(),
            _ => "",
        }
    }
}



// /// TEMPORARY
// fn to_string_retarded(v: &Vec<u8>) -> String {
//     if v.len() == 4 {
//         util::bytes_to_u32(&v[..]).to_string()
//     } else if v.len() == 8 && mem::size_of::<usize>() == 8 {
//         unsafe { util::bytes_to::<usize>(&v[..]).to_string() }
//     } else if v.len() == 3 * 8 {
//         unsafe { format!("{:?}", util::bytes_to_vec::<usize>(&v[..])) }
//     } else {
//         String::from_utf8(v.clone()).unwrap_or(format!("{:?}", v))
//     }
// }
