//! Custom enumerators not specifically based on OpenCL C-style enums.
//!
//! Bleh. Implementing these sucks.
//! 

use std::fmt;
use libc::{size_t, c_void};
use raw::{MemRaw, SamplerRaw, PlatformInfo};
use error::{Result as OclResult};
// use cl_h;

/// Here until everything can be implemented.
pub type TemporaryPlaceholderType = ();

/// Kernel argument option type.
///
/// The type argument `T` is ignored for `Mem`, `Sampler`, and `Other` 
/// (just put `usize` or anything).
///
pub enum KernelArg<'a, T: 'a> {
    /// Type `T` is ignored.
    Mem(MemRaw),
    /// Type `T` is ignored.
    Sampler(SamplerRaw),
    Scalar(&'a T),
    Vector(&'a [T]),
    /// Length in multiples of T (not bytes).
    Local(usize),
    /// `size`: size in bytes. Type `T` is ignored.
    Other { size: size_t, value: *const c_void },
}

/// Platform info result.
// #[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlatformInfoResult {
    Profile(String),
    Version(String),
    Name(String),
    Vendor(String),
    Extensions(String),
}

impl PlatformInfoResult {
    pub fn new(request_param: PlatformInfo, result_string: Vec<u8>) -> OclResult<PlatformInfoResult> {
        let string = String::from_utf8(result_string).expect("FIXME: src/raw/custom/enums.rs");

        Ok(match request_param {
            PlatformInfo::Profile => PlatformInfoResult::Profile(string),
            PlatformInfo::Version => PlatformInfoResult::Version(string),
            PlatformInfo::Name => PlatformInfoResult::Name(string),
            PlatformInfo::Vendor => PlatformInfoResult::Vendor(string),
            PlatformInfo::Extensions => PlatformInfoResult::Extensions(string),
        })
    }

    pub fn into_string(self) -> String {
        match self {
            PlatformInfoResult::Profile(string) => string,
            PlatformInfoResult::Version(string) => string,
            PlatformInfoResult::Name(string) => string,
            PlatformInfoResult::Vendor(string) => string,
            PlatformInfoResult::Extensions(string) => string,
        }
    }

    pub fn to_string(&self) -> String {
        self.as_str().to_string()
    }

    pub fn as_str(&self) -> &str {
        match self {
            &PlatformInfoResult::Profile(ref string) => string,
            &PlatformInfoResult::Version(ref string) => string,
            &PlatformInfoResult::Name(ref string) => string,
            &PlatformInfoResult::Vendor(ref string) => string,
            &PlatformInfoResult::Extensions(ref string) => string,
        }
    }
}

impl fmt::Display for PlatformInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A device info result.
///
/// [FIXME]: Implement this beast... someday...
pub enum DeviceInfoResult {
    PlaceholderVariant(Vec<u8>),
    Type(TemporaryPlaceholderType),
    VendorId(TemporaryPlaceholderType),
    MaxComputeUnits(TemporaryPlaceholderType),
    MaxWorkItemDimensions(TemporaryPlaceholderType),
    MaxWorkGroupSize(TemporaryPlaceholderType),
    MaxWorkItemSizes(TemporaryPlaceholderType),
    PreferredVectorWidthChar(TemporaryPlaceholderType),
    PreferredVectorWidthShort(TemporaryPlaceholderType),
    PreferredVectorWidthInt(TemporaryPlaceholderType),
    PreferredVectorWidthLong(TemporaryPlaceholderType),
    PreferredVectorWidthFloat(TemporaryPlaceholderType),
    PreferredVectorWidthDouble(TemporaryPlaceholderType),
    MaxClockFrequency(TemporaryPlaceholderType),
    AddressBits(TemporaryPlaceholderType),
    MaxReadImageArgs(TemporaryPlaceholderType),
    MaxWriteImageArgs(TemporaryPlaceholderType),
    MaxMemAllocSize(TemporaryPlaceholderType),
    Image2dMaxWidth(TemporaryPlaceholderType),
    Image2dMaxHeight(TemporaryPlaceholderType),
    Image3dMaxWidth(TemporaryPlaceholderType),
    Image3dMaxHeight(TemporaryPlaceholderType),
    Image3dMaxDepth(TemporaryPlaceholderType),
    ImageSupport(TemporaryPlaceholderType),
    MaxParameterSize(TemporaryPlaceholderType),
    MaxSamplers(TemporaryPlaceholderType),
    MemBaseAddrAlign(TemporaryPlaceholderType),
    MinDataTypeAlignSize(TemporaryPlaceholderType),
    SingleFpConfig(TemporaryPlaceholderType),
    GlobalMemCacheType(TemporaryPlaceholderType),
    GlobalMemCachelineSize(TemporaryPlaceholderType),
    GlobalMemCacheSize(TemporaryPlaceholderType),
    GlobalMemSize(TemporaryPlaceholderType),
    MaxConstantBufferSize(TemporaryPlaceholderType),
    MaxConstantArgs(TemporaryPlaceholderType),
    LocalMemType(TemporaryPlaceholderType),
    LocalMemSize(TemporaryPlaceholderType),
    ErrorCorrectionSupport(TemporaryPlaceholderType),
    ProfilingTimerResolution(TemporaryPlaceholderType),
    EndianLittle(TemporaryPlaceholderType),
    Available(TemporaryPlaceholderType),
    CompilerAvailable(TemporaryPlaceholderType),
    ExecutionCapabilities(TemporaryPlaceholderType),
    QueueProperties(TemporaryPlaceholderType),
    Name(TemporaryPlaceholderType),
    Vendor(TemporaryPlaceholderType),
    DriverVersion(TemporaryPlaceholderType),
    Profile(TemporaryPlaceholderType),
    Version(TemporaryPlaceholderType),
    Extensions(TemporaryPlaceholderType),
    Platform(TemporaryPlaceholderType),
    DoubleFpConfig(TemporaryPlaceholderType),
    HalfFpConfig(TemporaryPlaceholderType),
    PreferredVectorWidthHalf(TemporaryPlaceholderType),
    HostUnifiedMemory(TemporaryPlaceholderType),
    NativeVectorWidthChar(TemporaryPlaceholderType),
    NativeVectorWidthShort(TemporaryPlaceholderType),
    NativeVectorWidthInt(TemporaryPlaceholderType),
    NativeVectorWidthLong(TemporaryPlaceholderType),
    NativeVectorWidthFloat(TemporaryPlaceholderType),
    NativeVectorWidthDouble(TemporaryPlaceholderType),
    NativeVectorWidthHalf(TemporaryPlaceholderType),
    OpenclCVersion(TemporaryPlaceholderType),
    LinkerAvailable(TemporaryPlaceholderType),
    BuiltInKernels(TemporaryPlaceholderType),
    ImageMaxBufferSize(TemporaryPlaceholderType),
    ImageMaxArraySize(TemporaryPlaceholderType),
    ParentDevice(TemporaryPlaceholderType),
    PartitionMaxSubDevices(TemporaryPlaceholderType),
    PartitionProperties(TemporaryPlaceholderType),
    PartitionAffinityDomain(TemporaryPlaceholderType),
    PartitionType(TemporaryPlaceholderType),
    ReferenceCount(TemporaryPlaceholderType),
    PreferredInteropUserSync(TemporaryPlaceholderType),
    PrintfBufferSize(TemporaryPlaceholderType),
    ImagePitchAlignment(TemporaryPlaceholderType),
    ImageBaseAddressAlignment(TemporaryPlaceholderType),
}

