//! Custom enumerators not specifically based on OpenCL C-style enums.
//!
//! Bleh. Implementing these sucks.
//! 

use std::fmt;
use libc::{size_t, c_void};
use util;
use raw::{MemRaw, SamplerRaw, PlatformInfo, DeviceIdRaw, ContextInfo, ContextRaw, CommandQueueProperties};
use error::{Result as OclResult};
// use cl_h;

// Until everything can be implemented:
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
            &PlatformInfoResult::Profile(ref s) => s,
            &PlatformInfoResult::Version(ref s) => s,
            &PlatformInfoResult::Name(ref s) => s,
            &PlatformInfoResult::Vendor(ref s) => s,
            &PlatformInfoResult::Extensions(ref s) => s,
        }
    }
}

impl fmt::Debug for PlatformInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl fmt::Display for PlatformInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A device info result.
///
/// [FIXME]: Implement this beast... eventually...
pub enum DeviceInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
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

impl DeviceInfoResult {
    // [FIXME]: THIS IS A JANKY MOFO (a what?).
    // NOTE: Interestingly, this actually works sorta decently as long as there
    // isn't a 4, 8, or 24 character string. Ummm, yeah. Fix this.
    pub fn to_string(&self) -> String {
        match self {
            &DeviceInfoResult::TemporaryPlaceholderVariant(ref v) => {
                // TEMPORARY (and retarded):
                to_string_retarded(v)
            },
            _ => panic!("DeviceInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl fmt::Debug for DeviceInfoResult {
    /// [INCOMPLETE]: TEMPORARY
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for DeviceInfoResult {
    /// [INCOMPLETE]: TEMPORARY
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}



/// [UNSTABLE][INCOMPLETE] A context info result.
///
/// [FIXME]: Figure out what to do with the properties variant.
pub enum ContextInfoResult {
    ReferenceCount(u32),
    Devices(Vec<DeviceIdRaw>),
    // Properties(ContextInfoOrPropertiesPointerType),
    Properties(Vec<u8>),
    NumDevices(u32),
    // TemporaryPlaceholderVariant(Vec<u8>),
}

impl ContextInfoResult {
    pub fn new(request_param: ContextInfo, result: Vec<u8>) -> OclResult<ContextInfoResult> {
        Ok(match request_param {
            ContextInfo::ReferenceCount => {
                ContextInfoResult::ReferenceCount(util::bytes_to_u32(&result))
            },
            ContextInfo::Devices => {
                ContextInfoResult::Devices(
                    unsafe { util::bytes_into_vec::<DeviceIdRaw>(result) }
                )
            },
            ContextInfo::Properties => {
                // unsafe { ContextInfoResult::Properties(
                //     ContextInfoOrPropertiesPointerType.from_u32(util::bytes_into::
                //         <cl_h::cl_context_properties>(result))
                // ) }
                ContextInfoResult::Properties(result)
            },
            ContextInfo::NumDevices => {
                ContextInfoResult::NumDevices(util::bytes_to_u32(&result))
            },
        })
    }

    pub fn to_string(&self) -> String {
        match self {
            &ContextInfoResult::ReferenceCount(ref count) => count.to_string(),
            &ContextInfoResult::Devices(ref vec) => format!("{:?}", vec),
            &ContextInfoResult::Properties(ref props) => format!("{:?}", props),
            &ContextInfoResult::NumDevices(ref num) => num.to_string(),
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
        write!(f, "{}", &self.to_string())
    }
}


/// [UNSTABLE][INCOMPLETE] A command queue info result.
pub enum CommandQueueInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
    Context(ContextRaw),
    Device(DeviceIdRaw),
    ReferenceCount(u32),
    Properties(CommandQueueProperties),
}

impl CommandQueueInfoResult {
    // TODO: IMPLEMENT THIS PROPERLY.
    pub fn to_string(&self) -> String {
        match self {
            &CommandQueueInfoResult::TemporaryPlaceholderVariant(ref v) => {
               to_string_retarded(v)
            },
            _ => panic!("CommandQueueInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl fmt::Debug for CommandQueueInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}

impl fmt::Display for CommandQueueInfoResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.to_string())
    }
}



/// TEMPORARY
fn to_string_retarded(v: &Vec<u8>) -> String {
    if v.len() == 4 {
        util::bytes_to_u32(&v[..]).to_string()
    } else if v.len() == 8 {
        unsafe { util::bytes_to::<usize>(&v[..]).to_string() }
    } else if v.len() == 3 * 8 {
        unsafe { format!("{:?}", util::bytes_to_vec::<usize>(&v[..])) }
    } else {
        String::from_utf8(v.clone()).unwrap_or(format!("{:?}", v))
    }
}
