//! Custom enumerators not specifically based on OpenCL C-style enums.
//!
//! Bleh. Implementing these sucks.
//! 

use std::fmt;
use libc::{size_t, c_void};
use raw::{MemRaw, SamplerRaw, PlatformInfo};
use error::{Result as OclResult};
use cl_h;

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

/// [FIXME]: Implement this beast... someday...
pub enum DeviceInfoResult {
    Type = cl_h::CL_DEVICE_TYPE as isize,
    VendorId = cl_h::CL_DEVICE_VENDOR_ID as isize,
    MaxComputeUnits = cl_h::CL_DEVICE_MAX_COMPUTE_UNITS as isize,
    MaxWorkItemDimensions = cl_h::CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS as isize,
    MaxWorkGroupSize = cl_h::CL_DEVICE_MAX_WORK_GROUP_SIZE as isize,
    MaxWorkItemSizes = cl_h::CL_DEVICE_MAX_WORK_ITEM_SIZES as isize,
    PreferredVectorWidthChar = cl_h::CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR as isize,
    PreferredVectorWidthShort = cl_h::CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT as isize,
    PreferredVectorWidthInt = cl_h::CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT as isize,
    PreferredVectorWidthLong = cl_h::CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG as isize,
    PreferredVectorWidthFloat = cl_h::CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT as isize,
    PreferredVectorWidthDouble = cl_h::CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE as isize,
    MaxClockFrequency = cl_h::CL_DEVICE_MAX_CLOCK_FREQUENCY as isize,
    AddressBits = cl_h::CL_DEVICE_ADDRESS_BITS as isize,
    MaxReadImageArgs = cl_h::CL_DEVICE_MAX_READ_IMAGE_ARGS as isize,
    MaxWriteImageArgs = cl_h::CL_DEVICE_MAX_WRITE_IMAGE_ARGS as isize,
    MaxMemAllocSize = cl_h::CL_DEVICE_MAX_MEM_ALLOC_SIZE as isize,
    Image2dMaxWidth = cl_h::CL_DEVICE_IMAGE2D_MAX_WIDTH as isize,
    Image2dMaxHeight = cl_h::CL_DEVICE_IMAGE2D_MAX_HEIGHT as isize,
    Image3dMaxWidth = cl_h::CL_DEVICE_IMAGE3D_MAX_WIDTH as isize,
    Image3dMaxHeight = cl_h::CL_DEVICE_IMAGE3D_MAX_HEIGHT as isize,
    Image3dMaxDepth = cl_h::CL_DEVICE_IMAGE3D_MAX_DEPTH as isize,
    ImageSupport = cl_h::CL_DEVICE_IMAGE_SUPPORT as isize,
    MaxParameterSize = cl_h::CL_DEVICE_MAX_PARAMETER_SIZE as isize,
    MaxSamplers = cl_h::CL_DEVICE_MAX_SAMPLERS as isize,
    MemBaseAddrAlign = cl_h::CL_DEVICE_MEM_BASE_ADDR_ALIGN as isize,
    MinDataTypeAlignSize = cl_h::CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE as isize,
    SingleFpConfig = cl_h::CL_DEVICE_SINGLE_FP_CONFIG as isize,
    GlobalMemCacheType = cl_h::CL_DEVICE_GLOBAL_MEM_CACHE_TYPE as isize,
    GlobalMemCachelineSize = cl_h::CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE as isize,
    GlobalMemCacheSize = cl_h::CL_DEVICE_GLOBAL_MEM_CACHE_SIZE as isize,
    GlobalMemSize = cl_h::CL_DEVICE_GLOBAL_MEM_SIZE as isize,
    MaxConstantBufferSize = cl_h::CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE as isize,
    MaxConstantArgs = cl_h::CL_DEVICE_MAX_CONSTANT_ARGS as isize,
    LocalMemType = cl_h::CL_DEVICE_LOCAL_MEM_TYPE as isize,
    LocalMemSize = cl_h::CL_DEVICE_LOCAL_MEM_SIZE as isize,
    ErrorCorrectionSupport = cl_h::CL_DEVICE_ERROR_CORRECTION_SUPPORT as isize,
    ProfilingTimerResolution = cl_h::CL_DEVICE_PROFILING_TIMER_RESOLUTION as isize,
    EndianLittle = cl_h::CL_DEVICE_ENDIAN_LITTLE as isize,
    Available = cl_h::CL_DEVICE_AVAILABLE as isize,
    CompilerAvailable = cl_h::CL_DEVICE_COMPILER_AVAILABLE as isize,
    ExecutionCapabilities = cl_h::CL_DEVICE_EXECUTION_CAPABILITIES as isize,
    QueueProperties = cl_h::CL_DEVICE_QUEUE_PROPERTIES as isize,
    Name = cl_h::CL_DEVICE_NAME as isize,
    Vendor = cl_h::CL_DEVICE_VENDOR as isize,
    DriverVersion = cl_h::CL_DRIVER_VERSION as isize,
    Profile = cl_h::CL_DEVICE_PROFILE as isize,
    Version = cl_h::CL_DEVICE_VERSION as isize,
    Extensions = cl_h::CL_DEVICE_EXTENSIONS as isize,
    Platform = cl_h::CL_DEVICE_PLATFORM as isize,
    DoubleFpConfig = cl_h::CL_DEVICE_DOUBLE_FP_CONFIG as isize,
    HalfFpConfig = cl_h::CL_DEVICE_HALF_FP_CONFIG as isize,
    PreferredVectorWidthHalf = cl_h::CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF as isize,
    HostUnifiedMemory = cl_h::CL_DEVICE_HOST_UNIFIED_MEMORY as isize,
    NativeVectorWidthChar = cl_h::CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR as isize,
    NativeVectorWidthShort = cl_h::CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT as isize,
    NativeVectorWidthInt = cl_h::CL_DEVICE_NATIVE_VECTOR_WIDTH_INT as isize,
    NativeVectorWidthLong = cl_h::CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG as isize,
    NativeVectorWidthFloat = cl_h::CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT as isize,
    NativeVectorWidthDouble = cl_h::CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE as isize,
    NativeVectorWidthHalf = cl_h::CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF as isize,
    OpenclCVersion = cl_h::CL_DEVICE_OPENCL_C_VERSION as isize,
    LinkerAvailable = cl_h::CL_DEVICE_LINKER_AVAILABLE as isize,
    BuiltInKernels = cl_h::CL_DEVICE_BUILT_IN_KERNELS as isize,
    ImageMaxBufferSize = cl_h::CL_DEVICE_IMAGE_MAX_BUFFER_SIZE as isize,
    ImageMaxArraySize = cl_h::CL_DEVICE_IMAGE_MAX_ARRAY_SIZE as isize,
    ParentDevice = cl_h::CL_DEVICE_PARENT_DEVICE as isize,
    PartitionMaxSubDevices = cl_h::CL_DEVICE_PARTITION_MAX_SUB_DEVICES as isize,
    PartitionProperties = cl_h::CL_DEVICE_PARTITION_PROPERTIES as isize,
    PartitionAffinityDomain = cl_h::CL_DEVICE_PARTITION_AFFINITY_DOMAIN as isize,
    PartitionType = cl_h::CL_DEVICE_PARTITION_TYPE as isize,
    ReferenceCount = cl_h::CL_DEVICE_REFERENCE_COUNT as isize,
    PreferredInteropUserSync = cl_h::CL_DEVICE_PREFERRED_INTEROP_USER_SYNC as isize,
    PrintfBufferSize = cl_h::CL_DEVICE_PRINTF_BUFFER_SIZE as isize,
    ImagePitchAlignment = cl_h::CL_DEVICE_IMAGE_PITCH_ALIGNMENT as isize,
    ImageBaseAddressAlignment = cl_h::CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT as isize,
}

