//! Custom enumerators not specifically based on OpenCL C-style enums.
//!
//! #### Complete
//!
//! * PlatformInfoResult
//! * ContextInfoResult
//!
//!
//! #### Incomplete
//!
//! The following are using placeholder variants and types meaning everything
//! is just stored and formatted as raw bytes.
//!
//! * DeviceInfoResult
//! * CommandQueueInfoResult
//! * MemInfoResult
//! * ImageInfoResult
//! * SamplerInfoResult
//! * ProgramInfoResult
//! * ProgramBuildInfoResult
//! * KernelInfoResult
//! * KernelArgInfoResult
//! * KernelWorkGroupInfoResult
//! * EventInfoResult
//! * ProfilingInfoResult
//!
//!
//! Bleh. Implementing these sucks. On hold for a while.
//!
//! TODO: ADD ERROR VARIANT FOR EACH OF THE RESULT ENUMS.

#![allow(dead_code)]

use std;
use std::mem;
// use std::error::Error;
use std::convert::Into;
use libc::{size_t, c_void};
use num::FromPrimitive;
use util;
use core::{OclPrm, CommandQueueProperties, PlatformId, PlatformInfo, DeviceId, DeviceInfo,
    ContextInfo, Context, CommandQueue, CommandQueueInfo, CommandType, CommandExecutionStatus,
     Mem, MemInfo, Sampler, SamplerInfo,
    ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo,
    KernelArgAddressQualifier, KernelArgAccessQualifier, KernelArgTypeQualifier, ImageInfo,
    ImageFormat, EventInfo, ProfilingInfo,};
use error::{Result as OclResult, Error as OclError};
// use cl_h;



// Until everything can be implemented:
pub type TemporaryPlaceholderType = ();


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


// cl_context_properties enum  Property value  Description
//
// CL_CONTEXT_PLATFORM cl_platform_id  Specifies the platform to use.
//
// CL_CONTEXT_INTEROP_USER_SYNC    cl_bool Specifies whether the user is
// responsible for synchronization between OpenCL and other APIs. Please refer
// to the specific sections in the OpenCL 1.2 extension specification that
// describe sharing with other APIs for restrictions on using this flag.
//
//    - If CL_CONTEXT_INTEROP_USER_ SYNC is not specified, a default of
//      CL_FALSE is assumed.
//
// CL_CONTEXT_D3D10_DEVICE_KHR ID3D10Device*   If the cl_khr_d3d10_sharing
// extension is enabled, specifies the ID3D10Device* to use for Direct3D 10
// interoperability. The default value is NULL.
//
// CL_GL_CONTEXT_KHR   0, OpenGL context handle    OpenGL context to
// associated the OpenCL context with (available if the cl_khr_gl_sharing
// extension is enabled)
//
// CL_EGL_DISPLAY_KHR  EGL_NO_DISPLAY, EGLDisplay handle   EGLDisplay an
// OpenGL context was created with respect to (available if the
// cl_khr_gl_sharing extension is enabled)
//
// CL_GLX_DISPLAY_KHR  None, X handle  X Display an OpenGL context was created
// with respect to (available if the cl_khr_gl_sharing extension is enabled)
//
// CL_CGL_SHAREGROUP_KHR   0, CGL share group handle   CGL share group to
// associate the OpenCL context with (available if the cl_khr_gl_sharing
// extension is enabled)
//
// CL_WGL_HDC_KHR  0, HDC handle   HDC an OpenGL context was created with
// respect to (available if the cl_khr_gl_sharing extension is enabled)
//
// CL_CONTEXT_ADAPTER_D3D9_KHR IDirect3DDevice9 *  Specifies an
// IDirect3DDevice9 to use for D3D9 interop (if the cl_khr_dx9_media_sharing
// extension is supported).
//
// CL_CONTEXT_ADAPTER_D3D9EX_KHR   IDirect3DDeviceEx*  Specifies an
// IDirect3DDevice9Ex to use for D3D9 interop (if the cl_khr_dx9_media_sharing
// extension is supported).
//
// CL_CONTEXT_ADAPTER_DXVA_KHR IDXVAHD_Device *    Specifies an IDXVAHD_Device
// to use for DXVA interop (if the cl_khr_dx9_media_sharing extension is
// supported).
//
// CL_CONTEXT_D3D11_DEVICE_KHR ID3D11Device *  Specifies the ID3D11Device * to
// use for Direct3D 11 interoperability. The default value is NULL.
//
#[derive(Clone, Debug)]
pub enum ContextProperty {
    Platform(PlatformId),
    InteropUserSync(bool),
    D3d10DeviceKhr(TemporaryPlaceholderType),
    GlContextKhr(TemporaryPlaceholderType),
    EglDisplayKhr(TemporaryPlaceholderType),
    GlxDisplayKhr(TemporaryPlaceholderType),
    CglSharegroupKhr(TemporaryPlaceholderType),
    WglHdcKhr(TemporaryPlaceholderType),
    AdapterD3d9Khr(TemporaryPlaceholderType),
    AdapterD3d9exKhr(TemporaryPlaceholderType),
    AdapterDxvaKhr(TemporaryPlaceholderType),
    D3d11DeviceKhr(TemporaryPlaceholderType),
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
        // match result_string {
        //     Ok(rs) => {
        //         // let string = String::from_utf8(result_string).expect("FIXME: src/core/custom/enums.rs");
        //         let string = try!(String::from_utf8(rs));

        //         Ok(match request {
        //             PlatformInfo::Profile => PlatformInfoResult::Profile(string),
        //             PlatformInfo::Version => PlatformInfoResult::Version(string),
        //             PlatformInfo::Name => PlatformInfoResult::Name(string),
        //             PlatformInfo::Vendor => PlatformInfoResult::Vendor(string),
        //             PlatformInfo::Extensions => PlatformInfoResult::Extensions(string),
        //         })
        //     },
        //     Err(err) => {
        //         PlatformInfoResult::Error(Box::new(err.clone))
        //     }
        // }

        match result {
            Ok(result) => {
                let string = match String::from_utf8(result) {
                    Ok(s) => s,
                    Err(err) => return PlatformInfoResult::Error(Box::new(OclError::from(err))),
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
}

impl std::fmt::Debug for PlatformInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for PlatformInfoResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            &PlatformInfoResult::Profile(ref s) => write!(f, "{}", s),
            &PlatformInfoResult::Version(ref s) => write!(f, "{}", s),
            &PlatformInfoResult::Name(ref s) => write!(f, "{}", s),
            &PlatformInfoResult::Vendor(ref s) => write!(f, "{}", s),
            &PlatformInfoResult::Extensions(ref s) => write!(f, "{}", s),
            &PlatformInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
        }
    }
}

impl Into<String> for PlatformInfoResult {
    fn into(self) -> String {
        match self {
            PlatformInfoResult::Profile(string) => string,
            PlatformInfoResult::Version(string) => string,
            PlatformInfoResult::Name(string) => string,
            PlatformInfoResult::Vendor(string) => string,
            PlatformInfoResult::Extensions(string) => string,
            PlatformInfoResult::Error(err) => err.status_code(),
        }
    }
}


/// A device info result.
///
/// [FIXME]: Implement the rest of this beast... eventually...
#[derive(Debug)]
pub enum DeviceInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
    Type(TemporaryPlaceholderType),
    VendorId(TemporaryPlaceholderType),
    MaxComputeUnits(TemporaryPlaceholderType),
    MaxWorkItemDimensions(TemporaryPlaceholderType),
    MaxWorkGroupSize(usize),
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
    Error(Box<OclError>),
}

impl DeviceInfoResult {
    pub fn from_bytes(request: DeviceInfo, result: OclResult<Vec<u8>>)
            -> DeviceInfoResult
    {
        match result {
            Ok(result) => { match request {
            DeviceInfo::MaxWorkGroupSize => {
                let r0 = unsafe { util::bytes_to::<usize>(&result) };
                let size = unsafe { util::bytes_into::<usize>(result) };
                debug_assert_eq!(r0, size);
                // println!("\n\nDEVICEINFORESULT::FROM_BYTES(MAXWORKGROUPSIZE): r1: {}, r2: {}", r1, r2);
                DeviceInfoResult::MaxWorkGroupSize(size)
            },
            _ => DeviceInfoResult::TemporaryPlaceholderVariant(result),
        } }
            Err(err) => DeviceInfoResult::Error(Box::new(err)),
        }
    }
}

// // DON'T REIMPLEMENT THIS UNTIL ALL THE VARIANTS ARE WORKING
// impl std::fmt::Debug for DeviceInfoResult {
//     /// [INCOMPLETE]: TEMPORARY
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         // DON'T REIMPLEMENT THIS UNTIL ALL THE VARIANTS ARE WORKING
//         write!(f, "{}", &self.to_string())
//     }
// }

impl std::fmt::Display for DeviceInfoResult {
    /// [INCOMPLETE]: TEMPORARY
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            &DeviceInfoResult::TemporaryPlaceholderVariant(ref v) => {
                // TEMPORARY (and retarded):
                write!(f, "{}", to_string_retarded(v))
            },
            &DeviceInfoResult::MaxWorkGroupSize(size) => write!(f, "{}", size),
            &DeviceInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
            r @ _ => panic!("DeviceInfoResult: Converting '{:?}' to string not yet implemented.", r),
        }
    }
}

impl Into<String> for DeviceInfoResult {
    fn into(self) -> String {
        self.to_string()
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
            Ok(result) => { match request {
                ContextInfo::ReferenceCount => {
                    ContextInfoResult::ReferenceCount(util::bytes_to_u32(&result))
                },
                ContextInfo::Devices => {
                    ContextInfoResult::Devices(
                        unsafe { util::bytes_into_vec::<DeviceId>(result) }
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
            } }
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
        match self {
            &ContextInfoResult::ReferenceCount(ref count) => write!(f, "{}", count),
            &ContextInfoResult::Devices(ref vec) => write!(f, "{:?}", vec),
            &ContextInfoResult::Properties(ref props) => write!(f, "{:?}", props),
            &ContextInfoResult::NumDevices(ref num) => write!(f, "{}", num),
            &ContextInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
        }
    }
}

impl Into<String> for ContextInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}



/// [UNSTABLE][INCOMPLETE] A command queue info result.
pub enum CommandQueueInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
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
            Ok(result) => { match request {
                _ => CommandQueueInfoResult::TemporaryPlaceholderVariant(result),
                // CommandQueueInfo::ReferenceCount => {
                //     CommandQueueInfoResult::ReferenceCount(util::bytes_to_u32(&result))
                // },
                // CommandQueueInfo::Devices => {
                //     CommandQueueInfoResult::Devices(
                //         unsafe { util::bytes_into_vec::<DeviceId>(result) }
                //     )
                // },
                // CommandQueueInfo::Properties => {
                //     // unsafe { CommandQueueInfoResult::Properties(
                //     //     CommandQueueInfoOrPropertiesPointerType.from_u32(util::bytes_into::
                //     //         <cl_h::cl_context_properties>(result))
                //     // ) }
                //     CommandQueueInfoResult::Properties(result)
                // },
                // CommandQueueInfo::NumDevices => {
                //     CommandQueueInfoResult::NumDevices(util::bytes_to_u32(&result))
                // },
            } }
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
        match self {
            &CommandQueueInfoResult::TemporaryPlaceholderVariant(ref v) => {
               write!(f, "{}", to_string_retarded(v))
            },
            &CommandQueueInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
            _ => panic!("CommandQueueInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for CommandQueueInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}



/// [UNSTABLE][INCOMPLETE] A mem info result.
pub enum MemInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
    Type(TemporaryPlaceholderType),
    Flags(TemporaryPlaceholderType),
    Size(TemporaryPlaceholderType),
    HostPtr(TemporaryPlaceholderType),
    MapCount(TemporaryPlaceholderType),
    ReferenceCount(TemporaryPlaceholderType),
    Context(TemporaryPlaceholderType),
    AssociatedMemobject(TemporaryPlaceholderType),
    Offset(TemporaryPlaceholderType),
    Error(Box<OclError>),
}


impl MemInfoResult {
    pub fn from_bytes(request: MemInfo, result: OclResult<Vec<u8>>)
            -> MemInfoResult
    {
        match result {
            Ok(result) => { match request {
                _ => MemInfoResult::TemporaryPlaceholderVariant(result),
            } }
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
        match self {
            &MemInfoResult::TemporaryPlaceholderVariant(ref v) => {
               write!(f, "{}", to_string_retarded(v))
            },
            &MemInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
            _ => panic!("MemInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for MemInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}



/// [UNSTABLE][INCOMPLETE] An image info result.
pub enum ImageInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
    Format(ImageFormat),
    ElementSize(usize),
    RowPitch(usize),
    SlicePitch(usize),
    Width(usize),
    Height(usize),
    Depth(usize),
    ArraySize(usize),
    Buffer(Mem),
    NumMipLevels(u32),
    NumSamples(u32),
    Error(Box<OclError>),
}

impl ImageInfoResult {
    // TODO: IMPLEMENT THIS PROPERLY.
    pub fn from_bytes(request: ImageInfo, result: OclResult<Vec<u8>>) -> ImageInfoResult
    {
        match result {
            Ok(result) => { match request {
                ImageInfo::ElementSize => {
                    let size = unsafe { util::bytes_into::<usize>(result) };
                    ImageInfoResult::ElementSize(size)
                },
                _ => ImageInfoResult::TemporaryPlaceholderVariant(result),
            } }
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
        match self {
            &ImageInfoResult::TemporaryPlaceholderVariant(ref v) => {
               write!(f, "{}", to_string_retarded(v))
            },
            &ImageInfoResult::ElementSize(s) => write!(f, "{}", s),
            &ImageInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
            _ => panic!("ImageInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for ImageInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}


/// [UNSTABLE][INCOMPLETE] A sampler info result.
pub enum SamplerInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
    ReferenceCount(TemporaryPlaceholderType),
    Context(TemporaryPlaceholderType),
    NormalizedCoords(TemporaryPlaceholderType),
    AddressingMode(TemporaryPlaceholderType),
    FilterMode(TemporaryPlaceholderType),
    Error(Box<OclError>),
}

impl SamplerInfoResult {
    pub fn from_bytes(request: SamplerInfo, result: OclResult<Vec<u8>>)
            -> SamplerInfoResult
    {
        match result {
            Ok(result) => { match request {
                _ => SamplerInfoResult::TemporaryPlaceholderVariant(result),
            } }
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
        match self {
            &SamplerInfoResult::TemporaryPlaceholderVariant(ref v) => {
               write!(f, "{}", to_string_retarded(v))
            },
            &SamplerInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
            _ => panic!("SamplerInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for SamplerInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}



/// [UNSTABLE][INCOMPLETE] A program info result.
pub enum ProgramInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
    ReferenceCount(TemporaryPlaceholderType),
    Context(TemporaryPlaceholderType),
    NumDevices(TemporaryPlaceholderType),
    Devices(TemporaryPlaceholderType),
    Source(TemporaryPlaceholderType),
    BinarySizes(TemporaryPlaceholderType),
    Binaries(TemporaryPlaceholderType),
    NumKernels(TemporaryPlaceholderType),
    KernelNames(TemporaryPlaceholderType),
    Error(Box<OclError>),
}

impl ProgramInfoResult {
    pub fn from_bytes(request: ProgramInfo, result: OclResult<Vec<u8>>)
            -> ProgramInfoResult
    {
        match result {
            Ok(result) => { match request {
                _ => ProgramInfoResult::TemporaryPlaceholderVariant(result),
            } }
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
        match self {
            &ProgramInfoResult::TemporaryPlaceholderVariant(ref v) => {
               write!(f, "{}", to_string_retarded(v))
            },
            &ProgramInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
            _ => panic!("ProgramInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for ProgramInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}


/// [UNSTABLE][INCOMPLETE] A program build info result.
pub enum ProgramBuildInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
    BuildStatus(TemporaryPlaceholderType),
    BuildOptions(TemporaryPlaceholderType),
    BuildLog(String),
    BinaryType(TemporaryPlaceholderType),
    Error(Box<OclError>),
}

impl ProgramBuildInfoResult {
    pub fn from_bytes(request: ProgramBuildInfo, result: OclResult<Vec<u8>>)
            -> ProgramBuildInfoResult
    {
        match result {
            Ok(result) => { match request {
                ProgramBuildInfo::BuildLog => {
                    let string = match String::from_utf8(result) {
                        Ok(s) => s,
                        Err(err) => return ProgramBuildInfoResult::Error(Box::new(OclError::from(err))),
                    };

                    ProgramBuildInfoResult::BuildLog(string)
                },
                _ => ProgramBuildInfoResult::TemporaryPlaceholderVariant(result),
            } }
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
        match self {
            &ProgramBuildInfoResult::TemporaryPlaceholderVariant(ref v) => {
               write!(f, "{}", to_string_retarded(v))
            },
            &ProgramBuildInfoResult::BuildLog(ref s) => write!(f, "{}", s),
            &ProgramBuildInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
            _ => panic!("ProgramBuildInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for ProgramBuildInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}



/// [UNSTABLE][INCOMPLETE] A kernel info result.
pub enum KernelInfoResult {
    TemporaryPlaceholderVariant(Vec<u8>),
    FunctionName(String),
    NumArgs(TemporaryPlaceholderType),
    ReferenceCount(TemporaryPlaceholderType),
    Context(TemporaryPlaceholderType),
    Program(TemporaryPlaceholderType),
    Attributes(TemporaryPlaceholderType),
    Error(Box<OclError>),
}

impl KernelInfoResult {
    pub fn from_bytes(request: KernelInfo, result: OclResult<Vec<u8>>)
            -> KernelInfoResult
    {
        match result {
            Ok(result) => match request {
                // KernelInfo::MaxWorkGroupSize => {
                //     let r0 = unsafe { util::bytes_to::<usize>(&result) };
                //     let size = unsafe { util::bytes_into::<usize>(result) };
                //     debug_assert_eq!(r0, size);
                //     // println!("\n\nDEVICEINFORESULT::FROM_BYTES(MAXWORKGROUPSIZE): r1: {}, r2: {}", r1, r2);
                //     KernelInfoResult::MaxWorkGroupSize(size)
                // },
                KernelInfo::FunctionName => {
                    let string = match String::from_utf8(result) {
                        Ok(s) => s,
                        Err(err) => return KernelInfoResult::Error(Box::new(OclError::from(err))),
                    };

                    KernelInfoResult::FunctionName(string)
                },
                _ => KernelInfoResult::TemporaryPlaceholderVariant(result),
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
        match self {
            &KernelInfoResult::TemporaryPlaceholderVariant(ref v) => write!(f, "{}", to_string_retarded(v)),
            &KernelInfoResult::FunctionName(ref s) => write!(f, "{}", s),
            &KernelInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
            _ => panic!("KernelInfoResult: Converting this variant to string not yet implemented."),
        }
    }
}

impl Into<String> for KernelInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}



/// [UNSTABLE][INCOMPLETE] A kernel arg info result.
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
            Ok(result) => match request {
                KernelArgInfo::AddressQualifier => {
                    let r = unsafe { util::bytes_into::<u32>(result) };
                    match KernelArgAddressQualifier::from_u32(r) {
                        Some(kaaq) => KernelArgInfoResult::AddressQualifier(kaaq),
                        None => KernelArgInfoResult::Error(Box::new(
                            OclError::new(format!("Error converting '{}' to \
                                KernelArgAddressQualifier.", r)))),
                    }
                },
                KernelArgInfo::AccessQualifier => {
                    let r = unsafe { util::bytes_into::<u32>(result) };
                    match KernelArgAccessQualifier::from_u32(r) {
                        Some(kaaq) => KernelArgInfoResult::AccessQualifier(kaaq),
                        None => KernelArgInfoResult::Error(Box::new(
                            OclError::new(format!("Error converting '{}' to \
                                KernelArgAccessQualifier.", r)))),
                    }
                },
                KernelArgInfo::TypeName => {
                    let string = match String::from_utf8(result) {
                        Ok(s) => s,
                        Err(err) => return KernelArgInfoResult::Error(Box::new(OclError::from(err))),
                    };
                    KernelArgInfoResult::TypeName(string)
                },
                KernelArgInfo::TypeQualifier => {
                    let r = unsafe { util::bytes_into::<KernelArgTypeQualifier>(result) };
                    KernelArgInfoResult::TypeQualifier(r)
                },
                KernelArgInfo::Name => {
                    let string = match String::from_utf8(result) {
                        Ok(s) => s,
                        Err(err) => return KernelArgInfoResult::Error(Box::new(OclError::from(err))),
                    };
                    KernelArgInfoResult::Name(string)
                },
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
        match self {
            &KernelArgInfoResult::AddressQualifier(s) => write!(f, "{:?}", s),
            &KernelArgInfoResult::AccessQualifier(s) => write!(f, "{:?}", s),
            &KernelArgInfoResult::TypeName(ref s) => write!(f, "{}", s),
            &KernelArgInfoResult::TypeQualifier(s) => write!(f, "{:?}", s),
            &KernelArgInfoResult::Name(ref s) => write!(f, "{}", s),
            &KernelArgInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
        }
    }
}

impl Into<String> for KernelArgInfoResult {
    fn into(self) -> String {
        self.to_string()
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
            Ok(result) => match request {
                KernelWorkGroupInfo::WorkGroupSize => {
                    let r = unsafe { util::bytes_into::<usize>(result) };
                    KernelWorkGroupInfoResult::WorkGroupSize(r)
                },
                KernelWorkGroupInfo::CompileWorkGroupSize => {
                    let r = unsafe { util::bytes_into::<[usize; 3]>(result) };
                    KernelWorkGroupInfoResult::CompileWorkGroupSize(r)
                }
                KernelWorkGroupInfo::LocalMemSize => {
                    let r = unsafe { util::bytes_into::<u64>(result) };
                     KernelWorkGroupInfoResult::LocalMemSize(r)
                },
                KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple => {
                    let r = unsafe { util::bytes_into::<usize>(result) };
                    KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(r)
                },
                KernelWorkGroupInfo::PrivateMemSize => {
                    let r = unsafe { util::bytes_into::<u64>(result) };
                    KernelWorkGroupInfoResult::PrivateMemSize(r)
                },
                KernelWorkGroupInfo::GlobalWorkSize => {
                    let r = unsafe { util::bytes_into::<[usize; 3]>(result) };
                    KernelWorkGroupInfoResult::GlobalWorkSize(r)
                },
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
        match self {
            &KernelWorkGroupInfoResult::WorkGroupSize(s) => write!(f, "{}", s),
            &KernelWorkGroupInfoResult::CompileWorkGroupSize(s) => write!(f, "{:?}", s),
            &KernelWorkGroupInfoResult::LocalMemSize(s) => write!(f, "{}", s),
            &KernelWorkGroupInfoResult::PreferredWorkGroupSizeMultiple(s) => write!(f, "{}", s),
            &KernelWorkGroupInfoResult::PrivateMemSize(s) => write!(f, "{}", s),
            &KernelWorkGroupInfoResult::GlobalWorkSize(s) => write!(f, "{:?}", s),
            &KernelWorkGroupInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
        }
    }
}

impl Into<String> for KernelWorkGroupInfoResult {
    fn into(self) -> String {
        self.to_string()
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
            Ok(result) => { match request {
                EventInfo::CommandQueue => {
                    let ptr = unsafe { util::bytes_into::<*mut c_void>(result) };
                    EventInfoResult::CommandQueue(unsafe { CommandQueue::from_copied_ptr(ptr) })
                },
                EventInfo::CommandType => {
                    let code = unsafe { util::bytes_into::<u32>(result) };
                    match CommandType::from_u32(code) {
                        Some(ces) => EventInfoResult::CommandType(ces),
                        None => EventInfoResult::Error(Box::new(
                            OclError::new(format!("Error converting '{}' to CommandType.", code)))),
                    }
                },
                EventInfo::ReferenceCount => { EventInfoResult::ReferenceCount(
                        unsafe { util::bytes_into::<u32>(result) }
                ) },
                EventInfo::CommandExecutionStatus => {
                    let code = unsafe { util::bytes_into::<i32>(result) };
                    match CommandExecutionStatus::from_i32(code) {
                        Some(ces) => EventInfoResult::CommandExecutionStatus(ces),
                        None => EventInfoResult::Error(Box::new(
                            OclError::new(format!("Error converting '{}' to \
                                CommandExecutionStatus.", code)))),
                    }
                },
                EventInfo::Context => {
                    let ptr = unsafe { util::bytes_into::<*mut c_void>(result) };
                    EventInfoResult::Context(unsafe { Context::from_copied_ptr(ptr) })
                },
            } }
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
        match self {
            &EventInfoResult::CommandQueue(ref s) => write!(f, "{:?}", s),
            &EventInfoResult::CommandType(ref s) => write!(f, "{:?}", s),
            &EventInfoResult::ReferenceCount(ref s) => write!(f, "{}", s),
            &EventInfoResult::CommandExecutionStatus(ref s) => write!(f, "{:?}", s),
            &EventInfoResult::Context(ref s) => write!(f, "{:?}", s),
            &EventInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
        }
    }
}

impl Into<String> for EventInfoResult {
    fn into(self) -> String {
        self.to_string()
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
            Ok(result) => { match request {
                ProfilingInfo::Queued => ProfilingInfoResult::Queued(
                        unsafe { util::bytes_into::<u64>(result) }),
                ProfilingInfo::Submit => ProfilingInfoResult::Queued(
                        unsafe { util::bytes_into::<u64>(result) }),
                ProfilingInfo::Start => ProfilingInfoResult::Queued(
                        unsafe { util::bytes_into::<u64>(result) }),
                ProfilingInfo::End => ProfilingInfoResult::Queued(
                        unsafe { util::bytes_into::<u64>(result) }),
            } },
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
        match self {
            &ProfilingInfoResult::Queued(ref s) => write!(f, "{}", s),
            &ProfilingInfoResult::Submit(ref s) => write!(f, "{}", s),
            &ProfilingInfoResult::Start(ref s) => write!(f, "{}", s),
            &ProfilingInfoResult::End(ref s) => write!(f, "{}", s),
            &ProfilingInfoResult::Error(ref err) => write!(f, "{}", err.status_code()),
        }
    }
}

impl Into<String> for ProfilingInfoResult {
    fn into(self) -> String {
        self.to_string()
    }
}



/// TEMPORARY
fn to_string_retarded(v: &Vec<u8>) -> String {
    if v.len() == 4 {
        util::bytes_to_u32(&v[..]).to_string()
    } else if v.len() == 8 && mem::size_of::<usize>() == 8 {
        unsafe { util::bytes_to::<usize>(&v[..]).to_string() }
    } else if v.len() == 3 * 8 {
        unsafe { format!("{:?}", util::bytes_to_vec::<usize>(&v[..])) }
    } else {
        String::from_utf8(v.clone()).unwrap_or(format!("{:?}", v))
    }
}
