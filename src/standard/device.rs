//! An OpenCL device identifier.

// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::ops::{Deref, DerefMut};
use std::convert::Into;
use std::error::Error;
// use std::borrow::Borrow;
use error::{Error as OclError, Result as OclResult};
use standard::Platform;
use core::{self, DeviceId as DeviceIdCore, DeviceType, DeviceInfo, DeviceInfoResult, ClDeviceIdPtr};
use util;

const DEBUG_PRINT: bool = true;

/// Specifies [what boils down to] a list of devices.
///
/// The variants: `All`, `Index`, and `Indices` are context-specific, not robust, 
/// and may lead to a stack unwind if the context changes. They are useful for
/// convenience only [NOTE: This may change and they may soon round-robin by default, 
/// making them robust and sexy... well robust anyway][UPDATE: this will probably remain as is].
///
/// The `TypeFlags` variant is useful for specifying a list of devices using a bitfield
/// (`DeviceType`) and is the most robust / portable.
///
///
///
/// [FIXME: Add some links to the SDK]
/// [FIXME: Figure out what we're doing as far as round-robin/moduloing by default]
/// - UPDATE: Leave this to the builder or whatever else to determine and leave this
///   enum an exact index which panics.
///
#[derive(Debug, Clone)]
pub enum DeviceSpecifier {
    All,
    First,
    Single(Device),
    List(Vec<Device>),
    Indices(Vec<usize>),
    WrappingIndices(Vec<usize>),
    TypeFlags(DeviceType),
}

impl DeviceSpecifier {
    pub fn all(self) -> DeviceSpecifier {
        DeviceSpecifier::All
    }

    pub fn first(self) -> DeviceSpecifier {
        DeviceSpecifier::First
    }

    pub fn single(self, device: Device) -> DeviceSpecifier {
        DeviceSpecifier::Single(device)
    }

    pub fn list(self, list: Vec<Device>) -> DeviceSpecifier {
        DeviceSpecifier::List(list)
    }

    pub fn indices(self, indices: Vec<usize>) -> DeviceSpecifier {
        DeviceSpecifier::Indices(indices)
    }

    pub fn wrapping_indices(self, windices: Vec<usize>) -> DeviceSpecifier {
        DeviceSpecifier::WrappingIndices(windices)
    }

    pub fn type_flags(self, flags: DeviceType) -> DeviceSpecifier {
        DeviceSpecifier::TypeFlags(flags)
    }

    /// Returns the list of devices matching the parameters specified by this `DeviceSpecifier`
    ///
    /// ## Panics
    ///
    /// Any device indices within the `Index` and `Indices` variants must be within the range of the number of devices for the platform specified by `Platform`. If no `platform` has been specified, this behaviour is undefined and could end up using any platform at all.
    ///
    /// TODO: Swap some of the `try!`s for `.map`s.
    pub fn to_device_list(&self, platform: Option<Platform>) -> OclResult<Vec<Device>> {
        let platform = match platform {
            Some(p) => p.clone(),
            None => Platform::default(),
        };

        Ok(match self {
            &DeviceSpecifier::All => {
                Device::list_all(&platform)
            },
            &DeviceSpecifier::First => {
                try!(Device::list_select(&platform, None, &vec![0]))
            },
            &DeviceSpecifier::Single(ref device) => {
                vec![device.clone()]
            },
            &DeviceSpecifier::List(ref devices) => {
                // devices.iter().map(|d| d.clone()).collect() 
                devices.clone()
            },
            // &DeviceSpecifier::Index(idx) => {
            //     assert!(idx < device_list_all.len(), "ocl::Context::new: DeviceSpecifier::Index: \
            //         Device index out of range.");
            //     vec![device_list_all[idx].clone()]
            // },
            &DeviceSpecifier::Indices(ref idx_list) => {
                // idx_list.iter().map(|&idx| {
                //         assert!(idx < device_list_all.len(), "ocl::Context::new: \
                //             DeviceSpecifier::Indices: Device index out of range.");
                //         device_list_all[idx].clone()
                //     } ).collect()

                try!(Device::list_select(&platform, None, idx_list))
            },
            &DeviceSpecifier::WrappingIndices(ref idx_list) => {
                // idx_list.iter().map(|&idx| {
                //         assert!(idx < device_list_all.len(), "ocl::Context::new: \
                //             DeviceSpecifier::Indices: Device index out of range.");
                //         device_list_all[idx].clone()
                //     } ).collect()
                Device::list_select_wrap(&platform, None, idx_list)
            },
            &DeviceSpecifier::TypeFlags(flags) => {
                // Device::list_from_core(try!(
                //     core::get_device_ids(platform_id_core.clone(), Some(flags))
                // ))
                Device::list(&platform, Some(flags))
            },
        })
    }
}

impl Default for DeviceSpecifier {
    fn default() -> DeviceSpecifier {
        DeviceSpecifier::All
    }
}

impl From<usize> for DeviceSpecifier {
    fn from(index: usize) -> DeviceSpecifier {
        DeviceSpecifier::WrappingIndices(vec![index])
    }
}

impl<'a> From<&'a [usize]> for DeviceSpecifier {
    fn from(indices: &'a [usize]) -> DeviceSpecifier {
        DeviceSpecifier::WrappingIndices(indices.into())
    }
}

impl<'a> From<&'a Vec<usize>> for DeviceSpecifier {
    fn from(indices: &'a Vec<usize>) -> DeviceSpecifier {
        DeviceSpecifier::WrappingIndices(indices.clone())
    }
}

// impl<'a, T> Borrow<T> for &'a T where T: ?Sized

// impl<'a, B> From<B> for DeviceSpecifier where B: Borrow<usize> {
//     fn from(indices: B) -> DeviceSpecifier {
//         DeviceSpecifier::WrappingIndices(indices.clone())
//     }
// }

impl<'a> From<&'a [Device]> for DeviceSpecifier {
    fn from(devices: &'a [Device]) -> DeviceSpecifier {
        DeviceSpecifier::List(devices.into())
    }
}

// impl<'a> From<&'a [Device; 1]> for DeviceSpecifier {
//     fn from(devices: &'a [Device; 1]) -> DeviceSpecifier {
//         DeviceSpecifier::List(devices.into())
//     }
// }

impl<'a> From<&'a Vec<Device>> for DeviceSpecifier {
    fn from(devices: &'a Vec<Device>) -> DeviceSpecifier {
        DeviceSpecifier::List(devices.clone())
    }
}

impl From<Device> for DeviceSpecifier {
    fn from(device: Device) -> DeviceSpecifier {
        DeviceSpecifier::Single(device)
    }
}

impl<'a> From<&'a Device> for DeviceSpecifier {
    fn from(device: &'a Device) -> DeviceSpecifier {
        DeviceSpecifier::Single(device.clone())
    }
}

impl From<DeviceType> for DeviceSpecifier {
    fn from(flags: DeviceType) -> DeviceSpecifier {
        DeviceSpecifier::TypeFlags(flags)
    }
}


/// A device identifier.
#[derive(Clone, Copy, Debug)]
pub struct Device(DeviceIdCore);

impl Device {
    /// Returns the first available device on a platform
    pub fn first(platform: Platform) -> Device {
        let first_core = core::get_device_ids(&platform, None, None)
            .expect("ocl::Device::first: Error retrieving device list");
        Device(first_core[0])
    }

    /// Returns a `DeviceSpecifier` useful for precisely specifying a set
    /// of devices.
    pub fn specifier() -> DeviceSpecifier {
        DeviceSpecifier::default()
    }
    
    /// Resolves a list of indexes into a list of valid devices.
    ///
    /// `devices` is the set of all indexable devices.
    ///
    /// # Errors
    ///
    /// All indices in `idxs` must be valid.
    ///
    pub fn resolve_idxs(idxs: &[usize], devices: &[Device]) -> OclResult<Vec<Device>> {
        // idxs.iter().map(|&idx| devices.get(idx)).collect()
        let mut result = Vec::with_capacity(idxs.len());
        for &idx in idxs.iter() {
            match devices.get(idx) {
                Some(&device) => result.push(device),
                None => return OclError::err(format!("Error resolving device index: '{}'. Index out of \
                    range. Devices avaliable: '{}'.", idx, devices.len())),
            }
        }
        Ok(result)
    }

    /// Resolves a list of indexes into a list of valid devices.
    ///
    /// `devices` is the set of all indexable devices.
    ///
    /// Wraps indexes around using modulo (`%`) so that every index is valid.
    ///
    pub fn resolve_idxs_wrap(idxs: &[usize], devices: &[Device]) -> Vec<Device> {
        let valid_idxs = util::wrap_vals(idxs, devices.len());
        valid_idxs.iter().map(|&idx| devices[idx]).collect()
    }

    /// Returns a list of all devices avaliable for a given platform which
    /// optionally match the flags set in the bitfield, `device_types`.
    ///
    /// Setting `device_types` to `None` will return a list of all avaliable
    /// devices for `platform`
    pub fn list(platform: &Platform, device_types: Option<DeviceType>) -> Vec<Device> {
        let list_core = core::get_device_ids(platform.as_core(), device_types, None)
            .expect("Device::list: Error retrieving device list");
        let list = list_core.into_iter().map(|pr| Device(pr) ).collect();
        if DEBUG_PRINT { println!("\nDevices::list(): device_types: {:?} -> list: {:?}", 
            device_types, list); }
        list
    }

    /// Returns a list of all devices avaliable for a given `platform`.
    ///
    /// Equivalent to `::list(platform, None)`.
    pub fn list_all(platform: &Platform) -> Vec<Device> {
        // let list_core = core::get_device_ids(Some(platform.as_core()), None)
        //     .expect("Device::list_all: Error retrieving device list");        
        // list_core.into_iter().map(|pr| Device(pr) ).collect()
        Self::list(platform, None)
    }

    /// Returns a list of devices filtered by type then selected using a
    /// list of indexes.
    ///
    /// # Errors
    ///
    /// All indices in `idxs` must be valid.
    ///
    pub fn list_select(platform: &Platform, device_types: Option<DeviceType>,
            idxs: &[usize]) -> OclResult<Vec<Device>>
    {
        Self::resolve_idxs(idxs, &Self::list(platform, device_types))
    }

    /// Returns a list of devices filtered by type then selected using a
    /// wrapping list of indexes.
    ///
    /// Wraps indexes around (`%`) so that every index is valid.
    ///
    pub fn list_select_wrap(platform: &Platform, device_types: Option<DeviceType>,
            idxs: &[usize]) -> Vec<Device> 
    {
        let list = Self::resolve_idxs_wrap(idxs, &Self::list(platform, device_types));
        if DEBUG_PRINT { println!("\nDevices::list_select_wrap(): device_types: {:?} \
            -> list: {:?}", device_types, list); }
        list
    }

    // /// Creates a new `Device` from a `DeviceIdCore`.
    // ///
    // /// ## Safety 
    // ///
    // /// Not meant to be called unless you know what you're doing.
    // pub fn new(id_core: DeviceIdCore) -> Device {
    //     Device(id_core)
    // }

    /// Returns a list of `Device`s from a list of `DeviceIdCore`s
    pub fn list_from_core(devices: Vec<DeviceIdCore>) -> Vec<Device> {
        devices.into_iter().map(|p| Device(p)).collect()
    }

    /// Returns the device name.
    pub fn name(&self) -> String {
        // match core::get_device_info(&self.0, DeviceInfo::Name) {
        //     Ok(pi) => pi.into(),
        //     Err(err) => err.into(),
        // }
        core::get_device_info(&self.0, DeviceInfo::Name).into()
    }

    /// Returns the device vendor as a string.
    pub fn vendor(&self) -> String {
        // match core::get_device_info(&self.0, DeviceInfo::Vendor) {
        //     Ok(pi) => pi.into(),
        //     Err(err) => err.into(),
        // }
        core::get_device_info(&self.0, DeviceInfo::Vendor).into()
    }

    /// Returns the maximum workgroup size.
    pub fn max_wg_size(&self) -> usize {
        match self.info(DeviceInfo::MaxWorkGroupSize) {
            DeviceInfoResult::MaxWorkGroupSize(s) => s,
            DeviceInfoResult::Error(err) => panic!("ocl::Device::max_wg_size: {}", err.description()),
            _ => panic!("ocl::Device::max_wg_size: Unexpected 'DeviceInfoResult' variant."),
        }
    }

    /// Returns info about the device. 
    pub fn info(&self, info_kind: DeviceInfo) -> DeviceInfoResult {
        // match core::get_device_info(&self.0, info_kind) {
        //     Ok(pi) => pi,
        //     Err(err) => DeviceInfoResult::Error(Box::new(err)),
        // }
        core::get_device_info(&self.0, info_kind)
    }

    /// Returns a string containing a formatted list of device properties.
    pub fn to_string(&self) -> String {
        self.clone().into()
    }

    /// Returns the underlying `DeviceIdCore`.
    pub fn as_core(&self) -> &DeviceIdCore {
        &self.0
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("Type", &self.info(DeviceInfo::Type))
            .field("VendorId", &self.info(DeviceInfo::VendorId))
            .field("MaxComputeUnits", &self.info(DeviceInfo::MaxComputeUnits))
            .field("MaxWorkItemDimensions", &self.info(DeviceInfo::MaxWorkItemDimensions))
            .field("MaxWorkGroupSize", &self.info(DeviceInfo::MaxWorkGroupSize))
            .field("MaxWorkItemSizes", &self.info(DeviceInfo::MaxWorkItemSizes))
            .field("PreferredVectorWidthChar", &self.info(DeviceInfo::PreferredVectorWidthChar))
            .field("PreferredVectorWidthShort", &self.info(DeviceInfo::PreferredVectorWidthShort))
            .field("PreferredVectorWidthInt", &self.info(DeviceInfo::PreferredVectorWidthInt))
            .field("PreferredVectorWidthLong", &self.info(DeviceInfo::PreferredVectorWidthLong))
            .field("PreferredVectorWidthFloat", &self.info(DeviceInfo::PreferredVectorWidthFloat))
            .field("PreferredVectorWidthDouble", &self.info(DeviceInfo::PreferredVectorWidthDouble))
            .field("MaxClockFrequency", &self.info(DeviceInfo::MaxClockFrequency))
            .field("AddressBits", &self.info(DeviceInfo::AddressBits))
            .field("MaxReadImageArgs", &self.info(DeviceInfo::MaxReadImageArgs))
            .field("MaxWriteImageArgs", &self.info(DeviceInfo::MaxWriteImageArgs))
            .field("MaxMemAllocSize", &self.info(DeviceInfo::MaxMemAllocSize))
            .field("Image2dMaxWidth", &self.info(DeviceInfo::Image2dMaxWidth))
            .field("Image2dMaxHeight", &self.info(DeviceInfo::Image2dMaxHeight))
            .field("Image3dMaxWidth", &self.info(DeviceInfo::Image3dMaxWidth))
            .field("Image3dMaxHeight", &self.info(DeviceInfo::Image3dMaxHeight))
            .field("Image3dMaxDepth", &self.info(DeviceInfo::Image3dMaxDepth))
            .field("ImageSupport", &self.info(DeviceInfo::ImageSupport))
            .field("MaxParameterSize", &self.info(DeviceInfo::MaxParameterSize))
            .field("MaxSamplers", &self.info(DeviceInfo::MaxSamplers))
            .field("MemBaseAddrAlign", &self.info(DeviceInfo::MemBaseAddrAlign))
            .field("MinDataTypeAlignSize", &self.info(DeviceInfo::MinDataTypeAlignSize))
            .field("SingleFpConfig", &self.info(DeviceInfo::SingleFpConfig))
            .field("GlobalMemCacheType", &self.info(DeviceInfo::GlobalMemCacheType))
            .field("GlobalMemCachelineSize", &self.info(DeviceInfo::GlobalMemCachelineSize))
            .field("GlobalMemCacheSize", &self.info(DeviceInfo::GlobalMemCacheSize))
            .field("GlobalMemSize", &self.info(DeviceInfo::GlobalMemSize))
            .field("MaxConstantBufferSize", &self.info(DeviceInfo::MaxConstantBufferSize))
            .field("MaxConstantArgs", &self.info(DeviceInfo::MaxConstantArgs))
            .field("LocalMemType", &self.info(DeviceInfo::LocalMemType))
            .field("LocalMemSize", &self.info(DeviceInfo::LocalMemSize))
            .field("ErrorCorrectionSupport", &self.info(DeviceInfo::ErrorCorrectionSupport))
            .field("ProfilingTimerResolution", &self.info(DeviceInfo::ProfilingTimerResolution))
            .field("EndianLittle", &self.info(DeviceInfo::EndianLittle))
            .field("Available", &self.info(DeviceInfo::Available))
            .field("CompilerAvailable", &self.info(DeviceInfo::CompilerAvailable))
            .field("ExecutionCapabilities", &self.info(DeviceInfo::ExecutionCapabilities))
            .field("QueueProperties", &self.info(DeviceInfo::QueueProperties))
            .field("Name", &self.info(DeviceInfo::Name))
            .field("Vendor", &self.info(DeviceInfo::Vendor))
            .field("DriverVersion", &self.info(DeviceInfo::DriverVersion))
            .field("Profile", &self.info(DeviceInfo::Profile))
            .field("Version", &self.info(DeviceInfo::Version))
            .field("Extensions", &self.info(DeviceInfo::Extensions))
            .field("Platform", &self.info(DeviceInfo::Platform))
            .field("DoubleFpConfig", &self.info(DeviceInfo::DoubleFpConfig))
            .field("HalfFpConfig", &self.info(DeviceInfo::HalfFpConfig))
            .field("PreferredVectorWidthHalf", &self.info(DeviceInfo::PreferredVectorWidthHalf))
            .field("HostUnifiedMemory", &self.info(DeviceInfo::HostUnifiedMemory))
            .field("NativeVectorWidthChar", &self.info(DeviceInfo::NativeVectorWidthChar))
            .field("NativeVectorWidthShort", &self.info(DeviceInfo::NativeVectorWidthShort))
            .field("NativeVectorWidthInt", &self.info(DeviceInfo::NativeVectorWidthInt))
            .field("NativeVectorWidthLong", &self.info(DeviceInfo::NativeVectorWidthLong))
            .field("NativeVectorWidthFloat", &self.info(DeviceInfo::NativeVectorWidthFloat))
            .field("NativeVectorWidthDouble", &self.info(DeviceInfo::NativeVectorWidthDouble))
            .field("NativeVectorWidthHalf", &self.info(DeviceInfo::NativeVectorWidthHalf))
            .field("OpenclCVersion", &self.info(DeviceInfo::OpenclCVersion))
            .field("LinkerAvailable", &self.info(DeviceInfo::LinkerAvailable))
            .field("BuiltInKernels", &self.info(DeviceInfo::BuiltInKernels))
            .field("ImageMaxBufferSize", &self.info(DeviceInfo::ImageMaxBufferSize))
            .field("ImageMaxArraySize", &self.info(DeviceInfo::ImageMaxArraySize))
            .field("ParentDevice", &self.info(DeviceInfo::ParentDevice))
            .field("PartitionMaxSubDevices", &self.info(DeviceInfo::PartitionMaxSubDevices))
            .field("PartitionProperties", &self.info(DeviceInfo::PartitionProperties))
            .field("PartitionAffinityDomain", &self.info(DeviceInfo::PartitionAffinityDomain))
            .field("PartitionType", &self.info(DeviceInfo::PartitionType))
            .field("ReferenceCount", &self.info(DeviceInfo::ReferenceCount))
            .field("PreferredInteropUserSync", &self.info(DeviceInfo::PreferredInteropUserSync))
            .field("PrintfBufferSize", &self.info(DeviceInfo::PrintfBufferSize))
            .field("ImagePitchAlignment", &self.info(DeviceInfo::ImagePitchAlignment))
            .field("ImageBaseAddressAlignment", &self.info(DeviceInfo::ImageBaseAddressAlignment))
            .finish()
    }
}

unsafe impl ClDeviceIdPtr for Device {}

impl Into<String> for Device {
    fn into(self) -> String {
        format!("{}", self)
    }
}

impl Into<DeviceIdCore> for Device {
    fn into(self) -> DeviceIdCore {
        self.0
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl AsRef<Device> for Device {
    fn as_ref(&self) -> &Device {
        self
    }
}

impl Deref for Device {
    type Target = DeviceIdCore;

    fn deref(&self) -> &DeviceIdCore {
        &self.0
    }
}

impl DerefMut for Device {
    fn deref_mut(&mut self) -> &mut DeviceIdCore {
        &mut self.0
    }
}


// fn try_to_str(result: OclResult<DeviceInfoResult>) -> String {
//     match result {
//         Ok(pi) => pi.into(),
//         Err(err) => err.into(),
//     }
// }
