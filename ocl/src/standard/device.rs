//! An OpenCL device identifier and related types.

use std;
use std::ops::{Deref, DerefMut};
use std::borrow::Borrow;
use core::error::{Result as OclCoreResult};
use standard::Platform;
use ffi::cl_device_id;
use core::{self, DeviceId as DeviceIdCore, DeviceType, DeviceInfo, DeviceInfoResult, ClDeviceIdPtr};
use core::util;


// Perhaps add something like this to the `DeviceSpecifier`.
//
// Copied from `https://github.com/TyOverby/ocl-repro/blob/master/src/main.rs`:
//
// pub fn first_gpu() -> (Platform, Device) {
//     let mut out = vec![];
//     for plat in Platform::list() {
//         if let Ok(all_devices) = Device::list_all(&plat) {
//             for dev in all_devices {
//                 out.push((plat.clone(), dev));
//             }
//         }
//     }
//
//     // Prefer GPU
//     out.sort_by(|&(_, ref a), &(_, ref b)| {
//         let a_type = a.info(DeviceInfo::Type);
//         let b_type = b.info(DeviceInfo::Type);
//         if let (DeviceInfoResult::Type(a_type), DeviceInfoResult::Type(b_type)) = (a_type, b_type) {
//             b_type.cmp(&a_type)
//         } else {
//             (0).cmp(&0)
//         }
//     });
//
//     out.first().unwrap().clone()
// }

/// Specifies [what boils down to] a list of devices.
///
/// The `Indices` variant is context-specific, not robust, and may lead to a
/// panic if the context changes. It is useful for convenience only and not
/// recommended for general use. The `WrappingIndices` variant is somewhat
/// less dangerous but can still be somewhat machine-specific.
///
/// The `TypeFlags` variant is used for specifying a list of devices using a
/// bitfield (`DeviceType`) and is the most robust / portable.
///
///
/// [FIXME: Add some links to the SDK]
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
    /// Returns a `DeviceSpecifier::All` variant which specifies all
    /// devices on a platform.
    ///
    pub fn all(self) -> DeviceSpecifier {
        DeviceSpecifier::All
    }

    /// Returns a `DeviceSpecifier::First` variant which specifies only
    /// the first device on a platform.
    ///
    pub fn first(self) -> DeviceSpecifier {
        DeviceSpecifier::First
    }

    /// Returns a `DeviceSpecifier::Single` variant which specifies a single
    /// device.
    ///
    pub fn single(self, device: Device) -> DeviceSpecifier {
        DeviceSpecifier::Single(device)
    }

    /// Returns a `DeviceSpecifier::List` variant which specifies a list of
    /// devices.
    ///
    pub fn list(self, list: Vec<Device>) -> DeviceSpecifier {
        DeviceSpecifier::List(list)
    }

    /// Returns a `DeviceSpecifier::Indices` variant which specifies a list of
    /// devices by index.
    ///
    ///
    /// ### Safety
    ///
    /// This variant is context-specific, not robust, and may lead to a panic
    /// if the context changes. It is useful for convenience only and not
    /// recommended for general use.
    ///
    /// Though using the `Indices` variant is not strictly unsafe in the usual
    /// way (will not lead to memory bugs, etc.), it is marked unsafe as a
    /// warning. Recommendations for a more idiomatic way to express this
    /// potential footgun are welcome.
    ///
    /// Using `::wrapping_indices` is a more robust (but still potentially
    /// non-portable) solution.
    ///
    pub unsafe fn indices(self, indices: Vec<usize>) -> DeviceSpecifier {
        DeviceSpecifier::Indices(indices)
    }

    /// Returns a `DeviceSpecifier::WrappingIndices` variant, specifying a
    /// list of devices by indices which are wrapped around (simply using the
    /// modulo operator) so that every index is always valid.
    ///
    pub fn wrapping_indices(self, windices: Vec<usize>) -> DeviceSpecifier {
        DeviceSpecifier::WrappingIndices(windices)
    }

    /// Returns a `DeviceSpecifier::TypeFlags` variant which specifies a list
    /// of devices using a conventional bitfield.
    ///
    pub fn type_flags(self, flags: DeviceType) -> DeviceSpecifier {
        DeviceSpecifier::TypeFlags(flags)
    }

    /// Returns the list of devices matching the parameters specified by this
    /// `DeviceSpecifier`
    ///
    /// ### Panics
    ///
    /// Any device indices listed within the `Indices` variant must be within
    /// the range of the number of devices for the platform specified by
    /// `Platform`. If no `platform` has been specified, this behaviour is
    /// undefined and could end up using any platform at all.
    ///
    pub fn to_device_list<P: Borrow<Platform>>(&self, platform: Option<P>) -> OclCoreResult<Vec<Device>> {
        let platform = platform.map(|p| p.borrow().clone()).unwrap_or(Platform::first(false)?);

        match *self {
            DeviceSpecifier::All => {
                Device::list_all(&platform)
            },
            DeviceSpecifier::First => {
                Device::list_select(&platform, None, &[0])
            },
            DeviceSpecifier::Single(ref device) => {
                Ok(vec![device.clone()])
            },
            DeviceSpecifier::List(ref devices) => {
                Ok(devices.clone())
            },
            DeviceSpecifier::Indices(ref idx_list) => {
                Device::list_select(&platform, None, idx_list)
            },
            DeviceSpecifier::WrappingIndices(ref idx_list) => {
                Device::list_select_wrap(&platform, None, idx_list)
            },
            DeviceSpecifier::TypeFlags(flags) => {
                Device::list(&platform, Some(flags))
            },
        }
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

impl<'a> From<&'a [Device]> for DeviceSpecifier {
    fn from(devices: &'a [Device]) -> DeviceSpecifier {
        DeviceSpecifier::List(devices.into())
    }
}

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

impl From<DeviceIdCore> for DeviceSpecifier {
    fn from(device: DeviceIdCore) -> DeviceSpecifier {
        DeviceSpecifier::Single(device.into())
    }
}

impl<'a> From<&'a DeviceIdCore> for DeviceSpecifier {
    fn from(device: &'a DeviceIdCore) -> DeviceSpecifier {
        DeviceSpecifier::Single(device.clone().into())
    }
}

impl From<DeviceType> for DeviceSpecifier {
    fn from(flags: DeviceType) -> DeviceSpecifier {
        DeviceSpecifier::TypeFlags(flags)
    }
}


/// An individual device identifier (an OpenCL device_id).
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Device(DeviceIdCore);

impl Device {
    /// Returns the first available device on a platform.
    ///
    /// Panics upon OpenCL error.
    ///
    pub fn first<P: Borrow<Platform>>(platform: P) -> Device {
        let device_ids = core::get_device_ids(platform.borrow(), None, None)
            .expect("ocl::Device::first: Error retrieving device list");
        Device(device_ids[0])
    }

    /// Returns a single device specified by a wrapped index.
    ///
    /// Panics upon OpenCL error.
    ///
    pub fn by_idx_wrap<P: Borrow<Platform>>(platform: P, device_idx_wrap: usize) -> Device {
        let device_ids = core::get_device_ids(platform.borrow(), None, None)
            .expect("ocl::Device::by_idx_wrap: Error retrieving device list");
        let wrapped_idx = device_idx_wrap % device_ids.len();
        Device(device_ids[wrapped_idx])
    }

    /// Returns a `DeviceSpecifier` useful for precisely specifying a set
    /// of devices.
    pub fn specifier() -> DeviceSpecifier {
        DeviceSpecifier::default()
    }

    /// Resolves a list of indices into a list of valid devices.
    ///
    /// `devices` is the set of all indexable devices.
    ///
    ///
    /// ### Errors
    ///
    /// All indices in `idxs` must be valid. Use `resolve_idxs_wrap` for index
    /// lists which may contain out of bounds indices.
    ///
    pub fn resolve_idxs(idxs: &[usize], devices: &[Device]) -> OclCoreResult<Vec<Device>> {
        let mut result = Vec::with_capacity(idxs.len());
        for &idx in idxs.iter() {
            match devices.get(idx) {
                Some(&device) => result.push(device),
                None => return Err(format!("Error resolving device index: '{}'. Index out of \
                    range. Devices avaliable: '{}'.", idx, devices.len()).into()),
            }
        }
        Ok(result)
    }

    /// Resolves a list of indices into a list of valid devices.
    ///
    /// `devices` is the set of all indexable devices.
    ///
    /// Wraps indices around using modulo (`%`) so that every index is valid.
    ///
    pub fn resolve_idxs_wrap(idxs: &[usize], devices: &[Device]) -> Vec<Device> {
        let valid_idxs = util::wrap_vals(idxs, devices.len());
        valid_idxs.iter().map(|&idx| devices[idx]).collect()
    }

    /// Returns a list of all devices avaliable for a given platform which
    /// optionally match the flags set in the bitfield, `device_types`.
    ///
    /// Setting `device_types` to `None` will return a list of all avaliable
    /// devices for `platform` regardless of type.
    ///
    ///
    /// ### Errors
    ///
    /// Returns an `Err(ocl::core::Error::Status {...})` enum variant upon any
    /// OpenCL error. Calling [`.status()`] on the returned error will return
    /// an `Option(``[ocl::core::Status]``)` which can be unwrapped then
    /// matched to determine the precise reason for failure.
    ///
    /// [`.status()`]: enum.Error.html#method.status
    /// [`ocl::core::Status`]: enum.Status.html
    ///
    pub fn list<P: Borrow<Platform>>(platform: P, device_types: Option<DeviceType>) -> OclCoreResult<Vec<Device>> {
        let list_core = core::get_device_ids(platform.borrow(), device_types, None)
            .unwrap_or(vec![]);
        Ok(list_core.into_iter().map(Device).collect())
    }

    /// Returns a list of all devices avaliable for a given `platform`.
    ///
    /// Equivalent to `::list(platform, None)`.
    ///
    /// See [`::list`](struct.Device.html#method.list) for other
    /// error information.
    ///
    pub fn list_all<P: Borrow<Platform>>(platform: P) -> OclCoreResult<Vec<Device>> {
        Self::list(platform, None)
    }

    /// Returns a list of devices filtered by type then selected using a
    /// list of indices.
    ///
    ///
    /// ### Errors
    ///
    /// All indices in `idxs` must be valid.
    ///
    /// See [`::list`](struct.Device.html#method.list) for other
    /// error information.
    ///
    pub fn list_select<P: Borrow<Platform>>(platform: P, device_types: Option<DeviceType>,
                idxs: &[usize]) -> OclCoreResult<Vec<Device>> {
        Self::resolve_idxs(idxs, &Self::list(platform, device_types)?)
    }

    /// Returns a list of devices filtered by type then selected using a
    /// wrapping list of indices.
    ///
    /// Wraps indices around (`%`) so that every index is valid.
    ///
    ///
    /// ### Errors
    ///
    /// See [`::list`](struct.Device.html#method.list)
    ///
    pub fn list_select_wrap<P: Borrow<Platform>>(platform: P, device_types: Option<DeviceType>,
                idxs: &[usize]) -> OclCoreResult<Vec<Device>> {
        Ok(Self::resolve_idxs_wrap(idxs, &Self::list(platform, device_types)?))
    }

    /// Returns a list of `Device`s from a list of `DeviceIdCore`s
    pub fn list_from_core(mut devices: Vec<DeviceIdCore>) -> Vec<Device> {
        use std::mem;
        debug_assert!(mem::size_of::<DeviceIdCore>() == mem::size_of::<Device>());
        unsafe {
            let (ptr, len, cap) = (devices.as_mut_ptr(), devices.len(), devices.capacity());
            mem::forget(devices);
            Vec::from_raw_parts(ptr as *mut Device, len, cap)
        }
    }

    /// Returns the device name.
    pub fn name(&self) -> String {
        core::get_device_info(&self.0, DeviceInfo::Name).into()
    }

    /// Returns the device vendor as a string.
    pub fn vendor(&self) -> String {
        core::get_device_info(&self.0, DeviceInfo::Vendor).into()
    }

    /// Returns the maximum workgroup size or an error.
    pub fn max_wg_size(&self) -> OclCoreResult<usize> {
        match self.info(DeviceInfo::MaxWorkGroupSize) {
            DeviceInfoResult::MaxWorkGroupSize(r) => Ok(r),
            DeviceInfoResult::Error(err) => Err(*err),
            _ => panic!("Device::max_wg_size: Unexpected 'DeviceInfoResult' variant."),
        }
    }

    /// Returns the memory base address alignment offset or an error.
    pub fn mem_base_addr_align(&self) -> OclCoreResult<u32> {
        match self.info(DeviceInfo::MemBaseAddrAlign) {
            DeviceInfoResult::MemBaseAddrAlign(r) => Ok(r),
            DeviceInfoResult::Error(err) => Err(*err),
            _ => panic!("Device::mem_base_addr_align: Unexpected 'DeviceInfoResult' variant."),
        }
    }

    /// Returns whether or not the device is available for use.
    pub fn is_available(&self) -> OclCoreResult<bool> {
        match self.info(DeviceInfo::Available) {
            DeviceInfoResult::Available(r) => Ok(r),
            DeviceInfoResult::Error(err) => Err(*err),
            _ => panic!("Device::is_available: Unexpected 'DeviceInfoResult' variant."),
        }
    }

    /// Returns info about the device.
    pub fn info(&self, info_kind: DeviceInfo) -> DeviceInfoResult {
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

unsafe impl ClDeviceIdPtr for Device {
    fn as_ptr(&self) -> cl_device_id {
        self.0.as_raw()
    }
}

unsafe impl<'a> ClDeviceIdPtr for &'a Device {
    fn as_ptr(&self) -> cl_device_id {
        self.0.as_raw()
    }
}

impl From<DeviceIdCore> for Device {
    fn from(core: DeviceIdCore) -> Device {
        Device(core)
    }
}

// impl Into<String> for Device {
//     fn into(self) -> String {
//         format!("{}", self)
//     }
// }

impl From<Device> for String {
    fn from(d: Device) -> String {
        format!("{}", d)
    }
}

// impl Into<DeviceIdCore> for Device {
//     fn into(self) -> DeviceIdCore {
//         self.0
//     }
// }

impl From<Device> for DeviceIdCore {
    fn from(d: Device) -> DeviceIdCore {
        d.0
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
