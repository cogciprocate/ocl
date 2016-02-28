//! An OpenCL device identifier.

// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::convert::Into;
// use error::Result as OclResult;
use standard::{Platform};
use core::{self, DeviceId as DeviceIdCore, DeviceType, DeviceInfo, DeviceInfoResult, ClDeviceIdPtr};
// use util;


/// A device identifier.
#[derive(Clone, Debug)]
pub struct Device(DeviceIdCore);

impl Device {
    /// Creates a new `Device` from a `DeviceIdCore`.
    ///
    /// ### Safety 
    ///
    /// Not meant to be called unless you know what you're doing.
    pub fn new(id_core: DeviceIdCore) -> Device {
        Device(id_core)
    }

    /// Returns a list of all devices avaliable for a given platform which
    /// optionally match the flags set in the bitfield, `device_types`.
    ///
    /// Setting `device_types` to `None` will return a list of all avaliable
    /// devices for `platform`
    pub fn list(platform: Platform, device_types: Option<DeviceType>) -> Vec<Device> {
        let list_core = core::get_device_ids(Some(platform.as_core().clone()), device_types)
            .expect("Device::list: Error retrieving device list");

        list_core.into_iter().map(|pr| Device::new(pr) ).collect()
    }

    /// Returns a list of all devices avaliable for a given `platform`.
    ///
    /// Equivalent to `::list(platform, None)`.
    pub fn list_all(platform: &Platform) -> Vec<Device> {
        let list_core = core::get_device_ids(Some(platform.as_core().clone()), None)
            .expect("Device::list_all: Error retrieving device list");

        list_core.into_iter().map(|pr| Device::new(pr) ).collect()
    }

    /// Returns a list of `Device`s from a list of `DeviceIdCore`s
    pub fn list_from_core(devices: Vec<DeviceIdCore>) -> Vec<Device> {
        devices.into_iter().map(|p| Device::new(p)).collect()
    }

    /// Returns the device name.
    pub fn name(&self) -> String {
        match core::get_device_info(&self.0, DeviceInfo::Name) {
            Ok(pi) => pi.into(),
            Err(err) => err.into(),
        }
    }

    /// Returns the device vendor as a string.
    pub fn vendor(&self) -> String {
        match core::get_device_info(&self.0, DeviceInfo::Vendor) {
            Ok(pi) => pi.into(),
            Err(err) => err.into(),
        }
    }

    /// Returns info about the device. 
    pub fn info(&self, info_kind: DeviceInfo) -> DeviceInfoResult {
        match core::get_device_info(&self.0, info_kind) {
            Ok(pi) => pi,
            Err(err) => DeviceInfoResult::Error(Box::new(err)),
        }
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

// fn try_to_str(result: OclResult<DeviceInfoResult>) -> String {
//     match result {
//         Ok(pi) => pi.into(),
//         Err(err) => err.into(),
//     }
// }
