//! An OpenCL device identifier.

// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::convert::Into;
use error::Result as OclResult;
use standard::Platform;
use raw::{self, DeviceIdRaw, DeviceType, DeviceInfo, DeviceInfoResult};
// use util;

#[derive(Copy, Clone, Debug)]
/// A device identifier.
pub struct Device(DeviceIdRaw);

impl Device {
	/// Creates a new `Device` from a `DeviceIdRaw`.
	///
	/// ### Safety 
	///
	/// Not meant to be called unless you know what you're doing.
	pub unsafe fn new(id_raw: DeviceIdRaw) -> Device {
		Device(id_raw)
	}

	/// Returns a list of all devices avaliable for a given platform which
	/// optionally match the flags set in the bitfield, `device_types`.
	///
	/// Setting `device_types` to `None` will return a list of all avaliable
	/// devices for `platform`
	pub fn list(platform: Platform, device_types: Option<DeviceType>) -> Vec<Device> {
		let list_raw = raw::get_device_ids(platform.as_raw(), device_types)
			.expect("Device::list: Error retrieving device list");

		unsafe { list_raw.into_iter().map(|pr| Device::new(pr) ).collect() }
	}

	/// Returns a list of all devices avaliable for a given `platform`.
	///
	/// Equivalent to `::list(platform, None)`.
	pub fn list_all(platform: Platform) -> Vec<Device> {
		let list_raw = raw::get_device_ids(platform.as_raw(), None)
			.expect("Device::list_all: Error retrieving device list");

		unsafe { list_raw.into_iter().map(|pr| Device::new(pr) ).collect() }
	}

	/// Returns a string containing a formatted list of device properties.
	pub fn to_string(&self) -> String {
		self.clone().into()
	}

	/// Returns the underlying `DeviceIdRaw`.
	pub fn as_raw(&self) -> DeviceIdRaw {
		self.0
	}
}

impl Into<String> for Device {
	fn into(self) -> String {
		format!("{}", self)
	}
}

impl Into<DeviceIdRaw> for Device {
	fn into(self) -> DeviceIdRaw {
		self.as_raw()
	}
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		writeln!(f, "[FORMATTING NOT FULLY IMPLEMENTED] DEVICE:\n\
				DeviceInfo::Type: {}\n\
				DeviceInfo::VendorId: {}\n\
				DeviceInfo::MaxComputeUnits: {}\n\
				DeviceInfo::MaxWorkItemDimensions: {}\n\
				DeviceInfo::MaxWorkGroupSize: {}\n\
				DeviceInfo::MaxWorkItemSizes: {}\n\
				DeviceInfo::PreferredVectorWidthChar: {}\n\
				DeviceInfo::PreferredVectorWidthShort: {}\n\
				DeviceInfo::PreferredVectorWidthInt: {}\n\
				DeviceInfo::PreferredVectorWidthLong: {}\n\
				DeviceInfo::PreferredVectorWidthFloat: {}\n\
				DeviceInfo::PreferredVectorWidthDouble: {}\n\
				DeviceInfo::MaxClockFrequency: {}\n\
				DeviceInfo::AddressBits: {}\n\
				DeviceInfo::MaxReadImageArgs: {}\n\
				DeviceInfo::MaxWriteImageArgs: {}\n\
				DeviceInfo::MaxMemAllocSize: {}\n\
				DeviceInfo::Image2dMaxWidth: {}\n\
				DeviceInfo::Image2dMaxHeight: {}\n\
				DeviceInfo::Image3dMaxWidth: {}\n\
				DeviceInfo::Image3dMaxHeight: {}\n\
				DeviceInfo::Image3dMaxDepth: {}\n\
				DeviceInfo::ImageSupport: {}\n\
				DeviceInfo::MaxParameterSize: {}\n\
				DeviceInfo::MaxSamplers: {}\n\
				DeviceInfo::MemBaseAddrAlign: {}\n\
				DeviceInfo::MinDataTypeAlignSize: {}\n\
				DeviceInfo::SingleFpConfig: {}\n\
				DeviceInfo::GlobalMemCacheType: {}\n\
				DeviceInfo::GlobalMemCachelineSize: {}\n\
				DeviceInfo::GlobalMemCacheSize: {}\n\
				DeviceInfo::GlobalMemSize: {}\n\
				DeviceInfo::MaxConstantBufferSize: {}\n\
				DeviceInfo::MaxConstantArgs: {}\n\
				DeviceInfo::LocalMemType: {}\n\
				DeviceInfo::LocalMemSize: {}\n\
				DeviceInfo::ErrorCorrectionSupport: {}\n\
				DeviceInfo::ProfilingTimerResolution: {}\n\
				DeviceInfo::EndianLittle: {}\n\
				DeviceInfo::Available: {}\n\
				DeviceInfo::CompilerAvailable: {}\n\
				DeviceInfo::ExecutionCapabilities: {}\n\
				DeviceInfo::QueueProperties: {}\n\
				DeviceInfo::Name: {}\n\
				DeviceInfo::Vendor: {}\n\
				DeviceInfo::DriverVersion: {}\n\
				DeviceInfo::Profile: {}\n\
				DeviceInfo::Version: {}\n\
				DeviceInfo::Extensions: {}\n\
				DeviceInfo::Platform: {}\n\
				DeviceInfo::DoubleFpConfig: {}\n\
				DeviceInfo::HalfFpConfig: {}\n\
				DeviceInfo::PreferredVectorWidthHalf: {}\n\
				DeviceInfo::HostUnifiedMemory: {}\n\
				DeviceInfo::NativeVectorWidthChar: {}\n\
				DeviceInfo::NativeVectorWidthShort: {}\n\
				DeviceInfo::NativeVectorWidthInt: {}\n\
				DeviceInfo::NativeVectorWidthLong: {}\n\
				DeviceInfo::NativeVectorWidthFloat: {}\n\
				DeviceInfo::NativeVectorWidthDouble: {}\n\
				DeviceInfo::NativeVectorWidthHalf: {}\n\
				DeviceInfo::OpenclCVersion: {}\n\
				DeviceInfo::LinkerAvailable: {}\n\
				DeviceInfo::BuiltInKernels: {}\n\
				DeviceInfo::ImageMaxBufferSize: {}\n\
				DeviceInfo::ImageMaxArraySize: {}\n\
				DeviceInfo::ParentDevice: {}\n\
				DeviceInfo::PartitionMaxSubDevices: {}\n\
				DeviceInfo::PartitionProperties: {}\n\
				DeviceInfo::PartitionAffinityDomain: {}\n\
				DeviceInfo::PartitionType: {}\n\
				DeviceInfo::ReferenceCount: {}\n\
				DeviceInfo::PreferredInteropUserSync: {}\n\
				DeviceInfo::PrintfBufferSize: {}\n\
				DeviceInfo::ImagePitchAlignment: {}\n\
				DeviceInfo::ImageBaseAddressAlignment: {}\n\
			",
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Type)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::VendorId)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxComputeUnits)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxWorkItemDimensions)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxWorkGroupSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxWorkItemSizes)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PreferredVectorWidthChar)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PreferredVectorWidthShort)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PreferredVectorWidthInt)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PreferredVectorWidthLong)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PreferredVectorWidthFloat)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PreferredVectorWidthDouble)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxClockFrequency)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::AddressBits)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxReadImageArgs)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxWriteImageArgs)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxMemAllocSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Image2dMaxWidth)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Image2dMaxHeight)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Image3dMaxWidth)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Image3dMaxHeight)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Image3dMaxDepth)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ImageSupport)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxParameterSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxSamplers)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MemBaseAddrAlign)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MinDataTypeAlignSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::SingleFpConfig)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::GlobalMemCacheType)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::GlobalMemCachelineSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::GlobalMemCacheSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::GlobalMemSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxConstantBufferSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::MaxConstantArgs)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::LocalMemType)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::LocalMemSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ErrorCorrectionSupport)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ProfilingTimerResolution)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::EndianLittle)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Available)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::CompilerAvailable)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ExecutionCapabilities)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::QueueProperties)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Name)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Vendor)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::DriverVersion)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Profile)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Version)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Extensions)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::Platform)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::DoubleFpConfig)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::HalfFpConfig)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PreferredVectorWidthHalf)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::HostUnifiedMemory)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::NativeVectorWidthChar)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::NativeVectorWidthShort)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::NativeVectorWidthInt)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::NativeVectorWidthLong)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::NativeVectorWidthFloat)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::NativeVectorWidthDouble)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::NativeVectorWidthHalf)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::OpenclCVersion)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::LinkerAvailable)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::BuiltInKernels)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ImageMaxBufferSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ImageMaxArraySize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ParentDevice)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PartitionMaxSubDevices)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PartitionProperties)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PartitionAffinityDomain)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PartitionType)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ReferenceCount)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PreferredInteropUserSync)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::PrintfBufferSize)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ImagePitchAlignment)),
			try_to_str(raw::get_device_info(self.0, DeviceInfo::ImageBaseAddressAlignment)),
		)
    }
}

fn try_to_str(result: OclResult<DeviceInfoResult>) -> String {
	match result {
		Ok(pi) => pi.into(),
		Err(err) => err.into(),
	}
}
