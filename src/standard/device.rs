//! An OpenCL device identifier.

// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::convert::Into;
use error::Result as OclResult;
use standard::Platform;
use raw::{self, DeviceIdRaw, DeviceType, DeviceInfo, DeviceInfoResult};
use util;

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
		writeln!(f, "DEVICE:\n\
				{t}DeviceInfo::Type: {}\n\
				{t}DeviceInfo::VendorId: {}\n\
				{t}DeviceInfo::MaxComputeUnits: {}\n\
				{t}DeviceInfo::MaxWorkItemDimensions: {}\n\
				{t}DeviceInfo::MaxWorkGroupSize: {}\n\
				{t}DeviceInfo::MaxWorkItemSizes: {}\n\
				{t}DeviceInfo::PreferredVectorWidthChar: {}\n\
				{t}DeviceInfo::PreferredVectorWidthShort: {}\n\
				{t}DeviceInfo::PreferredVectorWidthInt: {}\n\
				{t}DeviceInfo::PreferredVectorWidthLong: {}\n\
				{t}DeviceInfo::PreferredVectorWidthFloat: {}\n\
				{t}DeviceInfo::PreferredVectorWidthDouble: {}\n\
				{t}DeviceInfo::MaxClockFrequency: {}\n\
				{t}DeviceInfo::AddressBits: {}\n\
				{t}DeviceInfo::MaxReadImageArgs: {}\n\
				{t}DeviceInfo::MaxWriteImageArgs: {}\n\
				{t}DeviceInfo::MaxMemAllocSize: {}\n\
				{t}DeviceInfo::Image2dMaxWidth: {}\n\
				{t}DeviceInfo::Image2dMaxHeight: {}\n\
				{t}DeviceInfo::Image3dMaxWidth: {}\n\
				{t}DeviceInfo::Image3dMaxHeight: {}\n\
				{t}DeviceInfo::Image3dMaxDepth: {}\n\
				{t}DeviceInfo::ImageSupport: {}\n\
				{t}DeviceInfo::MaxParameterSize: {}\n\
				{t}DeviceInfo::MaxSamplers: {}\n\
				{t}DeviceInfo::MemBaseAddrAlign: {}\n\
				{t}DeviceInfo::MinDataTypeAlignSize: {}\n\
				{t}DeviceInfo::SingleFpConfig: {}\n\
				{t}DeviceInfo::GlobalMemCacheType: {}\n\
				{t}DeviceInfo::GlobalMemCachelineSize: {}\n\
				{t}DeviceInfo::GlobalMemCacheSize: {}\n\
				{t}DeviceInfo::GlobalMemSize: {}\n\
				{t}DeviceInfo::MaxConstantBufferSize: {}\n\
				{t}DeviceInfo::MaxConstantArgs: {}\n\
				{t}DeviceInfo::LocalMemType: {}\n\
				{t}DeviceInfo::LocalMemSize: {}\n\
				{t}DeviceInfo::ErrorCorrectionSupport: {}\n\
				{t}DeviceInfo::ProfilingTimerResolution: {}\n\
				{t}DeviceInfo::EndianLittle: {}\n\
				{t}DeviceInfo::Available: {}\n\
				{t}DeviceInfo::CompilerAvailable: {}\n\
				{t}DeviceInfo::ExecutionCapabilities: {}\n\
				{t}DeviceInfo::QueueProperties: {}\n\
				{t}DeviceInfo::Name: {}\n\
				{t}DeviceInfo::Vendor: {}\n\
				{t}DeviceInfo::DriverVersion: {}\n\
				{t}DeviceInfo::Profile: {}\n\
				{t}DeviceInfo::Version: {}\n\
				{t}DeviceInfo::Extensions: {}\n\
				{t}DeviceInfo::Platform: {}\n\
				{t}DeviceInfo::DoubleFpConfig: {}\n\
				{t}DeviceInfo::HalfFpConfig: {}\n\
				{t}DeviceInfo::PreferredVectorWidthHalf: {}\n\
				{t}DeviceInfo::HostUnifiedMemory: {}\n\
				{t}DeviceInfo::NativeVectorWidthChar: {}\n\
				{t}DeviceInfo::NativeVectorWidthShort: {}\n\
				{t}DeviceInfo::NativeVectorWidthInt: {}\n\
				{t}DeviceInfo::NativeVectorWidthLong: {}\n\
				{t}DeviceInfo::NativeVectorWidthFloat: {}\n\
				{t}DeviceInfo::NativeVectorWidthDouble: {}\n\
				{t}DeviceInfo::NativeVectorWidthHalf: {}\n\
				{t}DeviceInfo::OpenclCVersion: {}\n\
				{t}DeviceInfo::LinkerAvailable: {}\n\
				{t}DeviceInfo::BuiltInKernels: {}\n\
				{t}DeviceInfo::ImageMaxBufferSize: {}\n\
				{t}DeviceInfo::ImageMaxArraySize: {}\n\
				{t}DeviceInfo::ParentDevice: {}\n\
				{t}DeviceInfo::PartitionMaxSubDevices: {}\n\
				{t}DeviceInfo::PartitionProperties: {}\n\
				{t}DeviceInfo::PartitionAffinityDomain: {}\n\
				{t}DeviceInfo::PartitionType: {}\n\
				{t}DeviceInfo::ReferenceCount: {}\n\
				{t}DeviceInfo::PreferredInteropUserSync: {}\n\
				{t}DeviceInfo::PrintfBufferSize: {}\n\
				{t}DeviceInfo::ImagePitchAlignment: {}\n\
				{t}DeviceInfo::ImageBaseAddressAlignment: {}\n\
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
			t = util::TAB,
		)
    }
}

fn try_to_str(result: OclResult<DeviceInfoResult>) -> String {
	match result {
		Ok(pi) => pi.into(),
		Err(err) => err.into(),
	}
}
