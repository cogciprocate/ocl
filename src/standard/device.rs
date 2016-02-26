//! An OpenCL device identifier.

// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::convert::Into;
use error::Result as OclResult;
use standard::{self, Platform};
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
    	let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
    		("\n", "\n", "\n")
    	} else {
    		("{ ", ", ", " }")
		};

		write!(f, "[FORMATTING NOT FULLY IMPLEMENTED][Device] {b}\
				DeviceInfo::Type: {}{d}\
				DeviceInfo::VendorId: {}{d}\
				DeviceInfo::MaxComputeUnits: {}{d}\
				DeviceInfo::MaxWorkItemDimensions: {}{d}\
				DeviceInfo::MaxWorkGroupSize: {}{d}\
				DeviceInfo::MaxWorkItemSizes: {}{d}\
				DeviceInfo::PreferredVectorWidthChar: {}{d}\
				DeviceInfo::PreferredVectorWidthShort: {}{d}\
				DeviceInfo::PreferredVectorWidthInt: {}{d}\
				DeviceInfo::PreferredVectorWidthLong: {}{d}\
				DeviceInfo::PreferredVectorWidthFloat: {}{d}\
				DeviceInfo::PreferredVectorWidthDouble: {}{d}\
				DeviceInfo::MaxClockFrequency: {}{d}\
				DeviceInfo::AddressBits: {}{d}\
				DeviceInfo::MaxReadImageArgs: {}{d}\
				DeviceInfo::MaxWriteImageArgs: {}{d}\
				DeviceInfo::MaxMemAllocSize: {}{d}\
				DeviceInfo::Image2dMaxWidth: {}{d}\
				DeviceInfo::Image2dMaxHeight: {}{d}\
				DeviceInfo::Image3dMaxWidth: {}{d}\
				DeviceInfo::Image3dMaxHeight: {}{d}\
				DeviceInfo::Image3dMaxDepth: {}{d}\
				DeviceInfo::ImageSupport: {}{d}\
				DeviceInfo::MaxParameterSize: {}{d}\
				DeviceInfo::MaxSamplers: {}{d}\
				DeviceInfo::MemBaseAddrAlign: {}{d}\
				DeviceInfo::MinDataTypeAlignSize: {}{d}\
				DeviceInfo::SingleFpConfig: {}{d}\
				DeviceInfo::GlobalMemCacheType: {}{d}\
				DeviceInfo::GlobalMemCachelineSize: {}{d}\
				DeviceInfo::GlobalMemCacheSize: {}{d}\
				DeviceInfo::GlobalMemSize: {}{d}\
				DeviceInfo::MaxConstantBufferSize: {}{d}\
				DeviceInfo::MaxConstantArgs: {}{d}\
				DeviceInfo::LocalMemType: {}{d}\
				DeviceInfo::LocalMemSize: {}{d}\
				DeviceInfo::ErrorCorrectionSupport: {}{d}\
				DeviceInfo::ProfilingTimerResolution: {}{d}\
				DeviceInfo::EndianLittle: {}{d}\
				DeviceInfo::Available: {}{d}\
				DeviceInfo::CompilerAvailable: {}{d}\
				DeviceInfo::ExecutionCapabilities: {}{d}\
				DeviceInfo::QueueProperties: {}{d}\
				DeviceInfo::Name: {}{d}\
				DeviceInfo::Vendor: {}{d}\
				DeviceInfo::DriverVersion: {}{d}\
				DeviceInfo::Profile: {}{d}\
				DeviceInfo::Version: {}{d}\
				DeviceInfo::Extensions: {}{d}\
				DeviceInfo::Platform: {}{d}\
				DeviceInfo::DoubleFpConfig: {}{d}\
				DeviceInfo::HalfFpConfig: {}{d}\
				DeviceInfo::PreferredVectorWidthHalf: {}{d}\
				DeviceInfo::HostUnifiedMemory: {}{d}\
				DeviceInfo::NativeVectorWidthChar: {}{d}\
				DeviceInfo::NativeVectorWidthShort: {}{d}\
				DeviceInfo::NativeVectorWidthInt: {}{d}\
				DeviceInfo::NativeVectorWidthLong: {}{d}\
				DeviceInfo::NativeVectorWidthFloat: {}{d}\
				DeviceInfo::NativeVectorWidthDouble: {}{d}\
				DeviceInfo::NativeVectorWidthHalf: {}{d}\
				DeviceInfo::OpenclCVersion: {}{d}\
				DeviceInfo::LinkerAvailable: {}{d}\
				DeviceInfo::BuiltInKernels: {}{d}\
				DeviceInfo::ImageMaxBufferSize: {}{d}\
				DeviceInfo::ImageMaxArraySize: {}{d}\
				DeviceInfo::ParentDevice: {}{d}\
				DeviceInfo::PartitionMaxSubDevices: {}{d}\
				DeviceInfo::PartitionProperties: {}{d}\
				DeviceInfo::PartitionAffinityDomain: {}{d}\
				DeviceInfo::PartitionType: {}{d}\
				DeviceInfo::ReferenceCount: {}{d}\
				DeviceInfo::PreferredInteropUserSync: {}{d}\
				DeviceInfo::PrintfBufferSize: {}{d}\
				DeviceInfo::ImagePitchAlignment: {}{d}\
				DeviceInfo::ImageBaseAddressAlignment: {}{e}\
			",
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Type)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::VendorId)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxComputeUnits)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxWorkItemDimensions)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxWorkGroupSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxWorkItemSizes)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthChar)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthShort)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthInt)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthLong)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthFloat)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthDouble)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxClockFrequency)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::AddressBits)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxReadImageArgs)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxWriteImageArgs)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxMemAllocSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Image2dMaxWidth)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Image2dMaxHeight)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Image3dMaxWidth)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Image3dMaxHeight)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Image3dMaxDepth)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ImageSupport)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxParameterSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxSamplers)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MemBaseAddrAlign)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MinDataTypeAlignSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::SingleFpConfig)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::GlobalMemCacheType)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::GlobalMemCachelineSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::GlobalMemCacheSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::GlobalMemSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxConstantBufferSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxConstantArgs)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::LocalMemType)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::LocalMemSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ErrorCorrectionSupport)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ProfilingTimerResolution)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::EndianLittle)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Available)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::CompilerAvailable)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ExecutionCapabilities)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::QueueProperties)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Name)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Vendor)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::DriverVersion)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Profile)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Version)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Extensions)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::Platform)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::DoubleFpConfig)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::HalfFpConfig)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthHalf)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::HostUnifiedMemory)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthChar)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthShort)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthInt)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthLong)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthFloat)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthDouble)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthHalf)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::OpenclCVersion)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::LinkerAvailable)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::BuiltInKernels)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ImageMaxBufferSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ImageMaxArraySize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ParentDevice)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PartitionMaxSubDevices)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PartitionProperties)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PartitionAffinityDomain)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PartitionType)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ReferenceCount)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredInteropUserSync)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::PrintfBufferSize)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ImagePitchAlignment)),
			try_to_str(core::get_device_info(&self.0, DeviceInfo::ImageBaseAddressAlignment)),
			b = begin,
			d = delim,
			e = end,
		)
    }
}

fn try_to_str(result: OclResult<DeviceInfoResult>) -> String {
	match result {
		Ok(pi) => pi.into(),
		Err(err) => err.into(),
	}
}
