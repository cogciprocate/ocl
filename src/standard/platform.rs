//! An OpenCL platform identifier.
//!
//! Documentation copied from [https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetPlatformInfo.html](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetPlatformInfo.html)

// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::convert::Into;
use raw::{self, PlatformIdRaw, PlatformInfo};
use util;

#[derive(Copy, Clone, Debug)]
/// A platform identifier.
pub struct Platform(PlatformIdRaw);

impl Platform {
	/// Creates a new `Platform` from a `PlatformIdRaw`.
	///
	/// ### Safety 
	///
	/// Not meant to be called unless you know what you're doing.
	pub unsafe fn new(id_raw: PlatformIdRaw) -> Platform {
		Platform(id_raw)
	}

	/// Returns a list of all platforms avaliable on the host machine.
	pub fn list() -> Vec<Platform> {
		let list_raw = raw::get_platform_ids()
			.expect("Platform::list: Error retrieving platform list");

		unsafe { list_raw.into_iter().map(|pr| Platform::new(pr) ).collect() }
	}

	/// Returns a string containing a formatted list of every platform property.
	pub fn to_string(&self) -> String {
		self.clone().into()
	}

	/// Returns the platform profile as a string. 
	///
	/// Returns the profile name supported by the implementation. The profile name returned can be one of the following strings:
	///
	/// * FULL_PROFILE - if the implementation supports the OpenCL specification (functionality defined as part of the core specification and does not require any extensions to be supported).
	/// 
	/// * EMBEDDED_PROFILE - if the implementation supports the OpenCL embedded profile. The embedded profile is defined to be a subset for each version of OpenCL.
	///
	pub fn profile(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Profile) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

		
	/// Returns the platform driver version as a string. 
	///
	/// Returns the OpenCL version supported by the implementation. This version string has the following format:
	///
	/// * OpenCL<space><major_version.minor_version><space><platform-specific information>
	///
	/// * The major_version.minor_version value returned will be 1.2.
	pub fn version(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Version) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

	/// Returns the platform name as a string.
	pub fn name(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Name) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

	/// Returns the platform vendor as a string.
	pub fn vendor(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Vendor) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

	/// Returns the list of platform extensions as a string.
	///
	/// Returns a space-separated list of extension names (the extension names themselves do not contain any spaces) supported by the platform. Extensions defined here must be supported by all devices associated with this platform.
	pub fn extensions(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Extensions) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

	/// Returns the underlying `PlatformIdRaw`.
	pub fn as_raw(&self) -> PlatformIdRaw {
		self.0
	}
}

impl Into<String> for Platform {
	fn into(self) -> String {
		format!("{}", self)
	}
}

impl Into<PlatformIdRaw> for Platform {
	fn into(self) -> PlatformIdRaw {
		self.as_raw()
	}
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // write!(f, "{}", &self.to_string())
        writeln!(f, "PLATFORM:\n\
				{t}Profile: {}\n\
				{t}Version: {}\n\
				{t}Name: {}\n\
				{t}Vendor: {}\n\
				{t}Extensions: {}\n\
			",
			self.profile(),
			self.version(),
			self.name(),
			self.vendor(),
			self.extensions(),
			t = util::TAB,
		)
    }
}

