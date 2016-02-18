// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::convert::Into;
use raw::{self, PlatformIdRaw, PlatformInfo};
use util;

#[derive(Copy, Clone, Debug)]
pub struct Platform (PlatformIdRaw);

impl Platform {
	pub unsafe fn new(id_raw: PlatformIdRaw) -> Platform {
		Platform(id_raw)
	}

	pub fn list() -> Vec<Platform> {
		let list_raw = raw::get_platform_ids()
			.expect("Platform::list: Error retrieving platform list");

		unsafe { list_raw.into_iter().map(|pr| Platform::new(pr) ).collect() }
	}

	pub fn as_raw(&self) -> PlatformIdRaw {
		self.0
	}

	pub fn to_string(&self) -> String {
		self.clone().into()
	}

	pub fn profile(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Profile) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

	pub fn version(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Version) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

	pub fn name(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Name) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

	pub fn vendor(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Vendor) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
	}

	pub fn extensions(&self) -> String {
		match raw::get_platform_info(self.0, PlatformInfo::Extensions) {
			Ok(pi) => pi.into(),
			Err(err) => err.into(),
		}
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
        writeln!(f, "Platform:\n\
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

