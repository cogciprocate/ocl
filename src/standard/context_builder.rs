//! A builder for `Context`.

use core::{self, PlatformId as PlatformIdCore, ContextProperties};
use error::{Result as OclResult};
use standard::{Context, Platform, Device, DeviceSpecifier};


/// A builder for `Context`.
///
/// Currently ignores all of the `cl_context_properties` except for platform. Use `Context::new` directly to specify `ContextProperties` in all its glory.
///
/// [WORK IN PROGRESS]
///
///
/// TODO: Implement index-searching-round-robin-ing methods (and thier '_exact' counterparts).
pub struct ContextBuilder {
	platform: Option<Platform>,
	device_spec: Option<DeviceSpecifier>,
}

impl ContextBuilder {
	/// Creates a new `ContextBuilder`
	///
	/// Use `Context::builder().build().unwrap()` for defaults.
	///
	/// ### Defaults
	///
	/// * The first avaliable platform
	/// * All devices associated with the first available platform
	/// * No notify callback function or user data.
	///
	///	TODO: That stuff above (find a valid context, devices, etc. first thing).
	/// 
	pub fn new() -> ContextBuilder {
		ContextBuilder {
			platform: None,
			device_spec: None,
		}
	}

	/// Returns a new `Context` with the parameters hitherinforthto specified (say what?).
	pub fn build(&self) -> OclResult<Context> {
		let platform = match self.platform {
			Some(ref plat) => plat.clone(),
			None => Platform::new(try!(core::get_first_platform())),
		};

		let properties: Option<ContextProperties> = Some(
				ContextProperties::new().platform::<PlatformIdCore>(platform.into())
			);

		// let device_spec = match self.device_spec {
		// 	Some(ref ds) => ds.clone(),
		// 	None => DeviceSpecifier::All,
		// };

		Context::new(properties, self.device_spec.clone(), None, None)
	}

	/// Specifies a platform.
	///
	/// ### Panics
	///
	/// [FIXME]
	///
	pub fn platform<'a>(&'a mut self, platform: Platform) -> &'a mut ContextBuilder {
		assert!(self.platform.is_none(), "ocl::ContextBuilder::platform: Platform already specified");
		self.platform = Some(platform);
		self
	}

	/// Specifies a device.
	///
	/// ### Panics
	///
	/// [FIXME]
	///
	pub fn device<'a>(&'a mut self, device: Device) -> &'a mut ContextBuilder {
		assert!(self.device_spec.is_none(), "ocl::ContextBuilder::device: Devices already specified");
		self.device_spec = Some(DeviceSpecifier::Single(device));
		self
	}

	/// Specifies a list of devices.
	///
	/// ### Panics
	///
	/// [FIXME]
	///
	pub fn devices<'a>(&'a mut self, devices: Vec<Device>) -> &'a mut ContextBuilder {
		assert!(self.device_spec.is_none(), "ocl::ContextBuilder::devices: Devices already specified");
		self.device_spec = Some(DeviceSpecifier::List(devices));
		self
	}

	/// Specifies a `DeviceSpecifer` which specifies, specifically, how exactly
	/// the relevant devices shall be specified.
	pub fn device_spec<'a>(&'a mut self, device_spec: DeviceSpecifier) -> &'a mut ContextBuilder {
		self.device_spec = Some(device_spec);
		self
	}

	// // [FIXME: Add these]
	//
	// pub fn device_idx_round_robin
	// pub fn context_idx_round_robin
	//
	//
}
