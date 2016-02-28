//! A builder for `Context`.

use core::{self, PlatformId as PlatformIdCore, ContextProperties};
use error::{Result as OclResult};
use standard::{Context, Platform, Device, DeviceSpecifier};


/// A builder for `Context`.
///
/// Currently ignores all of the `cl_context_properties` except for platform. 
/// Use `Context::new` directly to specify `ContextProperties` in all its glory.
/// [UPDATE ME]
///
/// [WORK IN PROGRESS]
///
///
/// TODO: Implement index-searching-round-robin-ing methods (and thier '_exact' counterparts).
pub struct ContextBuilder {
	properties: Option<ContextProperties>,
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
			properties: None,
			platform: None,
			device_spec: None,
		}
	}

	/// Returns a new `Context` with the parameters hitherinforthto specified (say what?).
	pub fn build(&self) -> OclResult<Context> {
		let properties = match self.properties {
			Some(ref props) => {
				assert!(self.platform.is_none(), "ocl::ContextBuilder::build: Internal error. 'platform' \
					and 'properties' have both been set.");
				Some(props.clone())
			},
			None => {
				let platform = match self.platform {
					Some(ref plat) => plat.clone(),
					None => Platform::new(try!(core::get_first_platform())),
				};
				Some(ContextProperties::new().platform::<PlatformIdCore>(platform.into()))
			},
		};

		Context::new(properties, self.device_spec.clone(), None, None)
	}

	/// Specifies a platform.
	///
	/// ### Panics
	///
	/// Panics if it has already been specified.
	///
	pub fn platform<'a>(&'a mut self, platform: Platform) -> &'a mut ContextBuilder {
		assert!(self.platform.is_none(), "ocl::ContextBuilder::platform: Platform already specified");
		assert!(self.properties.is_none(), "ocl::ContextBuilder::platform: Properties already specified");
		self.platform = Some(platform);
		self
	}

	/// Specify context properties directly.
	///
	/// ### Panics
	///
	/// Panics if properties have already been specified.
	///
	pub fn properties<'a>(&'a mut self, properties: ContextProperties) -> &'a mut ContextBuilder {
		assert!(self.platform.is_none(), "ocl::ContextBuilder::platform: Platform already specified");
		assert!(self.properties.is_none(), "ocl::ContextBuilder::platform: Properties already specified");
		self.properties = Some(properties);
		self
	}

	// /// Specifies a device.
	// ///
	// /// ### Panics
	// ///
	// /// Panics if any devices have already been specified.
	// ///
	// pub fn device<'a>(&'a mut self, device: Device) -> &'a mut ContextBuilder {
	// 	assert!(self.device_spec.is_none(), "ocl::ContextBuilder::device: Devices already specified");
	// 	self.device_spec = Some(DeviceSpecifier::Single(device));
	// 	self
	// }

	/// Specifies a list of devices.
	///
	/// ### Panics
	///
	/// Panics if any devices have already been specified.
	///
	pub fn device_list<'a>(&'a mut self, devices: Vec<Device>) -> &'a mut ContextBuilder {
		assert!(self.device_spec.is_none(), "ocl::ContextBuilder::device_list: Devices already specified");
		self.device_spec = Some(DeviceSpecifier::List(devices));
		self
	}

	/// Specifies a `DeviceSpecifer` which specifies how specifically
	/// the relevant devices shall be specified.
	///
	/// See [`DeviceSpecifier`](/ocl/enum.DeviceSpecifier.html) for actually
	/// useful documentation.
	///
	/// ### Panics
	///
	/// Panics if any devices have already been specified.
	///
	pub fn devices<'a>(&'a mut self, device_spec: DeviceSpecifier) -> &'a mut ContextBuilder {
		assert!(self.device_spec.is_none(), "ocl::ContextBuilder::devices: Devices already specified");
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
