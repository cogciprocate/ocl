//! A builder for `Context`.

use error::{Result as OclResult};
use standard::{Context, Device, DeviceSpecifier};


/// A builder for `Context`.
///
/// [WORK IN PROGRESS]
///
///
/// TODO: Implement index-searching-round-robin-ing methods (and thier '_exact'
/// counterparts).
pub struct ContextBuilder {
	device_spec: Option<DeviceSpecifier>,
}

impl ContextBuilder {
	/// Creates a new `ContextBuilder`
	///
	/// Use `Context::builder().build().unwrap()` for defaults.
	///
	/// ### Defaults
	///
	/// * The first avaliable context
	/// * All devices associated with the first available context
	/// * No notify callback function or user data.
	///
	///	TODO: That stuff above (find a valid context, devices, etc. first thing).
	/// 
	pub fn new() -> ContextBuilder {
		ContextBuilder {
			device_spec: Some(DeviceSpecifier::All),
		}
	}

	/// Returns a new `Context` with the parameters hitherinforthto specified (say what?).
	pub fn build(&self) -> OclResult<Context> {
		Context::new(None, None, None, None)
	}

	// // [ADD ME]
	//
	// pub fn device_idx_round_robin
	// pub fn context_idx_round_robin
	//
	//

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
}
