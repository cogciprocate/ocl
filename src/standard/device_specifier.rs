// use std::convert::Into;
use error::{Result as OclResult};
use core::{self, DeviceId as DeviceIdCore, PlatformId as PlatformIdCore, DeviceType};
use standard::Device;

pub enum DeviceSpecifier {
    All,
    Single(Device),
    List(Vec<Device>),
    Index(usize),
    Indices(Vec<usize>),
    TypeFlags(DeviceType),
}

impl DeviceSpecifier {
    pub fn to_device_list(&self, platform: Option<PlatformIdCore>) -> OclResult<Vec<DeviceIdCore>> {
        let device_list_all = try!(core::get_device_ids(platform.clone(), Some(core::DEVICE_TYPE_ALL)));

        Ok(match self {
            &DeviceSpecifier::All => {
                device_list_all
            },
            &DeviceSpecifier::Single(ref device) => {
                vec![device.as_core().clone()]
            },
            &DeviceSpecifier::List(ref devices) => {
                devices.iter().map(|d| d.as_core().clone()).collect() 
            },
            &DeviceSpecifier::Index(idx) => {
                assert!(idx < device_list_all.len(), "ocl::Context::new: DeviceSpecifier::Index: \
                    Device index out of range.");
                vec![device_list_all[idx].clone()]
            },
            &DeviceSpecifier::Indices(ref idx_list) => {
                idx_list.iter().map(|&idx| {
                        assert!(idx < device_list_all.len(), "ocl::Context::new: \
                            DeviceSpecifier::Indices: Device index out of range.");
                        device_list_all[idx].clone()
                    } ).collect()
            },
            &DeviceSpecifier::TypeFlags(flags) => {
                try!(core::get_device_ids(platform.clone(), Some(flags)))
            },
        } )
    }

	pub fn into_device_list(self, platform: Option<PlatformIdCore>) -> OclResult<Vec<DeviceIdCore>> {        
        self.to_device_list(platform)
	}
}
