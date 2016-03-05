// use std::convert::Into;
use error::{Result as OclResult};
use core::DeviceType;
use standard::{Device, Platform};

/// Specifies [what boils down to] a list of devices.
///
/// The variants: `All`, `Index`, and `Indices` are context-specific, not robust, 
/// and may lead to a stack unwind if the context changes. They are useful for
/// convenience only [NOTE: This may change and they may soon round-robin by default, 
/// making them robust and sexy... well robust anyway][UPDATE: this will probably remain as is].
///
/// The `TypeFlags` variant is useful for specifying a list of devices using a bitfield
/// (`DeviceType`) and is the most robust / portable.
///
///
///
/// [FIXME: Add some links to the SDK]
/// [FIXME: Figure out what we're doing as far as round-robin/moduloing by default]
/// - UPDATE: Leave this to the builder or whatever else to determine and leave this
///   enum an exact index which panics.
///
#[derive(Debug, Clone)]
pub enum DeviceSpecifier {
    All,
    First,
    Single(Device),
    List(Vec<Device>),
    Indices(Vec<usize>),
    WrappingIndices(Vec<usize>),
    TypeFlags(DeviceType),
}

impl DeviceSpecifier {
    pub fn all(self) -> DeviceSpecifier {
        DeviceSpecifier::All
    }

    pub fn first(self) -> DeviceSpecifier {
        DeviceSpecifier::First
    }

    pub fn single(self, device: Device) -> DeviceSpecifier {
        DeviceSpecifier::Single(device)
    }

    pub fn list(self, list: Vec<Device>) -> DeviceSpecifier {
        DeviceSpecifier::List(list)
    }

    pub fn indices(self, indices: Vec<usize>) -> DeviceSpecifier {
        DeviceSpecifier::Indices(indices)
    }

    pub fn wrapping_indices(self, windices: Vec<usize>) -> DeviceSpecifier {
        DeviceSpecifier::WrappingIndices(windices)
    }

    pub fn type_flags(self, flags: DeviceType) -> DeviceSpecifier {
        DeviceSpecifier::TypeFlags(flags)
    }

    /// Returns the list of devices matching the parameters specified by this `DeviceSpecifier`
    ///
    /// ## Panics
    ///
    /// Any device indices within the `Index` and `Indices` variants must be within the range of the number of devices for the platform specified by `Platform`. If no `platform` has been specified, this behaviour is undefined and could end up using any platform at all.
    ///
    /// TODO: Swap some of the `try!`s for `.map`s.
    pub fn to_device_list(&self, platform: Option<Platform>) -> OclResult<Vec<Device>> {
        let platform = match platform {
            Some(p) => p.clone(),
            None => Platform::first(),
        };

        Ok(match self {
            &DeviceSpecifier::All => {
                Device::list_all(&platform)
            },
            &DeviceSpecifier::First => {
                try!(Device::list_select(&platform, None, &vec![0]))
            },
            &DeviceSpecifier::Single(ref device) => {
                vec![device.clone()]
            },
            &DeviceSpecifier::List(ref devices) => {
                // devices.iter().map(|d| d.clone()).collect() 
                devices.clone()
            },
            // &DeviceSpecifier::Index(idx) => {
            //     assert!(idx < device_list_all.len(), "ocl::Context::new: DeviceSpecifier::Index: \
            //         Device index out of range.");
            //     vec![device_list_all[idx].clone()]
            // },
            &DeviceSpecifier::Indices(ref idx_list) => {
                // idx_list.iter().map(|&idx| {
                //         assert!(idx < device_list_all.len(), "ocl::Context::new: \
                //             DeviceSpecifier::Indices: Device index out of range.");
                //         device_list_all[idx].clone()
                //     } ).collect()

                try!(Device::list_select(&platform, None, idx_list))
            },
            &DeviceSpecifier::WrappingIndices(ref idx_list) => {
                // idx_list.iter().map(|&idx| {
                //         assert!(idx < device_list_all.len(), "ocl::Context::new: \
                //             DeviceSpecifier::Indices: Device index out of range.");
                //         device_list_all[idx].clone()
                //     } ).collect()
                Device::list_select_wrap(&platform, None, idx_list)
            },
            &DeviceSpecifier::TypeFlags(flags) => {
                // Device::list_from_core(try!(
                //     core::get_device_ids(platform_id_core.clone(), Some(flags))
                // ))
                Device::list(&platform, Some(flags))
            },
        })
    }
}

impl Default for DeviceSpecifier {
    fn default() -> DeviceSpecifier {
        DeviceSpecifier::All
    }
}