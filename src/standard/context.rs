//! An `OpenCL` context.

// TEMPORARY:
#![allow(dead_code)]

// use formatting::MT;
use std;
use std::ops::{Deref, DerefMut};
use core::{self, Context as ContextCore, ContextProperties, ContextInfo, ContextInfoResult, DeviceInfo,
           DeviceInfoResult, PlatformId as PlatformIdCore, PlatformInfo, PlatformInfoResult, CreateContextCallbackFn,
           UserDataPtr};
use error::{Result as OclResult, Error as OclError};
use standard::{Platform, Device, DeviceSpecifier};


/// A builder for `Context`.
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
    /// ## Defaults
    ///
    /// * The first avaliable platform
    /// * All devices associated with the first available platform
    /// * No notify callback function or user data.
    ///
    /// [TODO]:
    /// - That stuff above (find a valid context, devices, etc. first thing).
    /// - Handle context creation callbacks.
    ///
    pub fn new() -> ContextBuilder {
        ContextBuilder {
            properties: None,
            platform: None,
            device_spec: None,
        }
    }

    /// Returns a new `Context` with the parameters hitherinforthto specified (say what?).
    ///
    /// Returns a newly created context with the specified platform and set of device types.
    pub fn build(&self) -> OclResult<Context> {
        let properties = match self.properties {
            Some(ref props) => {
                assert!(self.platform.is_none(),
                        "ocl::ContextBuilder::build: Internal error. 'platform' \
                    and 'properties' have both been set.");
                Some(props.clone())
            }
            None => {
                let platform = match self.platform {
                    Some(ref plat) => plat.clone(),
                    None => Platform::default(),
                };
                Some(ContextProperties::new().platform::<PlatformIdCore>(platform.into()))
            }
        };

        Context::new(properties, self.device_spec.clone(), None, None)
    }

    /// Specifies a platform.
    ///
    /// ## Panics
    ///
    /// Panics if either platform or properties has already been specified.
    ///
    pub fn platform(&mut self, platform: Platform) -> &mut ContextBuilder {
        assert!(self.platform.is_none(),
                "ocl::ContextBuilder::platform: Platform already specified");
        assert!(self.properties.is_none(),
                "ocl::ContextBuilder::platform: Properties already specified");
        self.platform = Some(platform);
        self
    }

    /// Specify context properties directly.
    ///
    /// ## Panics
    ///
    /// Panics if either properties or platform has already been specified.
    ///
    pub fn properties(&mut self, properties: ContextProperties) -> &mut ContextBuilder {
        assert!(self.platform.is_none(),
                "ocl::ContextBuilder::platform: Platform already specified");
        assert!(self.properties.is_none(),
                "ocl::ContextBuilder::platform: Properties already specified");
        self.properties = Some(properties);
        self
    }

    /// Specifies a `DeviceSpecifer` which specifies how specifically
    /// the relevant devices shall be specified.
    ///
    /// See [`DeviceSpecifier`](/ocl/ocl/enum.DeviceSpecifier.html) for actually
    /// useful documentation.
    ///
    /// ## Panics
    ///
    /// Panics if any devices have already been specified.
    ///
    pub fn devices<D: Into<DeviceSpecifier>>(&mut self, device_spec: D) -> &mut ContextBuilder {
        assert!(self.device_spec.is_none(),
                "ocl::ContextBuilder::devices: Devices already specified");
        self.device_spec = Some(device_spec.into());
        self
    }

    // // [FIXME: Add these]
    //
    // pub fn device_idx_round_robin
    // pub fn context_idx_round_robin
    //
    //
}


/// A context for a particular platform and set of device types.
///
/// Thread safety and destruction for any enclosed pointers are all handled automatically.
/// Clone, store, and share between threads to your heart's content.
///
/// [TODO]: Consider removing contained copies of the device id list and
/// platform id. Can be easily ascertained via the API.
///
#[derive(Debug, Clone)]
pub struct Context {
    obj_core: ContextCore,
    platform: Option<Platform>,
    devices: Vec<Device>,
}

impl Context {
    /// Returns a [`ContextBuilder`](/ocl/ocl/struct.ContextBuilder.html).
    ///
    /// This is the preferred way to create a Context.
    pub fn builder() -> ContextBuilder {
        ContextBuilder::new()
    }

    /// Returns a newly created context.
    ///
    /// Prefer `Context::builder()...` instead of this method unless you know
    /// what you're doing. Please also immediately contact us if you do, in
    /// fact, know what you're doing so that you can be added to the
    /// development team as the one who does.
    ///
    /// ## Defaults
    ///
    /// * The 'NULL' platform (which is not to be relied on but is generally
    ///   the first avaliable).
    /// * All devices associated with the 'NULL' platform
    /// * No notify callback function or user data.
    ///
    /// Don't rely on these defaults, instead rely on the `ContextBuilder`
    /// defaults. In other words, use: `Context::builder().build().unwrap()`
    /// rather than `Context::new(None, None, None, None).unwrap()`.
    ///
    /// ## Panics
    ///
    /// [TEMPORARY] Passing a `Some` variant for `pfn_notify` or `user_data` is
    /// not yet supported.
    ///
    pub fn new(properties: Option<ContextProperties>,
               device_spec: Option<DeviceSpecifier>,
               pfn_notify: Option<CreateContextCallbackFn>,
               user_data: Option<UserDataPtr>)
               -> OclResult<Context> {
        assert!(pfn_notify.is_none() && user_data.is_none(),
                "Context creation callbacks not yet implemented.");

        let platform: Option<Platform> = match properties {
            Some(ref props) => props.get_platform().map(Platform::new),
            None => None,
        };

        let device_spec = match device_spec {
            Some(ds) => ds,
            None => DeviceSpecifier::All,
        };

        let device_list = try!(device_spec.to_device_list(platform.clone()));

        let obj_core = try!(core::create_context(&properties, &device_list, pfn_notify, user_data));

        Ok(Context {
            obj_core: obj_core,
            platform: platform,
            devices: device_list,
        })
    }

    /// Resolves a list of zero-based device indices into a list of Devices.
    ///
    /// If any index is out of bounds it will wrap around zero (%) to the next
    /// valid device index.
    ///
    pub fn resolve_wrapping_device_idxs(&self, idxs: &[usize]) -> Vec<Device> {
        Device::resolve_idxs_wrap(idxs, &self.devices)
    }

    /// Returns a device by its ordinal count within this context.
    ///
    /// Round-robins (%) to the next valid device.
    ///
    pub fn get_device_by_wrapping_index(&self, index: usize) -> Device {
        self.resolve_wrapping_device_idxs(&[index; 1])[0]
    }

    /// Returns info about the platform associated with the context.
    pub fn platform_info(&self, info_kind: PlatformInfo) -> PlatformInfoResult {
        // match core::get_platform_info(self.platform_id_core.clone(), info_kind) {
        // match core::get_platform_info(self.platform().clone(), info_kind) {
        //     Ok(pi) => pi,
        //     Err(err) => PlatformInfoResult::Error(Box::new(err)),
        // }
        core::get_platform_info(self.platform(), info_kind)
    }

    /// Returns info about the device indexed by `index` associated with this
    /// context.
    pub fn device_info(&self, index: usize, info_kind: DeviceInfo) -> DeviceInfoResult {
        let device = match self.devices.get(index) {
            Some(d) => d,
            None => {
                return DeviceInfoResult::Error(Box::new(OclError::new("Context::device_info: Invalid device index")));
            }
        };

        // match core::get_device_info(device, info_kind) {
        //     Ok(pi) => pi,
        //     Err(err) => DeviceInfoResult::Error(Box::new(err)),
        // }
        core::get_device_info(device, info_kind)
    }

    /// Returns info about the context.
    pub fn info(&self, info_kind: ContextInfo) -> ContextInfoResult {
        // match core::get_context_info(&self.obj_core, info_kind) {
        //     Ok(pi) => pi,
        //     Err(err) => ContextInfoResult::Error(Box::new(err)),
        // }
        core::get_context_info(&self.obj_core, info_kind)
    }

    // /// Returns a string containing a formatted list of context properties.
    // pub fn to_string(&self) -> String {
    //     String::new()
    // }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    pub fn core_as_ref(&self) -> &ContextCore {
        &self.obj_core
    }

    /// Returns the list of devices associated with this context.
    pub fn devices(&self) -> &[Device] {
        &self.devices[..]
    }

    /// Returns the platform this context is associated with.
    pub fn platform(&self) -> Option<Platform> {
        self.platform
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Context")
         .field("ReferenceCount", &self.info(ContextInfo::ReferenceCount))
         .field("Devices", &self.info(ContextInfo::Devices))
         .field("Properties", &self.info(ContextInfo::Properties))
         .field("NumDevices", &self.info(ContextInfo::NumDevices))
         .finish()
    }
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl Deref for Context {
    type Target = ContextCore;

    fn deref(&self) -> &ContextCore {
        &self.obj_core
    }
}

impl DerefMut for Context {
    fn deref_mut(&mut self) -> &mut ContextCore {
        &mut self.obj_core
    }
}

// impl PartialEq<Context> for Context {
//     fn eq(&self, other: &Context) -> bool {
//         self == other
//     }
// }
