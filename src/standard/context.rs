//! An `OpenCL` context.

use std;
use std::ops::{Deref, DerefMut};
use ffi::cl_context;
use core::{self, Context as ContextCore, ContextProperties, ContextPropertyValue, ContextInfo,
    ContextInfoResult, DeviceInfo, DeviceInfoResult, PlatformId as PlatformIdCore, PlatformInfo,
    PlatformInfoResult, CreateContextCallbackFn, UserDataPtr, OpenclVersion, ClContextPtr};
use core::error::{Result as OclResult, Error as OclError};
use standard::{Platform, Device, DeviceSpecifier};



/// A context for a particular platform and set of device types.
///
/// Thread safety and destruction for any enclosed pointers are all handled automatically.
/// Clone, store, and share between threads to your heart's content.
///
//
// * TODO: Remove contained copies of the device id list and platform id.
//   Can be easily ascertained via the API. [UPDATE]: devices list removed.
//   Need to parse the `ContextProperties` out of the
//   `ContextInfoResult::Properties` before we can eliminate `platform`.
//
#[derive(Debug, Clone)]
pub struct Context(ContextCore);

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
    /// not yet supported. File an issue if you need this.
    ///
    pub fn new(properties: Option<ContextProperties>, device_spec: Option<DeviceSpecifier>,
                pfn_notify: Option<CreateContextCallbackFn>, user_data: Option<UserDataPtr>)
            -> OclResult<Context>
    {
        assert!(pfn_notify.is_none() && user_data.is_none(),
            "Context creation callbacks not yet implemented - file issue if you need this.");

        let platform: Option<Platform> = match properties {
            Some(ref props) => props.get_platform().map(Platform::new),
            None => None,
        };

        let device_spec = match device_spec {
            Some(ds) => ds,
            None => DeviceSpecifier::All,
        };

        let device_list = try!(device_spec.to_device_list(platform));

        let obj_core = try!(core::create_context(properties.as_ref(), &device_list, pfn_notify, user_data));

        Ok(Context(obj_core))
    }

    /// Resolves a list of zero-based device indices into a list of Devices.
    ///
    /// If any index is out of bounds it will wrap around zero (%) to the next
    /// valid device index.
    ///
    pub fn resolve_wrapping_device_idxs(&self, idxs: &[usize]) -> Vec<Device> {
    // pub fn resolve_wrapping_device_idxs(&self, idxs: &[usize]) -> OclResult<Vec<Device>> {
        Device::resolve_idxs_wrap(idxs, &self.devices())
        // self.devices().map(|ds| Device::resolve_idxs_wrap(idxs, &ds))
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
        match self.platform() {
            Ok(plat_opt) => match plat_opt {
                Some(ref p) => core::get_platform_info(p, info_kind),
                None => PlatformInfoResult::from(OclError::from("Context::platform_info: \
                This context has no associated platform.")),
            },
            Err(e) => PlatformInfoResult::from(e),
        }
    }

    /// Returns info about the device indexed by `index` associated with this
    /// context.
    pub fn device_info(&self, index: usize, info_kind: DeviceInfo) -> DeviceInfoResult {
        match self.devices().get(index) {
            Some(d) => core::get_device_info(d, info_kind),
            None => {
                return DeviceInfoResult::Error(Box::new(
                    OclError::from("Context::device_info: Invalid device index")));
            },
        }

        // match self.devices() {
        //     Ok(ds) => {
        //         match ds.get(index) {
        //             Some(d) => core::get_device_info(d, info_kind),
        //             None => {
        //                 return DeviceInfoResult::Error(Box::new(
        //                     OclError::from("Context::device_info: Invalid device index")));
        //             },
        //         }
        //     },
        //     Err(err) => DeviceInfoResult::Error(Box::new(err)),
        // }
    }

    /// Returns info about the context.
    pub fn info(&self, info_kind: ContextInfo) -> ContextInfoResult {
        core::get_context_info(&self.0, info_kind)
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[deprecated(since="0.13.0", note="Use `::core` instead.")]
    pub fn core_as_ref(&self) -> &ContextCore {
        &self.0
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn core(&self) -> &ContextCore {
        &self.0
    }

    /// Returns the list of devices associated with this context.
    ///
    /// Panics upon any OpenCL error.
    pub fn devices(&self) -> Vec<Device> {
    // pub fn devices(&self) -> OclResult<Vec<Device>> {
        Device::list_from_core(self.0.devices().unwrap())
        // self.0.devices().map(|dl| Device::list_from_core(dl))
    }

    /// Returns the list of device versions associated with this context.
    pub fn device_versions(&self) -> OclResult<Vec<OpenclVersion>> {
        Device::list_from_core(self.0.devices()?).into_iter().map(|d| d.version()).collect()
    }

    /// Returns the platform this context is associated with.
    pub fn platform(&self) -> OclResult<Option<Platform>> {
        self.0.platform().map(|opt| opt.map(Platform::from))
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

impl From<ContextCore> for Context {
    fn from(c: ContextCore) -> Context {
        Context(c)
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
        &self.0
    }
}

impl DerefMut for Context {
    fn deref_mut(&mut self) -> &mut ContextCore {
        &mut self.0
    }
}

unsafe impl<'a> ClContextPtr for &'a Context {
    fn as_ptr(&self) -> cl_context {
        self.0.as_ptr()
    }
}



/// A builder for `Context`.
///
/// * TODO: Implement index-searching-round-robin-ing methods (and thier '_exact' counterparts).
#[must_use = "builders do nothing unless '::build' is called"]
pub struct ContextBuilder {
    properties: ContextProperties,
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
    /// * TODO:
    ///   - That stuff above (find a valid context, devices, etc. first thing).
    ///   - Handle context creation callbacks.
    ///
    pub fn new() -> ContextBuilder {
        let properties = ContextProperties::new()
            .platform::<PlatformIdCore>(Platform::default().into());

        ContextBuilder {
            properties: properties,
            device_spec: None,
        }
    }

    /// Returns a new `Context` with the parameters hitherinforthto specified (say what?).
    ///
    /// Returns a newly created context with the specified platform and set of device types.
    pub fn build(&self) -> OclResult<Context> {
        Context::new(Some(self.properties.clone()), self.device_spec.clone(), None, None)
    }

    /// Specify context properties directly.
    ///
    /// Overwrites any previously specified properties.
    ///
    pub fn properties<'a>(&'a mut self, properties: ContextProperties) -> &'a mut ContextBuilder {
        self.properties = properties;
        self
    }

    /// Specify a context property.
    ///
    /// Overwrites any property with the same variant (i.e.: if
    /// `ContextPropertyValue::Platform` was already set, it would be
    /// overwritten if `prop_val` is also `ContextPropertyValue::Platform`).
    ///
    pub fn property<'a>(&'a mut self, prop_val: ContextPropertyValue) -> &'a mut ContextBuilder {
        self.properties.set_property_value(prop_val);
        self
    }

    /// Specifies a platform.
    ///
    /// Overwrites any previously specified platform.
    ///
    pub fn platform(&mut self, platform: Platform) -> &mut ContextBuilder {
        self.properties.set_platform(platform);
        self
    }

    /// Specifies an OpenGL context to associate with.
    ///
    /// Overwrites any previously specified OpenGL context.
    ///
    pub fn gl_context(&mut self, gl_handle: u32) -> &mut ContextBuilder {
        self.properties.set_gl_context(gl_handle);
        self
    }

    /// Specifies a list of devices with which to associate the context.
    ///
    /// Devices may be specified in any number of ways including simply
    /// passing a device or slice of devices. See the [`impl From`] section of
    /// [`DeviceSpecifier`][device_specifier] for more information.
    ///
    ///
    /// ## Panics
    ///
    /// Devices must not have already been specified.
    ///
    /// [device_specifier_from]: enum.DeviceSpecifier.html#method.from
    /// [device_specifier]: enum.DeviceSpecifier.html
    ///
    pub fn devices<D: Into<DeviceSpecifier>>(&mut self, device_spec: D)
            -> &mut ContextBuilder
    {
        assert!(self.device_spec.is_none(), "ocl::ContextBuilder::devices: Devices already specified");
        self.device_spec = Some(device_spec.into());
        self
    }
}
