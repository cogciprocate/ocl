//! An `OpenCL` context.

use std;
use std::ops::{Deref, DerefMut};
use crate::ffi::cl_context;
use crate::core::{self, Context as ContextCore, ContextProperties, ContextPropertyValue, ContextInfo,
    ContextInfoResult, DeviceInfo, DeviceInfoResult, PlatformInfo, PlatformInfoResult,
    CreateContextCallbackFn, UserDataPtr, OpenclVersion, ClContextPtr, ClVersions};
use crate::core::error::{Result as OclCoreResult};
use crate::error::{Error as OclError, Result as OclResult};
use crate::standard::{Platform, Device, DeviceSpecifier};



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
            -> OclResult<Context> {
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

        let device_list = device_spec.to_device_list(platform)?;

        let obj_core = core::create_context(properties.as_ref(), &device_list, pfn_notify, user_data)?;

        Ok(Context(obj_core))
    }

    /// Resolves a list of zero-based device indices into a list of Devices.
    ///
    /// If any index is out of bounds it will wrap around zero (%) to the next
    /// valid device index.
    ///
    pub fn resolve_wrapping_device_idxs(&self, idxs: &[usize]) -> Vec<Device> {
        Device::resolve_idxs_wrap(idxs, &self.devices())
    }

    /// Returns a device by its ordinal count within this context.
    ///
    /// Round-robins (%) to the next valid device.
    ///
    pub fn get_device_by_wrapping_index(&self, index: usize) -> Device {
        self.resolve_wrapping_device_idxs(&[index; 1])[0]
    }

    /// Returns info about the platform associated with the context.
    pub fn platform_info(&self, info_kind: PlatformInfo) -> OclResult<PlatformInfoResult> {
        match self.platform() {
            Ok(plat_opt) => match plat_opt {
                Some(ref p) => core::get_platform_info(p, info_kind).map_err(OclError::from),
                None => Err(OclError::from("Context::platform_info: \
                    This context has no associated platform.")),
            },
            Err(err) => Err(err),
        }
    }

    /// Returns info about the device indexed by `index` associated with this
    /// context.
    pub fn device_info(&self, index: usize, info_kind: DeviceInfo) -> OclResult<DeviceInfoResult> {
        match self.devices().get(index) {
            Some(d) => core::get_device_info(d, info_kind).map_err(OclError::from),
            None => {
                Err(OclError::from("Context::device_info: Invalid device index"))
            },
        }
    }

    /// Returns info about the context.
    pub fn info(&self, info_kind: ContextInfo) -> OclResult<ContextInfoResult> {
        core::get_context_info(&self.0, info_kind).map_err(OclError::from)
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn as_core(&self) -> &ContextCore {
        &self.0
    }

    /// Returns the list of devices associated with this context.
    ///
    /// Panics upon any OpenCL error.
    pub fn devices(&self) -> Vec<Device> {
        Device::list_from_core(self.0.devices().unwrap())
    }

    /// Returns the list of device versions associated with this context.
    pub fn device_versions(&self) -> OclResult<Vec<OpenclVersion>> {
        Device::list_from_core(self.0.devices().map_err(OclError::from)?).into_iter()
            .map(|d| d.version().map_err(OclError::from)).collect()
    }

    /// Returns the platform this context is associated with.
    pub fn platform(&self) -> OclResult<Option<Platform>> {
        self.0.platform().map(|opt| opt.map(Platform::from)).map_err(OclError::from)
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

impl ClVersions for Context {
    fn device_versions(&self) -> OclCoreResult<Vec<OpenclVersion>> { self.0.device_versions() }
    fn platform_version(&self) -> OclCoreResult<OpenclVersion> { self.0.platform_version() }
}

impl<'a> ClVersions for &'a Context {
    fn device_versions(&self) -> OclCoreResult<Vec<OpenclVersion>> { self.0.device_versions() }
    fn platform_version(&self) -> OclCoreResult<OpenclVersion> { self.0.platform_version() }
}



/// A builder for `Context`.
///
// * TODO:
//   - Handle context creation callbacks.
//
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
    pub fn new() -> ContextBuilder {
        // Default platform will be set within `::build` if unspecified by that time.
        let properties = ContextProperties::new();

        ContextBuilder {
            properties,
            device_spec: None,
        }
    }

    /// Specifies all context properties directly.
    ///
    /// Overwrites all previously specified properties.
    ///
    pub fn properties<'a>(&'a mut self, properties: ContextProperties) -> &'a mut ContextBuilder {
        self.properties = properties;
        self
    }

    /// Specifies a context property.
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
    pub fn gl_context(&mut self, gl_handle: *mut crate::ffi::c_void) -> &mut ContextBuilder {
        self.properties.set_gl_context(gl_handle);
        self
    }

    /// Specifies a Display pointer for the GLX context.
    ///
    /// Overwrites any previously specified GLX context.
    ///
    pub fn glx_display(&mut self, glx_display: *mut crate::ffi::c_void) -> &mut ContextBuilder {
        self.properties.set_glx_display(glx_display);
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
            -> &mut ContextBuilder {
        assert!(self.device_spec.is_none(), "ocl::ContextBuilder::devices: Devices already specified");
        self.device_spec = Some(device_spec.into());
        self
    }

    /// Returns a new `Context` with the parameters hitherinforthto specified (say what?).
    ///
    /// Returns a newly created context with the specified platform and set of device types.
    ///
    // * TODO:
    //   - Handle context creation callbacks.
    //
    pub fn build(&self) -> OclResult<Context> {
        let mut props = self.properties.clone();

        if props.get_platform().is_none() {
            props.set_platform(Platform::default());
        }

        Context::new(Some(props), self.device_spec.clone(), None, None)
    }
}
