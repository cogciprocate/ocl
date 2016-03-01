//! An OpenCL context.

// TEMPORARY:
#![allow(dead_code)]

// use formatting::MT;
use std;
use std::ops::{Deref, DerefMut};
use core::{self, Context as ContextCore, ContextProperties, ContextInfo, ContextInfoResult, DeviceType, DeviceInfo, DeviceInfoResult, PlatformInfo, PlatformInfoResult, CreateContextCallbackFn, UserDataPtr};
use error::{Result as OclResult, Error as OclError};
use standard::{Platform, Device, DeviceSpecifier, ContextBuilder};


/// A context for a particular platform and set of device types.
///
/// Thread safety and destruction for any enclosed pointers are all handled automatically. 
/// Clone, store, and share between threads to your hearts content.
///
/// [FIXME]: Don't store device id list or platform id. Can be ascertained via the SDK.
/// 
#[derive(Debug, Clone)]
pub struct Context {
    obj_core: ContextCore,
    platform: Option<Platform>,
    devices: Vec<Device>,
}

impl Context {
    /// Returns a [`ContextBuilder`](/ocl/struct.ContextBuilder.html).
    ///
    /// This is the preferred way to create a Context.
    pub fn builder() -> ContextBuilder {
        ContextBuilder::new()
    }

    /// Returns a newly created context.
    ///
    /// [FIXME]: Yeah... documentation.
    ///
    /// Use `Context::builder()...` instead of this method unless you know what you're doing. Please also contact us or file an issue immediately if you do, in fact, know what you're doing so that you can be added to the development team as the one who does.
    ///
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
    pub fn new(properties: Option<ContextProperties>, device_spec: Option<DeviceSpecifier>, 
                pfn_notify: Option<CreateContextCallbackFn>, user_data: Option<UserDataPtr>
                ) -> OclResult<Context> {
        assert!(pfn_notify.is_none() && user_data.is_none(), 
            "Context creation callbacks not yet implemented.");

        let platform: Option<Platform> = match properties {
            Some(ref props) => props.get_platform().clone().map(|p| Platform::new(p)),
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

    /// [UNSTABLE]: About to be moved to builder
    /// [UNTESTED] Returns a newly created context.
    pub fn new_by_platform_and_device_list(platform: Platform, device_list: Vec<Device>
                ) -> OclResult<Context> {
        let devices = Device::list_from_core(device_list.into_iter().map(|d| d.into()).collect());
        let properties: Option<ContextProperties> = 
            Some(ContextProperties::new().platform(platform.as_core().clone()));
        let obj_core = try!(core::create_context(&properties, &devices, None, None));

        Ok(Context {
            obj_core: obj_core,
            platform: Some(platform),
            devices: devices,
        })
    }

    /// [UNSTABLE]: About to be moved to builder
    /// Returns a newly created context with a specified platform and set of device types.
    /// 
    /// [FIXME: Needs update]
    ///
    /// The desired platform may be specified by passing a valid index from a list 
    /// obtainable from the ocl::get_platform_ids() function and wrapping it with a 
    /// Some (ex. `Some(2)`). Pass `None` to use the first platform available (0). 
    /// 
    /// The device types mask may be specified using a union of any or all of the 
    /// following flags: [MOVED TO `DeviceType` DOCS].
    ///
    /// [FIXME: Update] Passing `None` will use the flag: `CL_DEVICE_TYPE_GPU`.
    ///
    /// # Examples
    /// 
    /// ```notest
    /// // use ocl;
    ///
    /// fn main() {
    ///     // Create a context with the first available platform and the default device type.
    ///     let ocl_context = ocl::Context::new_by_index_and_type(None, None);
    ///     
    ///     // Do fun stuff...
    /// }
    /// ```
    ///
    ///
    /// ```notest
    /// // use ocl;
    /// 
    /// fn main() {
    ///     //let platform_ids = ocl::get_platform_ids();
    /// 
    ///     let device_types = ocl::CL_DEVICE_TYPE_GPU | ocl::CL_DEVICE_TYPE_CPU;
    ///
    ///     // Create a context using the 1st platform and both CPU and GPU devices.
    ///     let ocl_context = ocl::Context::new_by_index_and_type(Some(0), Some(device_types));
    ///     
    ///     // ...
    /// }
    /// ``` 
    ///
    /// # Panics
    ///    - `get_devices_core_as_ref()` (work in progress)
    ///
    /// # Failures
    /// - No platforms.
    /// - Invalid platform index.
    /// - No devices.
    ///
    /// # TODO:
    /// - Add a more in-depth constructor which accepts an arbitrary list of devices (or sub-devices) and a list of cl_context_properties.
    ///
    /// # Maybe Someday TODO:
    /// - Handle context creation callbacks.
    ///
    pub fn new_by_index_and_type(platform_idx_opt: Option<usize>, device_types_opt: Option<DeviceType>) 
            -> OclResult<Context>
    {
        let platforms: Vec<Platform> = Platform::list_from_core(try!(core::get_platform_ids()));
        if platforms.len() == 0 { return OclError::err("\nNo OpenCL platforms found!\n"); }

        // println!("Platform list: {:?}", platforms);

        let platform: Platform = match platform_idx_opt {
            Some(pf_idx) => {
                match platforms.get(pf_idx) {
                    Some(pf) => {
                        pf.clone()
                    },
                    None => {
                        return OclError::err("Invalid OpenCL platform index specified. \
                            Use 'get_platform_ids()' for a list.")
                    },
                }
            },
            None => {
                debug_assert!(platforms.len() > 0, "Context::new_by_index_and_type(): Internal indexing error.");
                platforms[core::DEFAULT_PLATFORM_IDX].clone()
            },
        };

        // [DEBUG]: 
        // println!("CONTEXT::NEW: PLATFORM BEING USED: {:?}", platform_id_core);

        let properties = Some(ContextProperties::new().platform(platform.as_core().clone()));
        
        let devices: Vec<Device> = Device::list_from_core(try!(core::get_device_ids(
            &platform, device_types_opt, None)));
        if devices.len() == 0 { return OclError::err("\nNo OpenCL devices found!\n"); }

        // println!("# # # # # #  OCL::CONTEXT::NEW(): device list: {:?}", device_id_core_list);

        // [FIXME]: No callback or user data:
        let obj_core = try!(core::create_context(&properties, &devices, None, None));

        // [DEBUG]: 
        // println!("CONTEXT::NEW: CONTEXT: {:?}", obj_core);

        Ok(Context {
            obj_core: obj_core,
            platform: Some(platform),
            devices: devices,
        })
    }

    /// Resolves the zero-based device index into a list of Devices.
    pub fn resolve_device_idxs(&self, idxs: &[usize]) -> Vec<Device> {
        // let selected_idxs = match device_idxs.len() {
        //     0 => vec![0],
        //     _ => Vec::from(device_idxs),
        // };

        // let mut valid_devices = Vec::with_capacity(selected_idxs.len());

        // for selected_idx in selected_idxs {
        //     let valid_idx = selected_idx % self.devices.len();
        //     valid_devices.push(self.devices[valid_idx].clone());
        // }

        // valid_devices

        Device::resolve_idxs_wrap(idxs, &self.devices)
    }

    // /// Returns a list of valid devices regardless of whether or not the indexes 
    // /// passed are valid by performing a modulo operation on them and letting them
    // /// wrap around (round robin).
    // pub fn valid_devices(&self, selected_idxs: &[usize]) -> Vec<Device> {
    //     let mut valid_devices = Vec::with_capacity(selected_idxs.len());

    //     for selected_idx in selected_idxs {
    //         let valid_idx = selected_idx % self.devices.len();
    //         valid_devices.push(self.devices[valid_idx].clone());
    //     }

    //     valid_devices
    // }

    /// Returns a device by its ordinal count within this context.
    ///
    /// Round-robins (%) to the next valid device.
    ///
    pub fn get_device_by_index(&self, index: usize) -> Device {
        // [FIXME]: FIGURE OUT HOW TO DO THIS CORRECTLY
        let indices = [index; 1];
        self.resolve_device_idxs(&indices)[0].clone()
    }

    /// Returns info about the platform associated with the context.
    pub fn platform_info(&self, info_kind: PlatformInfo) -> PlatformInfoResult {
        // match core::get_platform_info(self.platform_id_core.clone(), info_kind) {
        match core::get_platform_info(self.platform().clone(), info_kind) {
            Ok(pi) => pi,
            Err(err) => PlatformInfoResult::Error(Box::new(err)),
        }
    }

    /// Returns info about a device associated with the context.
    pub fn device_info(&self, index: usize, info_kind: DeviceInfo) -> DeviceInfoResult {
        let device = match self.devices.get(index) {
            Some(d) => d,
            None => {
                return DeviceInfoResult::Error(Box::new(
                    OclError::new("Context::device_info: Invalid device index")));
            },
        };

        match core::get_device_info(device, info_kind) {
            Ok(pi) => pi,
            Err(err) => DeviceInfoResult::Error(Box::new(err)),
        }
    }

    /// Returns info about the context. 
    pub fn info(&self, info_kind: ContextInfo) -> ContextInfoResult {
        match core::get_context_info(&self.obj_core, info_kind) {
            Ok(pi) => pi,
            Err(err) => ContextInfoResult::Error(Box::new(err)),
        }
    }

    /// Returns a string containing a formatted list of context properties.
    pub fn to_string(&self) -> String {
        String::new()
    }

    /// Returns the current context as a `*mut libc::c_void`.
    pub fn core_as_ref(&self) -> &ContextCore {
        &self.obj_core
    }

    /// Returns a list of `*mut libc::c_void` corresponding to devices valid for use in this context.
    pub fn devices(&self) -> &[Device] {
        &self.devices[..]
    }

    /// Returns the platform our context is associated with.
    pub fn platform(&self) -> &Option<Platform> {
        &self.platform
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
