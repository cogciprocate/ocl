//! An OpenCL context.

// TEMPORARY:
#![allow(dead_code)]

// use formatting::MT;
use std;
use core::{self, Context as ContextCore, ContextProperties, ContextInfo, ContextInfoResult, PlatformId as PlatformIdCore, DeviceId as DeviceIdCore, DeviceType, DeviceInfo, DeviceInfoResult, PlatformInfo, PlatformInfoResult, CreateContextCallbackFn, UserDataPtr};
use error::{Result as OclResult, Error as OclError};
use standard::{self, Platform, DeviceSpecifier};


/// A context for a particular platform and set of device types.
///
/// Thread safety and destruction for any enclosed pointers are all handled automatically. 
/// Clone, store, and share between threads to your hearts content.
///
#[derive(Debug, Clone)]
pub struct Context {
    platform_id_core: Option<PlatformIdCore>,
    device_id_core_list: Vec<DeviceIdCore>,
    obj_core: ContextCore,
}

impl Context {
    /// [UNTESTED] Returns a newly created context.
    ///
    /// [FIXME]: Yeah... documentation.
    ///
    /// Use `Context::builder()...` instead of this method unless you know what you're doing. Please also contact us or file an issue immediately if you do, in fact, know what you're doing so that you can be added to the development team immediately as the one who does.
    ///
    /// # Panics
    ///
    /// [TEMPORARY] Passing a `Some` value for `pfn_notify` or `user_data` is 
    /// not yet supported.
    ///
    /// Any device indexes within `DeviceSpecifier` must be within the range of
    /// the number of devices for the platform specified in `properties`. If no
    /// platform has been specified, this behaviour is undefined and may be
    /// using an unintended platform. See `ContextProperties` for how to
    /// correctly specify the platform.
    pub fn new(properties: Option<ContextProperties>, device_spec: Option<DeviceSpecifier>, 
                pfn_notify: Option<CreateContextCallbackFn>, user_data: Option<UserDataPtr>
                ) -> OclResult<Context> {
        assert!(pfn_notify.is_none() && user_data.is_none(), 
            "Context creation callbacks not yet implemented.");

        let platform: Option<PlatformIdCore> = match properties {
            Some(ref props) => props.get_platform().clone(),
            None => None,
        };

        let device_spec = match device_spec {
            Some(ds) => ds,
            None => DeviceSpecifier::All,
        };

        let device_list = try!(device_spec.to_device_list(platform.clone()));

        let obj_core = try!(core::create_context(&properties, &device_list, pfn_notify, user_data));

        Ok(Context {
            platform_id_core: platform,
            device_id_core_list: device_list,
            obj_core: obj_core,
        })
    }


    /// [UNTESTED] Returns a newly created context.
    pub fn new_by_platform_and_device_list<D: Into<DeviceIdCore>>(platform: Platform, 
                    device_list: Vec<D>) -> OclResult<Context> {
        let devices: Vec<DeviceIdCore> = device_list.into_iter().map(|d| d.into()).collect();
        let properties: Option<ContextProperties> = 
            Some(ContextProperties::new().platform::<PlatformIdCore>(platform.clone().into()));
        let obj_core = try!(core::create_context(&properties, &devices, None, None));

        Ok(Context {
            platform_id_core: Some(platform.into()),
            device_id_core_list: devices,
            obj_core: obj_core,
        })
    }


    /// [FIXME: Docs] Returns a newly created context with a specified platform and set of device types.
    /// 
    /// [FIXME: Needs update]
    ///
    /// The desired platform may be specified by passing a valid index from a list 
    /// obtainable from the ocl::get_platform_ids() function and wrapping it with a 
    /// Some (ex. `Some(2)`). Pass `None` to use the first platform available (0). 
    /// 
    /// The device types mask may be specified using a union of any or all of the 
    /// following flags:
    /// - `CL_DEVICE_TYPE_DEFAULT`: The default OpenCL device in the system.
    /// - `CL_DEVICE_TYPE_CPU`: An OpenCL device that is the host processor. The host processor runs the OpenCL implementations and is a single or multi-core CPU.
    /// - `CL_DEVICE_TYPE_GPU`: An OpenCL device that is a GPU. By this we mean that the device can also be used to accelerate a 3D API such as OpenGL or DirectX.
    /// - `CL_DEVICE_TYPE_ACCELERATOR`: Dedicated OpenCL accelerators (for example the IBM CELL Blade). These devices communicate with the host processor using a peripheral interconnect such as PCIe.
    /// - `CL_DEVICE_TYPE_ALL`: A union of all flags.
    ///
    /// Passing `None` will use the flag: `CL_DEVICE_TYPE_GPU`.
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
        let platforms: Vec<PlatformIdCore> = try!(core::get_platform_ids());
        if platforms.len() == 0 { return OclError::err("\nNo OpenCL platforms found!\n"); }

        // println!("Platform list: {:?}", platforms);

        let platform_id_core: PlatformIdCore = match platform_idx_opt {
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

        let properties = Some(ContextProperties::new().platform(platform_id_core.clone()));
        // let properties = None;
        
        let device_id_core_list: Vec<DeviceIdCore> = try!(core::get_device_ids(Some(platform_id_core.clone()), 
            device_types_opt));
        if device_id_core_list.len() == 0 { return OclError::err("\nNo OpenCL devices found!\n"); }

        // println!("# # # # # #  OCL::CONTEXT::NEW(): device list: {:?}", device_id_core_list);

        // [FIXME]: No callback or user data:
        let obj_core = try!(core::create_context(&properties, &device_id_core_list, None, None));

        // [DEBUG]: 
        // println!("CONTEXT::NEW: CONTEXT: {:?}", obj_core);

        Ok(Context {
            platform_id_core: Some(platform_id_core),
            device_id_core_list: device_id_core_list,
            obj_core: obj_core,
        })
    }

    /// Resolves the zero-based device index into a DeviceIdCore (pointer).
    pub fn resolve_device_idxs(&self, device_idxs: &[usize]) -> Vec<DeviceIdCore> {
        match device_idxs.len() {
            0 => vec![self.device_id_core_list[core::DEFAULT_DEVICE_IDX].clone()],
            _ => self.valid_device_ids(&device_idxs),
        }
    }

    /// Returns a list of valid devices regardless of whether or not the indexes 
    /// passed are valid by performing a modulo operation on them and letting them
    /// wrap around (round robin).
    pub fn valid_device_ids(&self, selected_idxs: &[usize]) -> Vec<DeviceIdCore> {
        let mut valid_device_ids = Vec::with_capacity(selected_idxs.len());

        for selected_idx in selected_idxs {
            let valid_idx = selected_idx % self.device_id_core_list.len();
            valid_device_ids.push(self.device_id_core_list[valid_idx].clone());
        }

        valid_device_ids
    }

    /// Returns info about the platform associated with the context.
    pub fn platform_info(&self, info_kind: PlatformInfo) -> PlatformInfoResult {
        match core::get_platform_info(self.platform_id_core.clone(), info_kind) {
            Ok(pi) => pi,
            Err(err) => PlatformInfoResult::Error(Box::new(err)),
        }
    }

    /// Returns info about a device associated with the context.
    pub fn device_info(&self, index: usize, info_kind: DeviceInfo) -> DeviceInfoResult {
        let device = match self.device_id_core_list.get(index) {
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
        // self.clone().into()
        String::new()
    }

    /// Returns the current context as a `*mut libc::c_void`.
    pub fn core_as_ref(&self) -> &ContextCore {
        &self.obj_core
    }

    /// Returns a list of `*mut libc::c_void` corresponding to devices valid for use in this context.
    ///
    /// TODO: Rethink what this should return.
    pub fn devices_core_as_ref(&self) -> &[DeviceIdCore] {
        &self.device_id_core_list[..]
    }

    /// Returns the platform our context pertains to.
    ///
    /// TODO: Rethink what this should return.
    pub fn platform_core_as_ref(&self) -> Option<PlatformIdCore> {
        self.platform_id_core.clone()
    }  
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
            ("\n", "\n", "\n")
        } else {
            ("{ ", ", ", " }")
        };

        write!(f, "[Context]:{b}\
                Reference Count: {}{d}\
                Devices: {}{d}\
                Properties: {}{d}\
                Device Count: {}{e}\
            ",
            core::get_context_info(&self.obj_core, ContextInfo::ReferenceCount).unwrap(),
            core::get_context_info(&self.obj_core, ContextInfo::Devices).unwrap(),
            core::get_context_info(&self.obj_core, ContextInfo::Properties).unwrap(),
            core::get_context_info(&self.obj_core, ContextInfo::NumDevices).unwrap(),
            b = begin,
            d = delim,
            e = end,
        )
    }
}
