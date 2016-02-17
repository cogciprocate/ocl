//! An OpenCL context.

// use formatting::MT;
use raw::{self, ContextRaw, PlatformIdRaw, DeviceIdRaw, DeviceType};
use error::{Result as OclResult, Error as OclError};


/// An OpenCL context for a particular platform and set of device types.
///
/// Wraps a `ContextRaw` such as that returned by `raw::create_context`.
///
/// # Destruction
/// `::release()` must be manually called by consumer.
///
pub struct Context {
    platform_obj_raw: PlatformIdRaw,
    device_ids: Vec<DeviceIdRaw>,
    obj_raw: ContextRaw,
}

impl Context {
    /// Constructs a new `Context` within a specified platform and set of device types.
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
    ///     let ocl_context = ocl::Context::new(None, None);
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
    ///     let ocl_context = ocl::Context::new(Some(0), Some(device_types));
    ///     
    ///     // ...
    /// }
    /// ``` 
    ///
    /// # Panics
    ///    - `get_device_ids()` (work in progress)
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
    /// - Handle context callbacks.
    ///
    pub fn new(platform_idx_opt: Option<usize>, device_types_opt: Option<DeviceType>) 
            -> OclResult<Context>
    {
        let platforms: Vec<PlatformIdRaw> = try!(raw::get_platform_ids());
        if platforms.len() == 0 { return OclError::err("\nNo OpenCL platforms found!\n"); }

        // let platform = match platform_idx_opt {
        //     Some(pf_idx) => {
        //         match platforms.get(pf_idx) {
        //             Some(pf) => pf.clone(),
        //             None => return OclError::err("Invalid OpenCL platform index specified. \
        //                 Use 'get_platform_ids()' for a list."),
        //         }               
        //     },

        //     None => PlatformIdRaw::null(),
        // };

        let platform_obj_raw = match platform_idx_opt {
            Some(pf_idx) => {
                match platforms.get(pf_idx) {
                    Some(pf) => pf.clone(),
                    None => {
                        return OclError::err("Invalid OpenCL platform index specified. \
                            Use 'get_platform_ids()' for a list.")
                    },
                }
            },
            None => {
                debug_assert!(platforms.len() > 0, "Context::new(): Internal indexing error.");
                platforms[raw::DEFAULT_PLATFORM_IDX].clone()
            },
        };
        

        
        let device_ids: Vec<DeviceIdRaw> = try!(raw::get_device_ids(platform_obj_raw.clone(), 
            device_types_opt));
        if device_ids.len() == 0 { return OclError::err("\nNo OpenCL devices found!\n"); }

        // println!("# # # # # #  OCL::CONTEXT::NEW(): device list: {:?}", device_ids);

        let obj_raw = try!(raw::create_context(&device_ids));

        Ok(Context {
            platform_obj_raw: platform_obj_raw,
            device_ids: device_ids,
            obj_raw: obj_raw,
        })
    }

    /// Resolves the zero-based device index into a DeviceIdRaw (pointer).
    pub fn resolve_device_idxs(&self, device_idxs: &[usize]) -> Vec<DeviceIdRaw> {
        match device_idxs.len() {
            0 => vec![self.device_ids()[raw::DEFAULT_DEVICE_IDX]],
            _ => self.valid_device_ids(&device_idxs),
        }
    }

    /// Returns a list of valid devices regardless of whether or not the indexes 
    /// passed are valid by performing a modulo operation on them and letting them
    /// wrap around (round robin).
    pub fn valid_device_ids(&self, selected_idxs: &[usize]) -> Vec<DeviceIdRaw> {
        let mut valid_device_ids = Vec::with_capacity(selected_idxs.len());

        for selected_idx in selected_idxs {
            let valid_idx = selected_idx % self.device_ids.len();
            valid_device_ids.push(self.device_ids[valid_idx]);
        }

        valid_device_ids
    }

    /// Returns the current context as a `*mut libc::c_void`.
    pub fn obj_raw(&self) -> ContextRaw {
        self.obj_raw
    }

    /// Returns a list of `*mut libc::c_void` corresponding to devices valid for use in this context.
    pub fn device_ids(&self) -> &Vec<DeviceIdRaw> {
        &self.device_ids
    }

    /// Returns the platform our context pertains to.
    pub fn platform_obj_raw(&self) -> PlatformIdRaw {
        self.platform_obj_raw.clone()
    }   

    /// Releases the current context.
    pub fn release(&mut self) {     
        raw::release_context(self.obj_raw);
    }
}

