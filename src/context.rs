//! An OpenCL context.

// use formatting::MT;
use raw;
use cl_h::{self, cl_platform_id, cl_device_id, cl_device_type, cl_context};
use super::{Result as OclResult, Error as OclError};

// // cl_device_type - bitfield 
// pub const CL_DEVICE_TYPE_DEFAULT:                      cl_bitfield = 1 << 0;
// pub const CL_DEVICE_TYPE_CPU:                          cl_bitfield = 1 << 1;
// pub const CL_DEVICE_TYPE_GPU:                          cl_bitfield = 1 << 2;
// pub const CL_DEVICE_TYPE_ACCELERATOR:                  cl_bitfield = 1 << 3;
// pub const CL_DEVICE_TYPE_CUSTOM:                       cl_bitfield = 1 << 4;
// pub const CL_DEVICE_TYPE_ALL:                          cl_bitfield = 0xFFFFFFFF;
#[derive(Clone, Copy)]
pub enum DeviceType {
    Default = cl_h::CL_DEVICE_TYPE_DEFAULT as isize,
    Cpu = cl_h::CL_DEVICE_TYPE_CPU as isize,
    Gpu = cl_h::CL_DEVICE_TYPE_GPU as isize,
    Accelerator = cl_h::CL_DEVICE_TYPE_ACCELERATOR as isize,
    Custom = cl_h::CL_DEVICE_TYPE_CUSTOM as isize,
    All = cl_h::CL_DEVICE_TYPE_ALL as isize,
}

impl DeviceType {
    pub fn as_raw(&self) -> u64 {
        self.clone() as u64
    }
}

/// An OpenCL context for a particular platform and set of device types.
///
/// Wraps a 'cl_context' such as that returned by 'clCreateContext'.
///
/// # Destruction
/// `::release()` must be manually called by consumer.
///
pub struct Context {
    platform_opt: Option<cl_platform_id>,
    device_ids: Vec<cl_device_id>,
    context_obj: cl_context,
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
    pub fn new(platform_idx_opt: Option<usize>, device_types_opt: Option<cl_device_type>) 
            -> OclResult<Context>
    {
        let platforms = raw::get_platform_ids();
        if platforms.len() == 0 { return OclError::err("\nNo OpenCL platforms found!\n"); }

        let platform = match platform_idx_opt {
            Some(pf_idx) => {
                match platforms.get(pf_idx) {
                    Some(&pf) => pf,
                    None => return OclError::err("Invalid OpenCL platform index specified. \
                        Use 'get_platform_ids()' for a list."),
                }               
            },

            None => platforms[super::DEFAULT_PLATFORM],
        };
        
        let device_ids: Vec<cl_device_id> = raw::get_device_ids(platform, device_types_opt);
        if device_ids.len() == 0 { return OclError::err("\nNo OpenCL devices found!\n"); }

        // println!("# # # # # #  OCL::CONTEXT::NEW(): device list: {:?}", device_ids);

        let context_obj: cl_context = raw::create_context(&device_ids);

        Ok(Context {
            platform_opt: Some(platform),
            device_ids: device_ids,
            context_obj: context_obj,
        })
    }

    /// Resolves the zero-based device index into a cl_device_id (pointer).
    pub fn resolve_device_idxs(&self, device_idxs: &[usize]) -> Vec<cl_device_id> {
        match device_idxs.len() {
            0 => vec![self.device_ids()[super::DEFAULT_DEVICE]],
            _ => self.valid_device_ids(&device_idxs),
        }
    }

    /// Returns a list of valid devices regardless of whether or not the indexes 
    /// passed are valid by performing a modulo operation on them and letting them
    /// wrap around (round robin).
    pub fn valid_device_ids(&self, selected_idxs: &[usize]) -> Vec<cl_device_id> {
        let mut valid_device_ids = Vec::with_capacity(selected_idxs.len());

        for selected_idx in selected_idxs {
            let valid_idx = selected_idx % self.device_ids.len();
            valid_device_ids.push(self.device_ids[valid_idx]);
        }

        valid_device_ids
    }

    /// Returns the current context as a `*mut libc::c_void`.
    pub fn context_obj(&self) -> cl_context {
        self.context_obj
    }

    /// Returns a list of `*mut libc::c_void` corresponding to devices valid for use in this context.
    pub fn device_ids(&self) -> &Vec<cl_device_id> {
        &self.device_ids
    }

    /// Returns the platform our context pertains to.
    pub fn platform(&self) -> Option<cl_platform_id> {
        self.platform_opt
    }   

    /// Releases the current context.
    pub fn release(&mut self) {     
        unsafe {
            cl_h::clReleaseContext(self.context_obj);
        }
    }
}

