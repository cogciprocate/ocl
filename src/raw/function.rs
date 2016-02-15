use std::ptr;
use std::mem;
use std::io::Read;
use std::ffi::CString;
use std::iter;
use libc::{size_t, c_void};
use num::{FromPrimitive};

use cl_h::{self, cl_int, cl_uint, cl_platform_id, cl_device_id, cl_device_type, cl_device_info, 
    cl_context_info, cl_platform_info, cl_image_format, cl_image_desc, cl_mem_flags, cl_kernel,
    cl_program_build_info, cl_mem, cl_event, ClStatus};

use error::{Error as OclError, Result as OclResult};
use raw::{self, DEVICES_MAX, PlatformIdRaw, DeviceIdRaw, ContextRaw, MemFlags, 
    CommandQueueRaw, MemRaw, ProgramRaw, KernelRaw, EventRaw, SamplerRaw, KernelArg, DeviceType,
    ImageFormat, ImageDescriptor};

//=============================================================================
//=============================================================================
//============================ SUPPORT FUNCTIONS ==============================
//=============================================================================
//=============================================================================

/// Converts the `cl_int` errcode into a string containing the associated 
/// constant name.
fn errcode_string(errcode: cl_int) -> String {
    match ClStatus::from_i32(errcode) {
        Some(cls) => format!("{:?}", cls),
        None => format!("[Unknown Error Code: {}]", errcode as i64),
    }
}


/// Evaluates `errcode` and returns an `Err` if it is not 0.
fn errcode_try(message: &str, errcode: cl_int) -> OclResult<()> {
    if errcode != cl_h::ClStatus::CL_SUCCESS as cl_int {
        OclError::err(format!("\n\nOPENCL ERROR: {} failed with code: {}\n\n", 
            message, errcode_string(errcode)))
    } else {
        Ok(())
    }
}


/// Evaluates `errcode` and panics with a failure message if it is not 0.
fn errcode_assert(message: &str, errcode: cl_int) {
    errcode_try(message, errcode).unwrap();
}


/// Maps options of slices to pointers and a length.
fn resolve_queue_opts(wait_list: Option<&[EventRaw]>, new_event: Option<&mut EventRaw>)
        -> OclResult<(cl_uint, *const cl_event, *mut cl_event)>
{
    // If the wait list is empty or if its containing option is none, map to (0, null),
    // otherwise map to the length and pointer (driver doesn't want an empty list):    
    let (wait_list_len, wait_list_ptr) = match wait_list {
        Some(wl) => {
            if wl.len() > 0 {
                (wl.len() as cl_uint, wl.as_ptr() as *const cl_event)
            } else {
                (0, ptr::null_mut() as *const cl_event)
            }
        },
        None => (0, ptr::null_mut() as *const cl_event),
    };

    let new_event_ptr = match new_event {
        Some(ne) => ne as *mut _ as *mut cl_event,
        None => ptr::null_mut() as *mut cl_event,
    };

    Ok((wait_list_len, wait_list_ptr, new_event_ptr))
}


/// Converts an array option reference into a pointer to the contained array.
fn resolve_work_dims(work_dims: &Option<[usize; 3]>) -> *const size_t {
    match work_dims {
        &Some(ref w) => w as *const [usize; 3] as *const size_t,
        &None => 0 as *const size_t,
    }
}

//=============================================================================
//=============================================================================
//============================ OPENCL FUNCTIONS ===============================
//=============================================================================
//=============================================================================

//=============================================================================
//============================= Platform API ==================================
//=============================================================================

/// Returns a list of available platforms as 'raw' objects.
// TODO: Get rid of manual vec allocation now that PlatformIdRaw implements Clone.
pub fn get_platform_ids() -> Vec<PlatformIdRaw> {
    let mut num_platforms = 0 as cl_uint;
    
    // Get a count of available platforms:
    let mut errcode: cl_int = unsafe { 
        cl_h::clGetPlatformIDs(0, ptr::null_mut(), &mut num_platforms) 
    };
    errcode_assert("clGetPlatformIDs()", errcode);

    // Create a vec with the appropriate size:
    let mut null_vec: Vec<usize> = iter::repeat(0).take(num_platforms as usize).collect();
    let (ptr, cap, len) = (null_vec.as_mut_ptr(), null_vec.len(), null_vec.capacity());

    // Steal the vec's soul:
    let mut platforms: Vec<PlatformIdRaw> = unsafe {
        mem::forget(null_vec);
        Vec::from_raw_parts(ptr as *mut PlatformIdRaw, len, cap)
    };

    errcode = unsafe {
        cl_h::clGetPlatformIDs(
            num_platforms, 
            platforms.as_mut_ptr() as *mut cl_platform_id, 
            ptr::null_mut()
        )
    };
    errcode_assert("clGetPlatformIDs()", errcode);
    
    platforms
}




    // pub fn clGetPlatformInfo(platform: cl_platform_id,
    //                              param_name: cl_platform_info,
    //                              param_value_size: size_t,
    //                              param_value: *mut c_void,
    //                              param_value_size_ret: *mut size_t) -> cl_int;





//=============================================================================
//============================== Device APIs  =================================
//=============================================================================

/// Returns a list of available devices for a particular platform by id.
///
/// # Panics
///
///  -errcode_assert(): [FIXME]: Explaination needed (possibly at crate level?)
pub fn get_device_ids(
        platform: PlatformIdRaw, 
        // device_types_opt: Option<cl_device_type>)
        device_types_opt: Option<DeviceType>)
        -> Vec<DeviceIdRaw> 
{
    let device_type = device_types_opt.unwrap_or(raw::DEVICE_TYPE_DEFAULT);
    let mut devices_available: cl_uint = 0;

    let mut device_ids: Vec<DeviceIdRaw> = iter::repeat(DeviceIdRaw::null())
        .take(DEVICES_MAX as usize).collect();

    let errcode = unsafe { cl_h::clGetDeviceIDs(
            platform.as_ptr(), 
            device_type.bits() as cl_device_type,
            DEVICES_MAX, 
            device_ids.as_mut_ptr() as *mut cl_device_id,
            &mut devices_available,
    )};
    errcode_assert("clGetDeviceIDs()", errcode);

    // Trim vec len:
    unsafe { device_ids.set_len(devices_available as usize); }

    device_ids
}



    // pub fn clGetDeviceInfo(device: cl_device_id,
    //                    param_name: cl_device_info,
    //                    param_value_size: size_t,
    //                    param_value: *mut c_void,
    //                    param_value_size_ret: *mut size_t) -> cl_int;



    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clCreateSubDevices(cl_device_id                         /* in_device */,
    //                    const cl_device_partition_property * /* properties */,
    //                    cl_uint                              /* num_devices */,
    //                    cl_device_id *                       /* out_devices */,
    //                    cl_uint *                            /* num_devices_ret */) CL_API_SUFFIX__VERSION_1_2;



    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clRetainDevice(cl_device_id /* device */) CL_API_SUFFIX__VERSION_1_2;


    
    //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clReleaseDevice(cl_device_id /* device */) CL_API_SUFFIX__VERSION_1_2;



//=============================================================================
//============================== Context APIs  ================================
//=============================================================================


/// Returns a new context pointer valid for all devices in `device_ids`.
pub fn create_context(device_ids: &Vec<DeviceIdRaw>) -> ContextRaw {
    let mut errcode: cl_int = 0;

    let context = ContextRaw::new( unsafe { cl_h::clCreateContext(
            ptr::null(), 
            device_ids.len() as cl_uint, 
            device_ids.as_ptr()  as *const cl_device_id,
            mem::transmute(ptr::null::<fn()>()), 
            ptr::null_mut(), 
            &mut errcode)
    });
    errcode_assert("clCreateContext()", errcode);
    context
}



    // pub fn clCreateContextFromType(properties: *mut cl_context_properties,
    //                            device_type: cl_device_type,
    //                            pfn_notify: extern fn (*mut c_char, *mut c_void, size_t, *mut c_void),
    //                            user_data: *mut c_void,
    //                            errcode_ret: *mut cl_int) -> cl_context;



    // pub fn clRetainContext(context: cl_context) -> cl_int;




pub fn release_context(context: ContextRaw) {
    errcode_assert("clReleaseContext", unsafe {
        cl_h::clReleaseContext(context.as_ptr())
    });
}



    // pub fn clGetContextInfo(context: cl_context,
    //                     param_name: cl_context_info,
    //                     param_value_size: size_t,
    //                     param_value: *mut c_void,
    //                     param_value_size_ret: *mut size_t) -> cl_int;




//=============================================================================
//=========================== Command Queue APIs ==============================
//=============================================================================

/// Returns a new command queue pointer.
pub fn create_command_queue(
            context: ContextRaw, 
            device: DeviceIdRaw)
            -> OclResult<CommandQueueRaw>
{
    // Verify that the context is valid:
    try!(verify_context(context));

    let mut errcode: cl_int = 0;

    let cq = CommandQueueRaw::new(unsafe { cl_h::clCreateCommandQueue(
            context.as_ptr(), 
            device.as_ptr(),
            cl_h::CL_QUEUE_PROFILING_ENABLE, 
            &mut errcode
    )});
    errcode_assert("clCreateCommandQueue()", errcode);
    Ok(cq)
}




    // pub fn clRetainCommandQueue(command_queue: cl_command_queue) -> cl_int;




pub fn release_command_queue(queue: CommandQueueRaw) {
    errcode_assert("clReleaseCommandQueue", unsafe {
        cl_h::clReleaseCommandQueue(queue.as_ptr())
    });
}



    // pub fn clGetCommandQueueInfo(command_queue: cl_command_queue,
    //                          param_name: cl_command_queue_info,
    //                          param_value_size: size_t,
    //                          param_value: *mut c_void,
    //                          param_value_size_ret: *mut size_t) -> cl_int;




//=============================================================================
//=========================== Memory Object APIs ==============================
//=============================================================================

/// Returns a new buffer pointer with size (bytes): `len` * sizeof(T).
pub fn create_buffer<T>(
            context: ContextRaw,
            flags: MemFlags,
            len: usize,
            data: Option<&[T]>)
            -> OclResult<MemRaw>
{
    // Verify that the context is valid:
    try!(verify_context(context));

    let mut errcode: cl_int = 0;

    let host_ptr = match data {
        Some(d) => {
            assert!(d.len() == len, "ocl::create_buffer(): Data length mismatch.");
            d.as_ptr() as cl_mem
        },
        None => ptr::null_mut(),
    };

    let buf = MemRaw::new(unsafe { cl_h::clCreateBuffer(
            context.as_ptr(), 
            flags.bits() as cl_mem_flags,
            len * mem::size_of::<T>(),
            host_ptr, 
            &mut errcode,
    )});
    errcode_assert("create_buffer", errcode);

    Ok(buf)
}




    // pub fn clCreateSubBuffer(buffer: cl_mem,
    //                     flags: cl_mem_flags,
    //                     buffer_create_type: cl_buffer_create_type,
    //                     buffer_create_info: *mut c_void,
    //                     errcode_ret: *mut cl_int) -> cl_mem;




/// Returns a new image (mem) pointer.
// [WORK IN PROGRESS]
pub fn create_image<T>(
            context: ContextRaw,
            flags: MemFlags,
            // format: &cl_image_format,
            // desc: &cl_image_desc,
            format: ImageFormat,
            desc: ImageDescriptor,
            data: Option<&[T]>)
            -> OclResult<MemRaw>
{
    // Verify that the context is valid:
    try!(verify_context(context));

    let mut errcode: cl_int = 0;
    
    let data_ptr = match data {
        Some(d) => {
            // [FIXME]: CALCULATE CORRECT IMAGE SIZE AND COMPARE
            // assert!(d.len() == len, "ocl::create_image(): Data length mismatch.");
            d.as_ptr() as cl_mem
        },
        None => ptr::null_mut(),
    };

    let image_ptr = unsafe { cl_h::clCreateImage(
            context.as_ptr(),
            flags.bits() as cl_mem_flags,
            &format.as_raw() as *const cl_image_format,
            &desc.as_raw() as *const cl_image_desc,
            data_ptr,
            &mut errcode as *mut cl_int)
    }; 
    errcode_assert("create_image", errcode);

    assert!(!image_ptr.is_null());

    Ok(MemRaw::new(image_ptr))
}




    // pub fn clRetainMemObject(memobj: cl_mem) -> cl_int;





pub fn release_mem_object(mem: MemRaw) {
    errcode_assert("clReleaseMemObject", unsafe {
        cl_h::clReleaseMemObject(mem.as_ptr())
    });
}



    // pub fn clGetSupportedImageFormats(context: cl_context,
    //                               flags: cl_mem_flags,
    //                               image_type: cl_mem_object_type,
    //                               num_entries: cl_uint,
    //                               image_formats: *mut cl_image_format,
    //                               num_image_formats: *mut cl_uint) -> cl_int;



    // pub fn clGetMemObjectInfo(memobj: cl_mem,
    //                       param_name: cl_mem_info,
    //                       param_value_size: size_t,
    //                       param_value: *mut c_void,
    //                       param_value_size_ret: *mut size_t) -> cl_int;



    // pub fn clGetImageInfo(image: cl_mem,
    //                   param_name: cl_image_info,
    //                   param_value_size: size_t,
    //                   param_value: *mut c_void,
    //                   param_value_size_ret: *mut size_t) -> cl_int;



    // pub fn clSetMemObjectDestructorCallback(memobj: cl_mem,
    //                                     pfn_notify: extern fn (cl_mem, *mut c_void),
    //                                     user_data: *mut c_void) -> cl_int;



//=============================================================================
//============================== Sampler APIs =================================
//=============================================================================

    // pub fn clCreateSampler(context: cl_context,
    //                    normalize_coords: cl_bool,
    //                    addressing_mode: cl_addressing_mode,
    //                    filter_mode: cl_filter_mode,
    //                    errcode_ret: *mut cl_int) -> cl_sampler;



    // pub fn clRetainSampler(sampler: cl_sampler) -> cl_int;



    // pub fn clReleaseSampler(sampler: cl_sampler) ->cl_int;



    // pub fn clGetSamplerInfo(sampler: cl_sampler,
    //                     param_name: cl_sampler_info,
    //                     param_value_size: size_t,
    //                     param_value: *mut c_void,
    //                     param_value_size_ret: *mut size_t) -> cl_int;




//=============================================================================
//=========================== Program Object APIs =============================
//=============================================================================



    // pub fn clCreateProgramWithSource(context: cl_context,
    //                              count: cl_uint,
    //                              strings: *const *const c_char,
    //                              lengths: *const size_t,
    //                              errcode_ret: *mut cl_int) -> cl_program;




    // pub fn clCreateProgramWithBinary(context: cl_context,
    //                              num_devices: cl_uint,
    //                              device_list: *const cl_device_id,
    //                              lengths: *const size_t,
    //                              binaries: *const *const c_uchar,
    //                              binary_status: *mut cl_int,
    //                              errcode_ret: *mut cl_int) -> cl_program;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_program CL_API_CALL
    // clCreateProgramWithBuiltInKernels(cl_context            /* context */,
    //                                  cl_uint               /* num_devices */,
    //                                  const cl_device_id *  /* device_list */,
    //                                  const char *          /* kernel_names */,
    //                                  cl_int *              /* errcode_ret */) CL_API_SUFFIX__VERSION_1_2;



    // pub fn clRetainProgram(program: cl_program) -> cl_int;




pub fn release_program(program: ProgramRaw) {
    errcode_assert("clReleaseKernel", unsafe { 
        cl_h::clReleaseProgram(program.as_ptr())
    });
}



    // pub fn clBuildProgram(program: cl_program,
    //                   num_devices: cl_uint,
    //                   device_list: *const cl_device_id,
    //                   options: *const c_char,
    //                   pfn_notify: extern fn (cl_program, *mut c_void),
    //                   user_data: *mut c_void) -> cl_int;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clCompileProgram(cl_program           /* program */,
    //                 cl_uint              /* num_devices */,
    //                 const cl_device_id * /* device_list */,
    //                 const char *         /* options */, 
    //                 cl_uint              /* num_input_headers */,
    //                 const cl_program *   /* input_headers */,
    //                 const char **        /* header_include_names */,
    //                 void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
    //                 void *               /* user_data */) CL_API_SUFFIX__VERSION_1_2;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_program CL_API_CALL
    // clLinkProgram(cl_context           /* context */,
    //               cl_uint              /* num_devices */,
    //               const cl_device_id * /* device_list */,
    //               const char *         /* options */, 
    //               cl_uint              /* num_input_programs */,
    //               const cl_program *   /* input_programs */,
    //               void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
    //               void *               /* user_data */,
    //               cl_int *             /* errcode_ret */ ) CL_API_SUFFIX__VERSION_1_2;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clUnloadPlatformCompiler(cl_platform_id /* platform */) CL_API_SUFFIX__VERSION_1_2;




/// Creates, builds, and returns a new program pointer from `src_strings`.
///
/// TODO: Break out create and build parts separately.
pub fn create_build_program(
            src_strings: Vec<CString>,
            cmplr_opts: CString,
            context: ContextRaw, 
            device_ids: &Vec<DeviceIdRaw>)
            -> OclResult<ProgramRaw>
{
    // Verify that the context is valid:
    try!(verify_context(context));

    // Lengths (not including \0 terminator) of each string:
    let ks_lens: Vec<usize> = src_strings.iter().map(|cs| cs.as_bytes().len()).collect();  
    // Pointers to each string:
    let kern_string_ptrs: Vec<*const i8> = src_strings.iter().map(|cs| cs.as_ptr()).collect();

    let mut errcode: cl_int = 0;
    
    let program = ProgramRaw::new(unsafe { cl_h::clCreateProgramWithSource(
                context.as_ptr(), 
                kern_string_ptrs.len() as cl_uint,
                kern_string_ptrs.as_ptr() as *const *const i8,
                ks_lens.as_ptr() as *const usize,
                &mut errcode,
    )});
    errcode_assert("clCreateProgramWithSource()", errcode);

    errcode = unsafe { cl_h::clBuildProgram(
                program.as_ptr(),
                device_ids.len() as cl_uint,
                device_ids.as_ptr() as *const cl_device_id, 
                cmplr_opts.as_ptr() as *const i8,
                mem::transmute(ptr::null::<fn()>()), 
                ptr::null_mut(),
    )};  

    if errcode < 0 {
        program_build_err(program, device_ids).map(|_| program)
    } else {
        try!(errcode_try("clBuildProgram()", errcode));
        Ok(program) 
    }
}



    // pub fn clGetProgramInfo(program: cl_program,
    //                     param_name: cl_program_info,
    //                     param_value_size: size_t,
    //                     param_value: *mut c_void,
    //                     param_value_size_ret: *mut size_t) -> cl_int;



    // pub fn clGetProgramBuildInfo(program: cl_program,
    //                          device: cl_device_id,
    //                          param_name: cl_program_info,
    //                          param_value_size: size_t,
    //                          param_value: *mut c_void,
    //                          param_value_size_ret: *mut size_t) -> cl_int;


//=============================================================================
//=========================== Kernel Object APIs ==============================
//=============================================================================

/// Returns a new kernel pointer.
pub fn create_kernel(
            program: ProgramRaw, 
            name: &str)
            -> OclResult<KernelRaw>
{
    let mut err: cl_int = 0;

    let kernel_ptr = unsafe {
        cl_h::clCreateKernel(
            program.as_ptr(), 
            try!(CString::new(name.as_bytes())).as_ptr(), 
            &mut err,
        )
    };    
    let err_pre = format!("clCreateKernel('{}'):", &name);
    try!(errcode_try(&err_pre, err));
    Ok(KernelRaw::new(kernel_ptr))
}




    // pub fn clCreateKernelsInProgram(program: cl_program,
    //                             num_kernels: cl_uint,
    //                             kernels: *mut cl_kernel,
    //                             num_kernels_ret: *mut cl_uint) -> cl_int;



    // pub fn clRetainKernel(kernel: cl_kernel) -> cl_int;



pub fn release_kernel(kernel: KernelRaw) {
    errcode_assert("clReleaseKernel", unsafe {
        cl_h::clReleaseKernel(kernel.as_ptr())
    });
}


/// Modifies or creates a kernel argument.
///
/// `kernel_name` is for error reporting and is optional.
///
pub fn set_kernel_arg<T>(kernel: KernelRaw, arg_index: u32, arg: KernelArg<T>,
            kernel_name: Option<&str>) -> OclResult<()>
{
    let (arg_size, arg_value) = match arg {
        KernelArg::Mem(mem_obj) => {
            (mem::size_of::<MemRaw>() as size_t, 
            (&mem_obj.as_ptr() as *const *mut c_void) as *const c_void)
        },
        KernelArg::Sampler(smplr) => {
            (mem::size_of::<SamplerRaw>() as size_t, 
            (&smplr.as_ptr() as *const *mut c_void) as *const c_void)
        },
        KernelArg::Scalar(scalar) => {
            (mem::size_of::<T>() as size_t, 
            scalar as *const _ as *const c_void)
        },
        KernelArg::Vector(vector)=> {
            ((mem::size_of::<T>() * vector.len()) as size_t,
            vector as *const _ as *const c_void)
        },
        KernelArg::Local(length) => {
            ((mem::size_of::<T>() * length) as size_t,
            ptr::null())
        },
        KernelArg::Other { size, value } => (size, value),
    };

    let err = unsafe { cl_h::clSetKernelArg(
            kernel.as_ptr(), 
            arg_index,
            arg_size, 
            arg_value,
    )};
    let err_pre = format!("clSetKernelArg('{}'):", kernel_name.unwrap_or(""));
    errcode_try(&err_pre, err)
} 



    // pub fn clGetKernelInfo(kernel: cl_kernel,
    //                    param_name: cl_kernel_info,
    //                    param_value_size: size_t,
    //                    param_value: *mut c_void,
    //                    param_value_size_ret: *mut size_t) -> cl_int;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clGetKernelArgInfo(cl_kernel       /* kernel */,
    //                   cl_uint         /* arg_indx */,
    //                   cl_kernel_arg_info  /* param_name */,
    //                   size_t          /* param_value_size */,
    //                   void *          /* param_value */,
    //                   size_t *        /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_2;



    // pub fn clGetKernelWorkGroupInfo(kernel: cl_kernel,
    //                             device: cl_device_id,
    //                             param_name: cl_kernel_work_group_info,
    //                             param_value_size: size_t,
    //                             param_value: *mut c_void,
    //                             param_value_size_ret: *mut size_t) -> cl_int;



//=============================================================================
//=========================== Event Object APIs ===============================
//=============================================================================

pub fn wait_for_events(count: cl_uint, event_list: &[EventRaw]) {
    let errcode = unsafe {
        cl_h::clWaitForEvents(count, &(*event_list.as_ptr()).as_ptr())
    };

    errcode_assert("clWaitForEvents", errcode);
}




    // pub fn clGetEventInfo(event: cl_event,
    //                   param_name: cl_event_info,
    //                   param_value_size: size_t,
    //                   param_value: *mut c_void,
    //                   param_value_size_ret: *mut size_t) -> cl_int;



    // pub fn clCreateUserEvent(context: cl_context,
    //                      errcode_ret: *mut cl_int) -> cl_event;



    // pub fn clRetainEvent(event: cl_event) -> cl_int;




pub fn release_event(event: EventRaw) {
    errcode_assert("clReleaseEvent", unsafe {
        cl_h::clReleaseEvent(event.as_ptr())
    });
}




    // pub fn clSetUserEventStatus(event: cl_event,
    //                         execution_status: cl_int) -> cl_int;





pub unsafe fn set_event_callback(
            event: cl_event, 
            callback_trigger: cl_int, 
            callback_receiver: extern fn (cl_event, cl_int, *mut c_void),
            user_data: *mut c_void)
{
    let errcode = cl_h::clSetEventCallback(event, callback_trigger, 
        callback_receiver, user_data);

    errcode_assert("clSetEventCallback", errcode);
}




//=============================================================================
//============================= Profiling APIs ================================
//=============================================================================


    // pub fn clGetEventProfilingInfo(event: cl_event,
    //                            param_name: cl_profiling_info,
    //                            param_value_size: size_t,
    //                            param_value: *mut c_void,
    //                            param_value_size_ret: *mut size_t) -> cl_int;



//=============================================================================
//========================== Flush and Finish APIs ============================
//=============================================================================



    // pub fn clFlush(command_queue: cl_command_queue) -> cl_int;




pub fn finish(command_queue: CommandQueueRaw) {
    unsafe { 
        let errcode = cl_h::clFinish(command_queue.as_ptr());
        errcode_assert("clFinish()", errcode);
    }
}



//=============================================================================
//======================== Enqueued Commands APIs =============================
//=============================================================================


/// Enqueues a read from device memory referred to by `buffer` to device memory,
/// `data`.
///
/// # Safety
///
/// It's complicated. Short version: make sure the memory pointed to by the 
/// slice, `data`, doesn't get reallocated before `new_event` is complete.
///
/// [FIXME]: Add a proper explanation of all the ins and outs. 
///
/// [FIXME]: Return result
pub unsafe fn enqueue_read_buffer<T>(
            command_queue: CommandQueueRaw,
            buffer: &MemRaw, 
            block: bool,
            data: &[T],
            offset: usize,
            wait_list: Option<&[EventRaw]>, 
            new_event: Option<&mut EventRaw>,
        ) -> OclResult<()>
{
    let (wait_list_len, wait_list_ptr, new_event_ptr) = 
        resolve_queue_opts(wait_list, new_event).expect("[FIXME]: enqueue_read_buffer()");

    let errcode = cl_h::clEnqueueReadBuffer(
            command_queue.as_ptr(), 
            buffer.as_ptr(), 
            block as cl_uint, 
            offset, 
            (data.len() * mem::size_of::<T>()) as size_t, 
            data.as_ptr() as cl_mem, 
            wait_list_len,
            wait_list_ptr,
            new_event_ptr,
    );

    errcode_try("clEnqueueReadBuffer()", errcode)
}




    // pub fn clEnqueueReadBufferRect(command_queue: cl_command_queue,
    //                            buffer: cl_mem,
    //                            blocking_read: cl_bool,
    //                            buffer_origin: *mut size_t,
    //                            host_origin: *mut size_t,
    //                            region: *mut size_t,
    //                            buffer_slc_pitch: size_t,
    //                            buffer_slc_pitch: size_t,
    //                            host_slc_pitch: size_t,
    //                            host_slc_pitch: size_t,
    //                            ptr: *mut c_void,
    //                            num_events_in_wait_list: cl_uint,
    //                            event_wait_list: *const cl_event,
    //                            event: *mut cl_event) -> cl_int;




/// Enqueues a write from host memory, `data`, to device memory referred to by
/// `buffer`.
///
/// [FIXME]: Return result
pub fn enqueue_write_buffer<T>(
            command_queue: CommandQueueRaw,
            buffer: &MemRaw, 
            block: bool,
            data: &[T],
            offset: usize,
            wait_list: Option<&[EventRaw]>, 
            new_event: Option<&mut EventRaw>,
        ) -> OclResult<()>
{
    let (wait_list_len, wait_list_ptr, new_event_ptr) 
        = resolve_queue_opts(wait_list, new_event)
            .expect("[FIXME: Return result]: enqueue_write_buffer()");

    // let wait_list_len = match &wait_list {
    //     &Some(ref wl) => wl.len() as u32,
    //     &None => 0,
    // };

    unsafe {
        // let wait_list_ptr = wait_list as *const *mut c_void;
        // let new_event_ptr = new_event as *mut *mut c_void;

        let errcode = cl_h::clEnqueueWriteBuffer(
                    command_queue.as_ptr(),
                    buffer.as_ptr(),
                    block as cl_uint,
                    offset,
                    (data.len() * mem::size_of::<T>()) as size_t,
                    data.as_ptr() as cl_mem,
                    wait_list_len,
                    wait_list_ptr,
                    new_event_ptr,
        );

        errcode_try("clEnqueueWriteBuffer()", errcode)
    }
}




    // pub fn clEnqueueWriteBufferRect(command_queue: cl_command_queue,
    //                             blocking_write: cl_bool,
    //                             buffer_origin: *mut size_t,
    //                             host_origin: *mut size_t,
    //                             region: *mut size_t,
    //                             buffer_slc_pitch: size_t,
    //                             buffer_slc_pitch: size_t,
    //                             host_slc_pitch: size_t,
    //                             host_slc_pitch: size_t,
    //                             ptr: *mut c_void,
    //                             num_events_in_wait_list: cl_uint,
    //                             event_wait_list: *const cl_event,
    //                             event: *mut cl_event) -> cl_int;




/// [UNTESTED][UNUSED]
#[allow(dead_code)]
pub fn enqueue_copy_buffer(
                command_queue: CommandQueueRaw,
                src_buffer: cl_mem,
                dst_buffer: cl_mem,
                src_offset: usize,
                dst_offset: usize,
                len_copy_bytes: usize)
{
    unsafe {
        let errcode = cl_h::clEnqueueCopyBuffer(
            command_queue.as_ptr(),
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            len_copy_bytes as usize,
            0,
            ptr::null(),
            ptr::null_mut(),
        );
        errcode_assert("clEnqueueCopyBuffer()", errcode);
    }
}




    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueFillBuffer(cl_command_queue   /* command_queue */,
    //                 cl_mem             /* buffer */, 
    //                 const void *       /* pattern */, 
    //                 size_t             /* pattern_size */, 
    //                 size_t             /* offset */, 
    //                 size_t             /* size */, 
    //                 cl_uint            /* num_events_in_wait_list */, 
    //                 const cl_event *   /* event_wait_list */, 
    //                 cl_event *         /* event */) CL_API_SUFFIX__VERSION_1_2;



    // pub fn clEnqueueCopyBufferRect(command_queue: cl_command_queue,
    //                            src_buffer: cl_mem,
    //                            dst_buffer: cl_mem,
    //                            src_origin: *mut size_t,
    //                            dst_origin: *mut size_t,
    //                            region: *mut size_t,
    //                            src_slc_pitch: size_t,
    //                            src_slc_pitch: size_t,
    //                            dst_slc_pitch: size_t,
    //                            dst_slc_pitch: size_t,
    //                            num_events_in_wait_list: cl_uint,
    //                            event_wait_list: *const cl_event,
    //                            event: *mut cl_event) -> cl_int;



    // pub fn clEnqueueReadImage(command_queue: cl_command_queue,
    //                       image: cl_mem,
    //                       blocking_read: cl_bool,
    //                       origin: *mut size_t,
    //                       region: *mut size_t,
    //                       slc_pitch: size_t,
    //                       slc_pitch: size_t,
    //                       ptr: *mut c_void,
    //                       num_events_in_wait_list: cl_uint,
    //                       event_wait_list: *const cl_event,
    //                       event: *mut cl_event) -> cl_int;



    // pub fn clEnqueueWriteImage(command_queue: cl_command_queue,
    //                        image: cl_mem,
    //                        blocking_write: cl_bool,
    //                        origin: *mut size_t,
    //                        region: *mut size_t,
    //                        input_slc_pitch: size_t,
    //                        input_slc_pitch: size_t,
    //                        ptr: *mut c_void,
    //                        num_events_in_wait_list: cl_uint,
    //                        event_wait_list: *const cl_event,
    //                        event: *mut cl_event) -> cl_int;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueFillImage(cl_command_queue   /* command_queue */,
    //                   cl_mem             /* image */, 
    //                   const void *       /* fill_color */, 
    //                   const size_t *     /* origin[3] */, 
    //                   const size_t *     /* region[3] */, 
    //                   cl_uint            /* num_events_in_wait_list */, 
    //                   const cl_event *   /* event_wait_list */, 
    //                   cl_event *         /* event */) CL_API_SUFFIX__VERSION_1_2;



    // pub fn clEnqueueCopyImage(command_queue: cl_command_queue,
    //                       src_image: cl_mem,
    //                       dst_image: cl_mem,
    //                       src_origin: *mut size_t,
    //                       dst_origin: *mut size_t,
    //                       region: *mut size_t,
    //                       num_events_in_wait_list: cl_uint,
    //                       event_wait_list: *const cl_event,
    //                       event: *mut cl_event) -> cl_int;



    // pub fn clEnqueueCopyImageToBuffer(command_queue: cl_command_queue,
    //                               src_image: cl_mem,
    //                               dst_buffer: cl_mem,
    //                               src_origin: *mut size_t,
    //                               region: *mut size_t,
    //                               dst_offset: size_t,
    //                               num_events_in_wait_list: cl_uint,
    //                               event_wait_list: *const cl_event,
    //                               event: *mut cl_event) -> cl_int;



    // pub fn clEnqueueCopyBufferToImage(command_queue: cl_command_queue,
    //                               src_buffer: cl_mem,
    //                               dst_image: cl_mem,
    //                               src_offset: size_t,
    //                               dst_origin: *mut size_t,
    //                               region: *mut size_t,
    //                               num_events_in_wait_list: cl_uint,
    //                               event_wait_list: *const cl_event,
    //                               event: *mut cl_event) -> cl_int;



    // pub fn clEnqueueMapBuffer(command_queue: cl_command_queue,
    //                       buffer: cl_mem,
    //                       blocking_map: cl_bool,
    //                       map_flags: cl_map_flags,
    //                       offset: size_t,
    //                       cb: size_t,
    //                       num_events_in_wait_list: cl_uint,
    //                       event_wait_list: *const cl_event,
    //                       event: *mut cl_event,
    //                       errorcode_ret: *mut cl_int);



    // pub fn clEnqueueMapImage(command_queue: cl_command_queue,
    //                      image: cl_mem,
    //                      blocking_map: cl_bool,
    //                      map_flags: cl_map_flags,
    //                      origin: *mut size_t,
    //                      region: *mut size_t,
    //                      image_slc_pitch: size_t,
    //                      image_slc_pitch: size_t,
    //                      num_events_in_wait_list: cl_uint,
    //                      event_wait_list: *const cl_event,
    //                      event: *mut cl_event,
    //                      errorcode_ret: *mut cl_int);



    // pub fn clEnqueueUnmapMemObject(command_queue: cl_command_queue,
    //                            memobj: cl_mem,
    //                            mapped_ptr: *mut c_void,
    //                            num_events_in_wait_list: cl_uint,
    //                            event_wait_list: *const cl_event,
    //                            event: *mut cl_event) -> cl_int;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueMigrateMemObjects(cl_command_queue       /* command_queue */,
    //                           cl_uint                /* num_mem_objects */,
    //                           const cl_mem *         /* mem_objects */,
    //                           cl_mem_migration_flags /* flags */,
    //                           cl_uint                /* num_events_in_wait_list */,
    //                           const cl_event *       /* event_wait_list */,
    //                           cl_event *             /* event */) CL_API_SUFFIX__VERSION_1_2;




pub fn enqueue_kernel(
            command_queue: CommandQueueRaw,
            kernel: cl_kernel,
            work_dims: cl_uint,
            global_work_offset: Option<[usize; 3]>,
            global_work_dims: [usize; 3],
            local_work_dims: Option<[usize; 3]>,
            wait_list: Option<&[EventRaw]>, 
            new_event: Option<&mut EventRaw>,
            kernel_name: Option<&str>)
{
    let (wait_list_len, wait_list_ptr, new_event_ptr) = 
        resolve_queue_opts(wait_list, new_event).expect("[FIXME]: enqueue_kernel()");
    let gwo = resolve_work_dims(&global_work_offset);
    let gws = &global_work_dims as *const size_t;
    let lws = resolve_work_dims(&local_work_dims);

    unsafe {
        let errcode = cl_h::clEnqueueNDRangeKernel(
            command_queue.as_ptr(),
            kernel,
            work_dims,
            gwo,
            gws,
            lws,
            wait_list_len,
            wait_list_ptr,
            new_event_ptr,
        );

        let errcode_pre = format!("ocl::Kernel::enqueue()[{}]:", kernel_name.unwrap_or(""));
        errcode_assert(&errcode_pre, errcode);
    }
}




    // pub fn clEnqueueTask(command_queue: cl_command_queue,
    //                  kernel: cl_kernel,
    //                  num_events_in_wait_list: cl_uint,
    //                  event_wait_list: *const cl_event,
    //                  event: *mut cl_event) -> cl_int;



    // pub fn clEnqueueNativeKernel(command_queue: cl_command_queue,
    //                          user_func: extern fn (*mut c_void),
    //                          args: *mut c_void,
    //                          cb_args: size_t,
    //                          num_mem_objects: cl_uint,
    //                          mem_list: *const cl_mem,
    //                          args_mem_loc: *const *const c_void,
    //                          num_events_in_wait_list: cl_uint,
    //                          event_wait_list: *const cl_event,
    //                          event: *mut cl_event) -> cl_int;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueMarkerWithWaitList(cl_command_queue /* command_queue */,
    //          cl_uint           /* num_events_in_wait_list */,
    //          const cl_event *  /* event_wait_list */,
    //          cl_event *        /* event */) CL_API_SUFFIX__VERSION_1_2;



    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY cl_int CL_API_CALL
    // clEnqueueBarrierWithWaitList(
    //          cl_command_queue 
    //           // command_queue 
    //          ,                                
    //          cl_uint           
    //           // num_events_in_wait_list 
    //          ,
    //          const cl_event *  
    //           // event_wait_list 
    //          ,
    //          cl_event *        
    //           // event
    //      ) CL_API_SUFFIX__VERSION_1_2;



    // Extension function access
    //
    // Returns the extension function address for the given function name,
    // or NULL if a valid function can not be found. The client must
    // check to make sure the address is not NULL, before using or
    // or calling the returned function address.
    //
    // //################## NEW 1.2 ###################
    // extern CL_API_ENTRY void * CL_API_CALL 
    // clGetExtensionFunctionAddressForPlatform(cl_platform_id /* platform */,
    //                    const char *   
    //                     // func_name 
    //                    ) CL_API_SUFFIX__VERSION_1_2;
    


//=============================================================================
//=============================================================================
//============================ DERIVED FUNCTIONS ==============================
//=============================================================================
//=============================================================================
// MANY OF THESE NEED TO BE MORPHED INTO THE MORE GENERAL VERSIONS AND MOVED UP

pub fn get_max_work_group_size(device: DeviceIdRaw) -> usize {
    let mut max_work_group_size: usize = 0;

    let errcode = unsafe { 
        cl_h::clGetDeviceInfo(
            device.as_ptr(),
            cl_h::CL_DEVICE_MAX_WORK_GROUP_SIZE,
            mem::size_of::<usize>() as usize,
            &mut max_work_group_size as *mut _ as *mut c_void,
            ptr::null_mut(),
        ) 
    }; 

    errcode_assert("clGetDeviceInfo", errcode);

    max_work_group_size
}


#[allow(dead_code)]
/// [FIXME]: Why are we wrapping in this array? Fix this.
pub fn wait_for_event(event: EventRaw) {
    let event_array: [EventRaw; 1] = [event];

    let errcode = unsafe {
        cl_h::clWaitForEvents(1, &(*event_array.as_ptr()).as_ptr())
    };

    errcode_assert("clWaitForEvents", errcode);
}

/// Returns the status of `event`.
pub fn get_event_status(event: EventRaw) -> cl_int {
    let mut status: cl_int = 0;

    let errcode = unsafe { 
        cl_h::clGetEventInfo(
            event.as_ptr(),
            cl_h::CL_EVENT_COMMAND_EXECUTION_STATUS,
            mem::size_of::<cl_int>(),
            &mut status as *mut _ as *mut c_void,
            ptr::null_mut(),
        )
    };

    errcode_assert("clGetEventInfo", errcode);

    status
}


/// [UNFINISHED] Currently prints the platform name.
#[allow(dead_code)]
pub fn platform_info(platform: PlatformIdRaw) {
    let mut size = 0 as size_t;

    unsafe {
        let name = cl_h::CL_PLATFORM_NAME as cl_platform_info;
        let mut errcode = cl_h::clGetPlatformInfo(
                    platform.as_ptr(),
                    name,
                    0,
                    ptr::null_mut(),
                    &mut size,
        );
        errcode_assert("clGetPlatformInfo(size)", errcode);
        
        let mut param_value: Vec<u8> = iter::repeat(32u8).take(size as usize).collect();
        errcode = cl_h::clGetPlatformInfo(
                    platform.as_ptr(),
                    name,
                    size,
                    param_value.as_mut_ptr() as *mut c_void,
                    ptr::null_mut(),
        );
        errcode_assert("clGetPlatformInfo()", errcode);
        println!("*** Platform Name ({}): {}", name, String::from_utf8(param_value).unwrap());
    }
}


/// If the program pointed to by `cl_program` for any of the devices listed in 
/// `device_ids` has a build log of any length, it will be returned as an 
/// errcode result.
///
pub fn program_build_err(program: ProgramRaw, device_ids: &Vec<DeviceIdRaw>) -> OclResult<()> 
{
    let mut size = 0 as size_t;

    for &device_id in device_ids.iter() {
        unsafe {
            let name = cl_h::CL_PROGRAM_BUILD_LOG as cl_program_build_info;

            let mut errcode = cl_h::clGetProgramBuildInfo(
                program.as_ptr(),
                device_id.as_ptr(),
                name,
                0,
                ptr::null_mut(),
                &mut size,
            );
            errcode_assert("clGetProgramBuildInfo(size)", errcode);

            let mut pbi: Vec<u8> = iter::repeat(32u8).take(size as usize).collect();

            errcode = cl_h::clGetProgramBuildInfo(
                program.as_ptr(),
                device_id.as_ptr(),
                name,
                size,
                pbi.as_mut_ptr() as *mut c_void,
                ptr::null_mut(),
            );
            errcode_assert("clGetProgramBuildInfo()", errcode);

            if size > 1 {
                let pbi_nonull = try!(String::from_utf8(pbi).map_err(|e| e.to_string()));
                let pbi_errcode_string = format!(
                    "\n\n\
                    ###################### OPENCL PROGRAM BUILD DEBUG OUTPUT ######################\
                    \n\n{}\n\
                    ###############################################################################\
                    \n\n",
                    pbi_nonull);

                return OclError::err(pbi_errcode_string);
            }
        }
    }

    Ok(())
}


/// Returns a string containing requested information.
///
/// Currently lazily assumes everything is a char[] and converts to a String. 
/// Non-string info types need to be manually reconstructed from that. Yes this
/// is retarded.
///
/// [TODO (low priority)]: Needs to eventually be made more flexible and should return 
/// an enum with a variant corresponding to the type of info requested. Could 
/// alternatively return a generic type and just blindly cast to it.
#[allow(dead_code, unused_variables)] 
pub fn device_info(device_id: DeviceIdRaw, info_type: cl_device_info) -> String {
    let mut info_value_size: usize = 0;

    let errcode = unsafe { 
        cl_h::clGetDeviceInfo(
            device_id.as_ptr(),
            cl_h::CL_DEVICE_MAX_WORK_GROUP_SIZE,
            mem::size_of::<usize>() as usize,
            0 as cl_device_id,
            &mut info_value_size as *mut usize,
        ) 
    }; 

    errcode_assert("clGetDeviceInfo", errcode);

    String::new()
}


/// Returns context information.
///
/// [SDK Reference](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetContextInfo.html)
///
/// # Errors
///
/// Returns an error result for all the reasons listed in the SDK in addition 
/// to an additional error when called with `CL_CONTEXT_DEVICES` as described
/// in in the `verify_context()` documentation below.
///
/// TODO: Finish wiring up full functionality. Return a 'ContextInfo' enum result.
pub fn context_info(context: ContextRaw, info_kind: cl_context_info) 
        -> OclResult<()>
{
    let mut result_size = 0;

    // let info_kind: cl_context_info = cl_h::CL_CONTEXT_PROPERTIES;
    let errcode = unsafe { cl_h::clGetContextInfo(   
        context.as_ptr(),
        info_kind,
        0,
        0 as *mut c_void,
        &mut result_size as *mut usize,
    )};
    try!(errcode_try("clGetContextInfo", errcode));
    // println!("context_info(): errcode: {}, result_size: {}", errcode, result_size);

    let err_if_zero_result_size = info_kind == cl_h::CL_CONTEXT_DEVICES;

    if result_size > 10000 || (result_size == 0 && err_if_zero_result_size) {
        return OclError::err("\n\nocl::raw::context_info(): Possible invalid context detected. \n\
            Context info result size is either '> 10k bytes' or '== 0'. Almost certainly an \n\
            invalid context object. If not, please file an issue at: \n\
            https://github.com/cogciprocate/ocl/issues.\n\n");
    }

    let mut result: Vec<u8> = iter::repeat(0).take(result_size).collect();

    let errcode = unsafe { cl_h::clGetContextInfo(   
        context.as_ptr(),
        info_kind,
        result_size,
        result.as_mut_ptr() as *mut c_void,
        0 as *mut usize,
    )};
    try!(errcode_try("clGetContextInfo", errcode));
    // println!("context_info(): errcode: {}, result: {:?}", errcode, result);

    Ok(())
}


/// Verifies that the `context` is in fact a context object pointer.
///
/// # Assumptions
///
/// Some (most?) OpenCL implementations do not correctly error if non-context pointers are passed. This function relies on the fact that passing the `CL_CONTEXT_DEVICES` as the `param_name` to `clGetContextInfo` will (on my AMD implementation at least) often return a huge result size if `context` is not actually a `cl_context` pointer due to the fact that it's reading from some random memory location on non-context objects. Also checks for zero because a context must have at least one device (true?).
///
/// [UPDATE]: This function may no longer be necessary now that the raw pointers have wrappers but it still prevents a hard to track down bug so leaving it intact for now.
///
#[inline]
pub fn verify_context(context: ContextRaw) -> OclResult<()> {
    // context_info(context, cl_h::CL_CONTEXT_REFERENCE_COUNT)
    if cfg!(release) {
        Ok(())
    } else {
        context_info(context, cl_h::CL_CONTEXT_DEVICES)
    }
}

//=============================================================================
//=============================================================================
//=================================== EOF =====================================
//=============================================================================
//=============================================================================
