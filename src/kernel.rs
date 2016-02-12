//! An OpenCL kernel.

use std::ptr;
use std::mem;
use std::ffi;
use std::collections::HashMap;
use libc;

use raw;
use cl_h::{self, cl_mem, cl_kernel, cl_command_queue};
use super::{WorkSize, Buffer, OclNum, EventList, Program, Queue};

/// An OpenCL kernel.
///
/// # Destruction
/// Releases kernel object automatically upon drop.
///
/// [FIXME] TODO: Add more details, examples, etc.
/// [FIXME] TODO: Add information about panics and errors.
pub struct Kernel {
    kernel_obj: cl_kernel,
    name: String,
    arg_index: u32,
    named_args: HashMap<&'static str, u32>,
    arg_count: u32,
    command_queue: cl_command_queue,
    gwo: WorkSize,
    gws: WorkSize,
    lws: WorkSize,
}

impl Kernel {
    /// Returns a new kernel.
    // [FIXME] TODO: Implement proper error handling (return result etc.).
    pub fn new(name: String, program: &Program, queue: &Queue, 
                gws: WorkSize ) -> Kernel 
    {
        let mut err: i32 = 0;

        let kernel_obj = unsafe {
            cl_h::clCreateKernel(
                program.obj(), 
                ffi::CString::new(name.as_bytes()).unwrap().as_ptr(), 
                &mut err
            )
        };
        
        let err_pre = format!("Ocl::create_kernel({}):", &name);
        raw::errcode_assert(&err_pre, err);

        Kernel {
            kernel_obj: kernel_obj,
            name: name,
            arg_index: 0,
            named_args: HashMap::with_capacity(5),
            arg_count: 0u32,
            command_queue: queue.obj(),
            gwo: WorkSize::Unspecified,
            gws: gws,
            lws: WorkSize::Unspecified,
        }
    }

    /// Sets the global work offset (builder-style).
    pub fn gwo(mut self, gwo: WorkSize) -> Kernel {
        if gwo.dim_count() == self.gws.dim_count() {
            self.gwo = gwo
        } else {
            panic!("ocl::Kernel::gwo(): Work size mismatch.");
        }
        self
    }

    /// Sets the local work size (builder-style).
    pub fn lws(mut self, lws: WorkSize) -> Kernel {
        if lws.dim_count() == self.gws.dim_count() {
            self.lws = lws;
        } else {
            panic!("ocl::Kernel::lws(): Work size mismatch.");
        }
        self
    }

    /// Adds a new argument to the kernel specifying the buffer object represented
    /// by 'buffer' (builder-style). Argument is added to the bottom of the argument 
    /// order.
    pub fn arg_buf<T: OclNum>(mut self, buffer: &Buffer<T>) -> Kernel {
        self.new_arg_buffer(Some(buffer));
        self
    }

    /// Adds a new argument specifying the value: `scalar` (builder-style). Argument 
    /// is added to the bottom of the argument order.
    pub fn arg_scl<T: OclNum>(mut self, scalar: T) -> Kernel {
        self.new_arg_scalar(Some(scalar));
        self
    }

    /// Adds a new argument specifying the allocation of a local variable of size
    /// `length * sizeof(T)` bytes (builder_style).
    ///
    /// Local variables are used to share data between work items in the same 
    /// workgroup.
    pub fn arg_loc<T: OclNum>(mut self, length: usize) -> Kernel {
        self.new_arg_local::<T>(length);
        self
    }

    /// Adds a new named argument (in order) specifying the value: `scalar` 
    /// (builder-style).
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_scl_named<T: OclNum>(mut self, name: &'static str, scalar_opt: Option<T>) -> Kernel {
        let arg_idx = self.new_arg_scalar(scalar_opt);
        self.named_args.insert(name, arg_idx);
        self
    }

    /// Adds a new named buffer argument specifying the buffer object represented by 
    /// 'buffer' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_buf_named<T: OclNum>(mut self, name: &'static str,  buffer_opt: Option<&Buffer<T>>) -> Kernel {
        let arg_idx = self.new_arg_buffer(buffer_opt);
        self.named_args.insert(name, arg_idx);

        self
    }   

    /// Non-builder-style version of `::arg_buf()`.
    pub fn new_arg_buffer<T: OclNum>(&mut self, buffer_opt: Option<&Buffer<T>>) -> u32 {
        let buf = match buffer_opt {
            Some(buffer) => buffer.buffer_obj(),
            None => ptr::null_mut()
        };

        self.new_kernel_arg(
            mem::size_of::<cl_mem>() as libc::size_t, 
            (&buf as *const cl_mem) as *const libc::c_void,
        )
    }

    /// Non-builder-style version of `::arg_scl()`.
    pub fn new_arg_scalar<T: OclNum>(&mut self, scalar_opt: Option<T>) -> u32 {
        let scalar = match scalar_opt {
            Some(scl) => scl,
            None => Default::default(),
        };

        self.new_kernel_arg(
            mem::size_of::<T>() as libc::size_t,
            &scalar as *const _ as *const libc::c_void,
            //(scalar as *const super::cl_mem) as *const libc::c_void,
        )
    }

    /// Non-builder-style version of `::arg_loc()`.
    pub fn new_arg_local<T: OclNum>(&mut self, /*type_sample: T,*/ length: usize) -> u32 {

        self.new_kernel_arg(
            (mem::size_of::<T>() * length) as libc::size_t,
            ptr::null(),
        )
    }

    /// Adds a new argument to the kernel and returns the index.
    fn new_kernel_arg(&mut self, arg_size: libc::size_t, arg_value: *const libc::c_void) -> u32 {
        let a_i = self.arg_index;
        self.set_kernel_arg(a_i, arg_size, arg_value);
        self.arg_index += 1;
        a_i
    }

    /// Modifies the kernel argument named: `name`.
    // [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_scl_named<T: OclNum>(&mut self, name: &'static str, scalar: T) {
        //  TODO: ADD A CHECK FOR A VALID NAME (KEY)
        let arg_idx = self.named_args[name]; 

        self.set_kernel_arg(
            arg_idx,
            mem::size_of::<T>() as libc::size_t, 
            &scalar as *const _ as *const libc::c_void,
        )
    }

    /// Modifies the kernel argument named: `name`.
    // [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_buf_named<T: OclNum>(&mut self, name: &'static str, buffer: &Buffer<T>) {
        //  TODO: ADD A CHECK FOR A VALID NAME (KEY)
        let arg_idx = self.named_args[name];
        let buf = buffer.buffer_obj();

        self.set_kernel_arg(
            arg_idx,
            mem::size_of::<cl_mem>() as libc::size_t, 
            (&buf as *const cl_mem) as *const libc::c_void,
        )
    }

    // Modifies a kernel argument.
    fn set_kernel_arg(&mut self, arg_index: u32, arg_size: libc::size_t, 
                arg_value: *const libc::c_void) 
    {
        unsafe {
            let err = cl_h::clSetKernelArg(
                        self.kernel_obj, 
                        arg_index,
                        arg_size, 
                        arg_value,
            );

            let err_pre = format!("ocl::Kernel::set_kernel_arg('{}'):", &self.name);
            raw::errcode_assert(&err_pre, err);
        }
    }

//     /// Enqueues kernel using a non-default command queue.
//     ///
//     /// `cmd_queue` must have the same context and device as the
//     /// default command queue passed when creating kernel.
//     #[inline]
//     pub fn enqueue_with_cmd_queue(&self, queue: cl_command_queue, 
//                 wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) 
//     {
// //         // [FIXME] TODO: VERIFY THE DIMENSIONS OF ALL THE WORKSIZES
// //         let c_gws = self.gws.complete_worksize();
// //         let c_lws = self.lws.complete_worksize();
        
// //         let (wait_list_len, wait_list_ptr, new_event_ptr) = raw::resolve_queue_opts(
// //             wait_list.map(|el| el.events()), dest_list.map(|el| el.allot()));

// //         let gws = (&c_gws as *const (usize, usize, usize)) as *const libc::size_t;
// //         let lws = (&c_lws as *const (usize, usize, usize)) as *const libc::size_t;

// //         println!(
// //         r#"
// // ENQUEUING KERNEL: '{}'
// //     command_queue: {:?}
// //     kernel: {:?}
// //     work_dims: {}
// //     global_work_offset: '{:?}'
// //     global_work_size: '{:?}'
// //     local_work_size: '{:?}'
// //     wait_list_len: {:?}
// //     wait_list_ptr: {:?}
// //     new_event_ptr: {:?}
// //         "#, 
// //         &self.name,
// //         queue,
// //         self.kernel_obj,
// //         self.gws.dim_count(),
// //         self.gwo.as_ptr(),
// //         gws,
// //         lws,
// //         wait_list_len,
// //         wait_list_ptr,
// //         new_event_ptr,
// //         );

// //         unsafe {
// //             let err = cl_h::clEnqueueNDRangeKernel(
// //                         queue,
// //                         self.kernel_obj,
// //                         self.gws.dim_count(),
// //                         self.gwo.as_ptr(),
// //                         gws,
// //                         lws,
// //                         wait_list_len,
// //                         wait_list_ptr,
// //                         new_event_ptr,
// //             );

// //             let err_pre = format!("ocl::Kernel::enqueue()[{}]:", &self.name);
// //             raw::errcode_assert(&err_pre, err);
// //         }

//         raw::enqueue_kernel(queue, self.kernel_obj, self.gws.dim_count(), self.gwo.as_work_offset(),
//             self.gws.as_work_size(), self.lws.as_work_size(), wait_list.map(|el| el.events()),
//             dest_list.map(|el| el.allot()), Some(&self.name));
//     }

    /// Enqueues kernel on the default command queue.
    #[inline]
    pub fn enqueue(&self, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) {
        // self.enqueue_with_cmd_queue(self.command_queue, wait_list, dest_list);
        raw::enqueue_kernel(self.command_queue, self.kernel_obj, self.gws.dim_count(), 
            self.gwo.as_work_offset(), self.gws.as_work_size().unwrap(), self.lws.as_work_size(), 
            wait_list.map(|el| el.events()), dest_list.map(|el| el.allot()), Some(&self.name));
    }

    /// Returns the number of arguments specified for this kernel.
    #[inline]
    pub fn arg_count(&self) -> u32 {
        self.arg_count
    }

    pub unsafe fn release(&mut self) {
        cl_h::clReleaseKernel(self.kernel_obj);
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { self.release(); }
    }
}
