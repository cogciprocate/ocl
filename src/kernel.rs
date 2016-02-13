//! An OpenCL kernel.

// use std::ptr;
// use std::mem;
// use std::ffi;
use std::collections::HashMap;
// use libc::{size_t, c_void};

use raw::{self, KernelRaw, MemRaw, CommandQueueRaw, KernelArg};
use super::{Result as OclResult, WorkDims, Buffer, OclNum, EventList, Program, Queue};

/// An OpenCL kernel.
///
/// # Destruction
/// Releases kernel object automatically upon drop.
///
/// # Thread Safety
///
/// Do not share the kernel object pointer `obj` between threads. 
/// Specifically, do not attempt to create or modify kernel arguments
/// from more than one thread for a kernel.
///
/// TODO: Add more details, examples, etc.
/// TODO: Add information about panics and errors.
pub struct Kernel {
    obj_raw: KernelRaw,
    name: String,
    arg_index: u32,
    named_args: HashMap<&'static str, u32>,
    arg_count: u32,
    command_queue: CommandQueueRaw,
    gwo: WorkDims,
    gws: WorkDims,
    lws: WorkDims,
}

impl Kernel {
    /// Returns a new kernel.
    // TODO: Implement proper error handling (return result etc.).
    pub fn new(name: String, program: &Program, queue: &Queue, 
                gws: WorkDims ) -> OclResult<Kernel>
    {
        // let mut err: i32 = 0;

        // let obj = unsafe {
        //     cl_h::clCreateKernel(
        //         program.obj(), 
        //         ffi::CString::new(name.as_bytes()).unwrap().as_ptr(), 
        //         &mut err
        //     )
        // };
        
        // let err_pre = format!("Ocl::create_kernel({}):", &name);
        // raw::errcode_assert(&err_pre, err);
        let obj_raw = try!(raw::create_kernel(program.obj_raw(), &name));

        Ok(Kernel {
            obj_raw: obj_raw,
            name: name,
            arg_index: 0,
            named_args: HashMap::with_capacity(5),
            arg_count: 0u32,
            command_queue: queue.obj_raw(),
            gwo: WorkDims::Unspecified,
            gws: gws,
            lws: WorkDims::Unspecified,
        })
    }

    /// Sets the global work offset (builder-style).
    pub fn gwo(mut self, gwo: WorkDims) -> Kernel {
        if gwo.dim_count() == self.gws.dim_count() {
            self.gwo = gwo
        } else {
            panic!("ocl::Kernel::gwo(): Work size mismatch.");
        }
        self
    }

    /// Sets the local work size (builder-style).
    pub fn lws(mut self, lws: WorkDims) -> Kernel {
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
        self.new_arg_buf(Some(buffer));
        self
    }

    /// Adds a new argument specifying the value: `scalar` (builder-style). Argument 
    /// is added to the bottom of the argument order.
    pub fn arg_scl<T: OclNum>(mut self, scalar: T) -> Kernel {
        self.new_arg_scl(Some(scalar));
        self
    }

    /// Adds a new argument specifying the allocation of a local variable of size
    /// `length * sizeof(T)` bytes (builder_style).
    ///
    /// Local variables are used to share data between work items in the same 
    /// workgroup.
    pub fn arg_loc<T: OclNum>(mut self, length: usize) -> Kernel {
        self.new_arg_loc::<T>(length);
        self
    }

    /// Adds a new named argument (in order) specifying the value: `scalar` 
    /// (builder-style).
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_scl_named<T: OclNum>(mut self, name: &'static str, scalar_opt: Option<T>) -> Kernel {
        let arg_idx = self.new_arg_scl(scalar_opt);
        self.named_args.insert(name, arg_idx);
        self
    }

    /// Adds a new named buffer argument specifying the buffer object represented by 
    /// 'buffer' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_buf_named<T: OclNum>(mut self, name: &'static str,  buffer_opt: Option<&Buffer<T>>) -> Kernel {
        let arg_idx = self.new_arg_buf(buffer_opt);
        self.named_args.insert(name, arg_idx);

        self
    }     

    /// Modifies the kernel argument named: `name`.
    // [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_scl_named<T: OclNum>(&mut self, name: &'static str, scalar: T) {
        //  TODO: ADD A CHECK FOR A VALID NAME (KEY)
        let arg_idx = self.named_args[name]; 

        // raw::set_kernel_arg(
        //     self.obj_raw,
        //     arg_idx,
        //     // mem::size_of::<T>() as size_t, 
        //     // &scalar as *const _ as *const c_void,
        //     KernelArg::Scalar(&scalar),
        //     Some(&self.name),
        // ).unwrap();

        self.set_arg::<T>(arg_idx, KernelArg::Scalar(&scalar)).unwrap();
    }

    /// Modifies the kernel argument named: `name`.
    // [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_buf_named<T: OclNum>(&mut self, name: &'static str, 
                buffer_opt: Option<&Buffer<T>>) 
    {
        //  TODO: ADD A CHECK FOR A VALID NAME (KEY)
        let arg_idx = self.named_args[name];

        let buf_obj_raw = match buffer_opt {
            Some(buffer) => buffer.obj_raw(),
            None => MemRaw::null(),
        };

        self.set_arg::<T>(arg_idx, KernelArg::Mem(buf_obj_raw)).unwrap();

        // raw::set_kernel_arg::<T>(
        //     self.obj_raw,
        //     arg_idx,
        //     // mem::size_of::<MemRaw>() as size_t, 
        //     // (&buf_obj_raw.as_ptr() as *const *mut c_void) as *const c_void,
        //     KernelArg::Mem(buf_obj_raw),
        //     Some(&self.name),
        // ).unwrap();
    }

    /// Enqueues kernel on the default command queue.
    #[inline]
    pub fn enqueue(&self, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) {
        // self.enqueue_with_cmd_queue(self.command_queue, wait_list, dest_list);
        raw::enqueue_kernel(self.command_queue, self.obj_raw.as_ptr(), self.gws.dim_count(), 
            self.gwo.as_raw(), self.gws.as_raw().unwrap(), self.lws.as_raw(), 
            wait_list.map(|el| el.events()), dest_list.map(|el| el.allot()), Some(&self.name));
    }

    /// Returns the number of arguments specified for this kernel.
    #[inline]
    pub fn arg_count(&self) -> u32 {
        self.arg_count
    }    

     // Non-builder-style version of `::arg_buf()`.
    fn new_arg_buf<T: OclNum>(&mut self, buffer_opt: Option<&Buffer<T>>) -> u32 {
        let buf_obj = match buffer_opt {
            Some(buffer) => buffer.obj_raw(),
            None => MemRaw::null(),
        };

        // self.new_kernel_arg(
        //     mem::size_of::<MemRaw>() as size_t, 
        //     (&buf.as_ptr() as *const *mut c_void) as *const c_void,
        // )

        self.new_arg::<T>(KernelArg::Mem(buf_obj))
    }

    // Non-builder-style version of `::arg_scl()`.
    fn new_arg_scl<T: OclNum>(&mut self, scalar_opt: Option<T>) -> u32 {
        let scalar = match scalar_opt {
            Some(scl) => scl,
            None => Default::default(),
        };

        // self.new_arg::<T>(
        //     mem::size_of::<T>() as size_t,
        //     &scalar as *const _ as *const c_void,
        //     //(scalar as *const super::cl_mem) as *const c_void,
        // )

        self.new_arg::<T>(KernelArg::Scalar(&scalar))
    }

    // Non-builder-style version of `::arg_loc()`.
    fn new_arg_loc<T: OclNum>(&mut self, length: usize) -> u32 {
        // self.new_kernel_arg(
        //     (mem::size_of::<T>() * length) as size_t,
        //     ptr::null(),
        // )

        self.new_arg::<T>(KernelArg::Local(length))
    } 

    // Adds a new argument to the kernel and returns the index.
    fn new_arg<T>(&mut self, arg: KernelArg<T> /*arg_size: size_t, arg_value: *const c_void*/) -> u32 {
        let arg_idx = self.arg_index;

        raw::set_kernel_arg::<T>(self.obj_raw, arg_idx, 
                // KernelArg::Other { size: arg_size, value: arg_value }, 
                arg,
                Some(&self.name)
            ).unwrap();

        self.arg_index += 1;
        arg_idx
    } 

    fn set_arg<T>(&self, arg_idx: u32, arg: KernelArg<T>) -> OclResult<()> {
        raw::set_kernel_arg::<T>(self.obj_raw, arg_idx, arg, Some(&self.name))
    }

    // // Modifies a kernel argument.
    // // NOTE: Maintain mutability requirement to completely prevent simultaneous calls.
    // fn set_kernel_arg(&mut self, arg_index: u32, arg_size: size_t, 
    //             arg_value: *const c_void) 
    // {
    //     unsafe {
    //         let err = cl_h::clSetKernelArg(
    //                     self.obj_raw.as_ptr(), 
    //                     arg_index,
    //                     arg_size, 
    //                     arg_value,
    //         );

    //         let err_pre = format!("ocl::Kernel::set_kernel_arg('{}'):", &self.name);
    //         raw::errcode_assert(&err_pre, err);
    //     }
    // }    
}

impl Drop for Kernel {
    fn drop(&mut self) {
        raw::release_kernel(self.obj_raw);
    }
}
