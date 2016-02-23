//! An OpenCL kernel.

use std::convert::Into;
use std::collections::HashMap;
use core::{self, OclNum, Kernel as KernelCore, CommandQueue as CommandQueueCore, KernelArg};
use error::{Result as OclResult, Error as OclError};
use standard::{SimpleDims, Buffer, EventList, Program, Queue};

/// A kernel.
///
/// # Destruction
/// 
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
#[derive(Debug)]
pub struct Kernel {
    obj_core: KernelCore,
    name: String,
    arg_index: u32,
    named_args: HashMap<&'static str, u32>,
    arg_count: u32,
    command_queue: CommandQueueCore,
    gwo: SimpleDims,
    gws: SimpleDims,
    lws: SimpleDims,
}

impl Kernel {
    /// Returns a new kernel.
    // TODO: Implement proper error handling (return result etc.).
    pub fn new<S: Into<String>, D: Into<SimpleDims>>(name: S, program: &Program, queue: &Queue, 
                gws: D) -> OclResult<Kernel>
    {
        let name = name.into();
        let obj_core = try!(core::create_kernel(program.core_as_ref(), &name));

        Ok(Kernel {
            obj_core: obj_core,
            name: name,
            arg_index: 0,
            named_args: HashMap::with_capacity(5),
            arg_count: 0u32,
            command_queue: queue.core_as_ref().clone(),
            gwo: SimpleDims::Unspecified,
            gws: gws.into(),
            lws: SimpleDims::Unspecified,
        })
    }

    /// Sets the global work offset (builder-style).
    pub fn gwo(mut self, gwo: SimpleDims) -> Kernel {
        if gwo.dim_count() == self.gws.dim_count() {
            self.gwo = gwo
        } else {
            panic!("ocl::Kernel::gwo(): Work size mismatch.");
        }
        self
    }

    /// Sets the local work size (builder-style).
    pub fn lws(mut self, lws: SimpleDims) -> Kernel {
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
    // [FIXME]: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_scl_named<T: OclNum>(&mut self, name: &'static str, scalar: T) 
            -> OclResult<()> 
    {
        let arg_idx = try!(self.resolve_named_arg_idx(name));
        self.set_arg::<T>(arg_idx, KernelArg::Scalar(&scalar))
    }

    /// Modifies the kernel argument named: `name`.
    // [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_buf_named<T: OclNum>(&mut self, name: &'static str, 
                buffer_opt: Option<&Buffer<T>>)  -> OclResult<()>   
    {
        //  TODO: ADD A CHECK FOR A VALID NAME (KEY)
        // let arg_idx = self.named_args[name];
        let arg_idx = try!(self.resolve_named_arg_idx(name));

        match buffer_opt {
            Some(buffer) => {
                self.set_arg::<T>(arg_idx, KernelArg::Mem(buffer.core_as_ref()))
            },
            None => {
                // let mem_core_null = unsafe { MemCore::null() };
                self.set_arg::<T>(arg_idx, KernelArg::MemNull)
            },
        }
    }

    fn resolve_named_arg_idx(&self, name: &'static str) -> OclResult<u32> {
        match self.named_args.get(name) {
            Some(&ai) => Ok(ai),
            None => {
                OclError::err(format!("Kernel::set_arg_scl_named(): Invalid argument \
                    name: '{}'.", name))
            },
        }
    }

    /// Enqueues kernel on the default command queue.
    ///
    /// Execution of the kernel on the device will not occur until the events
    /// in `wait_list` have completed if it is specified. 
    ///
    /// Specify `dest_list` to have a new event added to that list associated
    /// with the completion of this kernel task.
    ///
    /// TODO: Implement 'alternative queue' version of this function.
    #[inline]
    pub fn enqueue_with_events(&self, wait_list: Option<&EventList>, 
                    dest_list: Option<&mut EventList>) {
        core::enqueue_kernel(&self.command_queue, &self.obj_core, self.gws.dim_count(), 
            self.gwo.work_dims(), self.gws.work_dims().unwrap(), self.lws.work_dims(), 
            wait_list.map(|el| el.core_as_ref()), dest_list.map(|el| el.core_as_mut()), Some(&self.name))
            .unwrap();
    }

    /// Enqueues kernel on the default command queue with no event lists.
    ///
    /// Equivalent to `::enqueue_with_events(None, None)`.
    ///
    /// TODO: Implement 'alternative queue' version of this function.
    #[inline]
    pub fn enqueue(&self) {
        self.enqueue_with_events(None, None);
    }

    /// Returns the number of arguments specified for this kernel.
    #[inline]
    pub fn arg_count(&self) -> u32 {
        self.arg_count
    }    

    // Non-builder-style version of `::arg_buf()`.
    fn new_arg_buf<T: OclNum>(&mut self, buffer_opt: Option<&Buffer<T>>) -> u32 {        
        // This value lives long enough to be copied by `clSetKernelArg`.
        // let buf_obj = match buffer_opt {
        //     Some(buffer) => buffer.core_as_ref(),
        //     None => unsafe { MemCore::null() },
        // };

        // self.new_arg::<T>(KernelArg::Mem(&buf_obj))

        match buffer_opt {
            Some(buffer) => {
                self.new_arg::<T>(KernelArg::Mem(buffer.core_as_ref()))
            },
            None => {
                // let mem_core_null = unsafe { MemCore::null() };
                self.new_arg::<T>(KernelArg::MemNull)
            },
        }
    }

    // Non-builder-style version of `::arg_scl()`.
    fn new_arg_scl<T: OclNum>(&mut self, scalar_opt: Option<T>) -> u32 {
        let scalar = match scalar_opt {
            Some(scl) => scl,
            None => Default::default(),
        };

        self.new_arg::<T>(KernelArg::Scalar(&scalar))
    }

    // Non-builder-style version of `::arg_loc()`.
    //
    // `length` lives long enough to be copied by `clSetKernelArg`.
    fn new_arg_loc<T: OclNum>(&mut self, length: usize) -> u32 {
        self.new_arg::<T>(KernelArg::Local(&length))
    } 

    // Adds a new argument to the kernel and returns the index.
    fn new_arg<T: OclNum>(&mut self, arg: KernelArg<T>) -> u32 {
        let arg_idx = self.arg_index;

        core::set_kernel_arg::<T>(&self.obj_core, arg_idx, 
            arg,
            Some(&self.name)
        ).unwrap();

        self.arg_index += 1;
        arg_idx
    } 

    fn set_arg<T: OclNum>(&self, arg_idx: u32, arg: KernelArg<T>) -> OclResult<()> {
        core::set_kernel_arg::<T>(&self.obj_core, arg_idx, arg, Some(&self.name))
    }

    pub fn core_as_ref(&self) -> &KernelCore {
        &self.obj_core
    }
}

// impl Drop for Kernel {
//     fn drop(&mut self) {
//         // println!("DROPPING KERNEL");
//         core::release_kernel(self.obj_core).unwrap();
//     }
// }
