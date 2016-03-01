//! An OpenCL kernel.

use std;
use std::convert::Into;
use std::collections::HashMap;
use core::{self, OclNum, Kernel as KernelCore, CommandQueue as CommandQueueCore, KernelArg, 
    KernelInfo, KernelInfoResult};
use error::{Result as OclResult, Error as OclError};
use standard::{SimpleDims, Buffer, Image, EventList, Program, Queue, WorkDims, Sampler};

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
    named_args: HashMap<&'static str, u32>,
    arg_count: u32,
    command_queue_obj_core: CommandQueueCore,
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
        let obj_core = try!(core::create_kernel(program, &name));

        Ok(Kernel {
            obj_core: obj_core,
            name: name,
            named_args: HashMap::with_capacity(5),
            arg_count: 0,
            command_queue_obj_core: queue.core_as_ref().clone(),
            gwo: SimpleDims::Unspecified,
            gws: gws.into(),
            lws: SimpleDims::Unspecified,
        })
    }

    /// Sets the global work offset (builder-style).
    pub fn gwo<D: Into<SimpleDims>>(mut self, gwo: D) -> Kernel {
        let gwo = gwo.into();

        if gwo.dim_count() == self.gws.dim_count() {
            self.gwo = gwo;
        } else {
            panic!("ocl::Kernel::gwo(): Work size mismatch.");
        }
        self
    }

    /// Sets the local work size (builder-style).
    pub fn lws<D: Into<SimpleDims>>(mut self, lws: D) -> Kernel {
        let lws = lws.into();

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

    /// Adds a new argument to the kernel specifying the image object represented
    /// by 'image' (builder-style). Argument is added to the bottom of the argument 
    /// order.
    pub fn arg_img(mut self, image: &Image) -> Kernel {
        self.new_arg_img(Some(image));
        self
    }

    /// Adds a new argument to the kernel specifying the sampler object represented
    /// by 'sampler' (builder-style). Argument is added to the bottom of the argument 
    /// order.
    pub fn arg_smp(mut self, sampler: &Sampler) -> Kernel {
        self.new_arg_smp(Some(sampler));
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
    pub fn arg_buf_named<T: OclNum>(mut self, name: &'static str, buffer_opt: Option<&Buffer<T>>) -> Kernel {
        let arg_idx = self.new_arg_buf(buffer_opt);
        self.named_args.insert(name, arg_idx);
        self
    }     

    /// Adds a new named image argument specifying the image object represented by 
    /// 'image' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_img_named(mut self, name: &'static str, image_opt: Option<&Image>) -> Kernel {
        let arg_idx = self.new_arg_img(image_opt);
        self.named_args.insert(name, arg_idx);
        self
    }    

    /// Adds a new named sampler argument specifying the sampler object represented by 
    /// 'sampler' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_smp_named(mut self, name: &'static str, sampler_opt: Option<&Sampler>) -> Kernel {
        let arg_idx = self.new_arg_smp(sampler_opt);
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
                self.set_arg::<T>(arg_idx, KernelArg::Mem(buffer))
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

    // pub fn enqueue_ndrange(&self, queue: Option<&Queue>, ) {
    //     let command_queue = match queue {
    //         Some(q) => q.core_as_ref(),
    //         None => &self.command_queue_obj_core,
    //     };        

    //     core::enqueue_kernel(command_queue, &self.obj_core, self.gws.dim_count(), 
    //         self.gwo.to_work_offset(), self.gws.to_work_size().unwrap(), self.lws.to_work_size(), 
    //         wait_list.map(|el| el.core_as_ref()), dest_list.map(|el| el.core_as_mut()), Some(&self.name))
    // }

    /// Enqueues kernel on the default command queue.
    ///
    /// Specify `queue` to use a non-default queue.
    ///
    /// Execution of the kernel on the device will not occur until the events
    /// in `wait_list` have completed if it is specified. 
    ///
    /// Specify `dest_list` to have a new event added to that list associated
    /// with the completion of this kernel task.
    ///
    #[inline]
    pub fn enqueue_events(&self, wait_list: Option<&EventList>, 
                    dest_list: Option<&mut EventList>) -> OclResult<()>
    {
        core::enqueue_kernel(&self.command_queue_obj_core, &self.obj_core, self.gws.dim_count(), 
            self.gwo.to_work_offset(), self.gws.to_work_size().unwrap(), self.lws.to_work_size(), 
            wait_list.map(|el| el.core_as_ref()), dest_list.map(|el| el.core_as_mut()), Some(&self.name))
    }

    /// Enqueues kernel on the default command queue with no event lists.
    ///
    /// Equivalent to `::enqueue_with_events(None, None)`.
    ///
    #[inline]
    pub fn enqueue(&self) {
        self.enqueue_events(None, None).expect("ocl::Kernel::enqueue");
    }

    /// Changes the default queue used when none is passed to `::enqueue_with`
    /// or when using `::enqueue`.
    ///
    /// Returns a ref for chaining i.e.:
    ///
    /// `buffer.set_queue(queue).flush_vec(....);`
    ///
    /// [NOTE]: Even when used as above, the queue is changed permanently,
    /// not just for the one call. Changing the queue is cheap so feel free
    /// to change as often as needed.
    ///
    /// The new queue must be associated with a device valid for the kernel's
    /// program.
    pub fn set_queue<'a>(&'a mut self, queue: &Queue) -> &'a mut Kernel {
        self.command_queue_obj_core = queue.core_as_ref().clone();
        self
    }

    /// Returns the number of arguments specified for this kernel.
    #[inline]
    pub fn arg_count(&self) -> u32 {
        self.arg_count
    }    

    // Non-builder-style version of `::arg_buf()`.
    fn new_arg_buf<T: OclNum>(&mut self, buffer_opt: Option<&Buffer<T>>) -> u32 {        
        match buffer_opt {
            Some(buffer) => {
                self.new_arg::<T>(KernelArg::Mem(buffer))
            },
            None => {
                // let mem_core_null = unsafe { MemCore::null() };
                self.new_arg::<T>(KernelArg::MemNull)
            },
        }
    }

    // Non-builder-style version of `::arg_img()`.
    fn new_arg_img(&mut self, image_opt: Option<&Image>) -> u32 {        
        match image_opt {
            Some(image) => {
                // Type is ignored:
                self.new_arg::<u8>(KernelArg::Mem(image))
            },
            None => {
                self.new_arg::<u8>(KernelArg::MemNull)
            },
        }
    }

    // Non-builder-style version of `::arg_img()`.
    fn new_arg_smp(&mut self, sampler_opt: Option<&Sampler>) -> u32 {
        match sampler_opt {
            Some(sampler) => {
                // Type is ignored:
                self.new_arg::<u8>(KernelArg::Sampler(sampler))
            },
            None => {
                self.new_arg::<u8>(KernelArg::SamplerNull)
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
        let arg_idx = self.arg_count;

        core::set_kernel_arg::<T>(&self.obj_core, arg_idx, 
            arg,
            Some(&self.name)
        ).unwrap();

        self.arg_count += 1;
        arg_idx
    } 

    fn set_arg<T: OclNum>(&self, arg_idx: u32, arg: KernelArg<T>) -> OclResult<()> {
        core::set_kernel_arg::<T>(&self.obj_core, arg_idx, arg, Some(&self.name))
    }

    pub fn core_as_ref(&self) -> &KernelCore {
        &self.obj_core
    }

    /// Returns info about this kernel.
    pub fn info(&self, info_kind: KernelInfo) -> KernelInfoResult {
        match core::get_kernel_info(&self.obj_core, info_kind) {
            Ok(res) => res,
            Err(err) => KernelInfoResult::Error(Box::new(err)),
        }        
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("FunctionName", &self.info(KernelInfo::FunctionName))
            .field("ReferenceCount", &self.info(KernelInfo::ReferenceCount))
            .field("Context", &self.info(KernelInfo::Context))
            .field("Program", &self.info(KernelInfo::Program))
            .field("Attributes", &self.info(KernelInfo::Attributes))
            .finish()
    }

        // AddressQualifier = cl_h::CL_KERNEL_ARG_ADDRESS_QUALIFIER as isize,
        // AccessQualifier = cl_h::CL_KERNEL_ARG_ACCESS_QUALIFIER as isize,
        // TypeName = cl_h::CL_KERNEL_ARG_TYPE_NAME as isize,
        // TypeQualifier = cl_h::CL_KERNEL_ARG_TYPE_QUALIFIER as isize,
        // Name = cl_h::CL_KERNEL_ARG_NAME as isize,
    // fn fmt_arg_info(&self, f: &mut std::fmt::Formatter, arg_idx: u32) -> std::fmt::Result {
    //     f.debug_struct("Kernel")
    //         .field("FunctionName", &self.info(KernelInfo::FunctionName))
    //         .field("ReferenceCount", &self.info(KernelInfo::ReferenceCount))
    //         .field("Context", &self.info(KernelInfo::Context))
    //         .field("Program", &self.info(KernelInfo::Program))
    //         .field("Attributes", &self.info(KernelInfo::Attributes))
    //         .finish()
    // }
}



impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

use std::ops::{Deref, DerefMut};

impl Deref for Kernel {
    type Target = KernelCore;

    fn deref(&self) -> &KernelCore {
        &self.obj_core
    }
}

impl DerefMut for Kernel {
    fn deref_mut(&mut self) -> &mut KernelCore {
        &mut self.obj_core
    }
}
