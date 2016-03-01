//! An OpenCL kernel.

use std;
use std::convert::Into;
use std::collections::HashMap;
use core::{self, OclNum, Kernel as KernelCore, CommandQueue as CommandQueueCore, KernelArg, 
    KernelInfo, KernelInfoResult};
use error::{Result as OclResult, Error as OclError};
use standard::{SimpleDims, Buffer, Image, EventList, Program, Queue, WorkDims, Sampler};



/// A kernel command builder used to queue a kernel with a mix of default
/// and optionally specified arguments.
pub struct KernelCmd<'k> {
    queue: &'k CommandQueueCore,
    kernel: &'k KernelCore,
    gwo: SimpleDims,
    gws: SimpleDims,
    lws: SimpleDims,
    wait_list: Option<&'k EventList>,
    dest_list: Option<&'k mut EventList>,
    name: &'k str,
}

impl<'k> KernelCmd<'k> {
    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'k Queue) -> KernelCmd<'k> {
        self.queue = queue.core_as_ref();
        self
    }

    /// Specifies a global work offset for this call only.
     pub fn gwo<D: Into<SimpleDims>>(mut self, gwo: D) -> KernelCmd<'k> {
        self.gwo = gwo.into();
        self
    }

    /// Specifies a global work size for this call only.
    pub fn gws<D: Into<SimpleDims>>(mut self, gws: D) -> KernelCmd<'k> {
        self.gws = gws.into();
        self
    }

    /// Specifies a local work size for this call only.
    pub fn lws<D: Into<SimpleDims>>(mut self, lws: D) -> KernelCmd<'k> {
        self.lws = lws.into();
        self
    }

    /// Specifies the list of events to wait on before the command will run.
    pub fn wait(mut self, wait_list: &'k EventList) -> KernelCmd<'k> {
        self.wait_list = Some(wait_list);
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn dest(mut self, dest_list: &'k mut EventList) -> KernelCmd<'k> {
        self.dest_list = Some(dest_list);
        self
    }

    /// Specifies the list of events to wait on before the command will run.
    pub fn wait_opt(mut self, wait_list: Option<&'k EventList>) -> KernelCmd<'k> {
        self.wait_list = wait_list;
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn dest_opt(mut self, dest_list: Option<&'k mut EventList>) -> KernelCmd<'k> {
        self.dest_list = dest_list;
        self
    }

    /// Enqueues this kernel command.
    pub fn enq(self) -> OclResult<()> {
        let dim_count = self.gws.dim_count();

        let gws = match self.gws.to_work_size() {
            Some(gws) => gws,
            None => return OclError::err("ocl::KernelCmd::enqueue: Global Work Size ('gws') \
                cannot be left unspecified. Set a default for the kernel or pass a valid parameter."),
        };

        // let wait_list = self.wait_list.map(|el| el.core_as_ref());
        // let dest_list = self.dest_list.map(|el| el.core_as_mut());

        core::enqueue_kernel(self.queue, self.kernel, dim_count, self.gwo.to_work_offset(), 
            gws, self.lws.to_work_size(), self.wait_list, self.dest_list, Some(self.name))
    }
}




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
    pub fn new<S: Into<String>, >(name: S, program: &Program, queue: &Queue, 
            ) -> OclResult<Kernel>
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
            gws: SimpleDims::Unspecified,
            lws: SimpleDims::Unspecified,
        })
    }

    /// Sets the default global work offset (builder-style).
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// making a call to enqueue or building a queue command with `::cmd`.
    pub fn gwo<D: Into<SimpleDims>>(mut self, gwo: D) -> Kernel {
        // let gwo = gwo.into();

        // if gwo.dim_count() == self.gws.dim_count() {
        //     self.gwo = gwo;
        // } else {
        //     panic!("ocl::Kernel::gwo(): Work size mismatch.");
        // }

        self.gwo = gwo.into();
        self
    }

    /// Sets the default global work size (builder-style).
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// making a call to enqueue or building a queue command with `::cmd`.
    pub fn gws<D: Into<SimpleDims>>(mut self, gws: D) -> Kernel {
        // if gws.dim_count() == self.gws.dim_count() {
        //     self.gws = gws;
        // } else {
        //     panic!("ocl::Kernel::gws(): Work size mismatch.");
        // }

        self.gws = gws.into();
        self
    }

    /// Sets the default local work size (builder-style).
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// making a call to enqueue or building a queue command with `::cmd`.
    pub fn lws<D: Into<SimpleDims>>(mut self, lws: D) -> Kernel {
        // let lws = lws.into();

        // if lws.dim_count() == self.gws.dim_count() {
        //     self.lws = lws;
        // } else {
        //     panic!("ocl::Kernel::lws(): Work size mismatch.");
        // }
        self.lws = lws.into();
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

    /// Adds a new named argument specifying the buffer object represented by 
    /// 'buffer' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_buf_named<T: OclNum>(mut self, name: &'static str, buffer_opt: Option<&Buffer<T>>) -> Kernel {
        let arg_idx = self.new_arg_buf(buffer_opt);
        self.named_args.insert(name, arg_idx);
        self
    }     

    /// Adds a new named argument specifying the image object represented by 
    /// 'image' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_img_named(mut self, name: &'static str, image_opt: Option<&Image>) -> Kernel {
        let arg_idx = self.new_arg_img(image_opt);
        self.named_args.insert(name, arg_idx);
        self
    }    

    /// Adds a new named argument specifying the sampler object represented by 
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


    pub fn cmd<'k>(&'k self) -> KernelCmd<'k> {
        KernelCmd { queue: &self.command_queue_obj_core, kernel: &self.obj_core, 
            gwo: self.gwo.clone(), gws: self.gws.clone(), lws: self.lws.clone(), 
            wait_list: None, dest_list: None, name: &self.name }
    }

    // /// Enqueues kernel on the default command queue.
    // ///
    // /// Specify `queue` to use a non-default queue.
    // ///
    // /// Execution of the kernel on the device will not occur until the events
    // /// in `wait_list` have completed if it is specified. 
    // ///
    // /// Specify `dest_list` to have a new event added to that list associated
    // /// with the completion of this kernel task.
    // ///
    // pub fn enqueue_ndrange<D: Into<SimpleDims>>(&self, queue: Option<&Queue>,  
    //             gwo: Option<D>, gws: Option<D>, lws: Option<D>, wait_list: Option<&EventList>, 
    //             dest_list: Option<&mut EventList>) -> OclResult<()>
    // {
    //     let queue = match queue {
    //         Some(q) => q.core_as_ref(),
    //         None => &self.command_queue_obj_core,
    //     };

    //     // If offset/size is passed, use passed value, if not use stored default:
    //     let gwo = gwo.map(|gwo| gwo.into()).unwrap_or(self.gwo);
    //     let gws = gws.map(|gws| gws.into()).unwrap_or(self.gws);
    //     let lws = lws.map(|lws| lws.into()).unwrap_or(self.lws);

    //     let dim_count = gws.dim_count();

    //     // If gws is still `None` we cannot continue.
    //     let gws = match gws.to_work_size() {
    //         Some(gws) => gws,
    //         None => return OclError::err("ocl::Kernel::enqueue_ndrange: Global Work Size ('gws') \
    //             cannot be left unspecified. Set a default for the kernel or pass a valid
    //             parameter."),
    //     };

    //     let wait_list = wait_list.map(|el| el.core_as_ref());
    //     let dest_list = dest_list.map(|el| el.core_as_mut());

    //     core::enqueue_kernel(queue, &self.obj_core, dim_count, gwo.to_work_offset(), 
    //         gws, lws.to_work_size(), wait_list, dest_list, Some(&self.name))
    // }

    // /// Enqueues kernel on the default command queue.
    // ///
    // /// Specify `queue` to use a non-default queue.
    // ///
    // /// Execution of the kernel on the device will not occur until the events
    // /// in `wait_list` have completed if it is specified. 
    // ///
    // /// Specify `dest_list` to have a new event added to that list associated
    // /// with the completion of this kernel task.
    // ///
    // pub fn enqueue_events(&self, wait_list: Option<&EventList>, 
    //                 dest_list: Option<&mut EventList>) -> OclResult<()>
    // {
    //     core::enqueue_kernel(&self.command_queue_obj_core, &self.obj_core, self.gws.dim_count(), 
    //         self.gwo.to_work_offset(), self.gws.to_work_size().unwrap(), self.lws.to_work_size(), 
    //         wait_list.map(|el| el.core_as_ref()), dest_list.map(|el| el.core_as_mut()), Some(&self.name))
    // }

    /// Enqueues kernel on the default command queue using only default
    /// parameters, panicing instead of returning a result upon error.
    ///
    /// Equivalent to `::cmd().enq()`
    ///
    /// # Panics
    /// 
    /// Panics on anything that would normally return an error. Use
    /// `::cmd().enq()` to get a result instead.
    #[inline]
    pub fn enqueue(&self) {
        // If gws is still `None` we cannot continue.
        let gws = match self.gws.to_work_size() {
            Some(gws) => gws,
            None => OclError::err("Global Work Size ('gws') cannot be left unspecified. \
                Set a default for the kernel before calling with '::set_queue'.")
                .expect("ocl::Kernel::enqueue"),
        };

        core::enqueue_kernel::<EventList, EventList>(&self.command_queue_obj_core, &self.obj_core,
            self.gws.dim_count(), self.gwo.to_work_offset(), gws, self.lws.to_work_size(), 
            None, None, Some(&self.name)) .expect("ocl::Kernel::enqueue")
    }

    /// Permanently changes the default queue.
    ///
    /// Returns a ref for chaining i.e.:
    ///
    /// `kernel.set_queue(queue).enqueue(....);`
    ///
    /// Even when used as above, the queue is changed permanently,
    /// not just for the one call. Changing the queue is cheap so feel free
    /// to change as often as needed.
    ///
    /// If you want to change the queue for only a single call, use: 
    /// `::cmd.queue(...)...enq()...`
    ///
    /// The new queue must be associated with a device associated with the
    /// kernel's program.
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
