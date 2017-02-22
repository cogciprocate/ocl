//! An `OpenCL` kernel.

use std;
use std::ops::{Deref, DerefMut};
use std::collections::HashMap;
use core::{self, OclPrm, Kernel as KernelCore, CommandQueue as CommandQueueCore, Mem as MemCore,
    KernelArg, KernelInfo, KernelInfoResult, KernelArgInfo, KernelArgInfoResult,
    KernelWorkGroupInfo, KernelWorkGroupInfoResult, AsMem, MemCmdAll, ClVersions};
use core::error::{Result as OclResult, Error as OclError};
use standard::{SpatialDims, Program, Queue, WorkDims, Sampler, Device, ClNullEventPtrEnum,
    ClWaitListPtrEnum};

const PRINT_DEBUG: bool = false;

/// A kernel command builder used to queue a kernel with a mix of default
/// and optionally specified arguments.
pub struct KernelCmd<'k> {
    queue: Option<&'k CommandQueueCore>,
    kernel: &'k KernelCore,
    gwo: SpatialDims,
    gws: SpatialDims,
    lws: SpatialDims,
    wait_list: Option<ClWaitListPtrEnum<'k>>,
    new_event: Option<ClNullEventPtrEnum<'k>>,
}

/// A kernel enqueue command.
///
/// [UNSTABLE]: Methods still being tuned.
impl<'k> KernelCmd<'k> {
    /// Specifies a queue to use for this call only.
    pub fn queue<'q, Q>(mut self, queue: &'q Q) -> KernelCmd<'k>
            where 'q: 'k, Q: 'k + AsRef<CommandQueueCore>
    {
        self.queue = Some(queue.as_ref());
        self
    }

    /// Specifies a global work offset for this call only.
     pub fn gwo<D: Into<SpatialDims>>(mut self, gwo: D) -> KernelCmd<'k> {
        self.gwo = gwo.into();
        self
    }

    /// Specifies a global work size for this call only.
    pub fn gws<D: Into<SpatialDims>>(mut self, gws: D) -> KernelCmd<'k> {
        self.gws = gws.into();
        self
    }

    /// Specifies a local work size for this call only.
    pub fn lws<D: Into<SpatialDims>>(mut self, lws: D) -> KernelCmd<'k> {
        self.lws = lws.into();
        self
    }


    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> KernelCmd<'k>
            where 'e: 'k, Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.wait_list = Some(ewait.into());
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt<'e, Ewl>(mut self, ewait: Option<Ewl>) -> KernelCmd<'k>
            where 'e: 'k, Ewl: Into<ClWaitListPtrEnum<'e>>
    {
        self.wait_list = ewait.map(|el| el.into());
        self
    }

    /// Specifies the destination list or empty event for a new, optionally
    /// created event associated with this command.
    // pub fn enew(mut self, new_event_dest: &'k mut ClNullEventPtr) -> KernelCmd<'k> {
    pub fn enew<'e, En>(mut self, new_event_dest: En) -> KernelCmd<'k>
            where 'e: 'k, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.new_event = Some(new_event_dest.into());
        self
    }

    /// Specifies a destination list for a new, optionally created event
    /// associated with this command.
    pub fn enew_opt<'e, En>(mut self, new_event_list: Option<En>) -> KernelCmd<'k>
            where 'e: 'k, En: Into<ClNullEventPtrEnum<'e>>
    {
        self.new_event = new_event_list.map(|e| e.into());
        self
    }

    /// Enqueues this kernel command.
    pub fn enq(self) -> OclResult<()> {
        let queue = match self.queue {
            Some(q) => q,
            None => return Err("KernelCmd::enq: No queue specified.".into()),
        };

        let dim_count = self.gws.dim_count();

        let gws = match self.gws.to_work_size() {
            Some(gws) => gws,
            None => return OclError::err_string("ocl::KernelCmd::enqueue: Global Work Size ('gws') \
                cannot be left unspecified. Set a default for the kernel or pass a valid parameter."),
        };

        if PRINT_DEBUG {
            println!("Enqueuing kernel: '{}'...",
                core::get_kernel_info(self.kernel, KernelInfo::FunctionName));
        }

        core::enqueue_kernel(queue, self.kernel, dim_count, self.gwo.to_work_offset(),
            &gws, self.lws.to_work_size(), self.wait_list, self.new_event)
    }
}



/// A kernel which represents a 'procedure'.
///
/// Corresponds to code which must have already been compiled into a program.
///
/// Set arguments using any of the `::arg...` (builder-style) or
/// `::set_arg...` functions or use `::set_arg` to set arguments by index.
///
/// ### Clonability
///
/// Cloning a kernel after creation should virtually never be necessary (and
/// may indicate that your design needs improvement). If an Rc<Kernel> is
/// insufficient and you really really need to clone and store a kernel, clone
/// the kernel core with `::core.clone()` and use
/// `ocl::core::enqueue_kernel(...)` to enqueue.
///
///
/// TODO: Add more details, examples, etc.
/// TODO: Add information about panics and errors.
/// TODO: Finish arg info formatting.
#[derive(Debug)]
pub struct Kernel {
    obj_core: KernelCore,
    named_args: HashMap<&'static str, u32>,
    mem_args: Vec<Option<MemCore>>,
    arg_count: u32,
    queue: Option<Queue>,
    gwo: SpatialDims,
    gws: SpatialDims,
    lws: SpatialDims,
}

// ######### IMPLEMENT THIS #########
// extern crate fnv;

// use std::collections::HashMap;
// use std::hash::BuildHasherDefault;
// use fnv::FnvHasher;

// type MyHasher = BuildHasherDefault<FnvHasher>;

// fn main() {
//     let mut map: HashMap<_, _, MyHasher> = HashMap::default();
//     map.insert(1, "Hello");
//     map.insert(2, ", world!");
//     println!("{:?}", map);
// }

impl Kernel {
    /// Returns a new kernel.
    pub fn new<S: Into<String>, >(name: S, program: &Program) -> OclResult<Kernel>
    {
        let name = name.into();
        let obj_core = try!(core::create_kernel(program, &name));

        Ok(Kernel {
            obj_core: obj_core,
            named_args: HashMap::with_capacity(5),
            arg_count: 0,
            mem_args: Vec::with_capacity(16),
            queue: None,
            gwo: SpatialDims::Unspecified,
            gws: SpatialDims::Unspecified,
            lws: SpatialDims::Unspecified,
        })
    }

    /// Sets the default queue to be used by all subsequent enqueue commands
    /// unless otherwise changed (with `::set_default_queue`) or overridden
    /// (by `::cmd().queue(...)...`).
    ///
    /// The queue must be associated with a device associated with the
    /// kernel's program.
    pub fn queue(mut self, queue: Queue) -> Kernel {
        self.queue = Some(queue);
        self
    }

    /// Sets the default global work offset (builder-style).
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// making a call to enqueue or building a queue command with `::cmd`.
    pub fn gwo<D: Into<SpatialDims>>(mut self, gwo: D) -> Kernel {
        self.gwo = gwo.into();
        self
    }

    /// Sets the default global work size (builder-style).
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// making a call to enqueue or building a queue command with `::cmd`.
    pub fn gws<D: Into<SpatialDims>>(mut self, gws: D) -> Kernel {
        self.gws = gws.into();
        self
    }

    /// Sets the default local work size (builder-style).
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// making a call to enqueue or building a queue command with `::cmd`.
    pub fn lws<D: Into<SpatialDims>>(mut self, lws: D) -> Kernel {
        self.lws = lws.into();
        self
    }

    /// Adds a new argument to the kernel specifying the buffer object represented
    /// by 'buffer' (builder-style). Argument is added to the bottom of the argument
    /// order.
    // pub fn arg_buf<T: OclPrm>(mut self, buffer: &Buffer<T>) -> Kernel {
    pub fn arg_buf<T, M>(mut self, buffer: M) -> Kernel
            where T: OclPrm, M: AsMem<T> + MemCmdAll
    {
        self.new_arg_buf::<T, _>(Some(buffer));
        self
    }

    /// Adds a new argument to the kernel specifying the image object represented
    /// by 'image' (builder-style). Argument is added to the bottom of the argument
    /// order.
    pub fn arg_img<T, M>(mut self, image: M) -> Kernel
        where T: OclPrm, M: AsMem<T> + MemCmdAll
    {
        self.new_arg_img::<T, _>(Some(image));
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
    pub fn arg_scl<T: OclPrm>(mut self, scalar: T) -> Kernel {
        self.new_arg_scl(Some(scalar));
        self
    }

    /// Adds a new argument specifying the value: `vector` (builder-style). Argument
    /// is added to the bottom of the argument order.
    pub fn arg_vec<T: OclPrm>(mut self, vector: T) -> Kernel {
        self.new_arg_vec(Some(vector));
        self
    }

    /// Adds a new argument specifying the allocation of a local variable of size
    /// `length * sizeof(T)` bytes (builder_style).
    ///
    /// Local variables are used to share data between work items in the same
    /// workgroup.
    pub fn arg_loc<T: OclPrm>(mut self, length: usize) -> Kernel {
        self.new_arg_loc::<T>(length);
        self
    }

    /// Adds a new named argument (in order) specifying the value: `scalar`
    /// (builder-style).
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_scl_named<T: OclPrm>(mut self, name: &'static str, scalar_opt: Option<T>) -> Kernel {
        let arg_idx = self.new_arg_scl(scalar_opt);
        self.named_args.insert(name, arg_idx);
        self
    }

    /// Adds a new named argument (in order) specifying the value: `vector`
    /// (builder-style).
    ///
    /// Named arguments can be easily modified later using `::set_arg_vec_named()`.
    pub fn arg_vec_named<T: OclPrm>(mut self, name: &'static str, vector_opt: Option<T>) -> Kernel {
        let arg_idx = self.new_arg_vec(vector_opt);
        self.named_args.insert(name, arg_idx);
        self
    }

    /// Adds a new named argument specifying the buffer object represented by
    /// 'buffer' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_buf_named()`.
    pub fn arg_buf_named<T, M>(mut self, name: &'static str, buffer_opt: Option<M>) -> Kernel
            where T: OclPrm, M: AsMem<T> + MemCmdAll
    {
        let arg_idx = self.new_arg_buf::<T, _>(buffer_opt);
        self.named_args.insert(name, arg_idx);
        self
    }

    /// Adds a new named argument specifying the image object represented by
    /// 'image' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_img_named()`.
    pub fn arg_img_named<T, M>(mut self, name: &'static str, image_opt: Option<M>) -> Kernel
            where T: OclPrm, M: AsMem<T> + MemCmdAll
    {
        let arg_idx = self.new_arg_img::<T, _>(image_opt);
        self.named_args.insert(name, arg_idx);
        self
    }

    /// Adds a new named argument specifying the sampler object represented by
    /// 'sampler' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_smp_named()`.
    pub fn arg_smp_named(mut self, name: &'static str, sampler_opt: Option<&Sampler>) -> Kernel {
        let arg_idx = self.new_arg_smp(sampler_opt);
        self.named_args.insert(name, arg_idx);
        self
    }

    /// Modifies the kernel argument named: `name`.
    ///
    /// ## Panics [FIXME]
    // [FIXME]: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_scl_named<'a, T: OclPrm>(&'a mut self, name: &'static str, scalar: T)
            -> OclResult<&'a mut Kernel>
    {
        let arg_idx = try!(self.resolve_named_arg_idx(name));
        self.set_arg::<T>(arg_idx, KernelArg::Scalar(scalar))
            .and(Ok(self))
    }

    /// Modifies the kernel argument named: `name`.
    ///
    /// ## Panics [FIXME]
    // [FIXME]: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_vec_named<'a, T: OclPrm>(&'a mut self, name: &'static str, vector: T)
            -> OclResult<&'a mut Kernel>
    {
        let arg_idx = try!(self.resolve_named_arg_idx(name));
        self.set_arg::<T>(arg_idx, KernelArg::Vector(vector))
            .and(Ok(self))
    }

    /// Modifies the kernel argument named: `name`.
    ///
    /// ## Panics [FIXME]
    // [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_buf_named<'a, T, M>(&'a mut self, name: &'static str,
            buffer_opt: Option<M>)
            -> OclResult<&'a mut Kernel>
            where T: OclPrm, M: AsMem<T> + MemCmdAll
    {
        //  TODO: ADD A CHECK FOR A VALID NAME (KEY)
        let arg_idx = try!(self.resolve_named_arg_idx(name));
        match buffer_opt {
            Some(buffer) => {
                self.set_arg::<T>(arg_idx, KernelArg::Mem(buffer.as_mem()))
            },
            None => {
                self.set_arg::<T>(arg_idx, KernelArg::MemNull)
            },
        }.and(Ok(self))
    }

    /// Modifies the kernel argument named: `name`.
    ///
    /// ## Panics [FIXME]
    // [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_img_named<'a, T, M>(&'a mut self, name: &'static str,
            image_opt: Option<M>)
            -> OclResult<&'a mut Kernel>
            where T: OclPrm, M: AsMem<T> + MemCmdAll
    {
        //  TODO: ADD A CHECK FOR A VALID NAME (KEY)
        let arg_idx = try!(self.resolve_named_arg_idx(name));
        match image_opt {
            Some(img) => {
                self.set_arg::<T>(arg_idx, KernelArg::Mem(img.as_mem()))
            },
            None => {
                self.set_arg::<T>(arg_idx, KernelArg::MemNull)
            },
        }.and(Ok(self))
    }

    /// Sets the value of a named sampler argument.
    ///
    /// ## Panics [FIXME]
    // [PLACEHOLDER] Set a named sampler argument
    #[allow(unused_variables)]
    pub fn set_arg_smp_named<'a, T: OclPrm>(&'a mut self, name: &'static str,
                sampler_opt: Option<&Sampler>) -> OclResult<&'a mut Kernel>
    {
        unimplemented!();
    }

    /// Returns a command builder which is used to chain parameters of an
    /// 'enqueue' command together.
    pub fn cmd(&self) -> KernelCmd {
        KernelCmd { queue: self.queue.as_ref().map(|q| q.as_ref()), kernel: &self.obj_core,
            gwo: self.gwo, gws: self.gws, lws: self.lws,
            wait_list: None, new_event: None }
    }

    /// Enqueues this kernel on the default queue and returns the result.
    ///
    /// Shorthand for `.cmd().enq()`
    ///
    pub fn enq(&self) -> OclResult<()> {
        // core::enqueue_kernel::<EventList>(&self.queue, &self.obj_core,
        //     self.gws.dim_count(), self.gwo.to_work_offset(), &self.gws.to_lens().unwrap(), self.lws.to_work_size(),
        //     None, None)
        self.cmd().enq()
    }

    /// Changes the default queue.
    ///
    /// Returns a ref for chaining i.e.:
    ///
    /// `kernel.set_default_queue(queue).enqueue(....);`
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
    ///
    pub fn set_default_queue(&mut self, queue: Queue) -> &mut Kernel {
        // self.command_queue_obj_core = queue.core().clone();
        self.queue = Some(queue);
        self
    }

    /// Returns the default queue for this kernel.
    pub fn default_queue(&self) -> Option<&Queue> {
        self.queue.as_ref()
    }

    /// Returns the default global work offset.
    pub fn get_gwo(&self) -> SpatialDims {
        self.gwo
    }

    /// Returns the default global work size.
    pub fn get_gws(&self) -> SpatialDims {
        self.gws
    }

    /// Returns the default local work size.
    pub fn get_lws(&self) -> SpatialDims {
        self.lws
    }

    /// Returns the number of arguments specified for this kernel.
    #[inline]
    pub fn arg_count(&self) -> u32 {
        self.arg_count
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[deprecated(since="0.13.0", note="Use `::core` instead.")]
    pub fn core_as_ref(&self) -> &KernelCore {
        &self.obj_core
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn core(&self) -> &KernelCore {
        &self.obj_core
    }

    /// Returns information about this kernel.
    pub fn info(&self, info_kind: KernelInfo) -> KernelInfoResult {
        core::get_kernel_info(&self.obj_core, info_kind)
    }

    /// Returns argument information for this kernel.
    pub fn arg_info(&self, arg_index: u32, info_kind: KernelArgInfo) -> KernelArgInfoResult {
        let device_versions = match self.obj_core.device_versions() {
            Ok(vers) => vers,
            Err(e) => return e.into(),
        };

        core::get_kernel_arg_info(&self.obj_core, arg_index, info_kind,
            Some(&device_versions))
    }

    /// Returns work group information for this kernel.
    pub fn wg_info(&self, device: Device, info_kind: KernelWorkGroupInfo)
            -> KernelWorkGroupInfoResult
    {
        core::get_kernel_work_group_info(&self.obj_core, device, info_kind)
    }

    /// Returns the name of this kernel.
    pub fn name(&self) -> String {
        core::get_kernel_info(&self.obj_core, KernelInfo::FunctionName).into()
    }

    /// Returns the argument index of a named argument if it exists.
    pub fn named_arg_idx(&self, name: &'static str) -> Option<u32> {
        self.named_args.get(name).cloned()
    }

    /// Sets an argument by index.
    pub fn set_arg<T: OclPrm>(&mut self, arg_idx: u32, arg: KernelArg<T>) -> OclResult<()> {
        // If the `KernelArg` is a `Mem` variant, clone the `MemCore` it
        // refers to, store it in `self.mem_args`, and create a new
        // `KernelArg::Mem` referring to the locally stored copy. This prevents
        // a buffer which has gone out of scope from being erroneously referred
        // to when this kernel is enqueued and causing either a misleading
        // error message or a hard to debug segfault depending on the
        // platform.
        let arg = match arg {
            KernelArg::Mem(mem) => {
                self.mem_args[arg_idx as usize] = Some(mem.clone());
                let mem_arg_ref = self.mem_args.get(arg_idx as usize).unwrap().as_ref().unwrap();
                KernelArg::Mem(mem_arg_ref)
            },
            arg => {
                self.mem_args[arg_idx as usize] = None;
                arg
            },
        };

        core::set_kernel_arg::<T>(&self.obj_core, arg_idx, arg)
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

    // fn fmt_arg_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    //     f.debug_struct("Kernel")
    //         .field("FunctionName", &self.info(KernelInfo::FunctionName))
    //         .field("ReferenceCount", &self.info(KernelInfo::ReferenceCount))
    //         .field("Context", &self.info(KernelInfo::Context))
    //         .field("Program", &self.info(KernelInfo::Program))
    //         .field("Attributes", &self.info(KernelInfo::Attributes))
    //         .finish()
    // }

    fn fmt_wg_info<D>(&self, f: &mut std::fmt::Formatter, devices: Vec<D>) -> std::fmt::Result
            where D: Into<Device>
    {
        for device in devices {
            let device = device.into();

            f.debug_struct("WorkGroup")
                .field("WorkGroupSize", &self.wg_info(device, KernelWorkGroupInfo::WorkGroupSize))
                .field("CompileWorkGroupSize", &self.wg_info(device, KernelWorkGroupInfo::CompileWorkGroupSize))
                .field("LocalMemSize", &self.wg_info(device, KernelWorkGroupInfo::LocalMemSize))
                .field("PreferredWorkGroupSizeMultiple",
                    &self.wg_info(device, KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple))
                .field("PrivateMemSize", &self.wg_info(device, KernelWorkGroupInfo::PrivateMemSize))
                // .field("GlobalWorkSize", &self.wg_info(device, KernelWorkGroupInfo::GlobalWorkSize))
                .finish()?;

        }

        Ok(())
    }

    /// Resolves the index of a named argument.
    fn resolve_named_arg_idx(&self, name: &'static str) -> OclResult<u32> {
        match self.named_args.get(name) {
            Some(&ai) => Ok(ai),
            None => {
                OclError::err_string(format!("Kernel::set_arg_scl_named(): Invalid argument \
                    name: '{}'.", name))
            },
        }
    }

    /// Non-builder-style version of `::arg_buf()`.
    fn new_arg_buf<T, M>(&mut self, buffer_opt: Option<M>) -> u32
            where T: OclPrm, M: AsMem<T> + MemCmdAll
    {
        match buffer_opt {
            Some(buffer) => {
                self.new_arg::<T>(KernelArg::Mem(buffer.as_mem()))
            },
            None => {
                self.new_arg::<T>(KernelArg::MemNull)
            },
        }
    }

    /// Non-builder-style version of `::arg_img()`.
    fn new_arg_img<T, M>(&mut self, image_opt: Option<M>) -> u32
        where T: OclPrm, M: AsMem<T> + MemCmdAll
    {
        match image_opt {
            Some(image) => {
                // Type is ignored:
                self.new_arg::<u8>(KernelArg::Mem(image.as_mem()))
            },
            None => {
                self.new_arg::<u8>(KernelArg::MemNull)
            },
        }
    }

    /// Non-builder-style version of `::arg_img()`.
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

    /// Non-builder-style version of `::arg_scl()`.
    fn new_arg_scl<T: OclPrm>(&mut self, scalar_opt: Option<T>) -> u32 {
        let scalar = match scalar_opt {
            Some(s) => s,
            None => Default::default(),
        };

        self.new_arg::<T>(KernelArg::Scalar(scalar))
    }

    /// Non-builder-style version of `::arg_vec()`.
    fn new_arg_vec<T: OclPrm>(&mut self, vector_opt: Option<T>) -> u32 {

        let vector = match vector_opt {
            Some(s) => s,
            None => Default::default(),
        };

        self.new_arg::<T>(KernelArg::Vector(vector))
    }

    /// Non-builder-style version of `::arg_loc()`.
    fn new_arg_loc<T: OclPrm>(&mut self, length: usize) -> u32 {
        self.new_arg::<T>(KernelArg::Local(&length))
    }

    /// Adds a new argument to the kernel and returns the index.
    fn new_arg<T: OclPrm>(&mut self, arg: KernelArg<T>) -> u32 {
        let arg_idx = self.arg_count;

        // Push an empty `mem_arg` to the list just to make room.
        self.mem_args.push(None);

        match self.set_arg(arg_idx, arg) {
            Ok(_) => (),
            Err(err) => {
                panic!("Kernel::new_arg(arg_idx: {}): {}",
                    arg_idx, err);
            }
        }

        self.arg_count += 1;
        debug_assert!(self.arg_count as usize == self.mem_args.len());
        arg_idx
    }
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        try!(self.fmt_info(f));
        try!(write!(f, " "));
        self.fmt_wg_info(f, self.obj_core.devices().unwrap())
    }
}

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
