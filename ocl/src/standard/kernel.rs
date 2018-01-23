//! An `OpenCL` kernel.

use std;
use std::ops::{Deref, DerefMut};
use std::any::Any;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use core::{self, OclPrm, Kernel as KernelCore, CommandQueue as CommandQueueCore, Mem as MemCore,
    KernelArg, KernelInfo, KernelInfoResult, KernelArgInfo, KernelArgInfoResult,
    KernelWorkGroupInfo, KernelWorkGroupInfoResult, AsMem, MemCmdAll, ClVersions};
use core::error::{Result as OclCoreResult, ErrorKind as OclCoreErrorKind};
use error::{Error as OclError, Result as OclResult};
use standard::{SpatialDims, Program, Queue, WorkDims, Sampler, Device, ClNullEventPtrEnum,
    ClWaitListPtrEnum};
pub use self::arg_type::{BaseType, Cardinality, ArgType};

const PRINT_DEBUG: bool = false;

/// A kernel command builder used to queue a kernel with a mix of default
/// and optionally specified arguments.
#[must_use = "commands do nothing unless enqueued"]
pub struct KernelCmd<'k> {
    queue: Option<&'k CommandQueueCore>,
    kernel: &'k KernelCore,
    gwo: SpatialDims,
    gws: SpatialDims,
    lws: SpatialDims,
    wait_events: Option<ClWaitListPtrEnum<'k>>,
    new_event: Option<ClNullEventPtrEnum<'k>>,
}

/// A kernel enqueue command.
///
/// [UNSTABLE]: Methods still being tuned.
impl<'k> KernelCmd<'k> {
    /// Specifies a queue to use for this call only.
    ///
    /// Overrides the kernel's default queue if one is set. If no default
    /// queue is set, this method **must** be called before enqueuing the
    /// kernel.
    pub fn queue<'q, Q>(mut self, queue: &'q Q) -> KernelCmd<'k>
            where 'q: 'k, Q: 'k + AsRef<CommandQueueCore> {
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

    /// Specifies an event or list of events to wait on before the command
    /// will run.
    ///
    /// When events generated using the `::enew` method of **other**,
    /// previously enqueued commands are passed here (either individually or
    /// as part of an [`EventList`]), this command will not execute until
    /// those commands have completed.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event_list = EventList::new();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event_list).enq()?;
    /// // Read from a buffer using `queue_2`, ensuring the read does not begin until
    /// // after the kernel command has completed:
    /// buffer.read(rwvec.clone()).queue(&queue_2).ewait(&event_list).enq_async()?;
    /// ```
    ///
    /// [`EventList`]: struct.EventList.html
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> KernelCmd<'k>
            where 'e: 'k, Ewl: Into<ClWaitListPtrEnum<'e>> {
        self.wait_events = Some(ewait.into());
        self
    }

    /// Specifies the destination to store a new, optionally created event
    /// associated with this command.
    ///
    /// The destination can be a mutable reference to an empty event (created
    /// using [`Event::empty`]) or a mutable reference to an event list.
    ///
    /// After this command is enqueued, the event in the destination can be
    /// passed to the `::ewait` method of another command. Doing so will cause
    /// the other command to wait until this command has completed before
    /// executing.
    ///
    /// Using events can compliment the use of queues to order commands by
    /// creating temporal dependencies between them (where commands in one
    /// queue must wait for the completion of commands in another). Events can
    /// also supplant queues altogether when, for example, using out-of-order
    /// queues.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Create an event list:
    /// let mut event = Event::empty();
    /// // Enqueue a kernel on `queue_1`, creating an event representing the kernel
    /// // command in our list:
    /// kernel.cmd().queue(&queue_1).enew(&mut event).enq()?;
    /// // Read from a buffer using `queue_2`, ensuring the read does not begin until
    /// // after the kernel command has completed:
    /// buffer.read(rwvec.clone()).queue(&queue_2).ewait(&event).enq_async()?;
    /// ```
    ///
    /// [`Event::empty`]: struct.Event.html#method.empty
    pub fn enew<'e, En>(mut self, new_event_dest: En) -> KernelCmd<'k>
            where 'e: 'k, En: Into<ClNullEventPtrEnum<'e>> {
        self.new_event = Some(new_event_dest.into());
        self
    }

    /// Enqueues this kernel command.
    ///
    /// # Safety
    ///
    /// All kernel code must be considered untrusted. Therefore the act of
    /// calling this function contains implied unsafety even though the API
    /// itself is safe.
    pub unsafe fn enq(self) -> OclResult<()> {
        let queue = match self.queue {
            Some(q) => q,
            None => return Err("KernelCmd::enq: No queue specified.".into()),
        };

        let dim_count = self.gws.dim_count();

        let gws = match self.gws.to_work_size() {
            Some(gws) => gws,
            None => return Err("ocl::KernelCmd::enqueue: Global Work Size ('gws') \
                cannot be left unspecified. Set a default for the kernel or pass a \
                valid parameter.".into()),
        };

        if PRINT_DEBUG {
            println!("Enqueuing kernel: '{}'...",
                core::get_kernel_info(self.kernel, KernelInfo::FunctionName)?);
        }

        core::enqueue_kernel(queue, self.kernel, dim_count, self.gwo.to_work_offset(),
            &gws, self.lws.to_work_size(), self.wait_events, self.new_event)
            .map_err(OclError::from)
    }
}


/// A kernel which represents a 'procedure'.
///
/// Corresponds to code which must have already been compiled into a program.
///
/// Set arguments using any of the `::arg...` (builder-style) or
/// `::set_arg...` functions or use `::set_arg` to set arguments by index.
///
/// `Kernel` includes features that a raw OpenCL kernel does not, including:
///
/// 1. Type-checked arguments (not just size-checked)
/// 2. Named arguments (with a `&'static str` name)
/// 3. Prevention of a potential (difficult to debug) segfault if a buffer or
///    image used by a kernel is dropped prematurely.
/// 4. Stored defaults for the:
///     - Queue
///     - Global Work Offset
///     - Global Work Size
///     - Local Work Size
//
// ### `Clone`, `Send`, and segfaults
//
// Every struct field of `Kernel` is safe to `Send` and `Clone` (after all of
// the arguments are specified) with the exception of `mem_args`. In order to
// keep references to buffers/images alive throughout the life of the kernel
// and prevent nasty, platform-dependent, and very hard to debug segfaults,
// storing `MemCore`s (buffers/images) is necessary. However, storing them
// means that there are compromises in other areas. The following are the
// options as I see them:
//
// 1. Store buffers/images in an Rc<RefCell>. This allows us to
//    `Clone` but not to `Send` between threads.
// 2. [CURRENT] Store buffers/images in an Arc<Mutex/RwLock> allowing both `Clone` and
//    `Send` at the cost of performance (could add up if users constantly
//    change arguments) [UPDATE]: Performance cost of this is below negligible.
// 3. [PREVIOUS] Disallow cloning and sending.
// 4. Don't store buffer/image references and let them segfault if the user
//    doesn't keep them alive properly.
//
// Please provide feedback by filing an [issue] if you have thoughts,
// suggestions, or alternative ideas.
//
// [issue]: https://github.com/cogciprocate/ocl/issues
//
// * TODO: Add more details, examples, etc.
// * TODO: Add information about panics and errors.
// * TODO: Finish arg info formatting.
//
// * TODO: Consider switching to option 1 above since sending kernels between
//   threads is not entirely safe anyway.
//
// From: https://www.khronos.org/registry/OpenCL/sdk/1.1/docs/man/xhtml/clSetKernelArg.html:
//
// An OpenCL API call is considered to be thread-safe if the internal state as
// managed by OpenCL remains consistent when called simultaneously by multiple
// host threads. OpenCL API calls that are thread-safe allow an application to
// call these functions in multiple host threads without having to implement
// mutual exclusion across these host threads i.e. they are also
// re-entrant-safe.
//
// All OpenCL API calls are thread-safe except clSetKernelArg, which is safe
// to call from any host thread, and is safe to call re-entrantly so long as
// concurrent calls operate on different cl_kernel objects. However, the
// behavior of the cl_kernel object is undefined if clSetKernelArg is called
// from multiple host threads on the same cl_kernel object at the same time.
//
// There is an inherent race condition in the design of OpenCL that occurs
// between setting a kernel argument and using the kernel with
// clEnqueueNDRangeKernel or clEnqueueTask. Another host thread might change
// the kernel arguments between when a host thread sets the kernel arguments
// and then enqueues the kernel, causing the wrong kernel arguments to be
// enqueued. Rather than attempt to share cl_kernel objects among multiple
// host threads, applications are strongly encouraged to make additional
// cl_kernel objects for kernel functions for each host thread.
//
#[derive(Debug)]
pub struct Kernel {
    obj_core: KernelCore,
    named_args: Option<HashMap<&'static str, u32>>,
    mem_args: Arc<Mutex<Vec<Option<MemCore>>>>,
    new_arg_count: u32,
    queue: Option<Queue>,
    gwo: SpatialDims,
    gws: SpatialDims,
    lws: SpatialDims,
    num_args: u32,
    arg_types: Vec<ArgType>,
    /// Bypasses argument type check if true:
    bypass_arg_check: bool,
}


impl Kernel {
    /// Returns a new kernel.
    pub fn new<S: AsRef<str>>(name: S, program: &Program) -> OclResult<Kernel> {
        // let name = name.into();
        let obj_core = core::create_kernel(program, name)?;

        let num_args = match core::get_kernel_info(&obj_core, KernelInfo::NumArgs) {
            Ok(KernelInfoResult::NumArgs(num)) => num,
            Err(err) => return Err(OclError::from(err)),
            _=> unreachable!(),
        };

        let mut arg_types = Vec::with_capacity(num_args as usize);
        let mut bypass_arg_check = false;

        // Cache argument types for later use, bypassing if the OpenCL version
        // is too low (v1.1).
        for arg_idx in 0..num_args {
            let arg_type = match ArgType::from_kern_and_idx(&obj_core, arg_idx) {
                Ok(at) => at,
                Err(e) => {
                    if let OclCoreErrorKind::VersionLow { .. } = *e.kind() {
                        bypass_arg_check = true;
                        break;
                    }
                    return Err(OclError::from(e));
                },
            };
            arg_types.push(arg_type);
        }

        let mem_args = vec![None; num_args as usize];

        Ok(Kernel {
            obj_core: obj_core,
            named_args: None,
            new_arg_count: 0,
            mem_args: Arc::new(Mutex::new(mem_args)),
            queue: None,
            gwo: SpatialDims::Unspecified,
            gws: SpatialDims::Unspecified,
            lws: SpatialDims::Unspecified,
            num_args: num_args,
            arg_types: arg_types,
            bypass_arg_check,
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
            where T: OclPrm + 'static, M: AsMem<T> + MemCmdAll {
        self.new_arg_buf::<T, _>(Some(buffer));
        self
    }

    /// Adds a new argument to the kernel specifying the image object represented
    /// by 'image' (builder-style). Argument is added to the bottom of the argument
    /// order.
    pub fn arg_img<T, M>(mut self, image: M) -> Kernel
            where T: OclPrm, M: AsMem<T> + MemCmdAll {
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
    pub fn arg_scl<T: OclPrm>(mut self, scalar: T) -> Kernel
            where T: OclPrm + 'static {
        self.new_arg_scl(Some(scalar));
        self
    }

    /// Adds a new argument specifying the value: `vector` (builder-style). Argument
    /// is added to the bottom of the argument order.
    pub fn arg_vec<T: OclPrm>(mut self, vector: T) -> Kernel
            where T: OclPrm + 'static {
        self.new_arg_vec(Some(vector));
        self
    }

    /// Adds a new argument specifying the allocation of a local variable of size
    /// `length * sizeof(T)` bytes (builder_style).
    ///
    /// Local variables are used to share data between work items in the same
    /// workgroup.
    pub fn arg_loc<T: OclPrm>(mut self, length: usize) -> Kernel
            where T: OclPrm + 'static {
        self.new_arg_loc::<T>(length);
        self
    }

    /// Adds a new named argument (in order) specifying the value: `scalar`
    /// (builder-style).
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    pub fn arg_scl_named<T: OclPrm>(mut self, name: &'static str, scalar_opt: Option<T>) -> Kernel
            where T: OclPrm + 'static {
        let arg_idx = self.new_arg_scl(scalar_opt);
        self.insert_named_arg(name, arg_idx);
        self
    }

    /// Adds a new named argument (in order) specifying the value: `vector`
    /// (builder-style).
    ///
    /// Named arguments can be easily modified later using `::set_arg_vec_named()`.
    pub fn arg_vec_named<T: OclPrm>(mut self, name: &'static str, vector_opt: Option<T>) -> Kernel
            where T: OclPrm + 'static {
        let arg_idx = self.new_arg_vec(vector_opt);
        self.insert_named_arg(name, arg_idx);
        self
    }

    /// Adds a new named argument specifying the buffer object represented by
    /// 'buffer' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_buf_named()`.
    pub fn arg_buf_named<T, M>(mut self, name: &'static str, buffer_opt: Option<M>) -> Kernel
            where T: OclPrm + 'static, M: AsMem<T> + MemCmdAll {
        let arg_idx = self.new_arg_buf::<T, _>(buffer_opt);
        self.insert_named_arg(name, arg_idx);
        self
    }

    /// Adds a new named argument specifying the image object represented by
    /// 'image' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_img_named()`.
    pub fn arg_img_named<T, M>(mut self, name: &'static str, image_opt: Option<M>) -> Kernel
            where T: OclPrm, M: AsMem<T> + MemCmdAll {
        let arg_idx = self.new_arg_img::<T, _>(image_opt);
        self.insert_named_arg(name, arg_idx);
        self
    }

    /// Adds a new named argument specifying the sampler object represented by
    /// 'sampler' (builder-style). Argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_smp_named()`.
    pub fn arg_smp_named(mut self, name: &'static str, sampler_opt: Option<&Sampler>) -> Kernel {
        let arg_idx = self.new_arg_smp(sampler_opt);
        self.insert_named_arg(name, arg_idx);
        self
    }

    /// Modifies the kernel argument named: `name`.
    ///
    /// ## Panics [FIXME]
    // [FIXME]: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_scl_named<'a, T>(&'a mut self, name: &'static str, scalar: T)
            -> OclResult<&'a mut Kernel>
            where T: OclPrm + 'static {
        let arg_idx = self.resolve_named_arg_idx(name)?;
        self._set_arg::<T>(arg_idx, KernelArg::Scalar(scalar))
            .and(Ok(self))
    }

    /// Modifies the kernel argument named: `name`.
    ///
    /// ## Panics [FIXME]
    // [FIXME]: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_vec_named<'a, T>(&'a mut self, name: &'static str, vector: T)
            -> OclResult<&'a mut Kernel>
            where T: OclPrm + 'static {
        let arg_idx = self.resolve_named_arg_idx(name)?;
        self._set_arg::<T>(arg_idx, KernelArg::Vector(vector))
            .and(Ok(self))
    }

    /// Modifies the kernel argument named: `name`.
    ///
    /// ## Panics [FIXME]
    // * [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_buf_named<'a, T, M>(&'a mut self, name: &'static str,
            buffer_opt: Option<M>)
            -> OclResult<&'a mut Kernel>
            where T: OclPrm + 'static, M: AsMem<T> + MemCmdAll {
        //  * TODO: ADD A CHECK FOR A VALID NAME (KEY)
        let arg_idx = self.resolve_named_arg_idx(name)?;
        match buffer_opt {
            Some(buffer) => {
                self._set_arg::<T>(arg_idx, KernelArg::Mem(buffer.as_mem()))
            },
            None => {
                self._set_arg::<T>(arg_idx, KernelArg::MemNull)
            },
        }.and(Ok(self))
    }

    /// Modifies the kernel argument named: `name`.
    ///
    /// ## Panics [FIXME]
    // * [FIXME] TODO: CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE
    pub fn set_arg_img_named<'a, T, M>(&'a mut self, name: &'static str,
            image_opt: Option<M>)
            -> OclResult<&'a mut Kernel>
            where T: OclPrm + 'static, M: AsMem<T> + MemCmdAll {
        // * TODO: ADD A CHECK FOR A VALID NAME (KEY)
        let arg_idx = self.resolve_named_arg_idx(name)?;
        match image_opt {
            Some(img) => {
                self._set_arg::<T>(arg_idx, KernelArg::Mem(img.as_mem()))
            },
            None => {
                self._set_arg::<T>(arg_idx, KernelArg::MemNull)
            },
        }.and(Ok(self))
    }

    /// Sets the value of a named sampler argument.
    ///
    /// ## Panics [FIXME]
    // [PLACEHOLDER] Set a named sampler argument
    #[allow(unused_variables)]
    pub fn set_arg_smp_named<'a, T: OclPrm>(&'a mut self, name: &'static str,
            sampler_opt: Option<&Sampler>) -> OclResult<&'a mut Kernel> {
        unimplemented!();
    }

    /// Returns a command builder which is used to chain parameters of an
    /// 'enqueue' command together.
    pub fn cmd(&self) -> KernelCmd {
        KernelCmd { queue: self.queue.as_ref().map(|q| q.as_ref()), kernel: &self.obj_core,
            gwo: self.gwo, gws: self.gws, lws: self.lws,
            wait_events: None, new_event: None }
    }

    /// Enqueues this kernel on the default queue and returns the result.
    ///
    /// Shorthand for `.cmd().enq()`
    ///
    /// # Safety
    ///
    /// All kernel code must be considered untrusted. Therefore the act of
    /// calling this function contains implied unsafety even though the API
    /// itself is safe.
    ///
    pub unsafe fn enq(&self) -> OclResult<()> {
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
    pub fn new_arg_count(&self) -> u32 {
        self.new_arg_count
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn as_core(&self) -> &KernelCore {
        &self.obj_core
    }

    /// Returns information about this kernel.
    pub fn info(&self, info_kind: KernelInfo) -> OclCoreResult<KernelInfoResult> {
        core::get_kernel_info(&self.obj_core, info_kind)
    }

    /// Returns argument information for this kernel.
    pub fn arg_info(&self, arg_index: u32, info_kind: KernelArgInfo)
            -> OclCoreResult<KernelArgInfoResult> {
        arg_info(self, arg_index, info_kind)
    }

    /// Returns work group information for this kernel.
    pub fn wg_info(&self, device: Device, info_kind: KernelWorkGroupInfo)
            -> OclCoreResult<KernelWorkGroupInfoResult> {
        core::get_kernel_work_group_info(&self.obj_core, device, info_kind)
    }

    /// Returns the name of this kernel.
    pub fn name(&self) -> OclCoreResult<String> {
        core::get_kernel_info(&self.obj_core, KernelInfo::FunctionName).map(|r| r.into())
    }

    /// Returns the number of arguments this kernel has.
    pub fn num_args(&self) -> OclCoreResult<u32> {
        match core::get_kernel_info(&self.obj_core, KernelInfo::NumArgs) {
            Ok(KernelInfoResult::NumArgs(num)) => Ok(num),
            Err(err) => Err(err),
            _=> unreachable!(),
        }
    }

    /// Returns the argument index of a named argument if it exists.
    pub fn named_arg_idx(&self, name: &'static str) -> Option<u32> {
        self.resolve_named_arg_idx(name).ok()
    }

    /// Verifies that a type matches the kernel arg info:
    ///
    /// This function does nothing and always returns `Ok` if the OpenCL
    /// version of any of the devices associated with this kernel is below
    /// 1.2.
    pub fn verify_arg_type<T: OclPrm + Any>(&self, arg_index: u32) -> OclResult<()> {
        if self.bypass_arg_check { return Ok(()); }

        let arg_type = self.arg_types.get(arg_index as usize)
            .ok_or(format!("Kernel arg index out of range. (kernel: {}, index: {})",
                self.name()?, arg_index))?;

        if arg_type.is_match::<T>() {
            Ok(())
        } else {
            let type_name = arg_type_name(&self.obj_core, arg_index)?;
            Err(format!("Kernel argument type mismatch. The argument at index [{}] \
                is a '{}' ({:?}).", arg_index, type_name, arg_type).into())
        }
    }

    /// Sets an argument by index without checks of any kind.
    ///
    /// Setting buffer or image (`cl_mem`) arguments this way may cause
    /// segfaults or errors if the buffer goes out of scope at any point
    /// before this kernel is dropped.
    ///
    /// This method also bypasses the check to determine if the type you are
    /// passing matches the type defined in your kernel.
    ///
    pub unsafe fn set_arg_unchecked<T: OclPrm>(&mut self, arg_idx: u32,
            arg: KernelArg<T>) -> OclResult<()> {
        core::set_kernel_arg::<T>(&self.obj_core, arg_idx, arg).map_err(OclError::from)
    }

    /// Sets an argument by index.
    fn _set_arg<T: OclPrm + 'static>(&mut self, arg_idx: u32, arg: KernelArg<T>) -> OclResult<()> {
        self.verify_arg_type::<T>(arg_idx)?;

        // If the `KernelArg` is a `Mem` variant, clone the `MemCore` it
        // refers to, store it in `self.mem_args`, and create a new
        // `KernelArg::Mem` referring to the locally stored copy. This prevents
        // a buffer which has gone out of scope from being erroneously referred
        // to when this kernel is enqueued and causing either a misleading
        // error message or a hard to debug segfault depending on the
        // platform.
        let arg = match arg {
            KernelArg::Mem(mem) => {
                self.mem_args.lock().unwrap()[arg_idx as usize] = Some(mem.clone());
                KernelArg::Mem(&mem)
            },
            arg => arg,
        };

        core::set_kernel_arg::<T>(&self.obj_core, arg_idx, arg).map_err(OclError::from)
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
            where D: Into<Device> {
        for device in devices {
            let device = device.into();
            if !device.vendor().unwrap()
                    .contains("NVIDIA") {
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
        }
        Ok(())
    }

    /// Panics if this kernel has already had all arguments assigned.
    fn assert_unlocked(&self) {
        assert!(self.new_arg_count < self.num_args, "This kernel has already had \
            all arguments specified. Check the argument list.");
    }

    /// Inserts a named argument into the name->idx map.
    fn insert_named_arg(&mut self, name: &'static str, arg_idx: u32) {
        if self.named_args.is_none() {
            self.named_args = Some(HashMap::with_capacity(8));
        }

        self.named_args.as_mut().unwrap().insert(name, arg_idx);
    }

    /// Resolves the index of a named argument with a friendly error message.
    fn resolve_named_arg_idx(&self, name: &'static str) -> OclResult<u32> {
        match self.named_args {
            Some(ref map) => {
                match map.get(name) {
                    Some(&ai) => Ok(ai),
                    None => Err(format!("Kernel::set_arg_scl_named(): Invalid argument \
                        name: '{}'.", name).into()),
                }
            },
            None => Err("Kernel::resolve_named_arg_idx: No named arguments declared.".into()),
        }
    }

    /// Non-builder-style version of `::arg_buf()`.
    fn new_arg_buf<T, M>(&mut self, buffer_opt: Option<M>) -> u32
            where T: OclPrm + 'static, M: AsMem<T> + MemCmdAll {
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
            where T: OclPrm, M: AsMem<T> + MemCmdAll {
        match image_opt {
            Some(image) => {
                // Type is ignored:
                self.new_arg::<u64>(KernelArg::Mem(image.as_mem()))
            },
            None => {
                self.new_arg::<u64>(KernelArg::MemNull)
            },
        }
    }

    /// Non-builder-style version of `::arg_img()`.
    fn new_arg_smp(&mut self, sampler_opt: Option<&Sampler>) -> u32 {
        match sampler_opt {
            Some(sampler) => {
                // Type is ignored:
                self.new_arg::<u64>(KernelArg::Sampler(sampler))
            },
            None => {
                self.new_arg::<u64>(KernelArg::SamplerNull)
            },
        }
    }

    /// Non-builder-style version of `::arg_scl()`.
    fn new_arg_scl<T>(&mut self, scalar_opt: Option<T>) -> u32
            where T: OclPrm + 'static {
        let scalar = match scalar_opt {
            Some(s) => s,
            None => Default::default(),
        };

        self.new_arg::<T>(KernelArg::Scalar(scalar))
    }

    /// Non-builder-style version of `::arg_vec()`.
    fn new_arg_vec<T>(&mut self, vector_opt: Option<T>) -> u32
            where T: OclPrm + 'static {

        let vector = match vector_opt {
            Some(s) => s,
            None => Default::default(),
        };

        self.new_arg::<T>(KernelArg::Vector(vector))
    }

    /// Non-builder-style version of `::arg_loc()`.
    fn new_arg_loc<T>(&mut self, length: usize) -> u32
            where T: OclPrm + 'static {
        self.new_arg::<T>(KernelArg::Local(&length))
    }

    /// Adds a new argument to the kernel and returns the index.
    fn new_arg<T>(&mut self, arg: KernelArg<T>) -> u32
            where T: OclPrm + 'static {
        self.assert_unlocked();
        let arg_idx = self.new_arg_count;

        match self._set_arg(arg_idx, arg) {
            Ok(_) => (),
            Err(err) => {
                panic!("Kernel::new_arg(kernel name: '{}' arg index: '{}'): {}",
                    self.name().unwrap(), arg_idx, err);
            }
        }

        self.new_arg_count += 1;
        assert!(self.new_arg_count <= self.num_args);
        arg_idx
    }
}

impl Clone for Kernel {
    // TODO: Create a new, identical, kernel core instead of cloning it.
    fn clone(&self) -> Kernel {
        assert!(self.new_arg_count == self.num_args, "Cannot clone kernel until all arguments \
            are specified. Use named arguments with 'None' values to specify arguments you \
            plan to assign a value to later.");

        Kernel {
            obj_core: self.obj_core.clone(),
            named_args: self.named_args.clone(),
            new_arg_count: self.new_arg_count.clone(),
            mem_args: self.mem_args.clone(),
            queue: self.queue.clone(),
            gwo: self.gwo.clone(),
            gws: self.gws.clone(),
            lws: self.lws.clone(),
            num_args: self.num_args.clone(),
            arg_types: self.arg_types.clone(),
            bypass_arg_check: self.bypass_arg_check.clone(),
        }
    }
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)?;
        write!(f, " ")?;
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


/// Returns argument information for a kernel.
pub fn arg_info(core: &KernelCore, arg_index: u32, info_kind: KernelArgInfo)
        -> OclCoreResult<KernelArgInfoResult> {
    let device_versions = match core.device_versions() {
        Ok(vers) => vers,
        Err(e) => return Err(e.into()),
    };

    core::get_kernel_arg_info(core, arg_index, info_kind,
        Some(&device_versions))
}

/// Returns the type name for a kernel argument at the specified index.
pub fn arg_type_name(core: &KernelCore, arg_index: u32) -> OclCoreResult<String> {
    match arg_info(core, arg_index, KernelArgInfo::TypeName) {
        Ok(KernelArgInfoResult::TypeName(type_name)) => Ok(type_name),
        Err(err) => Err(err.into()),
        _ => unreachable!(),
    }
}


pub mod arg_type {
    #![allow(unused_imports)]
    use std::any::{Any, TypeId};
    use ffi::{cl_char, cl_uchar, cl_short, cl_ushort, cl_int, cl_uint, cl_long, cl_ulong,
        cl_half, cl_float, cl_double, cl_bool, cl_bitfield};
    use core::{Error as OclCoreError, Result as OclCoreResult, Status, OclPrm, Kernel as KernelCore};
    use error::{Error as OclError, Result as OclResult};
    use standard::Sampler;
    use super::{arg_info, arg_type_name};

    pub use core::{
        Char, Char2, Char3, Char4, Char8, Char16,
        Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
        Short, Short2, Short3, Short4, Short8, Short16,
        Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
        Int, Int2, Int3, Int4, Int8, Int16,
        Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
        Long, Long2, Long3, Long4, Long8, Long16,
        Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
        Float, Float2, Float3, Float4, Float8, Float16,
        Double, Double2, Double3, Double4, Double8, Double16};

    /// The base type of an OpenCL primitive.
    #[derive(Clone, Debug, Copy, PartialEq, Eq)]
    pub enum BaseType {
        Char,
        Uchar,
        Short,
        Ushort,
        Int,
        Uint,
        Long,
        Ulong,
        Float,
        Double,
        Sampler,
        Image,
        Unknown,
    }

    /// The cartinality of an OpenCL primitive.
    ///
    /// (ex. `float4`: 4)
    #[derive(Clone, Debug, Copy, PartialEq, Eq)]
    pub enum Cardinality {
        One,
        Two,
        Three,
        Four,
        Eight,
        Sixteen,
    }

    /// The type of a kernel argument derived from its string representation.
    #[allow(dead_code)]
    #[derive(Clone, Debug)]
    pub struct ArgType {
        base_type: BaseType,
        cardinality: Cardinality,
        is_ptr: bool,
    }

    impl ArgType {
        /// Returns an `ArgType` that will always return `true` when calling
        /// `::is_match`.
        pub fn unknown() -> OclCoreResult<ArgType> {
            Ok(ArgType {
                base_type: BaseType::Unknown,
                cardinality: Cardinality::One,
                is_ptr: false,
            })
        }

        /// Ascertains a `KernelArgType` from the contents of a
        /// `KernelArgInfoResult::TypeName`.
        ///
        /// * TODO: Optimize or outsource this if possible. Is `::contains`
        /// the fastest way to parse these in this situation? Should
        /// `::starts_with` be used for base type names instead?
        pub fn from_str(type_name: &str) -> OclCoreResult<ArgType> {
            let is_ptr = type_name.contains("*");

            let card = if type_name.contains("16") {
                Cardinality::Sixteen
            } else if type_name.contains("8") {
                Cardinality::Eight
            } else if type_name.contains("4") {
                Cardinality::Four
            } else if type_name.contains("3") {
                Cardinality::Three
            } else if type_name.contains("2") {
                Cardinality::Two
            } else {
                Cardinality::One
            };

            let base = if type_name.contains("uchar") {
                BaseType::Uchar
            } else if type_name.contains("char") {
                BaseType::Char
            } else if type_name.contains("ushort") {
                BaseType::Ushort
            } else if type_name.contains("short") {
                BaseType::Short
            } else if type_name.contains("uint") {
                BaseType::Uint
            } else if type_name.contains("int") {
                BaseType::Int
            } else if type_name.contains("ulong") {
                BaseType::Ulong
            } else if type_name.contains("long") {
                BaseType::Long
            } else if type_name.contains("float") {
                BaseType::Float
            } else if type_name.contains("double") {
                BaseType::Double
            } else if type_name.contains("sampler") {
                BaseType::Sampler
            } else if type_name.contains("image") {
                BaseType::Image
            } else {
                BaseType::Unknown
            };

            Ok(ArgType {
                base_type: base,
                cardinality: card,
                is_ptr: is_ptr,
            })
        }

        /// Returns a new argument type specifier.
        ///
        /// This function calls `core::get_kernel_arg_info`. Some platforms
        /// (Apple, NVIDIA) either do not implement
        /// `core::get_kernel_arg_info` or error in irregular ways. Known
        /// irregular errors are checked for here. The result of a call to
        /// `ArgType::unknown()` (which matches any argument type) is returned
        /// if any are found.
        pub fn from_kern_and_idx(core: &KernelCore, arg_index: u32) -> OclCoreResult<ArgType> {
            use core::EmptyInfoResultError;
            use core::ErrorKind as OclCoreErrorKind;

            match arg_type_name(core, arg_index) {
                Ok(type_name) => ArgType::from_str(type_name.as_str()),
                Err(err) => {
                    // Escape hatches for known, platform-specific errors.
                    match *err.kind() {
                        OclCoreErrorKind::Api(ref api_err) => {
                            if api_err.status() == Status::CL_KERNEL_ARG_INFO_NOT_AVAILABLE {
                                return Ok(ArgType { base_type: BaseType::Unknown,
                                    cardinality: Cardinality::One, is_ptr: false })
                            }
                        }
                        OclCoreErrorKind::EmptyInfoResult(EmptyInfoResultError::KernelArg) => {
                            return ArgType::unknown();
                        },
                        _ => (),
                    }

                    Err(err)
                },
            }
        }

        /// Returns true if the type of `T` matches the base type of this `ArgType`.
        pub fn is_match<T: OclPrm + Any + 'static>(&self) -> bool {
            match self.base_type {
                BaseType::Char => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_char>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Char2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Char3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Char4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Char8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Char16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_char>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Uchar => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_uchar>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Uchar2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Uchar3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Uchar4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Uchar8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Uchar16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_uchar>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Short => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_short>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Short2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Short3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Short4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Short8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Short16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_short>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Ushort => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_ushort>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Ushort2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Ushort3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Ushort4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Ushort8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Ushort16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_ushort>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Int => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_int>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Int2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Int3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Int4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Int8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Int16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_int>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Uint => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_uint>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Uint2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Uint3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Uint4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Uint8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Uint16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_uint>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Long => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_long>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Long2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Long3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Long4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Long8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Long16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_long>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Ulong => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_ulong>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Ulong2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Ulong3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Ulong4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Ulong8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Ulong16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_ulong>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Float => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_float>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Float2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Float3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Float4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Float8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Float16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_float>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Double => {
                    let card_match = match self.cardinality {
                        Cardinality::One => TypeId::of::<cl_double>() == TypeId::of::<T>(),
                        Cardinality::Two => TypeId::of::<Double2>() == TypeId::of::<T>(),
                        Cardinality::Three => TypeId::of::<Double3>() == TypeId::of::<T>(),
                        Cardinality::Four => TypeId::of::<Double4>() == TypeId::of::<T>(),
                        Cardinality::Eight => TypeId::of::<Double8>() == TypeId::of::<T>(),
                        Cardinality::Sixteen => TypeId::of::<Double16>() == TypeId::of::<T>(),
                    };

                    if self.is_ptr {
                        TypeId::of::<cl_double>() == TypeId::of::<T>() || card_match
                    } else {
                        card_match
                    }
                },
                BaseType::Sampler => TypeId::of::<u64>() == TypeId::of::<T>(),
                BaseType::Image => TypeId::of::<u64>() == TypeId::of::<T>(),
                // Everything matches if type was undetermined (escape hatch):
                BaseType::Unknown => true,
            }
        }

        #[allow(dead_code)]
        pub fn is_ptr(&self) -> bool {
            self.is_ptr
        }
    }

    impl<'a> From<&'a str> for ArgType {
        /// Ascertains a `KernelArgType` from the contents of a
        /// `KernelArgInfoResult::TypeName`.
        ///
        /// Panics if the string is not valid.
        fn from(type_name: &'a str) -> ArgType {
            ArgType::from_str(type_name).unwrap()
        }
    }
}
