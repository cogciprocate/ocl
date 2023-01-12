//! An OpenCL kernel.

pub use self::arg_type::{ArgType, BaseType, Cardinality};
use crate::core::error::Error as OclCoreErrorKind;
use crate::core::ffi::c_void;
use crate::core::{
    self, util, ArgVal, AsMem, ClVersions, CommandQueue as CommandQueueCore, Kernel as KernelCore,
    KernelArgInfo, KernelArgInfoResult, KernelInfo, KernelInfoResult, KernelWorkGroupInfo,
    KernelWorkGroupInfoResult, Mem as MemCore, MemCmdAll, OclPrm,
};
use crate::error::{Error as OclError, Result as OclResult};
use crate::standard::{
    Buffer, ClNullEventPtrEnum, ClWaitListPtrEnum, Device, Image, Program, Queue, Sampler,
    SpatialDims, WorkDims,
};
use std;
use std::any::Any;
use std::any::TypeId;
use std::borrow::Borrow;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::ops::Deref;

/// An error related to a `Kernel`, `KernelBuilder`, or `KernelCmd`.
#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("No queue specified.")]
    CmdNoQueue,
    #[error(
        "Global Work Size cannot be left unspecified. Set a default for \
        the kernel or specify one when enqueuing command."
    )]
    CmdNoGws,
    #[error(
        "Unable to resolve argument named: '{0}'. Ensure that an argument with \
        that name has been declared before building kernel."
    )]
    NamedArgsInvalidArgName(String),
    #[error(
        "No named arguments have been declared. Declare named arguments \
        when before building kernel."
    )]
    NamedArgsNone,
    #[error("Kernel arg index out of range. (kernel: {0}, index: {1})")]
    ArgIdxOor(String, u32),
    #[error(
        "Kernel argument type mismatch. The argument named: '{arg_name}' at index: [{idx}] \
        should be a '{ty_name}' ({ty:?})."
    )]
    ArgTypeMismatch {
        idx: u32,
        arg_name: String,
        ty_name: String,
        ty: ArgType,
    },
    #[error("No program specified.")]
    BuilderNoProgram,
    #[error("No kernel name specified.")]
    BuilderNoKernelName,
    #[error("The wrong number of kernel arguments have been specified \
        (required: {required}, specified: {specified}). Use named arguments with 'None' or zero values to \
        declare arguments you plan to assign a value to at a later time.")]
    BuilderWrongArgCount { required: u32, specified: u32 },
}

/// A kernel command builder used to enqueue a kernel with a mix of default
/// and optionally specified arguments.
#[must_use = "commands do nothing unless enqueued"]
pub struct KernelCmd<'k> {
    kernel: &'k KernelCore,
    queue: Option<&'k CommandQueueCore>,
    gwo: SpatialDims,
    gws: SpatialDims,
    lws: SpatialDims,
    wait_events: Option<ClWaitListPtrEnum<'k>>,
    new_event: Option<ClNullEventPtrEnum<'k>>,
}

/// A kernel enqueue command.
impl<'k> KernelCmd<'k> {
    /// Specifies a queue to use for this call only.
    ///
    /// Overrides the kernel's default queue if one is set. If no default
    /// queue is set, this method **must** be called before enqueuing the
    /// kernel.
    pub fn queue<'q, Q>(mut self, queue: &'q Q) -> KernelCmd<'k>
    where
        'q: 'k,
        Q: 'k + AsRef<CommandQueueCore>,
    {
        self.queue = Some(queue.as_ref());
        self
    }

    /// Specifies a global work offset for this call only.
    #[deprecated(since = "0.18.0", note = "Use `::global_work_offset` instead.")]
    pub fn gwo<D: Into<SpatialDims>>(mut self, gwo: D) -> KernelCmd<'k> {
        self.gwo = gwo.into();
        self
    }

    /// Specifies a global work size for this call only.
    #[deprecated(since = "0.18.0", note = "Use `::global_work_size` instead.")]
    pub fn gws<D: Into<SpatialDims>>(mut self, gws: D) -> KernelCmd<'k> {
        self.gws = gws.into();
        self
    }

    /// Specifies a local work size for this call only.
    #[deprecated(since = "0.18.0", note = "Use `::local_work_size` instead.")]
    pub fn lws<D: Into<SpatialDims>>(mut self, lws: D) -> KernelCmd<'k> {
        self.lws = lws.into();
        self
    }

    /// Specifies a global work offset for this call only.
    pub fn global_work_offset<D: Into<SpatialDims>>(mut self, gwo: D) -> KernelCmd<'k> {
        self.gwo = gwo.into();
        self
    }

    /// Specifies a global work size for this call only.
    pub fn global_work_size<D: Into<SpatialDims>>(mut self, gws: D) -> KernelCmd<'k> {
        self.gws = gws.into();
        self
    }

    /// Specifies a local work size for this call only.
    pub fn local_work_size<D: Into<SpatialDims>>(mut self, lws: D) -> KernelCmd<'k> {
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
    where
        'e: 'k,
        Ewl: Into<ClWaitListPtrEnum<'e>>,
    {
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
    where
        'e: 'k,
        En: Into<ClNullEventPtrEnum<'e>>,
    {
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
            None => return Err(KernelError::CmdNoQueue.into()),
        };

        let dim_count = self.gws.dim_count();

        let gws = match self.gws.to_work_size() {
            Some(gws) => gws,
            None => return Err(KernelError::CmdNoGws.into()),
        };

        core::enqueue_kernel(
            queue,
            &self.kernel,
            dim_count,
            self.gwo.to_work_offset(),
            &gws,
            self.lws.to_work_size(),
            self.wait_events,
            self.new_event,
        )
        .map_err(OclError::from)
    }
}

/// Converts an argument index specifier to `u32`.
#[derive(Clone, Debug)]
pub enum ArgIdxSpecifier {
    Uint(u32),
    Str(Cow<'static, str>),
}

impl ArgIdxSpecifier {
    // Resolves this specifier into an integer index using the specified
    // argument map if necessary.
    fn to_idx(&self, named_args: &NamedArgs) -> OclResult<u32> {
        match self {
            ArgIdxSpecifier::Uint(idx) => Ok(*idx),
            ArgIdxSpecifier::Str(ref s) => named_args.resolve_idx(&s),
        }
    }
}

impl From<u32> for ArgIdxSpecifier {
    fn from(idx: u32) -> ArgIdxSpecifier {
        ArgIdxSpecifier::Uint(idx)
    }
}

impl From<i32> for ArgIdxSpecifier {
    fn from(idx: i32) -> ArgIdxSpecifier {
        ArgIdxSpecifier::Uint(idx as u32)
    }
}

impl From<usize> for ArgIdxSpecifier {
    fn from(idx: usize) -> ArgIdxSpecifier {
        ArgIdxSpecifier::Uint(idx as u32)
    }
}

impl From<&'static str> for ArgIdxSpecifier {
    fn from(s: &'static str) -> ArgIdxSpecifier {
        ArgIdxSpecifier::Str(s.into())
    }
}

impl From<String> for ArgIdxSpecifier {
    fn from(s: String) -> ArgIdxSpecifier {
        ArgIdxSpecifier::Str(s.into())
    }
}

/// Contains owned or shared argument values.
#[derive(Debug, Clone)]
enum ArgValKeeper<'b> {
    Shared(ArgVal<'b>),
    OwnedPrm(Vec<u8>),
    OwnedMem(MemCore),
}

impl<'b> ArgValKeeper<'b> {
    fn owned_prm<T>(prm: T) -> ArgValKeeper<'b>
    where
        T: OclPrm,
    {
        unsafe { ArgValKeeper::OwnedPrm(util::into_bytes(prm)) }
    }

    fn owned_mem<T>(buf: MemCore) -> ArgValKeeper<'b>
    where
        T: OclPrm,
    {
        ArgValKeeper::OwnedMem(buf)
    }

    fn mem_null() -> ArgValKeeper<'b> {
        ArgValKeeper::Shared(ArgVal::mem_null())
    }

    /// Returns the size (in bytes) and raw pointer to the contained kernel
    /// argument value.
    fn to_arg_val(&'b self) -> ArgVal<'b> {
        match *self {
            ArgValKeeper::Shared(ref av) => av.clone(),
            ArgValKeeper::OwnedPrm(ref bytes) => unsafe {
                ArgVal::from_raw(bytes.len(), bytes.as_ptr() as *const c_void, false)
            },
            ArgValKeeper::OwnedMem(ref mem) => ArgVal::mem(mem),
        }
    }
}

/// Wraps argument values of different types.
pub struct ArgValConverter<'b, T>
where
    T: OclPrm,
{
    val: ArgValKeeper<'b>,
    type_id: Option<TypeId>,
    mem: Option<MemCore>,
    _ty: PhantomData<T>,
}

impl<'b, T> From<Option<&'b Buffer<T>>> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    /// Converts from an Option<`Buffer`>.
    fn from(buf: Option<&'b Buffer<T>>) -> ArgValConverter<'b, T> {
        let val = match buf {
            Some(b) => ArgValKeeper::Shared(ArgVal::mem(b)),
            None => ArgValKeeper::mem_null(),
        };

        ArgValConverter {
            val,
            type_id: Some(TypeId::of::<T>()),
            mem: buf.map(|b| b.as_mem().clone()),
            _ty: PhantomData,
        }
    }
}

impl<'b, T> From<&'b Buffer<T>> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    /// Converts from a `Buffer`.
    fn from(buf: &'b Buffer<T>) -> ArgValConverter<'b, T> {
        ArgValConverter {
            val: ArgValKeeper::Shared(ArgVal::mem(buf)),
            type_id: Some(TypeId::of::<T>()),
            mem: Some(buf.as_mem().clone()),
            _ty: PhantomData,
        }
    }
}

impl<'b, T> From<&'b mut Buffer<T>> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    fn from(buf: &'b mut Buffer<T>) -> ArgValConverter<'b, T> {
        ArgValConverter::from(&*buf)
    }
}

// impl<'b, T> From<&'b &'b Buffer<T>> for ArgValConverter<'b, T> where T: OclPrm {
//     fn from(buf: &'b &'b Buffer<T>) -> ArgValConverter<'b, T> {
//         ArgValConverter::from(&**buf)
//     }
// }

impl<'b, T> From<Buffer<T>> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    /// Converts from an owned `Buffer`.
    fn from(buf: Buffer<T>) -> ArgValConverter<'b, T> {
        ArgValConverter {
            val: ArgValKeeper::owned_mem::<T>(buf.as_mem().clone()),
            type_id: Some(TypeId::of::<T>()),
            mem: Some(buf.as_mem().clone()),
            _ty: PhantomData,
        }
    }
}

impl<'b, T> From<Option<&'b Image<T>>> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    /// Converts from an Option<`Image`>.
    fn from(img: Option<&'b Image<T>>) -> ArgValConverter<'b, T> {
        let val = match img {
            Some(i) => ArgValKeeper::Shared(ArgVal::mem(i)),
            None => ArgValKeeper::mem_null(),
        };

        ArgValConverter {
            val,
            type_id: None,
            mem: img.map(|i| i.as_mem().clone()),
            _ty: PhantomData,
        }
    }
}

impl<'b, T> From<&'b Image<T>> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    /// Converts from an `Image`.
    fn from(img: &'b Image<T>) -> ArgValConverter<'b, T> {
        ArgValConverter {
            val: ArgValKeeper::Shared(ArgVal::mem(img)),
            type_id: None,
            mem: Some(img.as_mem().clone()),
            _ty: PhantomData,
        }
    }
}

impl<'b, T> From<&'b mut Image<T>> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    fn from(img: &'b mut Image<T>) -> ArgValConverter<'b, T> {
        ArgValConverter::from(&*img)
    }
}

// impl<'b, T> From<&'b &'b Image<T>> for ArgValConverter<'b, T> where T: OclPrm {
//     fn from(img: &'b &'b Image<T>) -> ArgValConverter<'b, T> {
//         ArgValConverter::from(&**img)
//     }
// }

impl<'b, T> From<Image<T>> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    /// Converts from an owned `Image`.
    fn from(img: Image<T>) -> ArgValConverter<'b, T> {
        ArgValConverter {
            val: ArgValKeeper::owned_mem::<T>(img.as_mem().clone()),
            type_id: Some(TypeId::of::<T>()),
            mem: Some(img.as_mem().clone()),
            _ty: PhantomData,
        }
    }
}

impl<'b, T> From<&'b T> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    /// Converts from a scalar or vector value.
    fn from(prm: &'b T) -> ArgValConverter<'b, T> {
        ArgValConverter {
            val: ArgValKeeper::Shared(ArgVal::scalar(prm)),
            type_id: Some(TypeId::of::<T>()),
            mem: None,
            _ty: PhantomData,
        }
    }
}

impl<'b, T> From<T> for ArgValConverter<'b, T>
where
    T: OclPrm,
{
    /// Converts from a scalar or vector value.
    fn from(prm: T) -> ArgValConverter<'b, T> {
        ArgValConverter {
            val: ArgValKeeper::owned_prm(prm),
            type_id: Some(TypeId::of::<T>()),
            mem: None,
            _ty: PhantomData,
        }
    }
}

/// A map of argument names -> indexes.
#[derive(Clone, Debug)]
struct NamedArgs(Option<HashMap<Cow<'static, str>, u32>>);

impl NamedArgs {
    /// Inserts a named argument into the map.
    fn insert(&mut self, name: Cow<'static, str>, arg_idx: u32) {
        if self.0.is_none() {
            self.0 = Some(HashMap::with_capacity(8));
        }
        self.0.as_mut().unwrap().insert(name, arg_idx);
    }

    /// Resolves the index of a named argument with a friendly error message.
    fn resolve_idx(&self, name: &str) -> OclResult<u32> {
        match self.0 {
            Some(ref map) => match map.get(name) {
                Some(&ai) => Ok(ai),
                None => Err(KernelError::NamedArgsInvalidArgName(name.into()).into()),
            },
            None => Err(KernelError::NamedArgsNone.into()),
        }
    }
}

/// Storage for `Mem` arguments.
//
// NOTE: `RefCell` is used to prevent `::set_arg*` methods from requiring a
// mutable reference (to `self`).
#[derive(Clone, Debug)]
struct MemArgs(Option<RefCell<BTreeMap<u32, MemCore>>>);

impl MemArgs {
    /// Inserts a `Mem` argument for storage if possible.
    fn insert(&self, idx: u32, mem: MemCore) {
        if let Some(ref map) = self.0 {
            map.borrow_mut().insert(idx, mem);
        }
    }

    /// Removes a `Mem` argument.
    fn remove(&self, idx: &u32) {
        if let Some(ref map) = self.0 {
            map.borrow_mut().remove(idx);
        }
    }
}

/// A kernel which represents a 'procedure'.
///
/// Corresponds to code which must have already been compiled into a program.
///
/// Set arguments using `::set_arg` or any of the `::set_arg...` methods.
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
///
/// ### `Clone` and `Send`
///
/// A `Kernel` may not be cloned but may be sent between threads. This ensures
/// that no two threads create a race condition by attempting to set an
/// argument and enqueue a kernel at the same time. Use the `KernelBuilder` to
/// create multiple identical kernels (`KernelBuilder` is clonable and
/// re-usable).
#[derive(Debug)]
pub struct Kernel {
    obj_core: KernelCore,
    named_args: NamedArgs,
    mem_args: MemArgs,
    queue: Option<Queue>,
    gwo: SpatialDims,
    gws: SpatialDims,
    lws: SpatialDims,
    arg_types: Option<Vec<ArgType>>,
}

impl Kernel {
    /// Returns a new `KernelBuilder`.
    pub fn builder<'p>() -> KernelBuilder<'p> {
        KernelBuilder::new()
    }

    /// Verifies that a type matches the kernel arg info:
    ///
    /// This function does nothing and always returns `Ok` if argument type
    /// checking is disabled by the user or for any other reason.
    ///
    /// Argument type checking will automatically be disabled if any of the
    /// devices in use do not support OpenCL version 1.2 or higher or if
    /// argument information is not available on the associated platform.
    fn verify_arg_type<T: OclPrm + Any>(&self, arg_idx: u32) -> OclResult<()> {
        if let Some(ref arg_types) = self.arg_types {
            let arg_type = arg_types
                .get(arg_idx as usize)
                .ok_or(KernelError::ArgIdxOor(self.name()?, arg_idx))?;

            if arg_type.is_match::<T>() {
                Ok(())
            } else {
                let ty_name = arg_type_name(&self.obj_core, arg_idx)?;
                let arg_name = arg_name(&self.obj_core, arg_idx)?;
                Err(KernelError::ArgTypeMismatch {
                    idx: arg_idx,
                    arg_name,
                    ty_name,
                    ty: arg_type.clone(),
                }
                .into())
            }
        } else {
            Ok(())
        }
    }

    /// Returns the argument index of a named argument if it exists.
    pub fn named_arg_idx(&self, name: &'static str) -> Option<u32> {
        self.named_args.resolve_idx(name).ok()
    }

    /// Sets an argument by index without checks of any kind.
    ///
    /// Setting buffer or image (`cl_mem`) arguments this way may cause
    /// segfaults or errors if the buffer goes out of scope at any point
    /// before this kernel is dropped.
    ///
    /// This also bypasses the check to determine if the type of the value you
    /// pass here matches the type defined in your kernel.
    pub unsafe fn set_arg_unchecked(&self, arg_idx: u32, arg_val: ArgVal) -> OclResult<()> {
        core::set_kernel_arg(&self.obj_core, arg_idx, arg_val).map_err(OclError::from)
    }

    /// Sets an argument by index.
    fn _set_arg<T: OclPrm>(&self, arg_idx: u32, arg_val: ArgVal) -> OclResult<()> {
        self.verify_arg_type::<T>(arg_idx)?;
        core::set_kernel_arg(&self.obj_core, arg_idx, arg_val).map_err(OclError::from)
    }

    /// Sets a `Buffer`, `Image`, scalar, or vector argument by index or by
    /// name.
    ///
    /// ### Example
    /// ```rust,ignore
    /// // Create a kernel with arguments corresponding to those in the kernel.
    /// // Just for fun, one argument will be 'named':
    /// let kern = ocl_pq.kernel_builder("multiply_by_scalar")
    ///     .arg(&0)
    ///     .arg(None::<&Buffer<f32>>)
    ///     .arg_named("result", None::<&Buffer<f32>>)
    ///     .build()?;
    ///
    /// // Set our named argument. The Option<_> wrapper is, well... optional:
    /// kern.set_arg("result", &result_buffer)?;
    /// // We can also set arguments (named or not) by index. Just for
    /// // demonstration, we'll set one using an option:
    /// kern.set_arg(0, &COEFF)?;
    /// kern.set_arg(1, Some(&source_buffer))?;
    /// kern.set_arg(2, &result_buffer)?;
    /// ```
    pub fn set_arg<'a, T, Ai, Av>(&self, idx: Ai, arg: Av) -> OclResult<()>
    where
        T: OclPrm,
        Ai: Into<ArgIdxSpecifier>,
        Av: Into<ArgValConverter<'a, T>>,
    {
        let arg_idx = idx.into().to_idx(&self.named_args)?;
        self.verify_arg_type::<T>(arg_idx)?;
        let arg: ArgValConverter<T> = arg.into();

        // If the `KernelArg` is a `Mem` variant, clone the `MemCore` it
        // refers to, store it in `self.mem_args`. This prevents a buffer
        // which has gone out of scope from being erroneously referred to when
        // this kernel is enqueued and causing either a misleading error
        // message or a hard to debug segfault depending on the platform.
        match arg.mem {
            Some(mem) => self.mem_args.insert(arg_idx, mem),
            None => self.mem_args.remove(&arg_idx),
        };

        let val = arg.val.to_arg_val();
        self._set_arg::<T>(arg_idx, val)
    }

    /// Modifies the kernel argument named: `name`.
    #[deprecated(since = "0.18.0", note = "Use `::set_arg` instead.")]
    pub fn set_arg_buf_named<'a, T, M>(
        &'a self,
        name: &'static str,
        buffer_opt: Option<M>,
    ) -> OclResult<()>
    where
        T: OclPrm,
        M: AsMem<T> + MemCmdAll,
    {
        let arg_idx = self.named_args.resolve_idx(name)?;
        match buffer_opt {
            Some(buffer) => {
                self.mem_args.insert(arg_idx, buffer.as_mem().clone());
                self._set_arg::<T>(arg_idx, ArgVal::mem(buffer.as_mem()))
            }
            None => {
                self.mem_args.remove(&arg_idx);
                self._set_arg::<T>(arg_idx, ArgVal::mem_null())
            }
        }
    }

    /// Modifies the kernel argument named: `name`.
    #[deprecated(since = "0.18.0", note = "Use `::set_arg` instead.")]
    pub fn set_arg_img_named<'a, T, M>(
        &'a self,
        name: &'static str,
        image_opt: Option<M>,
    ) -> OclResult<()>
    where
        T: OclPrm,
        M: AsMem<T> + MemCmdAll,
    {
        let arg_idx = self.named_args.resolve_idx(name)?;
        match image_opt {
            Some(img) => {
                self.mem_args.insert(arg_idx, img.as_mem().clone());
                self._set_arg::<T>(arg_idx, ArgVal::mem(img.as_mem()))
            }
            None => {
                self.mem_args.remove(&arg_idx);
                self._set_arg::<T>(arg_idx, ArgVal::mem_null())
            }
        }
    }

    /// Sets the value of a named sampler argument.
    #[deprecated(since = "0.18.0", note = "Use `::set_arg_sampler_named` instead.")]
    pub fn set_arg_smp_named<'a>(
        &'a self,
        name: &'static str,
        sampler_opt: Option<&Sampler>,
    ) -> OclResult<()> {
        let arg_idx = self.named_args.resolve_idx(name)?;
        match sampler_opt {
            Some(sampler) => self._set_arg::<u64>(arg_idx, ArgVal::sampler(sampler)),
            None => self._set_arg::<u64>(arg_idx, ArgVal::sampler_null()),
        }
    }

    /// Modifies the kernel argument named: `name`.
    #[deprecated(since = "0.18.0", note = "Use `::set_arg` instead.")]
    pub fn set_arg_scl_named<'a, T, B>(&'a self, name: &'static str, scalar: B) -> OclResult<()>
    where
        T: OclPrm,
        B: Borrow<T>,
    {
        let arg_idx = self.named_args.resolve_idx(name)?;
        self._set_arg::<T>(arg_idx, ArgVal::scalar(scalar.borrow()))
    }

    /// Modifies the kernel argument named: `name`.
    #[deprecated(since = "0.18.0", note = "Use `::set_arg` instead.")]
    pub fn set_arg_vec_named<'a, T, B>(&'a self, name: &'static str, vector: B) -> OclResult<()>
    where
        T: OclPrm,
        B: Borrow<T>,
    {
        let arg_idx = self.named_args.resolve_idx(name)?;
        self._set_arg::<T>(arg_idx, ArgVal::vector(vector.borrow()))
    }

    /// Sets the value of a named sampler argument.
    pub fn set_arg_sampler_named<'a, Ai>(
        &'a self,
        idx: Ai,
        sampler_opt: Option<&Sampler>,
    ) -> OclResult<()>
    where
        Ai: Into<ArgIdxSpecifier>,
    {
        // let arg_idx = self.named_args.resolve_idx(name)?;
        let arg_idx = idx.into().to_idx(&self.named_args)?;
        match sampler_opt {
            Some(sampler) => self._set_arg::<u64>(arg_idx, ArgVal::sampler(sampler)),
            None => self._set_arg::<u64>(arg_idx, ArgVal::sampler_null()),
        }
    }

    /// Returns a command builder which is used to chain parameters of an
    /// 'enqueue' command together.
    pub fn cmd(&self) -> KernelCmd {
        KernelCmd {
            kernel: &self.obj_core,
            queue: self.queue.as_ref().map(|q| q.as_ref()),
            gwo: self.gwo,
            gws: self.gws,
            lws: self.lws,
            wait_events: None,
            new_event: None,
        }
    }

    /// Enqueues this kernel on the default queue using the default work sizes
    /// and offsets.
    ///
    /// Shorthand for `.cmd().enq()`
    ///
    /// # Safety
    ///
    /// All kernel code must be considered untrusted. Therefore the act of
    /// calling this function contains implied unsafety even though the API
    /// itself is safe.
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

    /// Returns the default global work offset.
    #[deprecated(since = "0.18.0", note = "Use `::global_work_offset` instead.")]
    pub fn get_gwo(&self) -> SpatialDims {
        self.gwo
    }

    /// Returns the default global work size.
    #[deprecated(since = "0.18.0", note = "Use `::global_work_size` instead.")]
    pub fn get_gws(&self) -> SpatialDims {
        self.gws
    }

    /// Returns the default local work size.
    #[deprecated(since = "0.18.0", note = "Use `::local_work_size` instead.")]
    pub fn get_lws(&self) -> SpatialDims {
        self.lws
    }

    /// Sets the default global work offset.
    pub fn set_default_global_work_offset(&mut self, gwo: SpatialDims) -> &mut Kernel {
        self.gwo = gwo;
        self
    }

    /// Sets the default global work size.
    pub fn set_default_global_work_size(&mut self, gws: SpatialDims) -> &mut Kernel {
        self.gws = gws;
        self
    }

    /// Sets the default local work size.
    pub fn set_default_local_work_size(&mut self, lws: SpatialDims) -> &mut Kernel {
        self.lws = lws;
        self
    }

    /// Returns the default queue for this kernel if one has been set.
    pub fn default_queue(&self) -> Option<&Queue> {
        self.queue.as_ref()
    }

    /// Returns the default global work offset.
    pub fn default_global_work_offset(&self) -> SpatialDims {
        self.gwo
    }

    /// Returns the default global work size.
    pub fn default_global_work_size(&self) -> SpatialDims {
        self.gws
    }

    /// Returns the default local work size.
    pub fn default_local_work_size(&self) -> SpatialDims {
        self.lws
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn as_core(&self) -> &KernelCore {
        self
    }

    /// Returns information about this kernel.
    pub fn info(&self, info_kind: KernelInfo) -> OclResult<KernelInfoResult> {
        core::get_kernel_info(&self.obj_core, info_kind).map_err(OclError::from)
    }

    /// Returns work group information for this kernel.
    pub fn wg_info(
        &self,
        device: Device,
        info_kind: KernelWorkGroupInfo,
    ) -> OclResult<KernelWorkGroupInfoResult> {
        core::get_kernel_work_group_info(&self.obj_core, device, info_kind).map_err(OclError::from)
    }

    /// Returns argument information for this kernel.
    pub fn arg_info(
        &self,
        arg_idx: u32,
        info_kind: KernelArgInfo,
    ) -> OclResult<KernelArgInfoResult> {
        arg_info(&*self.as_core(), arg_idx, info_kind)
    }

    /// Returns the name of this kernel.
    pub fn name(&self) -> OclResult<String> {
        core::get_kernel_info(&self.obj_core, KernelInfo::FunctionName)
            .map(|r| r.into())
            .map_err(OclError::from)
    }

    /// Returns the number of arguments this kernel has.
    pub fn num_args(&self) -> OclResult<u32> {
        match core::get_kernel_info(&self.obj_core, KernelInfo::NumArgs) {
            Ok(KernelInfoResult::NumArgs(num)) => Ok(num),
            Err(err) => Err(err.into()),
            _ => unreachable!(),
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
    where
        D: Into<Device>,
    {
        for device in devices {
            let device = device.into();
            if !device.vendor().unwrap().contains("NVIDIA") {
                f.debug_struct("WorkGroup")
                    .field(
                        "WorkGroupSize",
                        &self.wg_info(device, KernelWorkGroupInfo::WorkGroupSize),
                    )
                    .field(
                        "CompileWorkGroupSize",
                        &self.wg_info(device, KernelWorkGroupInfo::CompileWorkGroupSize),
                    )
                    .field(
                        "LocalMemSize",
                        &self.wg_info(device, KernelWorkGroupInfo::LocalMemSize),
                    )
                    .field(
                        "PreferredWorkGroupSizeMultiple",
                        &self.wg_info(device, KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple),
                    )
                    .field(
                        "PrivateMemSize",
                        &self.wg_info(device, KernelWorkGroupInfo::PrivateMemSize),
                    )
                    // .field("GlobalWorkSize", &self.wg_info(device, KernelWorkGroupInfo::GlobalWorkSize))
                    .finish()?;
            }
        }
        Ok(())
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

/// A kernel builder.
///
///
/// ### Examples
///
/// ```rust,ignore
/// // Create a kernel:
/// let kernel = Kernel::builder()
///     .program(&program)
///     .name("process")
///     .queue(queue.clone())
///     .global_work_size(&gws_patch_count)
///     .arg(&(patch_size as i32))
///     .arg(&source_buffer)
///     .arg(&destination_buffer)
///     .build()?;
/// ```
///
/// Re-use and clone:
///
/// ```rust,ignore
/// // Create a builder for re-use:
/// let builder = Kernel::builder()
///     .program(&program)
///     .name("process")
///     .queue(queue.clone())
///     .global_work_size([512, 64, 64])
///     .arg(&(patch_size as i32))
///     .arg(&source_buffer)
///     .arg_named("dest", &destination_buffer);
///
/// // Create multiple kernels using the same builder:
/// let kernel_0 = builder.build()?;
/// let kernel_1 = builder.build()?;
///
/// // Clone the builder:
/// let mut builder_clone = builder.clone();
///
/// // Modify a default parameter:
/// builder_clone.global_work_size([1024, 128, 128]);
///
/// // Create another kernel using the cloned builder:
/// let kernel_2 = builder_clone.build()?;
///
/// // Modify one of the arguments using the created kernel directly
/// // (arguments cannot be modified using the builder):
/// kernel_2.set_arg("dest", &alternate_destination_buffer)?;
///
/// // Arguments can also be referred to by index:
/// kernel_2.set_arg(1, &alternate_source_buffer)?;
/// ```
///
#[derive(Debug)]
pub struct KernelBuilder<'b> {
    program: Option<&'b Program>,
    name: Option<String>,
    named_args: NamedArgs,
    mem_args: MemArgs,
    args: Vec<(ArgValKeeper<'b>, Option<TypeId>)>,
    queue: Option<Queue>,
    gwo: SpatialDims,
    gws: SpatialDims,
    lws: SpatialDims,
    disable_arg_check: bool,
}

impl<'b> KernelBuilder<'b> {
    /// Returns a new kernel builder.
    pub fn new() -> KernelBuilder<'b> {
        KernelBuilder {
            program: None,
            name: None,
            named_args: NamedArgs(None),
            args: Vec::with_capacity(16),
            mem_args: MemArgs(Some(RefCell::new(BTreeMap::new()))),
            queue: None,
            gwo: SpatialDims::Unspecified,
            gws: SpatialDims::Unspecified,
            lws: SpatialDims::Unspecified,
            disable_arg_check: false,
        }
    }

    /// Specifies a program object with a successfully built executable.
    pub fn program<'s>(&'s mut self, program: &'b Program) -> &'s mut KernelBuilder<'b> {
        self.program = Some(program);
        self
    }

    /// Specifies a function name in the program declared with the `__kernel`
    /// qualifier (e.g. `__kernel void add_values(...`).
    pub fn name<'s, S>(&'s mut self, name: S) -> &'s mut KernelBuilder<'b>
    where
        S: Into<String>,
    {
        self.name = Some(name.into());
        self
    }

    /// Sets the default queue to be used by all subsequent enqueue commands
    /// unless otherwise changed (with `::set_default_queue`) or overridden
    /// (by `::cmd().queue(...)...`).
    ///
    /// The queue must be associated with a device associated with the
    /// kernel's program.
    pub fn queue<'s>(&'s mut self, queue: Queue) -> &'s mut KernelBuilder<'b> {
        self.queue = Some(queue);
        self
    }

    /// Sets the default global work offset.
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// building a queue command with `::cmd`.
    #[deprecated(since = "0.18.0", note = "Use `::global_work_offset` instead.")]
    pub fn gwo<'s, D: Into<SpatialDims>>(&'s mut self, gwo: D) -> &'s mut KernelBuilder<'b> {
        self.gwo = gwo.into();
        self
    }

    /// Sets the default global work size.
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// building a queue command with `::cmd`.
    #[deprecated(since = "0.18.0", note = "Use `::global_work_size` instead.")]
    pub fn gws<'s, D: Into<SpatialDims>>(&'s mut self, gws: D) -> &'s mut KernelBuilder<'b> {
        self.gws = gws.into();
        self
    }

    /// Sets the default local work size.
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// building a queue command with `::cmd`.
    #[deprecated(since = "0.18.0", note = "Use `::local_work_size` instead.")]
    pub fn lws<'s, D: Into<SpatialDims>>(&'s mut self, lws: D) -> &'s mut KernelBuilder<'b> {
        self.lws = lws.into();
        self
    }

    /// Sets the default global work offset.
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// building a queue command with `::cmd`.
    pub fn global_work_offset<'s, D: Into<SpatialDims>>(
        &'s mut self,
        gwo: D,
    ) -> &'s mut KernelBuilder<'b> {
        self.gwo = gwo.into();
        self
    }

    /// Sets the default global work size.
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// building a queue command with `::cmd`.
    pub fn global_work_size<'s, D: Into<SpatialDims>>(
        &'s mut self,
        gws: D,
    ) -> &'s mut KernelBuilder<'b> {
        self.gws = gws.into();
        self
    }

    /// Sets the default local work size.
    ///
    /// Used when enqueuing kernel commands. Superseded if specified while
    /// building a queue command with `::cmd`.
    pub fn local_work_size<'s, D: Into<SpatialDims>>(
        &'s mut self,
        lws: D,
    ) -> &'s mut KernelBuilder<'b> {
        self.lws = lws.into();
        self
    }

    /// Adds a new argument to the kernel and returns the index.
    fn new_arg(
        &mut self,
        arg_val: ArgValKeeper<'b>,
        type_id: Option<TypeId>,
        mem: Option<MemCore>,
    ) -> u32 {
        let arg_idx = self.args.len() as u32;

        // If the `KernelArg` is a `Mem` variant, clone the `MemCore` it
        // refers to, store it in `self.mem_args`. This prevents a buffer
        // which has gone out of scope from being erroneously referred to when
        // this kernel is enqueued and causing either a misleading error
        // message or a hard to debug segfault depending on the platform.
        match mem {
            Some(mem) => self.mem_args.insert(arg_idx, mem),
            None => self.mem_args.remove(&arg_idx),
        };

        self.args.push((arg_val, type_id));
        arg_idx
    }

    /// Non-builder-style version of `::arg_buf()`.
    fn new_arg_buf<T, M>(&mut self, buffer_opt: Option<&'b M>) -> u32
    where
        T: OclPrm,
        M: 'b + AsMem<T> + MemCmdAll,
    {
        match buffer_opt {
            Some(buffer) => self.new_arg(
                ArgValKeeper::Shared(ArgVal::mem(buffer.as_mem())),
                Some(TypeId::of::<T>()),
                Some(buffer.as_mem().clone()),
            ),
            None => self.new_arg(ArgValKeeper::mem_null(), Some(TypeId::of::<T>()), None),
        }
    }

    /// Non-builder-style version of `::arg_img()`.
    fn new_arg_img<T, M>(&mut self, image_opt: Option<&'b M>) -> u32
    where
        T: OclPrm,
        M: 'b + AsMem<T> + MemCmdAll,
    {
        match image_opt {
            Some(image) => {
                // Type is ignored:
                self.new_arg(
                    ArgValKeeper::Shared(ArgVal::mem(image.as_mem())),
                    None,
                    Some(image.as_mem().clone()),
                )
            }
            None => self.new_arg(ArgValKeeper::mem_null(), None, None),
        }
    }

    /// Non-builder-style version of `::arg_img()`.
    fn new_arg_smp(&mut self, sampler_opt: Option<&'b Sampler>) -> u32 {
        match sampler_opt {
            Some(sampler) => {
                // Type is ignored:
                self.new_arg(ArgValKeeper::Shared(ArgVal::sampler(sampler)), None, None)
            }
            None => self.new_arg(ArgValKeeper::Shared(ArgVal::sampler_null()), None, None),
        }
    }

    /// Non-builder-style version of `::arg_scl()`.
    fn new_arg_scl<T>(&mut self, scalar: T) -> u32
    where
        T: OclPrm,
    {
        self.new_arg(
            ArgValKeeper::owned_prm(scalar),
            Some(TypeId::of::<T>()),
            None,
        )
    }

    /// Non-builder-style version of `::arg_vec()`.
    fn new_arg_vec<T>(&mut self, vector: T) -> u32
    where
        T: OclPrm,
    {
        self.new_arg(
            ArgValKeeper::owned_prm(vector),
            Some(TypeId::of::<T>()),
            None,
        )
    }

    /// Non-builder-style version of `::arg_loc()`.
    fn new_arg_loc<T>(&mut self, length: usize) -> u32
    where
        T: OclPrm,
    {
        self.new_arg(
            ArgValKeeper::Shared(ArgVal::local::<T>(&length)),
            None,
            None,
        )
    }

    /// Adds a new `Buffer`, `Image`, scalar, or vector argument to the
    /// kernel.
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// ### Example
    ///
    /// ```rust,ignore
    /// let kern = ocl_pq.kernel_builder("multiply_by_scalar")
    ///     .arg(&100.0f32)
    ///     .arg(&source_buffer)
    ///     .arg(&result_buffer)
    ///     .build()?;
    /// ```
    pub fn arg<'s, T, A>(&'s mut self, arg: A) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
        A: Into<ArgValConverter<'b, T>>,
    {
        let arg = arg.into();
        self.new_arg(arg.val, arg.type_id, arg.mem);
        self
    }

    /// Adds a new argument to the kernel specifying the buffer object represented
    /// by 'buffer'.
    ///
    /// The argument is added to the bottom of the argument order.
    #[deprecated(since = "0.18.0", note = "Use ::arg instead.")]
    pub fn arg_buf<'s, T, M>(&'s mut self, buffer: &'b M) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
        M: 'b + AsMem<T> + MemCmdAll,
    {
        self.new_arg_buf::<T, _>(Some(buffer));
        self
    }

    /// Adds a new argument to the kernel specifying the image object represented
    /// by 'image'.
    ///
    /// The argument is added to the bottom of the argument order.
    #[deprecated(since = "0.18.0", note = "Use ::arg instead.")]
    pub fn arg_img<'s, T, M>(&'s mut self, image: &'b M) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
        M: 'b + AsMem<T> + MemCmdAll,
    {
        self.new_arg_img::<T, _>(Some(image));
        self
    }

    /// Adds a new argument to the kernel specifying the sampler object represented
    /// by 'sampler'. Argument is added to the bottom of the argument
    /// order.
    #[deprecated(since = "0.18.0", note = "Use ::arg_sampler instead.")]
    pub fn arg_smp<'s>(&'s mut self, sampler: &'b Sampler) -> &'s mut KernelBuilder<'b> {
        self.new_arg_smp(Some(sampler));
        self
    }

    /// Adds a new argument specifying the value: `scalar`.
    ///
    /// The argument is added to the bottom of the argument order.
    #[deprecated(since = "0.18.0", note = "Use ::arg instead.")]
    pub fn arg_scl<'s, T>(&'s mut self, scalar: T) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
    {
        self.new_arg_scl(scalar);
        self
    }

    /// Adds a new argument specifying the value: `vector`.
    ///
    /// The argument is added to the bottom of the argument order.
    #[deprecated(since = "0.18.0", note = "Use ::arg instead.")]
    pub fn arg_vec<'s, T>(&'s mut self, vector: T) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
    {
        self.new_arg_vec(vector);
        self
    }

    /// Adds a new argument specifying the allocation of a local variable of size
    /// `length * sizeof(T)` bytes (builder_style).
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// Local variables are used to share data between work items in the same
    /// workgroup.
    #[deprecated(since = "0.18.0", note = "Use ::arg_local instead.")]
    pub fn arg_loc<'s, T>(&'s mut self, length: usize) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
    {
        self.new_arg_loc::<T>(length);
        self
    }

    /// Adds a new argument to the kernel specifying the sampler object represented
    /// by 'sampler'. Argument is added to the bottom of the argument
    /// order.
    pub fn arg_sampler<'s>(&'s mut self, sampler: &'b Sampler) -> &'s mut KernelBuilder<'b> {
        self.new_arg_smp(Some(sampler));
        self
    }

    /// Adds a new argument specifying the allocation of a local variable of size
    /// `length * sizeof(T)` bytes (builder_style).
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// Local variables are used to share data between work items in the same
    /// workgroup.
    pub fn arg_local<'s, T>(&'s mut self, length: usize) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
    {
        self.new_arg_loc::<T>(length);
        self
    }

    /// Adds a new *named* `Buffer`, `Image`, scalar, or vector argument to the
    /// kernel.
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// To set a `Buffer` or `Image` argument to `None` (null), you must use a
    /// type annotation (e.g. `None::<&Buffer<f32>>`). Scalar and vector
    /// arguments may not be null; use zero (e.g. `&0`) instead.
    ///
    /// Named arguments can be modified later using `::set_arg()`.
    ///
    /// ### Example
    ///
    /// ```rust,ignore
    /// // Create a kernel with arguments corresponding to those in the kernel.
    /// // One argument will be 'named':
    /// let kern = ocl_pq.kernel_builder("multiply_by_scalar")
    ///     .arg(&COEFF)
    ///     .arg(&source_buffer)
    ///     .arg_named("result", None::<&Buffer<f32>>)
    ///     .build()?;
    ///
    /// // Set our named argument. The Option<_> wrapper is, well... optional:
    /// kern.set_arg("result", &result_buffer)?;
    /// // We can also set arguments (named or not) by index:
    /// kern.set_arg(2, &result_buffer)?;
    /// ```
    pub fn arg_named<'s, T, S, A>(&'s mut self, name: S, arg: A) -> &'s mut KernelBuilder<'b>
    where
        S: Into<Cow<'static, str>>,
        T: OclPrm,
        A: Into<ArgValConverter<'b, T>>,
    {
        let arg = arg.into();
        let arg_idx = self.new_arg(arg.val, arg.type_id, arg.mem);
        self.named_args.insert(name.into(), arg_idx);
        self
    }

    /// Adds a new *named* argument specifying the buffer object represented by
    /// 'buffer'.
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_buf_named()`.
    #[deprecated(since = "0.18.0", note = "Use ::arg_named instead.")]
    pub fn arg_buf_named<'s, T, S, M>(
        &'s mut self,
        name: S,
        buffer_opt: Option<&'b M>,
    ) -> &'s mut KernelBuilder<'b>
    where
        S: Into<Cow<'static, str>>,
        T: OclPrm,
        M: 'b + AsMem<T> + MemCmdAll,
    {
        let arg_idx = self.new_arg_buf::<T, _>(buffer_opt);
        self.named_args.insert(name.into(), arg_idx);
        self
    }

    /// Adds a new *named* argument specifying the image object represented by
    /// 'image'.
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_img_named()`.
    #[deprecated(since = "0.18.0", note = "Use ::arg_named instead.")]
    pub fn arg_img_named<'s, T, S, M>(
        &'s mut self,
        name: S,
        image_opt: Option<&'b M>,
    ) -> &'s mut KernelBuilder<'b>
    where
        S: Into<Cow<'static, str>>,
        T: OclPrm,
        M: 'b + AsMem<T> + MemCmdAll,
    {
        let arg_idx = self.new_arg_img::<T, _>(image_opt);
        self.named_args.insert(name.into(), arg_idx);
        self
    }

    /// Adds a new *named* argument specifying the sampler object represented by
    /// 'sampler'.
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_smp_named()`.
    #[deprecated(since = "0.18.0", note = "Use ::arg_sampler_named instead.")]
    pub fn arg_smp_named<'s, S>(
        &'s mut self,
        name: S,
        sampler_opt: Option<&'b Sampler>,
    ) -> &'s mut KernelBuilder<'b>
    where
        S: Into<Cow<'static, str>>,
    {
        let arg_idx = self.new_arg_smp(sampler_opt);
        self.named_args.insert(name.into(), arg_idx);
        self
    }

    /// Adds a new *named* argument  specifying the value: `scalar`.
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// Scalar arguments may not be null, use zero (e.g. `&0`) instead.
    ///
    /// Named arguments can be easily modified later using `::set_arg_scl_named()`.
    #[deprecated(since = "0.18.0", note = "Use ::arg_named instead.")]
    pub fn arg_scl_named<'s, T>(
        &'s mut self,
        name: &'static str,
        scalar: T,
    ) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
    {
        let arg_idx = self.new_arg_scl(scalar);
        self.named_args.insert(name.into(), arg_idx);
        self
    }

    /// Adds a new *named* argument specifying the value: `vector`.
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// Vector arguments may not be null, use zero (e.g. `&0`) instead.
    ///
    /// Named arguments can be easily modified later using `::set_arg_vec_named()`.
    #[deprecated(since = "0.18.0", note = "Use ::arg_named instead.")]
    pub fn arg_vec_named<'s, T>(
        &'s mut self,
        name: &'static str,
        vector: T,
    ) -> &'s mut KernelBuilder<'b>
    where
        T: OclPrm,
    {
        let arg_idx = self.new_arg_vec(vector);
        self.named_args.insert(name.into(), arg_idx);
        self
    }

    /// Adds a new *named* argument specifying the sampler object represented by
    /// 'sampler'.
    ///
    /// The argument is added to the bottom of the argument order.
    ///
    /// Named arguments can be easily modified later using `::set_arg_smp_named()`.
    pub fn arg_sampler_named<'s>(
        &'s mut self,
        name: &'static str,
        sampler_opt: Option<&'b Sampler>,
    ) -> &'s mut KernelBuilder<'b> {
        let arg_idx = self.new_arg_smp(sampler_opt);
        self.named_args.insert(name.into(), arg_idx);
        self
    }

    /// Specifies whether or not to store a copy of memory objects (`Buffer`
    /// and `Image`).
    ///
    /// ### Safety
    ///
    /// Disabling memory object argument retention can lead to a misleading
    /// error message or a difficult to debug segfault (depending on the
    /// platform) *if* a memory object is dropped before a kernel referring to
    /// it is enqueued. Only disable this if you are certain all of the memory
    /// objects set as kernel arguments will outlive the `Kernel` itself.
    pub unsafe fn disable_mem_arg_retention<'s>(&'s mut self) -> &'s mut KernelBuilder<'b> {
        self.mem_args = MemArgs(None);
        self
    }

    /// Disables argument type checking when setting arguments.
    ///
    /// Because the performance cost of argument type checking is negligible,
    /// disabling is not recommended.
    ///
    /// Argument type checking will automatically be disabled if any of the
    /// devices in use do not support OpenCL version 1.2 or higher or if
    /// argument information is not available on the associated platform.
    pub unsafe fn disable_arg_type_check<'s>(&'s mut self) -> &'s mut KernelBuilder<'b> {
        self.disable_arg_check = true;
        self
    }

    /// Builds and returns a new `Kernel`
    pub fn build(&self) -> OclResult<Kernel> {
        let program = self.program.ok_or(KernelError::BuilderNoProgram)?;
        let name = self.name.as_ref().ok_or(KernelError::BuilderNoKernelName)?;

        let obj_core = core::create_kernel(program, name)?;

        let num_args = match core::get_kernel_info(&obj_core, KernelInfo::NumArgs) {
            Ok(KernelInfoResult::NumArgs(num)) => num,
            Err(err) => return Err(OclError::from(err)),
            _ => unreachable!(),
        };

        if self.args.len() as u32 != num_args {
            return Err(KernelError::BuilderWrongArgCount {
                required: num_args,
                specified: self.args.len() as u32,
            }
            .into());
        }

        let mut arg_types = Vec::with_capacity(num_args as usize);
        let mut all_arg_types_unknown = true;
        let mut disable_arg_check = self.disable_arg_check;

        // Cache argument types for later use, bypassing if the OpenCL version
        // is too low (v1.1).
        for arg_idx in 0..num_args {
            let arg_type = match ArgType::from_kern_and_idx(&obj_core, arg_idx) {
                Ok(at) => {
                    if !at.is_unknown() {
                        all_arg_types_unknown = false;
                    }
                    at
                }
                Err(err) => {
                    if let OclError::OclCore(ref core_err) = err {
                        if let OclCoreErrorKind::VersionLow { .. } = core_err {
                            disable_arg_check = true;
                            break;
                        }
                    }
                    return Err(err);
                }
            };
            arg_types.push(arg_type);
        }

        // Check argument types then set arguments.
        for (arg_idx, &(ref arg, ref type_id_opt)) in self.args.iter().enumerate() {
            if !disable_arg_check {
                if let Some(type_id) = *type_id_opt {
                    if !arg_types[arg_idx].matches(type_id) {
                        let ty_name = arg_type_name(&obj_core, arg_idx as u32)?;
                        let arg_name = arg_name(&obj_core, arg_idx as u32)?;
                        return Err(KernelError::ArgTypeMismatch {
                            idx: arg_idx as u32,
                            arg_name,
                            ty_name,
                            ty: arg_types[arg_idx].clone(),
                        }
                        .into());
                    }
                }
            }

            let val = arg.to_arg_val();

            // Some platforms do not like having a `null` argument set for mem objects.
            if !val.is_mem_null() {
                core::set_kernel_arg(&obj_core, arg_idx as u32, val)?;
            }
        }

        let arg_types = if all_arg_types_unknown || disable_arg_check {
            None
        } else {
            Some(arg_types)
        };

        Ok(Kernel {
            obj_core,
            named_args: self.named_args.clone(),
            mem_args: self.mem_args.clone(),
            queue: self.queue.clone(),
            gwo: self.gwo,
            gws: self.gws,
            lws: self.lws,
            arg_types,
        })
    }
}

/// Returns argument information for a kernel.
pub fn arg_info(
    core: &KernelCore,
    arg_idx: u32,
    info_kind: KernelArgInfo,
) -> OclResult<KernelArgInfoResult> {
    let device_versions = match core.device_versions() {
        Ok(vers) => vers,
        Err(e) => return Err(e.into()),
    };

    core::get_kernel_arg_info(core, arg_idx, info_kind, Some(&device_versions))
        .map_err(OclError::from)
}

/// Returns the type name for a kernel argument at the specified index.
pub fn arg_type_name(core: &KernelCore, arg_idx: u32) -> OclResult<String> {
    match arg_info(core, arg_idx, KernelArgInfo::TypeName) {
        Ok(KernelArgInfoResult::TypeName(type_name)) => Ok(type_name),
        Err(err) => Err(err),
        _ => unreachable!(),
    }
}

/// Returns the type name for a kernel argument at the specified index.
pub fn arg_name(core: &KernelCore, arg_idx: u32) -> OclResult<String> {
    match arg_info(core, arg_idx, KernelArgInfo::Name) {
        Ok(KernelArgInfoResult::Name(name)) => Ok(name),
        Err(err) => Err(err),
        _ => unreachable!(),
    }
}

pub mod arg_type {
    #![allow(unused_imports)]
    use super::{arg_info, arg_type_name};
    use crate::core::{
        Error as OclCoreError, Kernel as KernelCore, OclPrm, Result as OclCoreResult, Status,
    };
    use crate::error::{Error as OclError, Result as OclResult};
    use crate::ffi::{
        cl_bitfield, cl_bool, cl_char, cl_double, cl_float, cl_half, cl_int, cl_long, cl_short,
        cl_uchar, cl_uint, cl_ulong, cl_ushort,
    };
    use crate::standard::Sampler;
    use std::any::{Any, TypeId};

    pub use crate::core::{
        Char, Char16, Char2, Char3, Char4, Char8, Double, Double16, Double2, Double3, Double4,
        Double8, Float, Float16, Float2, Float3, Float4, Float8, Int, Int16, Int2, Int3, Int4,
        Int8, Long, Long16, Long2, Long3, Long4, Long8, Short, Short16, Short2, Short3, Short4,
        Short8, Uchar, Uchar16, Uchar2, Uchar3, Uchar4, Uchar8, Uint, Uint16, Uint2, Uint3, Uint4,
        Uint8, Ulong, Ulong16, Ulong2, Ulong3, Ulong4, Ulong8, Ushort, Ushort16, Ushort2, Ushort3,
        Ushort4, Ushort8,
    };

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

    /// The cardinality of an OpenCL primitive.
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
        pub fn unknown() -> OclResult<ArgType> {
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
        pub fn from_str(type_name: &str) -> OclResult<ArgType> {
            let is_ptr = type_name.contains('*');

            let card = if type_name.contains("16") {
                Cardinality::Sixteen
            } else if type_name.contains('8') {
                Cardinality::Eight
            } else if type_name.contains('4') {
                Cardinality::Four
            } else if type_name.contains('3') {
                Cardinality::Three
            } else if type_name.contains('2') {
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
                is_ptr,
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
        pub fn from_kern_and_idx(core: &KernelCore, arg_idx: u32) -> OclResult<ArgType> {
            use crate::core::EmptyInfoResultError;
            use crate::core::Error as OclCoreErrorKind;

            match arg_type_name(core, arg_idx) {
                Ok(type_name) => ArgType::from_str(type_name.as_str()),
                Err(err) => {
                    // Escape hatches for known, platform-specific errors.
                    if let OclError::OclCore(ref core_err) = err {
                        match *core_err {
                            OclCoreErrorKind::Api(ref api_err) => {
                                if api_err.status() == Status::CL_KERNEL_ARG_INFO_NOT_AVAILABLE {
                                    return ArgType::unknown().map_err(OclError::from);
                                }
                            }
                            OclCoreErrorKind::EmptyInfoResult(EmptyInfoResultError::KernelArg) => {
                                return ArgType::unknown();
                            }
                            _ => (),
                        }
                    }

                    Err(err)
                }
            }
        }

        /// Returns true if `type_id` matches the base type of this `ArgType`.
        pub fn matches(&self, type_id: TypeId) -> bool {
            macro_rules! matches {
                (
                    $primary:ty,
                    $struct:ty, $struct2:ty,
                    $struct3:ty, $struct4:ty,
                    $struct8:ty, $struct16:ty
                ) => {{
                    let card_match = match self.cardinality {
                        Cardinality::One => {
                            TypeId::of::<$primary>() == type_id
                                || TypeId::of::<$struct>() == type_id
                        }
                        Cardinality::Two => TypeId::of::<$struct2>() == type_id,
                        Cardinality::Three => TypeId::of::<$struct3>() == type_id,
                        Cardinality::Four => TypeId::of::<$struct4>() == type_id,
                        Cardinality::Eight => TypeId::of::<$struct8>() == type_id,
                        Cardinality::Sixteen => TypeId::of::<$struct16>() == type_id,
                    };

                    if self.is_ptr {
                        card_match
                            || TypeId::of::<$primary>() == type_id
                            || TypeId::of::<$struct>() == type_id
                    } else {
                        card_match
                    }
                }};
            }

            match self.base_type {
                BaseType::Char => matches!(cl_char, Char, Char2, Char3, Char4, Char8, Char16),
                BaseType::Uchar => {
                    matches!(cl_uchar, Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16)
                }

                BaseType::Short => {
                    matches!(cl_short, Short, Short2, Short3, Short4, Short8, Short16)
                }
                BaseType::Ushort => {
                    matches!(cl_ushort, Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16)
                }

                BaseType::Int => matches!(cl_int, Int, Int2, Int3, Int4, Int8, Int16),
                BaseType::Uint => matches!(cl_uint, Uint, Uint2, Uint3, Uint4, Uint8, Uint16),

                BaseType::Long => matches!(cl_long, Long, Long2, Long3, Long4, Long8, Long16),
                BaseType::Ulong => {
                    matches!(cl_ulong, Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16)
                }

                BaseType::Float => {
                    matches!(cl_float, Float, Float2, Float3, Float4, Float8, Float16)
                }
                BaseType::Double => {
                    matches!(cl_double, Double, Double2, Double3, Double4, Double8, Double16)
                }

                BaseType::Sampler => TypeId::of::<u64>() == type_id,
                BaseType::Image => TypeId::of::<u64>() == type_id,

                // Everything matches if type was undetermined (escape hatch):
                BaseType::Unknown => true,
            }
        }

        /// Returns true if the type of `T` matches the base type of this `ArgType`.
        pub fn is_match<T: OclPrm + Any + 'static>(&self) -> bool {
            self.matches(TypeId::of::<T>())
        }

        #[allow(dead_code)]
        pub fn is_ptr(&self) -> bool {
            self.is_ptr
        }

        pub fn is_unknown(&self) -> bool {
            match self.base_type {
                BaseType::Unknown => true,
                _ => false,
            }
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

#[cfg(test)]
mod tests {
    use crate::ProQue;

    #[test]
    fn set_local_arg() -> crate::Result<()> {
        let src = r#"
            __kernel void local_args(__local double* localVec) {

            }
        "#;

        let pro_que = ProQue::builder().src(src).build()?;

        let kernel_diff = pro_que
            .kernel_builder("local_args")
            .global_work_size(1)
            .arg_local::<f64>(64)
            .build()?;

        unsafe {
            kernel_diff.enq()?;
        }
        Ok(())
    }
}
