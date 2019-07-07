//! A convenient wrapper for `Program` and `Queue`.

use std::ops::Deref;
use crate::error::{Error as OclError, Result as OclResult};
use crate::core::{OclPrm, CommandQueueProperties};
use crate::standard::{Platform, Device, Context, ProgramBuilder, Program, Queue, Kernel, Buffer,
    MemLen, SpatialDims, WorkDims, DeviceSpecifier, KernelBuilder, BufferBuilder};

static DIMS_ERR_MSG: &'static str = "This 'ProQue' has not had any dimensions specified. Use
    'ProQueBuilder::dims' during creation or 'ProQue::set_dims' after creation to specify.";

const DEBUG_PRINT: bool = false;


/// An all-in-one chimera of the `Program`, `Queue`, `Context` and
/// (optionally) `SpatialDims` types.
///
/// Handy when you only need a single context, program, and queue for your
/// project or when using a unique context and program on each device.
///
/// All `ProQue` functionality is also provided separately by the `Context`, `Queue`,
/// `Program`, and `SpatialDims` types.
///
///
/// # Creation
///
/// There are two ways to create a `ProQue`:
///
/// 1. [Recommended] Use `ProQue::builder` or `ProQueBuilder::new()`.
/// 2. Call `::new` and pass pre-created components.
///
///
/// # Destruction
///
/// Now handled automatically. Freely use, store, clone, discard, share among
/// threads... put some on your toast... whatever.
///
#[derive(Clone, Debug)]
pub struct ProQue {
    context: Context,
    queue: Queue,
    program: Program,
    dims: Option<SpatialDims>,
}

impl ProQue {
    /// Returns a new `ProQueBuilder`.
    ///
    /// This is the recommended way to create a new `ProQue`.
    ///
    /// Calling `ProQueBuilder::build()` will return a new `ProQue`.
    pub fn builder<'b>() -> ProQueBuilder<'b> {
        ProQueBuilder::new()
    }

    /// Creates a new ProQue from individual parts.
    ///
    /// Use `::builder` instead unless you know what you're doing. Creating
    /// from components associated with different devices or contexts will
    /// cause errors later on.
    ///
    pub fn new<D: Into<SpatialDims>>(context: Context, queue: Queue, program: Program,
            dims: Option<D>) -> ProQue {
        ProQue {
            context,
            queue,
            program,
            dims: dims.map(|d| d.into()),
        }
    }

    // /// Creates a kernel with pre-assigned dimensions.
    // pub fn create_kernel(&self, name: &str) -> OclResult<Kernel> {
    //     let kernel = Kernel::new(name.to_string(), &self.program)?
    //         .queue(self.queue.clone());
    //     match self.dims {
    //         Some(d) => Ok(kernel.gws(d)),
    //         None => Ok(kernel),
    //     }
    // }

    /// Returns a new `KernelBuilder` with the name, program, default queue,
    /// and global work size pre-configured.
    ///
    /// Use `::arg` to specify arguments and `::build` to build the kernel.
    ///
    /// ### Example
    ///
    /// ```rust,ignore
    /// let kernel = pro_que.kernel_builder("add")
    ///    .arg(&buffer)
    ///    .arg(&10.0f32)
    ///    .build()?;
    /// ```
    ///
    /// See [`KernelBuilder`] documentation for more
    ///
    /// [`KernelBuilder`]: struct.KernelBuilder.html
    pub fn kernel_builder<S>(&self, name: S) -> KernelBuilder
            where S: Into<String> {
        let mut kb = Kernel::builder();
        kb.name(name);
        kb.program(&self.program);
        kb.queue(self.queue.clone());

        if let Some(d) = self.dims {
            kb.global_work_size(d);
        }

        kb
    }

    /// Returns a new buffer.
    ///
    /// The default dimensions and queue from this `ProQue` will be used.
    ///
    /// The buffer will be filled with zeros upon creation, blocking the
    /// current thread until completion.
    ///
    /// Use `Buffer::builder()` (or `BufferBuilder::new()`) for access to the
    /// full range of buffer creation options.
    ///
    /// # Errors
    ///
    /// This `ProQue` must have been pre-configured with default dimensions.
    /// If not, set them with `::set_dims`, or just create a buffer using
    /// `Buffer::builder()` instead.
    ///
    pub fn create_buffer<T: OclPrm>(&self) -> OclResult<Buffer<T>> {
        let len = self.dims_result()?.to_len();
        Buffer::<T>::builder()
            .queue(self.queue.clone())
            .len(len)
            .fill_val(Default::default())
            .build()
    }

    /// Returns a new `BufferBuilder` with the default queue and length
    /// pre-configured.
    ///
    /// Use `.fill_val(Default::default())` to fill buffer with zeros.
    ///
    /// Use `.build()` to create the buffer.
    ///
    /// ### Panics
    ///
    /// This `ProQue` must have been pre-configured with default dimensions.
    /// If not, set them with `::set_dims`, or just create a buffer using
    /// `Buffer::builder()` instead.
    ///
    pub fn buffer_builder<T: OclPrm>(&self) -> BufferBuilder<T> {
        let len = self.dims_result()
            .expect("`ProQue` dimensions not specified. Please specify dimensions \
                using `::set_dims` before calling this method.")
            .to_len();
        Buffer::<T>::builder()
            .queue(self.queue.clone())
            .len(len)
    }

    /// Sets the default dimensions used when creating buffers and kernels.
    pub fn set_dims<S: Into<SpatialDims>>(&mut self, dims: S) {
        self.dims = Some(dims.into());
    }

    /// Returns the maximum workgroup size supported by the device associated
    /// with this `ProQue`.
    ///
    /// [UNSTABLE]: Evaluate usefulness.
    pub fn max_wg_size(&self) -> OclResult<usize> {
        self.queue.device().max_wg_size().map_err(OclError::from)
    }

    /// Returns a reference to the queue associated with this ProQue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the contained context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the current program build.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Returns the current `dims` or panics.
    ///
    /// [UNSTABLE]: Evaluate which 'dims' method to keep. Leaning towards this
    /// version at the moment.
    pub fn dims(&self) -> &SpatialDims {
        self.dims_result().expect(DIMS_ERR_MSG)
    }

    /// Returns the current `dims` or an error.
    ///
    /// [UNSTABLE]: Evaluate which 'dims' method to keep. Leaning towards the
    /// above, panicking version at the moment.
    pub fn dims_result(&self) -> OclResult<&SpatialDims> {
        match self.dims {
            Some(ref dims) => Ok(dims),
            None => Err(DIMS_ERR_MSG.into()),
        }
    }
}

impl MemLen for ProQue {
    fn to_len(&self) -> usize {
        self.dims().to_len()
    }
    fn to_len_padded(&self, incr: usize) -> usize {
        self.dims().to_len_padded(incr)
    }
    fn to_lens(&self) -> [usize; 3] {
        self.dims_result().expect("ocl::ProQue::to_lens()")
            .to_lens().expect("ocl::ProQue::to_lens()")
    }
}

impl WorkDims for ProQue {
    fn dim_count(&self) -> u32 {
        self.dims_result().expect("ProQue::dim_count").dim_count()
    }

    fn to_work_size(&self) -> Option<[usize; 3]> {
        self.dims_result().expect("ProQue::to_work_size").to_work_size()
    }

    fn to_work_offset(&self) -> Option<[usize; 3]> {
        self.dims_result().expect("ProQue::to_work_offset").to_work_offset()
    }
}

impl Deref for ProQue {
    type Target = Queue;

    fn deref(&self) -> &Queue {
        &self.queue
    }
}


/// A builder for `ProQue`.
#[must_use = "builders do nothing unless '::build' is called"]
pub struct ProQueBuilder<'b> {
    platform: Option<Platform>,
    context: Option<Context>,
    device_spec: Option<DeviceSpecifier>,
    program_builder: Option<ProgramBuilder<'b>>,
    dims: Option<SpatialDims>,
    queue_properties: Option<CommandQueueProperties>,
}

impl<'b> ProQueBuilder<'b> {
    /// Returns a new `ProQueBuilder` with an empty / default configuration.
    ///
    /// The minimum amount of configuration possible before calling `::build` is to
    /// simply assign some source code using `::src`.
    ///
    /// For full configuration options, separately create a `Context` and
    /// `ProgramBuilder` (do not `::build` the `ProgramBuilder`, its device
    /// list must be set by this `ProQueBuilder` to assure consistency) then
    /// pass them as arguments to the `::context` and `::prog_bldr` methods
    /// respectively.
    ///
    pub fn new() -> ProQueBuilder<'b> {
        ProQueBuilder {
            platform: None,
            context: None,
            device_spec: None,
            program_builder: None,
            dims: None,
            queue_properties: None,
        }
    }

    /// Sets the platform to be used and returns the builder.
    ///
    /// # Panics
    ///
    /// If context is set, this will panic upon building. Only one or the other
    /// can be configured.
    pub fn platform(&mut self, platform: Platform) -> &mut ProQueBuilder<'b> {
        self.platform = Some(platform);
        self
    }

    /// Sets the context and returns the `ProQueBuilder`.
    ///
    /// # Panics
    ///
    /// If platform is set, this will panic upon building. Only one or the other
    /// can be configured.
    pub fn context(&mut self, context: Context) -> &mut ProQueBuilder<'b> {
        self.context = Some(context);
        self
    }

    /// Sets a device or devices to be used and returns a `ProQueBuilder`
    /// reference.
    ///
    /// Must specify only a single device.
    ///
    pub fn device<D: Into<DeviceSpecifier>>(&mut self, device_spec: D)
            -> &mut ProQueBuilder<'b>
    {
        assert!(self.device_spec.is_none(), "ocl::ProQue::devices: Devices already specified");
        self.device_spec = Some(device_spec.into());
        self
    }

    /// Adds some source code to be compiled and returns the `ProQueBuilder`.
    ///
    /// Creates a `ProgramBuilder` if one has not already been added. Attempts
    /// to call `::program_builder` after calling this method will cause a panic.
    ///
    /// If you need a more complex build configuration or to add multiple
    /// source files. Pass an *unbuilt* `ProgramBuilder` to the
    /// `::program_builder` method (described below).
    pub fn src<S: Into<String>>(&mut self, src: S) -> &mut ProQueBuilder<'b> {
        if self.program_builder.is_some() {
            panic!("ocl::ProQueBuilder::src: Cannot set src if a 'ProgramBuilder' is already \
                defined. Please use the '::program_builder' method for more complex build \
                configurations.");
        } else {
            let mut pb = Program::builder();
            pb.src(src);
            self.program_builder = Some(pb);
        }
        self
    }

    /// Adds a pre-configured `ProgramBuilder` and returns the `ProQueBuilder`.
    ///
    /// ## Panics
    ///
    /// This `ProQueBuilder` may not already contain a `ProgramBuilder`.
    ///
    /// `program_builder` must not have any device indices configured (via its
    /// `::device_idxs` method). `ProQueBuilder` will only build programs for
    /// the device specified by `::device_idx` or the default device if none has
    /// been specified.
    pub fn prog_bldr(&mut self, program_builder: ProgramBuilder<'b>) -> &mut ProQueBuilder<'b> {
        assert!(self.program_builder.is_none(), "ProQueBuilder::prog_bldr(): Cannot set the \
            'ProgramBuilder' using this method after one has already been set or after '::src' has \
            been called.");

        assert!(program_builder.get_device_spec().is_none(), "ProQueBuilder::prog_bldr(): The \
            'ProgramBuilder' passed may not have any device indices set as they will be unused. \
            See 'ProQueBuilder' documentation for more information.");

        self.program_builder = Some(program_builder);
        self
    }

    /// Sets the built-in dimensions.
    ///
    /// This is optional.
    ///
    /// Use if you want to be able to call the `::create_kernel` or
    /// `::create_buffer` methods on the `ProQue` created by this builder.
    /// Dimensions can alternatively be set after building by using the
    /// `ProQue::set_dims` method.
    ///
    pub fn dims<D: Into<SpatialDims>>(&mut self, dims: D) -> &mut ProQueBuilder<'b> {
        self.dims = Some(dims.into());
        self
    }

    /// Sets the command queue properties.
    ///
    /// Optional.
    ///
    pub fn queue_properties(&mut self, props: CommandQueueProperties) -> &mut ProQueBuilder<'b> {
        self.queue_properties = Some(props);
        self
    }


    /// Returns a new `ProQue`.
    ///
    /// ## Errors
    ///
    /// A `ProgramBuilder` or some source code must have been specified with
    /// `::prog_bldr` or `::src` before building.
    ///
    pub fn build(&self) -> OclResult<ProQue> {
        let program_builder = match self.program_builder {
            Some(ref program_builder) => program_builder,
            None => return Err("ProQueBuilder::build(): No program builder or kernel source defined. \
                OpenCL programs must have some source code to be compiled. Use '::src' to directly \
                add source code or '::program_builder' for more complex builds. Please see the \
                'ProQueBuilder' and 'ProgramBuilder' documentation for more information.".into()),
        };

        // If no platform is set or no context platform is set, use the first available:
        let platform = match self.platform {
            Some(ref plt) => {
                assert!(self.context.is_none(), "ocl::ProQueBuilder::build: \
                    platform and context cannot both be set.");
                *plt
            },
            None => match self.context {
                Some(ref context) => {
                    let plat = context.platform()?;

                    if DEBUG_PRINT { println!("ProQue::build(): plat: {:?}, default: {:?}",
                        plat, Platform::default()); }

                    plat.unwrap_or_default()
                },
                None => Platform::default(),
            },
        };


        // Resolve the device and ensure only one was specified.
        let device = match self.device_spec {
            Some(ref ds) => {
                let device_list = ds.to_device_list(Some(platform))?;

                if device_list.len() == 1 {
                    device_list[0]
                } else {
                    return Err(format!("Invalid number of devices specified ({}). Each 'ProQue' \
                        can only be associated with a single device. Use 'Context', 'Program', and \
                        'Queue' separately for multi-device configurations.",
                        device_list.len()).into());
                }
            },
            None => Device::first(platform)?,
        };

        if DEBUG_PRINT { println!("ProQue::build(): device: {:?}", device); }

        // If no context was set, creates one using the above platform and the
        // pre-set device index (default [0]).
        let context = match self.context {
            Some(ref ctx) => {
                assert!(ctx.devices().contains(&device));
                ctx.clone()
            }
            None => {
                Context::builder()
                    .platform(platform)
                    .devices(device)
                    .build()?
            },
        };

        if DEBUG_PRINT { println!("ProQue::build(): context.devices(): {:?}", context.devices()); }

        let queue = Queue::new(&context, device, self.queue_properties)?;

        // println!("PROQUEBUILDER: About to load SRC_STRINGS.");
        let src_strings = program_builder.get_src_strings().map_err(|e| e.to_string())?;
        // println!("PROQUEBUILDER: About to load CMPLR_OPTS.");
        let cmplr_opts = program_builder.get_compiler_options().map_err(|e| e.to_string())?;
        // println!("PROQUEBUILDER: All done.");

        let program = Program::with_source(
            &context,
            &src_strings,
            Some(&[device]),
            &cmplr_opts,
        )?;

        Ok(ProQue::new(context, queue, program, self.dims))
    }
}

