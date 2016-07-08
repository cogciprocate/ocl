//! A convenient wrapper for `Program` and `Queue`.

use std::convert::Into;
use std::ops::Deref;
use error::{Result as OclResult, Error as OclError};
use core::OclPrm;
use standard::{Platform, Device, Context, ProgramBuilder, Program, Queue, Kernel, Buffer,
    MemLen, SpatialDims, WorkDims, DeviceSpecifier};

static DIMS_ERR_MSG: &'static str = "This 'ProQue' has not had any dimensions specified. Use
    'ProQueBuilder::dims' during creation or 'ProQue::set_dims' after creation to specify.";

const DEBUG_PRINT: bool = false;

/// A builder for `ProQue`.
pub struct ProQueBuilder {
    platform: Option<Platform>,
    context: Option<Context>,
    // device_idx: usize,
    device_spec: Option<DeviceSpecifier>,
    program_builder: Option<ProgramBuilder>,
    dims: Option<SpatialDims>,
}

impl ProQueBuilder {
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
    pub fn new() -> ProQueBuilder {
        ProQueBuilder {
            platform: None,
            context: None,
            // device_idx: 0,
            device_spec: None,
            program_builder: None,
            dims: None,
        }
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
            // Some(program_builder) => ProQueBuilder::_build(self.context, self.device_idx, program_builder),
            Some(ref program_builder) => program_builder,
            None => return OclError::err("ProQueBuilder::build(): No program builder or kernel source defined. \
                OpenCL programs must have some source code to be compiled. Use '::src' to directly \
                add source code or '::program_builder' for more complex builds. Please see the \
                'ProQueBuilder' and 'ProgramBuilder' documentation for more information."),
        };

        // If no platform is set or no context platform is set, use the first available:
        let platform = match self.platform {
            Some(ref plt) => {
                assert!(self.context.is_none(), "ocl::ProQueBuilder::build: \
                    platform and context cannot both be set.");
                plt.clone()
            },
            None => match self.context {
                Some(ref context) => match context.platform() {
                    Some(platform) => platform,
                    None => Platform::default(),
                },
                None => Platform::default(),
            },
        };


        // Resolve the device and ensure only one was specified.
        let device = match self.device_spec {
            Some(ref ds) => {
                let device_list = try!(ds.to_device_list(Some(platform)));

                if device_list.len() == 1 {
                    device_list[0]
                } else {
                    return OclError::err(format!("Invalid number of devices specified ({}). Each 'ProQue' \
                        can only be associated with a single device. Use 'Context', 'Program', and \
                        'Queue' separately for multi-device configurations.", device_list.len()));
                }
            },
            None => Device::first(platform),
        };

        if DEBUG_PRINT { println!("ProQue::build(): device: {:?}", device); }

        // If no context was set, creates one using the above platform and
        // the pre-set device index (default [0]).
        let context = match self.context {
            Some(ref ctx) => ctx.clone(),
            None => {
                try!(Context::builder()
                    .platform(platform)
                    .devices(device)
                    .build())
            },
        };

        if DEBUG_PRINT { println!("ProQue::build(): context.devices(): {:?}", context.devices()); }

        let queue = try!(Queue::new(&context, device));

        // println!("PROQUEBUILDER: About to load SRC_STRINGS.");
        let src_strings = try!(program_builder.get_src_strings().map_err(|e| e.to_string()));
        // println!("PROQUEBUILDER: About to load CMPLR_OPTS.");
        let cmplr_opts = try!(program_builder.get_compiler_options().map_err(|e| e.to_string()));
        // println!("PROQUEBUILDER: All done.");

        let program = try!(Program::new(
            src_strings,
            cmplr_opts,
            &context,
            &[device],
        ));

        Ok(ProQue::new(context, queue, program, self.dims))
    }

    /// Sets the platform to be used and returns the builder.
    ///
    /// # Panics
    ///
    /// If context is set, this will panic upon building. Only one or the other
    /// can be configured.
    pub fn platform(&mut self, platform: Platform) -> &mut ProQueBuilder {
        self.platform = Some(platform);
        self
    }

    /// Sets the context and returns the `ProQueBuilder`.
    ///
    /// # Panics
    ///
    /// If platform is set, this will panic upon building. Only one or the other
    /// can be configured.
    pub fn context(&mut self, context: Context) -> &mut ProQueBuilder {
        self.context = Some(context);
        self
    }

    /// Sets a device or devices to be used and returns a `ProQueBuilder`
    /// reference.
    ///
    /// Must specify only a single device.
    ///
    pub fn device<D: Into<DeviceSpecifier>>(&mut self, device_spec: D)
            -> &mut ProQueBuilder
    {
        assert!(self.device_spec.is_none(), "ocl::ProQue::devices: Devices already specified");
        self.device_spec = Some(device_spec.into());
        self
    }

    // /// Sets a device index to be used and returns the `ProQueBuilder`.
    // ///
    // /// Defaults to `0`, the first available.
    // ///
    // /// This index WILL round robin, in other words, it cannot be invalid.
    // /// If you need to guarantee a certain device, create your parts without
    // /// using this builder and just call `ProQue::new` directly.
    // pub fn device_idx<'p>(&'p mut self, device_idx: usize) -> &'p mut ProQueBuilder {
    //     self.device_idx = device_idx;
    //     self
    // }

    /// Adds some source code to be compiled and returns the `ProQueBuilder`.
    ///
    /// Creates a `ProgramBuilder` if one has not already been added. Attempts
    /// to call `::program_builder` after calling this method will cause a panic.
    ///
    /// If you need a more complex build configuration or to add multiple
    /// source files. Pass an *unbuilt* `ProgramBuilder` to the
    /// `::program_builder` method (described below).
    pub fn src<S: Into<String>>(&mut self, src: S) -> &mut ProQueBuilder {
        if self.program_builder.is_some() {
            panic!("ocl::ProQueBuilder::src: Cannot set src if a 'ProgramBuilder' is already \
                defined. Please use the '::program_builder' method for more complex build \
                configurations.");
        } else {
            self.program_builder = Some(Program::builder().src(src))
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
    pub fn prog_bldr(&mut self, program_builder: ProgramBuilder) -> &mut ProQueBuilder {
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
    pub fn dims<D: Into<SpatialDims>>(&mut self, dims: D) -> &mut ProQueBuilder {
        self.dims = Some(dims.into());
        self
    }
}



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
    pub fn builder() -> ProQueBuilder {
        ProQueBuilder::new()
    }

    /// Creates a new ProQue from individual parts.
    ///
    /// Use builder unless you know what you're doing. Creating parts which are
    /// from different devices or contexts will cause errors later on.
    ///
    pub fn new<D: Into<SpatialDims>>(context: Context, queue: Queue, program: Program,
                    dims: Option<D>) -> ProQue
    {
        ProQue {
            context: context,
            queue: queue,
            program: program,
            dims: dims.map(|d| d.into()),
        }
    }

    /// Creates a kernel with pre-assigned dimensions.
    pub fn create_kernel(&self, name: &str) -> OclResult<Kernel> {
        let kernel = try!(Kernel::new(name.to_string(), &self.program, &self.queue));

        match self.dims {
            Some(d) => Ok(kernel.gws(d)),
            None => Ok(kernel),
        }
    }

    /// Returns a new buffer
    ///
    /// The default dimensions for this `ProQue` will be used.
    ///
    /// # Errors
    ///
    /// This `ProQue` must have been pre-configured with default dimensions to
    /// use this method. If not, set them with `::set_dims`, or just create a
    /// buffer using `Buffer::new()`.
    ///
    #[cfg(any(feature = "opencl_1_2"))]
    pub fn create_buffer<T: OclPrm>(&self) -> OclResult<Buffer<T>> {
        let dims = try!(self.dims_result());
        Buffer::<T>::new(&self.queue, None, &dims, None)
    }

    /// Sets the default dimensions used when creating buffers and kernels.
    pub fn set_dims<S: Into<SpatialDims>>(&mut self, dims: S) {
        self.dims = Some(dims.into());
    }

    /// Returns the maximum workgroup size supported by the device associated
    /// with this `ProQue`.
    ///
    /// [UNSTABLE]: Evaluate usefulness.
    pub fn max_wg_size(&self) -> usize {
        self.queue.device().max_wg_size()
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
    /// above, panicing version at the moment.
    pub fn dims_result(&self) -> OclResult<&SpatialDims> {
        match self.dims {
            Some(ref dims) => Ok(dims),
            None => OclError::err(DIMS_ERR_MSG),
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



    // / Creates a new queue on the device with `device_idx` (see
    // / [`Queue`](/ocl/ocl/struct.Queue.html)
    // / documentation) and returns a new Program/Queue hybrid.
    // /
    // / `::build_program` must be called before this ProQue can be used.
    // /
    // / [FIXME]: Elaborate upon the following:
    // /
    // / - device_idx wraps around (round robins)
    // / - one device only per ProQue
    // / - when is built-in Context used / destroyed
    // /
    // / [UNSTABLE]: Prefer using `ProQueBuilder`.
    // pub fn new(context: Context, device_idx: Option<usize>) -> ProQue {
    //     let queue = Queue::new_by_device_index(context, device_idx);

    //     ProQue {
    //         context: context,
    //         queue: queue,
    //         program: None,
    //         dims: None,
    //     }
    // }



    // /// Builds and stores the program defined by `builder`.
    // ///
    // /// ## Panics
    // /// This `ProQue` must not already contain a program.
    // ///
    // /// `program_builder` must not have any device indices configured (via its
    // /// `::device_idxs` method). `ProQue` will only build programs for the device
    // /// previously configured or the default device if none had been specified.
    // ///
    // /// ## Stability
    // ///
    // /// The usefulness of this method is questionable now that we have a builder.
    // /// It may be depricated.
    // ///
    // /// [UNSTABLE]: Prefer using `ProQueBuilder`.
    // pub fn build_program(&mut self, builder: &ProgramBuilder) -> OclResult<()> {
    //     if self.program.is_some() {
    //         return OclError::err("ProQue::build_program(): Pre-existing build detected. Use \
    //             '.clear_build()' first.");
    //     }

    //     if builder.get_devices().len() > 0 {
    //         return OclError::err("ProQue::build_program(): The 'ProgramBuilder' passed \
    //             may not have any device indices set as they will be ignored. See 'ProQue' \
    //             documentation for more information.");
    //     }

    //     self.program = Some(try!(Program::new(
    //         try!(builder.get_src_strings().map_err(|e| e.to_string())),
    //         try!(builder.get_compiler_options().map_err(|e| e.to_string())),
    //         self.queue.context_core_as_ref(),
    //         &vec![self.queue.device().clone()],
    //     )));

    //     Ok(())
    // }


    // /// Clears the current program build. Any kernels created with the pre-existing program will continue to work but new kernels will require a new program to be built. This can occasionally be useful for creating different programs based on the same source but with different constants.
    // ///
    // /// ## Stability
    // ///
    // /// [UNSTABLE]: Usefulness and safety questionable.
    // ///
    // pub fn clear_build(&mut self) {
    //     // match self.program {
    //     //     Some(ref mut program) => {
    //     //         program.release();
    //     //     },

    //     //     None => (),
    //     // }
    //     self.program = None;
    // }




    // /// Returns a new Kernel with name: `name` and global work size: `gws`.
    // ///
    // /// # Panics
    // ///
    // /// Panics if the contained program has not been created / built or if
    // /// there is a problem creating the kernel.
    // pub fn create_kernel_with_dims<D: Into<SpatialDims>>(&self, name: &str, gws: D) -> Kernel {
    //     let program = match self.program {
    //         Some(ref prg) => prg,
    //         None => {
    //             panic!("\nProQue::create_kernel(): Cannot add new kernel until \
    //             OpenCL program is built. Use: \
    //             '{{your_proque}}.build_program({{your_program_builder}});'.\n")
    //         },
    //     };

    //     Kernel::new(name.to_string(), &program, &self.queue, gws.into()).unwrap()
    // }


// impl Deref for ProQue {
//     type Target = SpatialDims;

//     fn deref(&self) -> &SpatialDims {
//         match self.dims {
//             Some(ref dims) => dims,
//             None => panic!(DIMS_ERR_MSG),
//         }
//     }
// }

// impl Deref for ProQue {
//     type Target = Context;

//     fn deref(&self) -> &Context {
//         &self.context
//     }
// }

// impl Deref for ProQue {
//     type Target = Program;

//     fn deref(&self) -> &Program {
//         &self.program
//     }
// }
