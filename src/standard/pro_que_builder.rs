//! The builder for `ProQue`.

use std::convert::Into;
use error::{Result as OclResult, Error as OclError};
use standard::{Platform, Context, ProgramBuilder, Program, Queue, ProQue, SimpleDims, DeviceSpecifier};

/// A builder for `ProQue`.
pub struct ProQueBuilder {
    platform: Option<Platform>,
    context: Option<Context>,
    device_idx: usize,
    program_builder: Option<ProgramBuilder>,
    dims: Option<SimpleDims>,
}

impl ProQueBuilder {
    /// Returns a new `ProQueBuilder` with an empty / default configuration.
    ///
    /// The minimum amount of configuration possible before calling `::build` is to 
    /// simply assign some source code using `::src`.
    ///
    /// For full configuration options, create the context and program_builder
    /// separately and pass them as options.
    ///
    pub fn new() -> ProQueBuilder {
        ProQueBuilder { 
            platform: None,
            context: None,
            device_idx: 0,
            program_builder: None,
            dims: None,
        }
    }

    /// Returns a new `ProQue`.
    ///
    /// ## Errors
    ///
    /// A `ProgramBuilder` or some source code must have been specified with `::program_builder` or `::src` before building.
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

        // If no platform was set, uses the first available.
        let platform = match self.platform {
            Some(ref plt) => {
                assert!(self.context.is_none(), "ocl::ProQueBuilder::build: \
                    platform and context cannot both be set.");
                plt.clone()
            },
            None => Platform::list()[0].clone(),
        };

        // If no context was set, creates one using the above platform and
        // the pre-set device index (default [0]).
        let context = match self.context {
            Some(ref ctx) => ctx.clone(),
            None => {
                // try!(Context::new(Some(ContextProperties::new().platform(platform)),
                //     Some(DeviceSpecifier::Index(self.device_idx)), None, None))

                try!(Context::builder()
                    .platform(platform)
                    .devices(DeviceSpecifier::Indices(vec![self.device_idx]))
                    .build())
            },
        };

        let device = context.get_device_by_index(self.device_idx);

        let queue = try!(Queue::new(&context, Some(device.clone())));

        let program = try!(Program::from_parts(
            try!(program_builder.get_src_strings().map_err(|e| e.to_string())), 
            try!(program_builder.get_compiler_options().map_err(|e| e.to_string())), 
            &context, 
            &vec![device],
        ));

        Ok(ProQue::new(context, queue, program, self.dims.clone()))
    }

    /// Sets the platform to be used and returns the builder.
    ///
    /// # Panics
    ///
    /// If context is set, this will panic upon building. Only one or the other
    /// can be configured.
    pub fn platform<'p>(&'p mut self, platform: Platform) -> &'p mut ProQueBuilder {
        self.platform = Some(platform);
        self
    }

    /// Sets the context and returns the `ProQueBuilder`.
    ///
    /// # Panics
    ///
    /// If platform is set, this will panic upon building. Only one or the other
    /// can be configured.
    pub fn context<'p>(&'p mut self, context: Context) -> &'p mut ProQueBuilder {
        self.context = Some(context);
        self
    }

    /// Sets a device index to be used and returns the `ProQueBuilder`.
    ///
    /// Defaults to `0`, the first available.
    ///
    /// This index WILL round robin, in other words, it cannot be invalid.
    /// If you need to guarantee a certain device, create your parts without
    /// using this builder and just call `ProQue::new` directly.
    pub fn device_idx<'p>(&'p mut self, device_idx: usize) -> &'p mut ProQueBuilder {
        self.device_idx = device_idx;
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
    pub fn src<'p, S: Into<String>>(&'p mut self, src: S) -> &'p mut ProQueBuilder {
        match self.program_builder {
            Some(ref mut program_builder) => { program_builder.src(src); },
            None => self.program_builder = {
                let mut pb = Program::builder();
                pb.src(src);
                Some(pb)
            },
        };

        self
    }

    /// Adds a pre-configured `ProgramBuilder` and returns the `ProQueBuilder`.
    ///
    /// ## Panics
    ///
    /// This `ProQueBuilder` may not already contain a `ProgramBuilder`.
    ///
    /// `program_builder` must not have any device indexes configured (via its
    /// `::device_idxs` method). `ProQueBuilder` will only build programs for
    /// the device specified by `::device_idx` or the default device if none has
    /// been specified.
    pub fn program_builder<'p>(&'p mut self, program_builder: ProgramBuilder) -> &'p mut ProQueBuilder {
        assert!(self.program_builder.is_none(), "ProQueBuilder::program_builder(): Cannot set the \
            'ProgramBuilder' using this method after one has already been set or after '::src' has \
            been called.");

        assert!(program_builder.get_devices().len() == 0, "ProQueBuilder::program_builder(): The \
            'ProgramBuilder' passed may not have any device indexes set as they will be unused. \
            See 'ProQueBuilder' documentation for more information.");

        self.program_builder = Some(program_builder);
        self
    } 

    /// Sets the built-in dimensions.
    ///
    /// This is optional.
    ///
    /// Used if you want to do a quick and dirty `create_kernel` or 
    /// `create_buffer` directly on the `ProQue`.
    pub fn dims<'p, D: Into<SimpleDims>>(&'p mut self, dims: D) -> &'p mut ProQueBuilder {
        self.dims = Some(dims.into());
        self
    }

    // pub fn build_with(self, program_builder: ProgramBuilder) -> OclResult<ProQue> {
    //  if self.program_builder.is_some() { 
    //      return OclError::err("ProQueBuilder::build_with(): This method cannot be used if a \
    //          'ProgramBuilder' has already been specified using '::src' or '::program_builder'. \
    //          Use '::build' instead.");
    //  }

    //  ProQueBuilder::_build(self.context, self.device_idx, program_builder)
    // }

    // fn _build(context: Option<&'c Context>, device_idx: Option<usize>,
    //          program_builder: ProgramBuilder) -> OclResult<ProQue> 
    // {
    //  let mut context_opt: Option<Context> = None;

    //  let queue = match context {
    //      Some(ctx) => Queue::new_by_device_index(ctx, device_idx),
    //      None => {
    //          context_opt = Some(try!(Context::new_by_index_and_type(None, None)));
    //          Queue::new_by_device_index(&context_opt.as_ref().unwrap(), None)
    //      },
    //  };

    //  let program_opt = Some(try!(Program::from_parts(
    //      try!(program_builder.get_src_strings().map_err(|e| e.to_string())), 
    //      try!(program_builder.get_compiler_options().map_err(|e| e.to_string())), 
    //      queue.context_core_as_ref(), 
    //      &vec![queue.device_id()],
    //  )));

    //  Ok(ProQue::from_parts(context_opt, queue, program_opt))
    // }
}
