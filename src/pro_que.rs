//! A convenience raw chimera of `Program` and `Queue`.
use raw;
use super::{Context, Kernel, WorkDims, ProgramBuilder, ProQueBuilder, Program, Queue, 
    Result as OclResult, Error as OclError};

/// A convenience raw chimera of the `Program`, `Queue`, and optionally,
/// `Context` types .
///
/// Handy when creating only a single context, program, and queue or when
/// using a unique program build on each device.
///
/// All `ProQue` functionality is also provided separately by the `Context`, `Queue`, 
/// and `Program` types.
/// 
/// # Creation
/// There are two ways to create a `ProQue`:
/// 1. First call `::new` and pass a `Context` and device index. Next call 
///    `::build` and pass a `ProgramBuilder`.
/// 2. [FIXME]: UPDATE THIS
///
/// # Destruction
/// `::release` must be manually called by consumer.
///
pub struct ProQue {
    context: Option<Context>,
    queue: Queue,
    program: Option<Program>,
}

impl ProQue {
    /// Returns a new ProQueBuilder.
    ///
    /// Calling `ProQueBuilder::build()` will return a new ProQue.
    // pub fn builder() -> ProQueBuilder {
    //  ProQueBuilder::new()
    // }
    pub fn builder<'c>() -> ProQueBuilder<'c> {
        ProQueBuilder::new()
    }

    /// Creates a new queue on the device with `device_idx` (see `Queue` documentation)
    /// and returns a new Program/Queue hybrid.
    ///
    /// `::build` must be called before this ProQue can be used.
    //
    /// [FIXME]: Elaborate upon the following:
    ///    - device_idx wraps around (round robins)
    ///    - one device only per ProQue
    pub fn new(context: &Context, device_idx: Option<usize>) -> ProQue {
        let queue = Queue::new(context, device_idx);

        ProQue {
            queue: queue,
            program: None,
            context: None,
        }
    }

    /// Creates a new ProQue from individual parts.
    pub fn from_parts(context: Option<Context>, queue: Queue, program: Option<Program>) -> ProQue {

        ProQue {
            context: context,
            queue: queue,
            program: program,
        }
    }

    /// [UNSTABLE]: Prefer using `ProQueBuilder`.
    /// Builds contained program with `program_builder`.
    ///
    /// # Panics
    /// This `ProQue` must not already contain a program.
    ///
    /// `program_builder` must not have any device indexes configured (via its
    /// `::device_idxs` method). `ProQue` will only build programs for the device
    /// previously configured or the default device if none had been specified.
    ///
    /// # Stability
    ///
    /// The usefulness of this method is questionable now that we have a builder. 
    /// It may be depricated.
    pub fn build_program(&mut self, program_builder: &ProgramBuilder) -> OclResult<()> {
        if self.program.is_some() { 
            return OclError::err("ProQue::build_program(): Pre-existing build detected. Use \
                '.clear_build()' first.");
        }

        if program_builder.get_device_idxs().len() > 0 {
            return OclError::err("ProQue::build_program(): The 'ProgramBuilder' passed \
                may not have any device indexes set as they will be ignored. See 'ProQue' \
                documentation for more information.");
        }
        
        self.program = Some(try!(Program::from_parts(
            try!(program_builder.get_src_strings().map_err(|e| e.to_string())), 
            try!(program_builder.get_compiler_options().map_err(|e| e.to_string())), 
            self.queue.context_obj(), 
            &vec![self.queue.device_id_obj_raw()],
        )));

        Ok(())
    }

    /// [UNSTABLE]: Usefulness and safety questionable.
    /// Clears the current program build. Any kernels created with the pre-existing program will continue to work but new kernels will require a new program to be built. This can occasionally be useful for creating different programs based on the same source but with different constants.
    /// 
    /// # Stability
    ///
    /// The safety and usefulness of this method is questionable. It may be depricated.
    pub fn clear_build(&mut self) {
        match self.program {
            Some(ref mut program) => { 
                program.release();              
            },

            None => (),
        }
        self.program = None;
    }

    /// Returns a new Kernel with name: `name` and global work size: `gws`.
    // [FIXME] TODO: Return result instead of panic.
    pub fn create_kernel(&self, name: &str, gws: WorkDims) -> OclResult<Kernel> {
        let program = match self.program {
            Some(ref prg) => prg,
            None => {
                return OclError::err("\nProQue::create_kernel(): Cannot add new kernel until \
                OpenCL program is built. Use: \
                '{{your_proque}}.build_program({{your_program_builder}});'.\n")
            },
        };

        Kernel::new(name.to_string(), &program, &self.queue, gws)   
    }

    /// Returns the maximum workgroup size supported by the device on which the
    /// contained queue exists.
    pub fn max_work_group_size(&self) -> usize {
        raw::get_max_work_group_size(self.queue.device_id_obj_raw())
    }

    /// Returns the queue created when constructing this ProQue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the current program build, if any.
    pub fn program(&self) -> &Option<Program> {
        &self.program
    }

    /// Release all components.
    pub fn release(&mut self) {     
        self.queue.release();
        self.clear_build();

        if let Some(ref mut context) = self.context {
            context.release();
        }
    }
}
