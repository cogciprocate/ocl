//! A convenient wrapper for `Program` and `Queue`.

use raw;
use standard::{Context, Kernel, WorkDims, ProgramBuilder, ProQueBuilder, Program, Queue};
use error::{Result as OclResult, Error as OclError};

/// An all-in-one chimera of the `Program`, `Queue`, and (optionally) `Context` types.
///
/// Handy when creating only a single context, program, and queue or when
/// using a unique program build on each device.
///
/// All `ProQue` functionality is also provided separately by the `Context`, `Queue`, 
/// and `Program` types.
/// 
/// # Creation
/// There are two ways to create a `ProQue`:
///
/// 1. First call `::new` and pass a `Context` and device index then call 
///    `::build` and pass a `ProgramBuilder`.
/// 2. Call `::builder` [FIXME]: Complete description and give examples.
///
/// # Destruction
///
/// `::release` must currently be called manually by consumer (this is temporary).
///
/// [FIXME]: Finish implementing new destruction sequence.
///
pub struct ProQue {
    context: Option<Context>,
    queue: Queue,
    program: Option<Program>,
}

impl ProQue {
    /// Returns a new `ProQueBuilder`.
    ///
    /// Calling `ProQueBuilder::build()` will return a new `ProQue`.
    pub fn builder<'c>() -> ProQueBuilder<'c> {
        ProQueBuilder::new()
    }

    /// Creates a new queue on the device with `device_idx` (see 
    /// [`Queue`](http://docs.cogciprocate.com/ocl/struct.Queue.html) 
    /// documentation) and returns a new Program/Queue hybrid.
    ///
    /// `::build_program` must be called before this ProQue can be used.
    ///
    /// [FIXME]: Elaborate upon the following:
    ///
    /// - device_idx wraps around (round robins)
    /// - one device only per ProQue
    /// - when is built-in Context used / destroyed
    ///
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

    /// Builds and stores the program defined by `builder`.
    ///
    /// ### Panics
    /// This `ProQue` must not already contain a program.
    ///
    /// `program_builder` must not have any device indexes configured (via its
    /// `::device_idxs` method). `ProQue` will only build programs for the device
    /// previously configured or the default device if none had been specified.
    ///
    /// ### Stability
    ///
    /// The usefulness of this method is questionable now that we have a builder. 
    /// It may be depricated.
    ///
    /// [UNSTABLE]: Prefer using `ProQueBuilder`.
    pub fn build_program(&mut self, builder: &ProgramBuilder) -> OclResult<()> {
        if self.program.is_some() { 
            return OclError::err("ProQue::build_program(): Pre-existing build detected. Use \
                '.clear_build()' first.");
        }

        if builder.get_device_idxs().len() > 0 {
            return OclError::err("ProQue::build_program(): The 'ProgramBuilder' passed \
                may not have any device indexes set as they will be ignored. See 'ProQue' \
                documentation for more information.");
        }
        
        self.program = Some(try!(Program::from_parts(
            try!(builder.get_src_strings().map_err(|e| e.to_string())), 
            try!(builder.get_compiler_options().map_err(|e| e.to_string())), 
            self.queue.context_obj_raw(), 
            &vec![self.queue.device_id_raw().clone()],
        )));

        Ok(())
    }

    /// Clears the current program build. Any kernels created with the pre-existing program will continue to work but new kernels will require a new program to be built. This can occasionally be useful for creating different programs based on the same source but with different constants.
    /// 
    /// ### Stability
    ///
    /// [UNSTABLE]: Usefulness and safety questionable.
    ///
    pub fn clear_build(&mut self) {
        // match self.program {
        //     Some(ref mut program) => { 
        //         program.release();              
        //     },

        //     None => (),
        // }
        self.program = None;
    }

    /// Returns a new Kernel with name: `name` and global work size: `gws`.
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

    /// Returns the maximum workgroup size supported by the device associated
    /// with this `ProQue`.
    pub fn max_work_group_size(&self) -> usize {
        raw::get_max_work_group_size(self.queue.device_id_raw())
    }

    /// Returns a reference to the queue associated with this ProQue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the contained context, if any.
    pub fn context(&self) -> &Option<Context> {
        &self.context
    }

    /// Returns the current program build, if any.
    pub fn program(&self) -> &Option<Program> {
        &self.program
    }

    // /// Releases all components.
    // pub fn release(&mut self) {     
    //     // self.queue.release();
    //     // self.clear_build();

    //     if let Some(ref mut context) = self.context {
    //         context.release();
    //     }
    // }
}
