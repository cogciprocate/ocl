//! A convenience wrapper chimera of `Program` and `Queue`.
use wrapper;
use super::{Context, Kernel, WorkSize, ProgramBuilder, ProQueBuilder, Program, Queue, 
	OclResult, OclError};

/// A convenience wrapper chimera of the `Program`, `Queue`, and optionally,
/// `Context` types .
///
/// Handy when creating only a single context, program, and queue or when
/// using a unique program build on each device.
///
/// All `ProQue` functionality is also provided separately by the `Context`, `Queue`, 
/// and `Program` types.
///
/// # Destruction
/// `::release` must be manually called by consumer.
///
// #[derive(Clone)]
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
	// 	ProQueBuilder::new()
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

	/// Builds contained program with `program_builder`.
	///
	/// # Panics
	/// This `ProQue` must not already contain a program.
	///
	/// `program_builder` must not have any device indexes configured (via its
	/// `::device_idxs` method). `ProQue` will only build programs for the device
	/// previously configured or the default device if none had been specified.
	pub fn build_program(&mut self, program_builder: ProgramBuilder) -> OclResult<()> {
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
			try!(program_builder.src_strings().map_err(|e| e.to_string())), 
			try!(program_builder.compiler_options().map_err(|e| e.to_string())), 
			self.queue.context_obj(), 
			&vec![self.queue.device_id()],
		)));

		Ok(())
	}

	/// Clears the current program build.
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
	pub fn create_kernel(&self, name: &str, gws: WorkSize) -> Kernel {
		let program = match self.program {
			Some(ref prg) => prg,
			None => panic!("\nOcl::create_kernel(): Cannot add new kernel until OpenCL program is built. \
				Use: '{your_Ocl_instance}.build_program({your_ProgramBuilder_instance})'.\n"),
		};

		Kernel::new(name.to_string(), &program, &self.queue, gws)	
	}

	/// Returns the maximum workgroup size supported by the device on which the
	/// contained queue exists.
	pub fn get_max_work_group_size(&self) -> usize {
		wrapper::get_max_work_group_size(self.queue.device_id())
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
	// Note: Do not move this to a Drop impl in case this ProQue has been cloned.
	pub fn release(&mut self) {		
		self.queue.release();
		self.clear_build();

		if let Some(ref mut context) = self.context {
			context.release();
		}
	}
}
