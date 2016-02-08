//! A convenience wrapper chimera of `Program` and `Queue`.
use wrapper;
use super::{Context, Kernel, WorkSize, BuildConfig, Program, Queue};


/// A convenience wrapper chimera of `Program` and `Queue`.
///
/// Useful when using a unique program build on each device.
#[derive(Clone)]
pub struct ProQue {
	queue: Queue,
	program_opt: Option<Program>,
}

impl ProQue {
	/// Creates a new queue on the device with `device_idx` (see `Queue` documentation)
	/// and returns a new Program/Queue hybrid.
	///
	/// `::build()` must be called before this ProQue can be used.
	//
	// TODO: Doc note: mention that:
	//    - device_idx wraps around
	//    - one device only
	pub fn new(context: &Context, device_idx: Option<usize>) -> ProQue {
		let queue = Queue::new(context, device_idx);

		ProQue {
			queue: queue,
			program_opt: None,
		}
	}

	/// Builds contained program with `build_config`.
	///
	/// # Panics
	/// This ProQue must contain a program.
	pub fn build(&mut self, build_config: BuildConfig) -> Result<(), String> {
		if self.program_opt.is_some() { 
			return Err(format!("Ocl::build(): Pre-existing build detected. Use: \
				'{{your_Ocl_instance}} = {{your_Ocl_instance}}.clear_build()' first."))
		}		

		self.program_opt = Some(try!(Program::from_parts(
			try!(build_config.kernel_strings().map_err(|e| e.to_string())), 
			try!(build_config.compiler_options().map_err(|e| e.to_string())), 
			self.queue.context_obj(), 
			&vec![self.queue.device_id()],
		)));

		Ok(())
	}	

	/// Clears the current program build.
	pub fn clear_build(&mut self) {
		match self.program_opt {
			Some(ref mut program) => { 
				program.release(); 				
			},

			None => (),
		}
		self.program_opt = None;
	}

	/// Returns a new Kernel with name: `name` and global work size: `gws`.
	// [FIXME] TODO: Return result instead of panic.
	pub fn create_kernel(&self, name: &str, gws: WorkSize) -> Kernel {
		let program = match self.program_opt {
			Some(ref prg) => prg,
			None => panic!("\nOcl::create_kernel(): Cannot add new kernel until OpenCL program is built. \
				Use: '{your_Ocl_instance}.build({your_BuildConfig_instance})'.\n"),
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
		&self.program_opt
	}

	/// Release all components.
	// Note: Do not move this to a Drop impl in case this ProQue has been cloned.
	pub fn release(&mut self) {		
		self.queue.release();
		self.clear_build();
	}
}
