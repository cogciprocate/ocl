
use super::{ Context, Kernel, WorkSize, BuildOptions, Program, Queue };


/// A convenience wrapper chimera of `Program` and `Queue`.
#[derive(Clone)]
pub struct ProQue {
	queue: Queue,
	program_opt: Option<Program>,
}

impl ProQue {
	///	Documentation coming.
	/// 	Doc Note: Will wrap device_idx around.
	pub fn new(context: &Context, device_idx: Option<usize>) -> ProQue {
		let queue = Queue::new(context, device_idx);

		ProQue {
			queue: queue,
			program_opt: None,
		}
	}

	pub fn build(&mut self, build_options: BuildOptions) -> Result<(), String> {
		if self.program_opt.is_some() { 
			return Err(format!("Ocl::build(): Pre-existing build detected. Use: \
				'{{your_Ocl_instance}} = {{your_Ocl_instance}}.clear_build()' first."))
		}		

		// [FIXME]: Unwrap
		self.program_opt = Some(try!(Program::from_parts(
			try!(build_options.kernel_strings().map_err(|e| e.to_string())), 
			try!(build_options.compiler_options().map_err(|e| e.to_string())), 
			self.queue.context_obj(), 
			self.queue.device_id()
		)));

		Ok(())
	}

	// [FIXME] TODO: Return result instead of panic.
	pub fn create_kernel(&self, name: String, gws: WorkSize) -> Kernel {
		let program = match self.program_opt {
			Some(ref prg) => prg,
			None => panic!("\nOcl::create_kernel(): Cannot add new kernel until OpenCL program is built. \
				Use: '{your_Ocl_instance}.build({your_BuildOptions_instance})'.\n"),
		};

		Kernel::new(name, &program, &self.queue, gws)	
	}

	pub fn clear_build(&mut self) {
		match self.program_opt {
			Some(ref mut program) => { 
				program.release(); 				
			},

			None => (),
		}
		self.program_opt = None;
	}	

	pub fn get_max_work_group_size(&self) -> usize {
		super::get_max_work_group_size(self.queue.device_id())
	}

	pub fn queue(&self) -> &Queue {
		&self.queue
	}

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
