use std::convert::Into;
use super::{Context, ProgramBuilder, Program, Queue, ProQue, OclResult, OclError};

/// Builder for ProQue.
pub struct ProQueBuilder<'c> {
	context: Option<&'c Context>,
	device_idx: Option<usize>,
	program_builder: Option<ProgramBuilder>,
}

impl<'c> ProQueBuilder<'c> {
	pub fn new() -> ProQueBuilder<'c> {
		ProQueBuilder { 
			context: None,
			device_idx: None,
			program_builder: None,
		}
	}

	pub fn context(mut self, context: &'c Context) -> ProQueBuilder<'c> {
		self.context = Some(context);
		self
	}

	pub fn src<S: Into<String>>(mut self, src: S) -> ProQueBuilder<'c> {
		self.program_builder = match self.program_builder {
			Some(program_builder) => Some(program_builder.src(src)),
			None => Some(Program::builder().src(src)),
		};

		self
	}

	pub fn program_builder(mut self, program_builder: ProgramBuilder) -> ProQueBuilder<'c> {
		self.program_builder = Some(program_builder);
		self
	}

	pub fn device_idx(mut self, device_idx: usize) -> ProQueBuilder<'c> {
		self.device_idx = Some(device_idx);
		self
	}

	pub fn build(self) -> OclResult<ProQue> {
		match self.program_builder {
			Some(program_builder) => ProQueBuilder::_build(self.context, self.device_idx, program_builder),
			None => return OclError::err("ProQueBuilder::build: No program builder or kernel source defined. \
				OpenCL programs must have some source code to be compiled. Use '::src' to directly \
				add source code or '::program_builder' for more complex builds. Please see the \
				documentation for 'ProgramBuilder'."),
		}
	}

	pub fn build_with(self, program_builder: ProgramBuilder) -> OclResult<ProQue> {
		ProQueBuilder::_build(self.context, self.device_idx, program_builder)
	}

	fn _build(context: Option<&'c Context>, device_idx: Option<usize>,
				program_builder: ProgramBuilder) -> OclResult<ProQue> 
	{
		let mut context_opt: Option<Context> = None;

		let queue = match context {
			Some(ctx) => Queue::new(ctx, device_idx),
			None => {
				context_opt = Some(try!(Context::new(None, None)));
				Queue::new(&context_opt.as_ref().unwrap(), None)
			},
		};

		let program_opt = Some(try!(Program::from_parts(
			try!(program_builder.src_strings().map_err(|e| e.to_string())), 
			try!(program_builder.compiler_options().map_err(|e| e.to_string())), 
			queue.context_obj(), 
			&vec![queue.device_id()],
		)));

		Ok(ProQue::from_parts(context_opt, queue, program_opt))
	}
}
