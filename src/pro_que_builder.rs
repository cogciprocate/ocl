//! The builder for `ProQue`.

use std::convert::Into;
use super::{Context, ProgramBuilder, Program, Queue, ProQue, OclResult, OclError};

/// A builder for `ProQue`.
pub struct ProQueBuilder<'c> {
	context: Option<&'c Context>,
	device_idx: Option<usize>,
	program_builder: Option<ProgramBuilder>,
}

impl<'c> ProQueBuilder<'c> {
	/// Returns a new `ProQueBuilder` with an empty / default configuration.
	///
	/// The minimum amount of configuration possible before calling `::build` is to 
	/// simply assign some source using `::src`.
	pub fn new() -> ProQueBuilder<'c> {
		ProQueBuilder { 
			context: None,
			device_idx: None,
			program_builder: None,
		}
	}

	/// Sets the context and returns the `ProQueBuilder`.
	pub fn context(mut self, context: &'c Context) -> ProQueBuilder<'c> {
		self.context = Some(context);
		self
	}

	/// Adds some source code to be compiled and returns the `ProQueBuilder`.
	///
	/// Creates a `ProgramBuilder` if one has not already been added. Attempts
	/// to call `::program_builder` after calling this method will cause a panic.
	pub fn src<S: Into<String>>(mut self, src: S) -> ProQueBuilder<'c> {
		self.program_builder = match self.program_builder {
			Some(program_builder) => Some(program_builder.src(src)),
			None => Some(Program::builder().src(src)),
		};

		self
	}

	/// Sets a device index to be used and returns the `ProQueBuilder`.
	pub fn device_idx(mut self, device_idx: usize) -> ProQueBuilder<'c> {
		self.device_idx = Some(device_idx);
		self
	}

	/// Adds a pre-configured `ProgramBuilder` and returns the `ProQueBuilder`.
	///
	/// # Panics
	/// This `ProQueBuilder` may not already contain a `ProgramBuilder`.
	///
	/// `program_builder` must not have any device indexes configured (via its
	/// `::device_idxs` method). `ProQueBuilder` will only build programs for
	/// the device specified by `::device_idx` or the default device if none has
	/// been specified.
	pub fn program_builder(mut self, program_builder: ProgramBuilder) -> ProQueBuilder<'c> {
		assert!(self.program_builder.is_none(), "ProQueBuilder::program_builder(): Cannot set the \
			'ProgramBuilder' using this method after one has already been set or after '::src' has \
			been called.");

		assert!(program_builder.get_device_idxs().len() > 0, "ProQueBuilder::program_builder(): The \
			'ProgramBuilder' passed may not have any device indexes set as they will be ignored. \
			See 'ProQueBuilder' documentation for more information.");

		self.program_builder = Some(program_builder);
		self
	}

	/// Returns a new `ProQue`.
	///
	/// # Errors
	/// A `ProgramBuilder` or some source code must have been specified before building
	/// using `::program_builder` or `::src`.
	pub fn build(self) -> OclResult<ProQue> {
		match self.program_builder {
			Some(program_builder) => ProQueBuilder::_build(self.context, self.device_idx, program_builder),
			None => return OclError::err("ProQueBuilder::build(): No program builder or kernel source defined. \
				OpenCL programs must have some source code to be compiled. Use '::src' to directly \
				add source code or '::program_builder' for more complex builds. Please see the \
				'ProgramBuilder' documentation for more information."),
		}
	}

	// pub fn build_with(self, program_builder: ProgramBuilder) -> OclResult<ProQue> {
	// 	if self.program_builder.is_some() { 
	// 		return OclError::err("ProQueBuilder::build_with(): This method cannot be used if a \
	// 			'ProgramBuilder' has already been specified using '::src' or '::program_builder'. \
	// 			Use '::build' instead.");
	// 	}

	// 	ProQueBuilder::_build(self.context, self.device_idx, program_builder)
	// }

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
