// use std;
use std::ptr;
use std::mem;
use std::io::{ Read };
use std::fs::{ File };
use std::path::{ Path };
use std::ffi;
use std::collections::{ HashSet };
use std::error::{ Error };
// use libc;

use super::{ cl_h, cl_device_id, cl_context, cl_program, cl_command_queue, cl_mem, 
	cl_int, Context, Kernel, OclNum, WorkSize, BuildOptions, DEFAULT_DEVICE };


// Name is tenative
pub struct ProQueue {
	context: cl_context,
	device: cl_device_id,
	cmd_queue: cl_command_queue,
	program: Option<cl_program>,
}

impl ProQueue {
	///	Documentation coming.
	/// 	Doc Note: Will wrap device_idx around.
	///
	pub fn new(context: &Context, device_idx: Option<usize>) -> ProQueue {
		let device: cl_device_id = match device_idx {
			Some(dvc_idx) => context.valid_device(dvc_idx),
			None => context.devices()[DEFAULT_DEVICE],
		};

		let cmd_queue: cl_command_queue = super::create_command_queue(context.context(), device); 

		ProQueue {
			context: context.context(),
			device: device,
			cmd_queue: cmd_queue,
			program: None,
		}
	}

	pub fn build(&mut self, build_options: BuildOptions) -> Result<(), String> {
		if self.program.is_some() { panic!("\nOcl::build(): Pre-existing build detected. Use: \
			'{your_Ocl_instance} = {your_Ocl_instance}.clear_build()' first.") }		

		let kern_c_str = try!(parse_kernel_files(&build_options));

		// [FIXME] TEMPORARY UNWRAP:
		let prgm = super::create_program(kern_c_str.as_ptr(), build_options.to_build_string(), 
			self.context, self.device).unwrap();

		// [FIXME] TEMPORARY UNWRAP:
		super::program_build_info(prgm, self.device).unwrap();

		self.program = Some(prgm);

		Ok(())
	}

	// [FIXME] TODO: Return result instead of panic.
	pub fn create_kernel(&self, name: String, gws: WorkSize) -> Kernel {
		let program = match self.program {
			Some(prg) => prg,
			None => panic!("\nOcl::create_kernel(): Cannot add new kernel until OpenCL program is built. \
				Use: '{your_Ocl_instance}.build({your_BuildOptions_instance})'.\n"),
		};

		let mut err: cl_h::cl_int = 0;

		let kernel = unsafe {
			cl_h::clCreateKernel(
				program, 
				ffi::CString::new(name.as_bytes()).unwrap().as_ptr(), 
				&mut err
			)
		};
		
		let err_pre = format!("Ocl::create_kernel({}):", &name);
		super::must_succ(&err_pre, err);

		Kernel::new(kernel, name, self.cmd_queue, gws)	
	}

	pub fn clone(&self) -> ProQueue {
		ProQueue {
			context:  self.context,
			device:  self.device,
			program:  self.program,
			cmd_queue: self.cmd_queue,
		}
	}

	pub fn clear_build(mut self) {
		self.release_program();
		self.program = None;
	}

	pub fn create_write_buffer<T: OclNum>(&self, data: &[T]) -> cl_h::cl_mem {
		super::create_write_buffer(data, self.context)
	}

	pub fn create_read_buffer<T: OclNum>(&self, data: &[T]) -> cl_h::cl_mem {
		super::create_read_buffer(data, self.context)
	}

	pub fn release_components(&self) {
		self.release_program();
		unsafe {
			cl_h::clReleaseCommandQueue(self.cmd_queue);
		}
	}

	pub fn release_program(&self) {
		unsafe { 
			if self.program.is_some() { cl_h::clReleaseProgram(self.program.unwrap()); }

		}
	}

	// [FIXME] TODO: Remove transmute()
	pub fn get_max_work_group_size(&self) -> u32 {
		let max_work_group_size: usize = 0;

		let err = unsafe { 
			cl_h::clGetDeviceInfo(
				self.device,
				cl_h::CL_DEVICE_MAX_WORK_GROUP_SIZE,
				mem::size_of::<usize>() as usize,
				mem::transmute(&max_work_group_size),
				ptr::null_mut(),
			) 
		}; 

		super::must_succ("clGetDeviceInfo", err);

		max_work_group_size as u32
	}

	pub fn cmd_queue(&self) -> cl_h::cl_command_queue {
		self.cmd_queue
	}

	pub fn context(&self) -> cl_h::cl_context {
		self.context
	}
}


fn parse_kernel_files(build_options: &BuildOptions) -> Result<ffi::CString, String> {
	let mut kern_str: Vec<u8> = Vec::with_capacity(10000);
	let mut kern_history: HashSet<&String> = HashSet::with_capacity(20);

	let dd_string = build_options.cl_file_header();
	kern_str.push_all(&dd_string);
	// print!("OCL::PARSE_KERNEL_FILES(): KERNEL FILE DIRECTIVES HEADER: \n{}", 
	// 	String::from_utf8(dd_string).ok().unwrap());

	for kfn in build_options.kernel_file_names().iter().rev() {
		// let file_name = format!("{}/{}/{}", env!("P"), "bismit/cl", f_n);
		// let valid_kfp = try!(valid_kernel_file_path(&kfn));

		// {
			if kern_history.contains(kfn) { continue; }
			let valid_kfp = Path::new(kfn);

			let mut kern_file = match File::open(&valid_kfp) {
				Err(why) => return Err(format!("Couldn't open '{}': {}", 
					kfn, Error::description(&why))),
				Ok(file) => file,
			};

			match kern_file.read_to_end(&mut kern_str) {
	    		Err(why) => return Err(format!("Couldn't read '{}': {}", 
	    			kfn, Error::description(&why))),
			    Ok(_) => (), //println!("{}OCL::BUILD(): parsing {}: {} bytes read.", MT, &file_name, bytes),
			}
		// }

		kern_history.insert(&kfn);
	}

	Ok(ffi::CString::new(kern_str).expect("Ocl::new(): ocl::parse_kernel_files(): ffi::CString::new(): Error."))
}


// Search the valid paths to find a kernel file.
// fn valid_kernel_file_path(kernel_file_name: &String) -> Result<Path, String> {
// 	Ok(std::path::Path::new(kernel_file_name))
// }
