use std;
use std::ptr;
use std::mem;
use std::io::{ Read };
use std::fs::{ File };
use std::ffi;
// use std::iter;
// use std::collections::{ HashMap, HashSet };
use std::collections::{ HashSet };
// use std::fmt::{ Display };
use std::error::{ Error };
// use num::{ self, Integer, FromPrimitive };
use libc;

// use super::cl_h::{ self, cl_platform_id, cl_device_id, cl_context, cl_program, 
// 	cl_kernel, cl_command_queue, cl_float, cl_mem, cl_event, cl_char, cl_uchar, 
// 	cl_short, cl_ushort, cl_int, cl_uint, cl_long, CLStatus, 
// 	clSetKernelArg, clEnqueueNDRangeKernel };
use super::{ cl_h, cl_platform_id, cl_device_id, cl_context, cl_program, 
	cl_kernel, cl_command_queue, cl_mem, cl_int, cl_uint, Kernel, Envoy, OclNum, WorkSize, BuildOptions };

// use super::kernel::{ Kernel };
// use super::envoy::{ Envoy, OclNum };
// use super::work_size::{ WorkSize };
// use super::build_options::{ BuildOptions, /*BuildOption*/ };
//use super::cortical_dimensions::{ CorticalDimensions };

pub static MT: &'static str = "    ";

const DEFAULT_PLATFORM: usize = 0;
const DEFAULT_DEVICE: usize = 0;

pub struct OclContext {
	//platforms: Vec<cl_platform_id>,
	platform: cl_platform_id,
	devices: Vec<cl_device_id>,
	context: cl_context,
}

impl OclContext {
	pub fn new(platform_idx: Option<usize>) -> OclContext {
		let platforms = super::get_platform_ids();
		if platforms.len() == 0 { panic!("\nNo OpenCL platforms found!\n"); }

		let platform = match platform_idx {
			Some(pf_idx) => platforms[pf_idx],
			None => platforms[DEFAULT_PLATFORM],
		};
		
		let devices: Vec<cl_device_id> = super::get_device_ids(platform);
		if devices.len() == 0 { panic!("\nNo OpenCL devices found!\n"); }

		println!("{}OCL::NEW(): device list: {:?}", MT, devices);

		let context: cl_context = super::create_context(&devices);

		OclContext {
			//platforms: platforms,
			platform: platform,
			devices: devices,
			context:  context,
		}
	}

	pub fn release_components(&mut self) {
		
    	unsafe {
			cl_h::clReleaseContext(self.context);
		}
		print!("[platform]");
	}


	pub fn context(&self) -> cl_context {
		self.context
	}

	pub fn devices(&self) -> &Vec<cl_device_id> {
		&self.devices
	}

	pub fn platform(&self) -> cl_platform_id {
		self.platform
	}

	pub fn valid_device(&self, selected_idx: usize) -> cl_device_id {
		let valid_idx = selected_idx % self.devices.len();
		self.devices[valid_idx]
	}
}


pub struct OclProgQueue {
	//context: OclContext,
	context: cl_context,
	device: cl_device_id,
	queue: cl_command_queue,
	program: Option<cl_program>,
}

impl OclProgQueue {
	pub fn new(context: &OclContext, device_idx: Option<usize>) -> OclProgQueue {
		let device: cl_device_id = match device_idx {
			Some(dvc_idx) => context.valid_device(dvc_idx),
			None => context.devices()[DEFAULT_DEVICE],
		};

		let queue: cl_command_queue = super::new_command_queue(context.context(), device); 

		OclProgQueue {
			context: context.context(),
			device: device,
			queue: queue,
			program: None,
		}
	}

	pub fn build(&mut self, build_options: BuildOptions) /*-> Ocl*/ {
		if self.program.is_some() { panic!("\nOcl::build(): Pre-existing build detected. Use: \
			'{your_Ocl_instance} = {your_Ocl_instance}.clear_build()' first.") }		

		let kern_c_str = parse_kernel_files(&build_options);

		println!("{}OCL::BUILD(): DEVICE: {:#?}", MT, self.device);

		let prg = super::new_program(kern_c_str.as_ptr(), build_options.to_build_string(), self.context, self.device);

		super::program_build_info(prg, self.device);

		self.program = Some(prg);
	}

	pub fn new_kernel(&self, name: String, gws: WorkSize) -> Kernel {
		let program = match self.program {
			Some(prg) => prg,
			None => panic!("\nOcl::new_kernel(): Cannot add new kernel until OpenCL program is built. \
				Use: '{your_Ocl_instance}.build({your_BuildOptions_instance})'.\n"),
		};

		let mut err: cl_h::cl_int = 0;

		let kernel = unsafe {
			cl_h::clCreateKernel(
				program, 
				ffi::CString::new(name.as_bytes()).ok().unwrap().as_ptr(), 
				&mut err
			)
		};
		
		let err_pre = format!("Ocl::new_kernel({}):", &name);
		super::must_succ(&err_pre, err);

		Kernel::new(kernel, name, self.queue, gws)	
	}

	pub fn clone(&self) -> OclProgQueue {
		OclProgQueue {
			context:  self.context,
			device:  self.device,
			program:  self.program,
			queue: self.queue,
		}
	}

	// pub fn clone_with_device(&self) -> OclProgQueue {
	// 	//let devices: Vec<device> = super::get_device_ids(self.cur_platform);

	// 	OclProgQueue {
	// 		devices: self.devices.clone(),
	// 		device: self.device,
	// 		context:  self.context.clone(),
	// 		program:  self.program,
	// 		queue: self.queue,
	// 	}
	// }

	pub fn clear_build(mut self) {
		self.release_program();
		self.program = None;
	}


	pub fn new_write_buffer<T: OclNum>(&self, data: &[T]) -> cl_h::cl_mem {
		super::new_write_buffer(data, self.context)
	}

	pub fn new_read_buffer<T: OclNum>(&self, data: &[T]) -> cl_h::cl_mem {
		super::new_read_buffer(data, self.context)
	}

	pub fn enqueue_write_buffer<T: OclNum>(
					&self,
					src: &Envoy<T>,
	) {

		unsafe {
			let err = cl_h::clEnqueueWriteBuffer(
						self.queue,
						src.buf(),
						cl_h::CL_TRUE,
						0,
						(src.vec().len() * mem::size_of::<T>()) as libc::size_t,
						src.vec().as_ptr() as *const libc::c_void,
						0 as cl_h::cl_uint,
						ptr::null(),
						ptr::null_mut(),
			);
			super::must_succ("clEnqueueWriteBuffer()", err);
		}
	}


	pub fn enqueue_read_buffer<T: OclNum>(
					&self,
					data: &[T],
					buffer: cl_h::cl_mem, 
	) {
		super::enqueue_read_buffer(data, buffer, self.queue, 0);
	}

	pub fn enqueue_copy_buffer<T: OclNum>(
					&self,
					src: &Envoy<T>,		//	src_buffer: cl_mem,
					dst: &Envoy<T>,		//	dst_buffer: cl_mem,
					src_offset: usize,
					dst_offset: usize,
					len_copy_bytes: usize,
	) {
		unsafe {
			let err = cl_h::clEnqueueCopyBuffer(
				self.queue,
				src.buf(),				//	src_buffer,
				dst.buf(),				//	dst_buffer,
				src_offset as u64,
				dst_offset as u64,
				len_copy_bytes as u64,
				0,
				ptr::null(),
				ptr::null_mut(),
			);
			super::must_succ("clEnqueueCopyBuffer()", err);
		}
	}

	pub fn enqueue_kernel(
				&self,
				kernel: cl_h::cl_kernel, 
				gws: usize,
	) { 
		super::enqueue_kernel(kernel, self.queue, gws);
	}

	pub fn release_components(&self) {
		self.release_program();
		print!("[program]");
		unsafe {
			cl_h::clReleaseCommandQueue(self.queue);
		}
		print!("[queue]");
	}

	pub fn release_program(&self) {
		unsafe { 
			if self.program.is_some() { cl_h::clReleaseProgram(self.program.unwrap()); }

		}
	}

	pub fn get_max_work_group_size(&self) -> u32 {
		let max_work_group_size: u64 = 0;

		let err = unsafe { 
			cl_h::clGetDeviceInfo(
				self.device,
				cl_h::CL_DEVICE_MAX_WORK_GROUP_SIZE,
				mem::size_of::<u64>() as u64,
				mem::transmute(&max_work_group_size),
				ptr::null_mut(),
			) 
		}; 

		super::must_succ("clGetDeviceInfo", err);

		max_work_group_size as u32
	}

	pub fn queue(&self) -> cl_h::cl_command_queue {
		self.queue
	}

	pub fn context(&self) -> cl_h::cl_context {
		self.context
	}
}


fn parse_kernel_files(build_options: &BuildOptions) -> ffi::CString {
	let mut kern_str: Vec<u8> = Vec::with_capacity(10000);
	let mut kern_history: HashSet<String> = HashSet::with_capacity(20);

	let dd_string = build_options.cl_file_header();
	kern_str.push_all(&dd_string);
	// print!("OCL::PARSE_KERNEL_FILES(): KERNEL FILE DIRECTIVES HEADER: \n{}", 
	// 	String::from_utf8(dd_string).ok().unwrap());

	for f_n in build_options.kernel_file_names().iter().rev() {
		let file_name = format!("{}/{}/{}", env!("P"), "bismit/cl", f_n);

		{
			if kern_history.contains(&file_name) { continue; }
			let kern_file_path = std::path::Path::new(&file_name);

			let mut kern_file = match File::open(&kern_file_path) {
				Err(why) => panic!("\nCouldn't open '{}': {}", &file_name, Error::description(&why)),
				Ok(file) => file,
			};

			match kern_file.read_to_end(&mut kern_str) {
	    		Err(why) => panic!("\ncouldn't read '{}': {}", &file_name, Error::description(&why)),
			    Ok(bytes) => println!("{}OCL::BUILD(): parsing {}: {} bytes read.", MT, &file_name, bytes),
			}
		}

		kern_history.insert(file_name);
	}

	ffi::CString::new(kern_str).ok().expect("Ocl::new(): ocl::parse_kernel_files(): ffi::CString::new(): Error.")
}
