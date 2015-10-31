// use std;
use std::ptr;
use std::mem;
// use std::io::{ Read };
// use std::fs::{ File };
// use std::ffi;
// use std::iter;
use std::collections::{ HashMap };
use num::{ /*self,*/ Integer, /*FromPrimitive,*/ Zero };
use libc;
//use std::default::{ Default };

// use cmn;
use super::{ WorkSize, Envoy, OclNum };


pub struct Kernel {
	kernel: super::cl_kernel,
	name: String,
	arg_index: u32,
	named_args: HashMap<&'static str, u32>,
	arg_count: u32,
	command_queue: super::cl_command_queue,
	gwo: WorkSize,
	gws: WorkSize,
	lws: WorkSize,
}

impl Kernel {
	pub fn new(kernel: super::cl_kernel, name: String, command_queue: super::cl_command_queue, gws: WorkSize
	) -> Kernel {

		//println!("                  KERNEL::NEW(): adding kernel: {}, gws: {:?}", name, gws);
		Kernel {
			kernel: kernel,
			name: name,
			arg_index: 0,
			named_args: HashMap::with_capacity(5),
			arg_count: 0u32,
			command_queue: command_queue,
			gwo: WorkSize::Unspecified,
			gws: gws,
			lws: WorkSize::Unspecified,
		}
	}

	pub fn gwo(mut self, gwo: WorkSize) -> Kernel {
		if gwo.dims() == self.gws.dims() {
			self.gwo = gwo
		} else {
			panic!("ocl::Kernel::gwo(): Work size mismatch.");
		}
		self
	}

	pub fn lws(mut self, lws: WorkSize) -> Kernel {
		if lws.dims() == self.gws.dims() {
			self.lws = lws;
		} else {
			panic!("ocl::Kernel::lws(): Work size mismatch.");
		}
		self
	}

	pub fn arg_env<T: OclNum>(mut self, envoy: &Envoy<T>) -> Kernel {
		self.new_arg_envoy(Some(envoy));
		self
	}

	pub fn arg_scl<T: Integer>(mut self, scalar: T) -> Kernel {
		self.new_arg_scalar(Some(scalar));
		self
	}

	pub fn arg_scl_named<T: Integer>(mut self, name: &'static str, scalar_opt: Option<T>) -> Kernel {
		let arg_idx = self.new_arg_scalar(scalar_opt);
		self.named_args.insert(name, arg_idx);
		self
	}

	pub fn arg_env_named<T: OclNum>(mut self, name: &'static str,  envoy_opt: Option<&Envoy<T>>) -> Kernel {
		let arg_idx = self.new_arg_envoy(envoy_opt);
		self.named_args.insert(name, arg_idx);

		self
	}

	pub fn arg_loc<T: Integer>(mut self, /*type_sample: T,*/ length: usize) -> Kernel {
		self.new_arg_local::<T>(/*type_sample,*/ length);
		self
	}


	pub fn new_arg_envoy<T: OclNum>(&mut self, envoy_opt: Option<&Envoy<T>>) -> u32 {
		let buf = match envoy_opt {
			Some(envoy) => envoy.buf(),
			None => ptr::null_mut()
		};

		self.new_kernel_arg(
			mem::size_of::<super::cl_mem>() as libc::size_t, 
			(&buf as *const super::cl_mem) as *const libc::c_void,
		)
	}

	pub fn new_arg_scalar<T: Integer>(&mut self, scalar_opt: Option<T>) -> u32 {
		let val = match scalar_opt {
			Some(scalar) => scalar,
			None => Zero::zero(),
		};

		unsafe {
			//let scal = &scalar;

			self.new_kernel_arg(
				mem::size_of::<T>() as libc::size_t,
				//(scal as *const super::cl_mem) as *const libc::c_void,
				mem::transmute(&val),
				//(scalar as *const super::cl_mem) as *const libc::c_void,
			)
		}
	}

	pub fn new_arg_local<T: Integer>(&mut self, /*type_sample: T,*/ length: usize) -> u32 {

		self.new_kernel_arg(
			(mem::size_of::<T>() * length) as libc::size_t,
			ptr::null(),
		)
	}


	fn new_kernel_arg(&mut self, arg_size: libc::size_t, arg_value: *const libc::c_void) -> u32 {
		let a_i = self.arg_index;

		/*let err = unsafe {
			super::clSetKernelArg(
						self.kernel, 
						self.arg_index, 
						arg_size, 
						arg_value,
			)
		};
		let err_pre = format!("\nocl::Kernel::new_kernel_arg()[{}]: ", self.name);
		super::must_succ(&err_pre, err);
		*/
		//println!("Adding Kernel Argument: {}", self.arg_index);

		self.set_kernel_arg(a_i, arg_size, arg_value);

		self.arg_index += 1;
		a_i
	}

	// // <<<<< CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE >>>>>
	pub fn set_arg_scl_named<T: Integer>(&mut self, name: &'static str, scalar: T) {
		//	TODO: ADD A CHECK FOR A VALID NAME (KEY)
		let arg_idx = self.named_args[name]; 

		// let scalar_ptr: *const libc::c_void = &scalar as *const _ as *const libc::c_void;

		self.set_kernel_arg(
			arg_idx,
			mem::size_of::<T>() as libc::size_t, 
			// mem::transmute(val),
			&scalar as *const _ as *const libc::c_void,
		)
	}

	// <<<<< CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE >>>>>
	// pub fn set_arg_scl_named(&mut self, name: &'static str, scalar: i32) {
	// 	//	TODO: ADD A CHECK FOR A VALID NAME (KEY)
	// 	let arg_idx = self.named_args[name]; 

	// 	// let scalar_ptr: *const libc::c_void = &scalar as *const _ as *const libc::c_void;

	// 	self.set_kernel_arg(
	// 		arg_idx,
	// 		mem::size_of::<i32>() as libc::size_t, 
	// 		// mem::transmute(val),
	// 		&scalar as *const _ as *const libc::c_void,
	// 	)
	// }

	// <<<<< CHECK THAT NAME EXISTS AND GIVE A BETTER ERROR MESSAGE >>>>>
	pub fn set_arg_env_named<T: OclNum>(&mut self, name: &'static str, envoy: &Envoy<T>) {
			//	TODO: ADD A CHECK FOR A VALID NAME (KEY)
		
		// println!("\nset_arg_env_named(): name: {}, named_args: {:?}", name, self.named_args);
		let arg_idx = self.named_args[name];
		let buf = envoy.buf();

		self.set_kernel_arg(
			arg_idx,
			mem::size_of::<super::cl_mem>() as libc::size_t, 
			(&buf as *const super::cl_mem) as *const libc::c_void,
		)

		//self.set_kernel_arg(arg_idx, buf)
	}

	fn set_kernel_arg(&mut self, arg_index: super::cl_uint, arg_size: libc::size_t, arg_value: *const libc::c_void) {
		unsafe {
			let err = super::clSetKernelArg(
						self.kernel, 
						arg_index,
						arg_size, 
						arg_value,
						/*mem::size_of::<T>() as libc::size_t, 
						mem::transmute(&val),*/
			);

			let err_pre = format!("ocl::Kernel::set_kernel_arg('{}'):", &self.name);
			super::must_succ(&err_pre, err);
		}
	}

	pub fn enqueue(&self) {
	//pub fn enqueue(&self) -> super::cl_event {

			// TODO: VERIFY THE DIMENSIONS OF ALL THE WORKSIZES

		let c_gws = self.gws.complete_worksize();
		let gws = (&c_gws as *const (usize, usize, usize)) as *const libc::size_t;

		let c_lws = self.lws.complete_worksize();
		let lws = (&c_lws as *const (usize, usize, usize)) as *const libc::size_t;

		//let mut event: super::cl_event = ptr::null_mut();

		unsafe {
			let err = super::clEnqueueNDRangeKernel(
						self.command_queue,
						self.kernel,
						self.gws.dims(),				//	dims,
						self.gwo.as_ptr(),
						gws,
						lws,
						0,
						ptr::null(),
						ptr::null_mut(),
						//&mut event as *mut super::cl_event, // LEAKS!
			);

			let err_pre = format!("ocl::Kernel::enqueue()[{}]:", &self.name);
			super::must_succ(&err_pre, err);
		}
		//event
	}

	pub fn arg_count(&self) -> u32 {
		self.arg_count
	}	
}



	/*pub fn enqueue_wait(&self, event_wait_list: Vec<super::cl_event>) -> super::cl_event {

			// TODO: VERIFY THE DIMENSIONS OF ALL THE WORKSIZES

		let c_gws = self.gws.complete_worksize();
		let gws = (&c_gws as *const (usize, usize, usize)) as *const libc::size_t;

		let c_lws = self.lws.complete_worksize();
		let lws = (&c_lws as *const (usize, usize, usize)) as *const libc::size_t;

		let mut event: super::cl_event = ptr::null_mut();

		unsafe {
			let err = super::clEnqueueNDRangeKernel(
						self.command_queue,
						self.kernel,
						self.gws.dims(),				//	dims,
						self.gwo.as_ptr(),
						gws,
						lws,
						event_wait_list.len() as super::cl_uint,
						//std::num::cast(event_wait_list.len()).expect("ocl::Kernel::enqueue_wait()"),
						event_wait_list.as_ptr(),
						&mut event as *mut super::cl_event,		// LEAKS!
			);

			let err_pre = format!("ocl::Kernel::enqueue_wait()[{}]: ", &self.name);
			super::must_succ(&err_pre, err);
		}
		event
	}*/
