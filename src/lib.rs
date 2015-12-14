#![feature(vec_push_all, zero_one)]
use std::ptr;
use std::mem;
use std::io::{ Read };
use std::ffi::{ CString };
use std::iter;
use std::fmt::{ Display, Debug, /*UpperHex*/ };
use std::num::{ Zero };
use num::{ NumCast, FromPrimitive, ToPrimitive };
use rand::distributions::range::{ SampleRange };

use self::cl_h::{ cl_platform_id, cl_device_id, cl_device_type, cl_context, cl_program, 
	cl_command_queue, cl_mem, cl_event };

pub use self::context::{ Context };
pub use self::program::{ Program };
pub use self::queue::{ Queue };
pub use self::cl_h::{ cl_float, cl_char, cl_uchar, cl_short, cl_ushort, cl_int, cl_uint, cl_long, 
	cl_bitfield, CLStatus, CL_DEVICE_TYPE_DEFAULT, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, 
	CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_ALL };
pub use self::kernel::{ Kernel };
pub use self::envoy::{ Envoy };
pub use self::pro_que::{ ProQue };
pub use self::simple_dims::{ SimpleDims };
pub use self::work_size::{ WorkSize };
pub use self::build_config::{ BuildConfig, BuildOpt };
pub use self::errors::{ DimError };
pub use self::event_list::{ EventList };
pub use self::formatting as fmt;

// [FIXME]: TODO: Create an additional crate build configuration for tests
// #[cfg(test)]
pub use self::envoy::tests::{ EnvoyTest };

#[macro_use] 
extern crate enum_primitive;
extern crate libc;
extern crate num;
extern crate rand;

mod context;
mod program;
mod queue;
pub mod cl_h;
pub mod envoy;
mod pro_que;
mod simple_dims;
mod kernel;
mod work_size;
mod build_config;
mod errors;
pub mod formatting;
mod event_list;
#[cfg(test)]
mod tests;



//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// pub static CL_DEVICE_TYPE_DEFAULT:                       cl_device_type = 1 << 0;
// 		CL_DEVICE_TYPE_DEFAULT:	The default OpenCL device in the system.
// pub static CL_DEVICE_TYPE_CPU:                           cl_device_type = 1 << 1;
// 		CL_DEVICE_TYPE_CPU:	An OpenCL device that is the host processor. The host processor runs the OpenCL implementations and is a single or multi-core CPU.
// pub static CL_DEVICE_TYPE_GPU:                           cl_device_type = 1 << 2;
// 		CL_DEVICE_TYPE_GPU:	An OpenCL device that is a GPU. By this we mean that the device can also be used to accelerate a 3D API such as OpenGL or DirectX.
// pub static CL_DEVICE_TYPE_ACCELERATOR:                   cl_device_type = 1 << 3;
// 		CL_DEVICE_TYPE_ACCELERATOR:	Dedicated OpenCL accelerators (for example the IBM CELL Blade). These devices communicate with the host processor using a peripheral interconnect such as PCIe.
// pub static CL_DEVICE_TYPE_ALL:                           cl_device_type = 0xFFFFFFFF;
// 		CL_DEVICE_TYPE_ALL
const DEFAULT_DEVICE_TYPE: cl_device_type = 1 << 0; // CL_DEVICE_TYPE_DEFAULT

const DEVICES_MAX: u32 = 16;
const DEFAULT_PLATFORM: usize = 0;
const DEFAULT_DEVICE: usize = 0;

//=============================================================================
//================================= TRAITS ====================================
//=============================================================================

pub trait OclNum: Copy + Clone + PartialOrd  + NumCast + Default + Zero + Display + Debug
	+ FromPrimitive + ToPrimitive + SampleRange {}

impl<T> OclNum for T where T: Copy + Clone + PartialOrd + NumCast + Default + Zero + Display + Debug
	+ FromPrimitive + ToPrimitive + SampleRange {}

pub trait EnvoyDims {
	fn padded_envoy_len(&self, usize) -> usize;
}

// + From<u32> + From<i32> + From<usize> + From<i8> + From<u8> + Into<usize> + Into<i8>

//=============================================================================
//=========================== UTILITY FUNCTIONS ===============================
//=============================================================================

/// Pads `len` to make it evenly divisible by `incr`.
pub fn padded_len(len: usize, incr: usize) -> usize {
	let len_mod = len % incr;

	if len_mod == 0 {
		len
	} else {
		let pad = incr - len_mod;
		let padded_len = len + pad;
		debug_assert_eq!(padded_len % incr, 0);
		padded_len
	}
}

//=============================================================================
//============================== OCL FUNCTIONS ================================
//=============================================================================

// Create Platform and get ID
fn get_platform_ids() -> Vec<cl_h::cl_platform_id> {
	let mut num_platforms = 0 as cl_h::cl_uint;
	
	let mut err: cl_h::cl_int = unsafe { cl_h::clGetPlatformIDs(0, ptr::null_mut(), &mut num_platforms) };
	must_succeed("clGetPlatformIDs()", err);

	let mut platforms: Vec<cl_h::cl_platform_id> = Vec::with_capacity(num_platforms as usize);
	for _ in 0..num_platforms as usize { platforms.push(0 as cl_platform_id); }

	unsafe {
		err = cl_h::clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), ptr::null_mut()); 
		must_succeed("clGetPlatformIDs()", err);
	}
	
	platforms
}

// GET_DEVICE_IDS():
/// # Panics
/// 	//-must_succeed (needs addressing)
fn get_device_ids(
			platform: cl_h::cl_platform_id, 
			device_types_opt: Option<cl_device_type>,
		) -> Vec<cl_h::cl_device_id> 
{
	let device_type = device_types_opt.unwrap_or(DEFAULT_DEVICE_TYPE);
	
	let mut devices_available: cl_h::cl_uint = 0;
	let mut devices_array: [cl_h::cl_device_id; DEVICES_MAX as usize] = [0 as cl_h::cl_device_id; DEVICES_MAX as usize];

	let err = unsafe { cl_h::clGetDeviceIDs(
		platform, 
		device_type, 
		DEVICES_MAX, 
		devices_array.as_mut_ptr(), 
		&mut devices_available
	) };

	must_succeed("clGetDeviceIDs()", err);

	let mut device_ids: Vec<cl_h::cl_device_id> = Vec::with_capacity(devices_available as usize);

	for i in 0..devices_available as usize {
		device_ids.push(devices_array[i]);
	}

	device_ids
}


fn create_context(device_ids: &Vec<cl_h::cl_device_id>) -> cl_h::cl_context {
	let mut err: cl_h::cl_int = 0;

	unsafe {
		let context: cl_h::cl_context = cl_h::clCreateContext(
						ptr::null(), 
						device_ids.len() as cl_h::cl_uint, 
						device_ids.as_ptr(),
						mem::transmute(ptr::null::<fn()>()), 
						ptr::null_mut(), 
						&mut err);
		must_succeed("clCreateContext()", err);
		context
	}
}


fn new_program(
			kern_strings: Vec<CString>,
			cmplr_opts: CString,
			context: cl_h::cl_context, 
			device_ids: &Vec<cl_h::cl_device_id>
		) -> Result<cl_h::cl_program, String>
{
	// Lengths (not including \0 terminator) of each string:
	let ks_lens: Vec<usize> = kern_strings.iter().map(|cs| cs.as_bytes().len()).collect();	
	// Pointers to each string:
	let kern_strs: Vec<*const i8> = kern_strings.iter().map(|cs| cs.as_ptr()).collect();

	let mut err = 0i32;

	unsafe {
		let program: cl_h::cl_program = cl_h::clCreateProgramWithSource(
					context, 
					kern_strs.len() as u32,
					kern_strs.as_ptr() as *const *const i8,
					ks_lens.as_ptr() as *const usize,
					&mut err,
		);
		must_succeed("clCreateProgramWithSource()", err);

		err = cl_h::clBuildProgram(
					program,
					device_ids.len() as u32,
					device_ids.as_ptr(), 
					cmplr_opts.as_ptr() as *const i8,
					mem::transmute(ptr::null::<fn()>()), 
					ptr::null_mut(),
		);		

		if err < 0 {
			program_build_info(program, device_ids).map(|_| program)
		} else {
			must_succeed("clBuildProgram()", err);
			// Unreachable:
			Ok(program) 
		}
	}
}


fn create_command_queue(
			context: cl_h::cl_context, 
			device: cl_h::cl_device_id,
		) -> cl_h::cl_command_queue 
{
	let mut err: cl_h::cl_int = 0;

	unsafe {
		let cq: cl_h::cl_command_queue = cl_h::clCreateCommandQueue(
					context, 
					device, 
					cl_h::CL_QUEUE_PROFILING_ENABLE, 
					&mut err
		);

		must_succeed("clCreateCommandQueue()", err);
		cq
	}
}


fn create_buffer<T>(data: &[T], context: cl_h::cl_context, flags: cl_h::cl_bitfield
		) -> cl_h::cl_mem 
{
	let mut err: cl_h::cl_int = 0;

	unsafe {
		let buf = cl_h::clCreateBuffer(
					context, 
					flags | cl_h::CL_MEM_COPY_HOST_PTR | cl_h::CL_MEM_ALLOC_HOST_PTR, 
					(data.len() * mem::size_of::<T>()) as usize,
					data.as_ptr() as *mut libc::c_void, 
					//ptr::null_mut(),
					&mut err,
		);
		must_succeed("create_buffer", err);
		buf
	}
}

fn enqueue_write_buffer<T>(
			command_queue: cl_h::cl_command_queue,
			buffer: cl_h::cl_mem, 
			wait: bool,
			data: &[T],
			offset: usize,
			wait_list: Option<&EventList>, 
			dest_list: Option<&mut EventList>,
		)
{
	let (blocking_write, wait_list_len, wait_list_ptr, new_event_ptr) 
		= resolve_queue_opts(wait, wait_list, dest_list);

	unsafe {
		let err = cl_h::clEnqueueWriteBuffer(
					command_queue,
					buffer,
					blocking_write,
					offset,
					(data.len() * mem::size_of::<T>()) as libc::size_t,
					data.as_ptr() as *const libc::c_void,
					wait_list_len,
					wait_list_ptr,
					new_event_ptr,
		);

		must_succeed("clEnqueueWriteBuffer()", err);
	}
}


fn enqueue_read_buffer<T>(
			command_queue: cl_h::cl_command_queue,
			buffer: cl_h::cl_mem, 
			wait: bool,
			data: &[T],
			offset: usize,
			wait_list: Option<&EventList>, 
			dest_list: Option<&mut EventList>,
		)
{
	let (blocking_read, wait_list_len, wait_list_ptr, new_event_ptr) 
		= resolve_queue_opts(wait, wait_list, dest_list);

	unsafe {
		let err = cl_h::clEnqueueReadBuffer(
					command_queue, 
					buffer, 
					blocking_read, 
					offset, 
					(data.len() * mem::size_of::<T>()) as libc::size_t, 
					data.as_ptr() as *mut libc::c_void, 
					wait_list_len,
					wait_list_ptr,
					new_event_ptr,
		);

		must_succeed("clEnqueueReadBuffer()", err);
	}
}

fn resolve_queue_opts(block: bool, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>)
		-> (cl_h::cl_bool, cl_uint, *const cl_event, *mut cl_event)
{
	let blocking_operation = if block { cl_h::CL_TRUE } else { cl_h::CL_FALSE };

	let (wait_list_len, wait_list_ptr): (u32, *const cl_h::cl_event) = match wait_list {
		Some(wl) => {
			if wl.count() > 0 {
				(wl.count() as u32, wl.as_ptr())
			} else {
				(0, ptr::null())
			}
		},
		None => (0, ptr::null()),
	};

	let new_event_ptr: *mut cl_h::cl_event = match dest_list {
		Some(el) => el.allot().as_mut_ptr(),
		None => ptr::null_mut(),
	};

	(blocking_operation, wait_list_len, wait_list_ptr, new_event_ptr)
}


// [FIXME]: TODO: Evaluate usefulness
#[allow(dead_code)]
fn enqueue_copy_buffer<T: OclNum>(
				command_queue: cl_h::cl_command_queue,
				src: &Envoy<T>,		//	src_buffer: cl_mem,
				dst: &Envoy<T>,		//	dst_buffer: cl_mem,
				src_offset: usize,
				dst_offset: usize,
				len_copy_bytes: usize) 
{
	unsafe {
		let err = cl_h::clEnqueueCopyBuffer(
			command_queue,
			src.buffer_obj(),				//	src_buffer,
			dst.buffer_obj(),				//	dst_buffer,
			src_offset,
			dst_offset,
			len_copy_bytes as usize,
			0,
			ptr::null(),
			ptr::null_mut(),
		);
		must_succeed("clEnqueueCopyBuffer()", err);
	}
}

fn get_max_work_group_size(device: cl_h::cl_device_id) -> usize {
	let mut max_work_group_size: usize = 0;

	let err = unsafe { 
		cl_h::clGetDeviceInfo(
			device,
			cl_h::CL_DEVICE_MAX_WORK_GROUP_SIZE,
			mem::size_of::<usize>() as usize,
			// mem::transmute(&max_work_group_size),
			&mut max_work_group_size as *mut _ as *mut libc::c_void,
			ptr::null_mut(),
		) 
	}; 

	must_succeed("clGetDeviceInfo", err);

	max_work_group_size
}



fn finish(command_queue: cl_h::cl_command_queue) {
	unsafe { 
		let err = cl_h::clFinish(command_queue);
		must_succeed("clFinish()", err);
	}
}


fn wait_for_events(wait_list: &EventList) {
	let err = unsafe {
		cl_h::clWaitForEvents(wait_list.count(), wait_list.as_ptr())
	};

	must_succeed("clWaitForEvents", err);
}


#[allow(dead_code)]
fn wait_for_event(event: cl_h::cl_event) {
	let event_array: [cl_h::cl_event; 1] = [event];

	let err = unsafe {
		cl_h::clWaitForEvents(1, event_array.as_ptr())
	};

	must_succeed("clWaitForEvents", err);
}

fn get_event_status(event: cl_h::cl_event) -> cl_int {
	let mut status: cl_int = 0;

	let err = unsafe { 
		cl_h::clGetEventInfo(
			event,
			cl_h::CL_EVENT_COMMAND_EXECUTION_STATUS,
			mem::size_of::<cl_int>(),
			&mut status as *mut _ as *mut libc::c_void,
			ptr::null_mut(),
		)
	};

	must_succeed("clGetEventInfo", err);

	status
}


unsafe fn set_event_callback(
			event: cl_h::cl_event, 
			callback_trigger: cl_int, 
			callback_receiver: extern fn (cl_event, cl_int, *mut libc::c_void),
			user_data: *mut libc::c_void,
		)
{
	let err = cl_h::clSetEventCallback(event, callback_trigger, callback_receiver, user_data);

	must_succeed("clSetEventCallback", err);
}

fn release_event(event: cl_h::cl_event) {
	let err = unsafe {
		cl_h::clReleaseEvent(event)
	};

	must_succeed("clReleaseEvent", err);
}




// extern {
//    pub fn clSetEventCallback_ex(event: cl_event,
//                           command_exec_callback_type: cl_int,
//                           pfn_notify: extern fn (cl_event, cl_int, *mut libc::c_void),
//                           user_data: *mut libc::c_void) -> cl_int;
// }


fn release_mem_object(obj: cl_h::cl_mem) {
	unsafe {
		cl_h::clReleaseMemObject(obj);
	}
}

// [FIXME]: TODO: Evaluate usefulness
#[allow(dead_code)]
fn platform_info(platform: cl_h::cl_platform_id) {
	let mut size = 0 as libc::size_t;

	unsafe {
		let name = cl_h::CL_PLATFORM_NAME as cl_h::cl_device_info;
        let mut err = cl_h::clGetPlatformInfo(
					platform,
					name,
					0,
					ptr::null_mut(),
					&mut size,
		);
		must_succeed("clGetPlatformInfo(size)", err);
		
		let mut param_value: Vec<u8> = iter::repeat(32u8).take(size as usize).collect();
        err = cl_h::clGetPlatformInfo(
					platform,
					name,
					size,
					param_value.as_mut_ptr() as *mut libc::c_void,
					ptr::null_mut(),
		);
        must_succeed("clGetPlatformInfo()", err);
        println!("*** Platform Name ({}): {}", name, String::from_utf8(param_value).unwrap());
    }
}

fn program_build_info(program: cl_h::cl_program, device_ids: &Vec<cl_h::cl_device_id>) -> Result<(), String> {
	let mut size = 0 as libc::size_t;

	for &device_id in device_ids.iter() {
		unsafe {
			let name = cl_h::CL_PROGRAM_BUILD_LOG as cl_h::cl_program_build_info;

			let mut err = cl_h::clGetProgramBuildInfo(
				program,
				device_id,
				name,
				0,
				ptr::null_mut(),
				&mut size,
			);
			must_succeed("clGetProgramBuildInfo(size)", err);

			let mut pbi: Vec<u8> = iter::repeat(32u8).take(size as usize).collect();

			err = cl_h::clGetProgramBuildInfo(
				program,
				device_id,
				name,
				size,
				pbi.as_mut_ptr() as *mut libc::c_void,
				ptr::null_mut(),
			);
			must_succeed("clGetProgramBuildInfo()", err);

			if size > 1 {
				let pbi_nonull = try!(String::from_utf8(pbi).map_err(|e| e.to_string()));
				let pbi_err_string = format!("OPENCL PROGRAM BUILD ERROR: \n\n {}", pbi_nonull);
				println!("\n{}", pbi_err_string);
				return Err(pbi_err_string);
			}
		}
	}

	Ok(())
}

fn must_succeed(message: &str, err_code: cl_h::cl_int) {
	if err_code != cl_h::CLStatus::CL_SUCCESS as cl_h::cl_int {
		//format!("##### \n{} failed with code: {}\n\n #####", message, err_string(err_code));
		panic!(format!("\n\n#####> {} failed with code: {}\n\n", message, err_string(err_code)));
	}
}

fn err_string(err_code: cl_int) -> String {
	match CLStatus::from_i32(err_code) {
		Some(cls) => format!("{:?}", cls),
		None => format!("[Unknown Error Code: {}]", err_code as i64),
	}
}


// fn create_write_buffer<T>(data: &[T], context: cl_h::cl_context) -> cl_h::cl_mem {
// 	let mut err: cl_h::cl_int = 0;
// 	unsafe {
// 		let buf = cl_h::clCreateBuffer(
// 					context, 
// 					cl_h::CL_MEM_READ_ONLY | cl_h::CL_MEM_COPY_HOST_PTR, 
// 					(data.len() * mem::size_of::<T>()) as usize,
// 					data.as_ptr() as *mut libc::c_void, 
// 					//ptr::null_mut(),
// 					&mut err,
// 		);
// 		must_succeed("create_write_buffer", err);
// 		buf
// 	}
// }

// fn create_read_buffer<T>(data: &[T], context: cl_h::cl_context) -> cl_h::cl_mem {
// 	let mut err: cl_h::cl_int = 0;
// 	unsafe {
// 		let buf = cl_h::clCreateBuffer(
// 					context, 
// 					cl_h::CL_MEM_WRITE_ONLY, 
// 					(data.len() * mem::size_of::<T>()) as usize, 
// 					ptr::null_mut(), 
// 					&mut err,
// 		);
// 		must_succeed("create_read_buffer", err);
// 		buf
// 	}
// }



// fn release_kernel(kernel: cl_h::cl_kernel) {
// 	unsafe {
// 		cl_h::clReleaseKernel(kernel);
// 	}
// }

// fn release_components(
// 			kernel: cl_h::cl_kernel, 
// 			command_queue: cl_h::cl_command_queue, 
// 			program: cl_h::cl_program, 
// 			context: cl_h::cl_context)
// {
// 	unsafe {
// 		cl_h::clReleaseKernel(kernel);
// 		cl_h::clReleaseCommandQueue(command_queue);
// 		cl_h::clReleaseProgram(program);
// 		cl_h::clReleaseContext(context);
// 	}
// }	


// fn create_kernel(program: cl_h::cl_program, kernel_name: &str) -> cl_h::cl_kernel {
// 	let mut err: cl_h::cl_int = 0;
// 	unsafe {
// 		let kernel = cl_h::clCreateKernel(program, CString::new(kernel_name.as_bytes()).ok().expect("ocl::create_kernel(): clCreateKernel").as_ptr(), &mut err);
// 		let err_pre = format!("clCreateKernel({}):", kernel_name);
// 		must_succeed(&err_pre, err);
// 		kernel
// 	}
// }

// fn set_kernel_arg<T>(arg_index: cl_h::cl_uint, buffer: T, kernel: cl_h::cl_kernel) {
// 	unsafe {
// 		let err = cl_h::clSetKernelArg(
// 					kernel, 
// 					arg_index, 
// 					mem::size_of::<T>() as usize, 
// 					mem::transmute(&buffer),
// 		);
// 		must_succeed("clSetKernelArg()", err);
// 	}
// }


// fn mem_object_info_size(object: cl_h::cl_mem) -> libc::size_t {
// 	unsafe {
// 		let mut size: libc::size_t = 0;
// 		let err = cl_h::clGetMemObjectInfo(
// 					object,
// 					cl_h::CL_MEM_SIZE,
// 					mem::size_of::<libc::size_t>() as libc::size_t,
// 					(&mut size as *mut usize) as *mut libc::c_void,
// 					ptr::null_mut()
// 		);
// 		must_succeed("clGetMemObjectInfo", err);
// 		size
// 	}
// }

// fn len(object: cl_h::cl_mem) -> usize {
// 	mem_object_info_size(object) as usize / mem::size_of::<f32>()
// }



// [FIXME] TODO: DEPRICATE
// fn enqueue_kernel_simple(
// 				command_queue: cl_h::cl_command_queue, 
// 				kernel: cl_h::cl_kernel, 
// 				gws: usize)
// {
// 	unsafe {
// 		let err = cl_h::clEnqueueNDRangeKernel(
// 					command_queue,
// 					kernel,
// 					1 as cl_uint,
// 					ptr::null(),
// 					mem::transmute(&gws),
// 					ptr::null(),
// 					0,
// 					ptr::null(),
// 					ptr::null_mut(),
// 		);
// 		must_succeed("clEnqueueNDRangeKernel()", err);
// 	}
// }


// fn cstring_to_string(cs: Vec<u8>) -> String {
// 	String::from_utf8(cs).unwrap()
// }


// fn print_junk(
// 			platform: cl_h::cl_platform_id, 
// 			device: cl_h::cl_device_id, 
// 			program: cl_h::cl_program, 
// 			kernel: cl_h::cl_kernel)
// {
// 	println!("");
// 	let mut size = 0 as libc::size_t;

// 	// Get Platform Name
// 	platform_info(platform);
// 	// Get Device Name
// 	let name = cl_h::CL_DEVICE_NAME as cl_h::cl_device_info;

// 	let mut err = unsafe { cl_h::clGetDeviceInfo(
// 		device,
// 		name,
// 		0,
// 		ptr::null_mut(),
// 		&mut size,
// 	) }; 

// 	must_succeed("clGetPlatformInfo(size)", err);

// 	unsafe {
//         let mut device_info: Vec<u8> = iter::repeat(32u8).take(size as usize).collect();

//         err = cl_h::clGetDeviceInfo(
// 			device,
// 			name,
// 			size,
// 			device_info.as_mut_ptr() as *mut libc::c_void,
// 			ptr::null_mut(),
// 		);

//         must_succeed("clGetDeviceInfo()", err);
//         println!("*** Device Name ({}): {}", name, cstring_to_string(device_info));
// 	}

// 	//Get Program Info
// 	unsafe {
// 		let name = cl_h::CL_PROGRAM_SOURCE as cl_h::cl_program_info;

//         err = cl_h::clGetProgramInfo(
// 					program,
// 					name,
// 					0,
// 					ptr::null_mut(),
// 					&mut size,
// 		);
// 		must_succeed("clGetProgramInfo(size)", err);
			
//         let mut program_info: Vec<u8> = iter::repeat(32u8).take(size as usize).collect();

//         err = cl_h::clGetProgramInfo(
// 					program,
// 					name,
// 					size,
// 					program_info.as_mut_ptr() as *mut libc::c_void,
// 					//program_info as *mut libc::c_void,
// 					ptr::null_mut(),
// 		);
//         must_succeed("clGetProgramInfo()", err);
//         println!("*** Program Info ({}): \n {}", name, cstring_to_string(program_info));
// 	}
// 	println!("");
// 	//Get Kernel Name
// 	unsafe {
// 		let name = cl_h::CL_KERNEL_NUM_ARGS as cl_h::cl_uint;

//         err = cl_h::clGetKernelInfo(
// 					kernel,
// 					name,
// 					0,
// 					ptr::null_mut(),
// 					&mut size,
// 		);
// 		must_succeed("clGetKernelInfo(size)", err);

//         let kernel_info = 5 as cl_h::cl_uint;

//         err = cl_h::clGetKernelInfo(
// 					kernel,
// 					name,
// 					size,
// 					mem::transmute(&kernel_info),
// 					ptr::null_mut(),
// 		);
		
//         must_succeed("clGetKernelInfo()", err);
//         println!("*** Kernel Info: ({})\n{}", name, kernel_info);
// 	}
// 	println!("");
// }
