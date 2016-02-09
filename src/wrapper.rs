//! Wrapper functions for the OpenCL C ABI.
//! 
//! Some functions *may* break Rust's usual safety promises and have not been
//! comprehensively tested or evaluated.
//! 
//! Objects created should not generally be moved or shared between threads.
//!

use std::ptr;
use std::mem;
use std::io::Read;
use std::ffi::CString;
use std::iter;
use libc;
use num::{FromPrimitive};

use cl_h::{self, cl_platform_id, cl_device_id, cl_device_type, cl_device_info, cl_context,
	cl_platform_info,
	cl_program, cl_program_build_info, cl_command_queue, cl_mem, cl_event, cl_bool,
	cl_int, cl_uint, cl_bitfield, CLStatus};

use super::{DEFAULT_DEVICE_TYPE, DEVICES_MAX, EventList, OclNum, Buffer, OclError, OclResult};


//=============================================================================
//============================== OCL FUNCTIONS ================================
//=============================================================================

// Create Platform and get ID
#[inline]
pub fn get_platform_ids() -> Vec<cl_platform_id> {
	let mut num_platforms = 0 as cl_uint;
	
	let mut err: cl_int = unsafe { cl_h::clGetPlatformIDs(0, ptr::null_mut(), &mut num_platforms) };
	must_succeed("clGetPlatformIDs()", err);

	let mut platforms: Vec<cl_platform_id> = Vec::with_capacity(num_platforms as usize);
	for _ in 0..num_platforms as usize { platforms.push(0 as cl_platform_id); }

	unsafe {
		err = cl_h::clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), ptr::null_mut()); 
		must_succeed("clGetPlatformIDs()", err);
	}
	
	platforms
}

// GET_DEVICE_IDS():
/// # Panics
///  -must_succeed(): [FIXME]: Explaination needed (possibly at crate level?)
#[inline]
pub fn get_device_ids(
			platform: cl_platform_id, 
			device_types_opt: Option<cl_device_type>,
		) -> Vec<cl_device_id> 
{
	let device_type = device_types_opt.unwrap_or(DEFAULT_DEVICE_TYPE);
	
	let mut devices_available: cl_uint = 0;
	let mut devices_array: [cl_device_id; DEVICES_MAX as usize] = [0 as cl_device_id; DEVICES_MAX as usize];

	let err = unsafe { cl_h::clGetDeviceIDs(
		platform, 
		device_type, 
		DEVICES_MAX, 
		devices_array.as_mut_ptr(), 
		&mut devices_available
	) };

	must_succeed("clGetDeviceIDs()", err);

	let mut device_ids: Vec<cl_device_id> = Vec::with_capacity(devices_available as usize);

	for i in 0..devices_available as usize {
		device_ids.push(devices_array[i]);
	}

	device_ids
}

#[inline]
pub fn create_context(device_ids: &Vec<cl_device_id>) -> cl_context {
	let mut err: cl_int = 0;

	unsafe {
		let context: cl_context = cl_h::clCreateContext(
						ptr::null(), 
						device_ids.len() as cl_uint, 
						device_ids.as_ptr(),
						mem::transmute(ptr::null::<fn()>()), 
						ptr::null_mut(), 
						&mut err);
		must_succeed("clCreateContext()", err);
		context
	}
}

#[inline]
pub fn new_program(
			kern_strings: Vec<CString>,
			cmplr_opts: CString,
			context: cl_context, 
			device_ids: &Vec<cl_device_id>
		) -> OclResult<cl_program>
{
	// Lengths (not including \0 terminator) of each string:
	let ks_lens: Vec<usize> = kern_strings.iter().map(|cs| cs.as_bytes().len()).collect();	
	// Pointers to each string:
	let kern_strs: Vec<*const i8> = kern_strings.iter().map(|cs| cs.as_ptr()).collect();

	let mut err = 0i32;

	unsafe {
		let program: cl_program = cl_h::clCreateProgramWithSource(
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
			program_build_info(program, device_ids, true).map(|_| program)
		} else {
			must_succeed("clBuildProgram()", err);
			// Unreachable:
			Ok(program) 
		}
	}
}

#[inline]
pub fn create_command_queue(
			context: cl_context, 
			device: cl_device_id,
		) -> cl_command_queue 
{
	let mut err: cl_int = 0;

	unsafe {
		let cq: cl_command_queue = cl_h::clCreateCommandQueue(
					context, 
					device, 
					cl_h::CL_QUEUE_PROFILING_ENABLE, 
					&mut err
		);

		must_succeed("clCreateCommandQueue()", err);
		cq
	}
}

// Note: the dval_len.0 is not actually used. If buffer is created in this way it will be
// uninitialized. Yes, this is janky.
#[inline]
pub fn create_buffer<T>(
			data: Option<&[T]>, 
			type_and_len: Option<(T, usize)>,
			context: cl_context, 
			flags: cl_bitfield
		) -> cl_mem 
{
	assert!(!(data.is_some() && type_and_len.is_some()));
	let mut err: cl_int = 0;

	let explicit_len_bytes = match type_and_len {
		Some((_, len)) => Some(len * mem::size_of::<T>()),
		None => None,
	};

	let (size, ptr) = match data {
		Some(d) => (match explicit_len_bytes {
				Some(size) => size,
				None => d.len() * mem::size_of::<T>(),
			}, 
			d.as_ptr() as *mut libc::c_void),
		None => (match explicit_len_bytes {
				Some(size) => size,
				None => panic!("ocl::create_buffer(): No data or type and size given."),
			}, 
			ptr::null_mut()),
	};

	unsafe {
		let buf = cl_h::clCreateBuffer(
					context, 
					flags,
					size,
					ptr, 
					//ptr::null_mut(),
					&mut err,
		);
		must_succeed("create_buffer", err);
		buf
	}
}

// [WORK IN PROGRESS]
#[inline]
pub fn create_image_2d() -> cl_mem {

	// pub fn clCreateImage2D(context: cl_context,
	// 					   flags: cl_mem_flags,
	// 					   image_format: *mut cl_image_format,
	// 					   image_width: size_t,
	// 					   image_depth: size_t,
	// 					   image_slc_pitch: size_t,
	// 					   host_ptr: *mut c_void,
	// 					   errcode_ret: *mut cl_int) -> cl_mem;

	// TEMPORARY
	0 as *mut libc::c_void
}

// [WORK IN PROGRESS]
#[inline]
pub fn create_image_3d() -> cl_mem {
	
	// pub fn clCreateImage3D(context: cl_context,
	// 					   flags: cl_mem_flags,
	// 					   image_format: *mut cl_image_format,
	// 					   image_width: size_t,
	// 					   image_depth: size_t,
	// 					   image_depth: size_t,
	// 					   image_slc_pitch: size_t,
	// 					   image_depth: size_t,
	// 					   image_slc_pitch: size_t,
	// 					   image_slc_pitch: size_t,
	// 					   host_ptr: *mut c_void,
	// 					   errcode_ret: *mut cl_int) -> cl_mem;

	// TEMPORARY
	0 as *mut libc::c_void
}


#[inline]
pub fn enqueue_write_buffer<T>(
			command_queue: cl_command_queue,
			buffer: cl_mem, 
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

#[inline]
pub fn enqueue_read_buffer<T>(
			command_queue: cl_command_queue,
			buffer: cl_mem, 
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

#[inline]
pub fn resolve_queue_opts(block: bool, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>)
		-> (cl_bool, cl_uint, *const cl_event, *mut cl_event)
{
	let blocking_operation = if block { cl_h::CL_TRUE } else { cl_h::CL_FALSE };

	let (wait_list_len, wait_list_ptr): (u32, *const cl_event) = match wait_list {
		Some(wl) => {
			if wl.count() > 0 {
				(wl.count() as u32, wl.as_ptr())
			} else {
				(0, ptr::null())
			}
		},
		None => (0, ptr::null()),
	};

	let new_event_ptr: *mut cl_event = match dest_list {
		Some(el) => el.allot().as_mut_ptr(),
		None => ptr::null_mut(),
	};

	(blocking_operation, wait_list_len, wait_list_ptr, new_event_ptr)
}


// [FIXME]: TODO: Evaluate usefulness
#[allow(dead_code)]
#[inline]
pub fn enqueue_copy_buffer<T: OclNum>(
				command_queue: cl_command_queue,
				src: &Buffer<T>,		//	src_buffer: cl_mem,
				dst: &Buffer<T>,		//	dst_buffer: cl_mem,
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

#[inline]
pub fn get_max_work_group_size(device: cl_device_id) -> usize {
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

#[inline]
pub fn finish(command_queue: cl_command_queue) {
	unsafe { 
		let err = cl_h::clFinish(command_queue);
		must_succeed("clFinish()", err);
	}
}

#[inline]
pub fn wait_for_events(wait_list: &EventList) {
	let err = unsafe {
		cl_h::clWaitForEvents(wait_list.count(), wait_list.as_ptr())
	};

	must_succeed("clWaitForEvents", err);
}


#[allow(dead_code)]
#[inline]
pub fn wait_for_event(event: cl_event) {
	let event_array: [cl_event; 1] = [event];

	let err = unsafe {
		cl_h::clWaitForEvents(1, event_array.as_ptr())
	};

	must_succeed("clWaitForEvents", err);
}

#[inline]
pub fn get_event_status(event: cl_h::cl_event) -> cl_int {
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

#[inline]
pub unsafe fn set_event_callback(
			event: cl_event, 
			callback_trigger: cl_int, 
			callback_receiver: extern fn (cl_event, cl_int, *mut libc::c_void),
			user_data: *mut libc::c_void,
		)
{
	let err = cl_h::clSetEventCallback(event, callback_trigger, callback_receiver, user_data);

	must_succeed("clSetEventCallback", err);
}

#[inline]
pub fn release_event(event: cl_event) {
	let err = unsafe {
		cl_h::clReleaseEvent(event)
	};

	must_succeed("clReleaseEvent", err);
}

#[inline]
pub fn release_mem_object(obj: cl_mem) {
	unsafe {
		cl_h::clReleaseMemObject(obj);
	}
}

// [FIXME]: TODO: Evaluate usefulness
#[allow(dead_code)]
#[inline]
pub fn platform_info(platform: cl_platform_id) {
	let mut size = 0 as libc::size_t;

	unsafe {
		let name = cl_h::CL_PLATFORM_NAME as cl_platform_info;
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

#[inline]
pub fn program_build_info(program: cl_program, device_ids: &Vec<cl_device_id>, 
			print_output: bool) -> OclResult<()> 
{
	let mut size = 0 as libc::size_t;

	for &device_id in device_ids.iter() {
		unsafe {
			let name = cl_h::CL_PROGRAM_BUILD_LOG as cl_program_build_info;

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
				let pbi_err_string = format!("OpenCL Program Build Info for '{:?}':\
				\n\n{}\n", device_id, pbi_nonull);

				if print_output {
					println!(

"\n\n\
###################### OPENCL PROGRAM BUILD DEBUG OUTPUT ######################\n\n\
{}\
###############################################################################\
\n\n"

					, pbi_err_string);
				}

				return OclError::err(pbi_err_string);
			}
		}
	}

	Ok(())
}

/// Returns a string containing requested information.
///
/// Currently lazily assumes everything is a char[] then into a String. Non-string
/// info types need to be manually reconstructed from that. Yes this is retarded.
///
/// [TODO (low priority)]: Needs to eventually be made more flexible and should return 
/// an enum with a variant corresponding to the type of info requested. Could 
/// alternatively return a generic type and just blindly cast to it.
#[allow(dead_code, unused_variables)] 
pub fn device_info(device: cl_device_id, info_type: cl_device_info) -> String {
	let mut info_value_size: usize = 0;

	let err = unsafe { 
		cl_h::clGetDeviceInfo(
			device,
			cl_h::CL_DEVICE_MAX_WORK_GROUP_SIZE,
			mem::size_of::<usize>() as usize,
			// mem::transmute(&max_work_group_size),
			0 as *mut libc::c_void,
			&mut info_value_size as *mut usize,
		) 
	}; 

	must_succeed("clGetDeviceInfo", err);

	String::new()
}

#[inline]
pub fn must_succeed(message: &str, err_code: cl_int) {
	if err_code != cl_h::CLStatus::CL_SUCCESS as cl_int {
		//format!("##### \n{} failed with code: {}\n\n #####", message, err_string(err_code));
		panic!(format!("\n\n#####> {} failed with code: {}\n\n", message, err_string(err_code)));
	}
}

#[inline]
fn err_string(err_code: cl_int) -> String {
	match CLStatus::from_i32(err_code) {
		Some(cls) => format!("{:?}", cls),
		None => format!("[Unknown Error Code: {}]", err_code as i64),
	}
}
