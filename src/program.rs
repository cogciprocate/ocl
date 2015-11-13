use std::ffi::{ CString };

use super::{ cl_h, BuildConfig, Context };

#[derive(Clone)]
pub struct Program {
	obj: cl_h::cl_program,
	context_obj: cl_h::cl_context,
	// device_id: cl_h::cl_device_id,
}

// [FIXME] TODO: ERROR HANDLING
impl Program {
	pub fn new(build_options: BuildConfig, context: &Context, device_idx: Option<usize>
			) -> Result<Program, String> 
	{
		let device_id = context.resolve_device_id(device_idx);

		// [FIXME]: Unwrap
		Program::from_parts(
			try!(build_options.kernel_strings().map_err(|e| e.to_string())), 
			try!(build_options.compiler_options().map_err(|e| e.to_string())), 
			context.obj(), 
			&vec![device_id])
	}

	pub fn from_parts(
				kernel_strings: Vec<CString>, 
				cmplr_opts: CString, 
				context_obj: cl_h::cl_context, 
				devices: &Vec<cl_h::cl_device_id>,
			) -> Result<Program, String> 
	{
		// let kern_c_str = try!(parse_kernel_files(&build_options));

		// [FIXME] TEMPORARY UNWRAP:
		let obj = super::create_program(kernel_strings, cmplr_opts, 
			context_obj, devices).unwrap();

		// [FIXME] TEMPORARY UNWRAP:
		// [FIXME] IS THIS A DUPLICATE CALL?
		super::program_build_info(obj, devices).unwrap();

		Ok(Program {
			obj: obj,
			context_obj: context_obj,
			// device_id: device_id,
		})
	}

	pub fn release(&mut self) {
		unsafe { 
			cl_h::clReleaseProgram(self.obj);
		}
	}

	pub fn obj(&self) -> cl_h::cl_program {
		self.obj
	}
}


// fn parse_kernel_files(build_options: &BuildConfig) -> Result<CString, String> {
// 	let mut kern_str: Vec<u8> = Vec::with_capacity(10000);
// 	let mut kern_history: HashSet<&String> = HashSet::with_capacity(20);

// 	let dd_string = build_options.cl_file_header();
// 	kern_str.push_all(&dd_string);
// 	// print!("OCL::PARSE_KERNEL_FILES(): KERNEL FILE DIRECTIVES HEADER: \n{}", 
// 	// 	String::from_utf8(dd_string).ok().unwrap());

// 	for kfn in build_options.kernel_file_names().iter().rev() {
// 		// let file_name = format!("{}/{}/{}", env!("P"), "bismit/cl", f_n);
// 		// let valid_kfp = try!(valid_kernel_file_path(&kfn));

// 		// {
// 			if kern_history.contains(kfn) { continue; }
// 			let valid_kfp = Path::new(kfn);

// 			let mut kern_file = match File::open(&valid_kfp) {
// 				Err(why) => return Err(format!("Couldn't open '{}': {}", 
// 					kfn, Error::description(&why))),
// 				Ok(file) => file,
// 			};

// 			match kern_file.read_to_end(&mut kern_str) {
// 	    		Err(why) => return Err(format!("Couldn't read '{}': {}", 
// 	    			kfn, Error::description(&why))),
// 			    Ok(_) => (), //println!("{}OCL::BUILD(): parsing {}: {} bytes read.", MT, &file_name, bytes),
// 			}
// 		// }

// 		kern_history.insert(&kfn);
// 	}

// 	Ok(CString::new(kern_str).expect("Ocl::new(): ocl::parse_kernel_files(): CString::new(): Error."))
// }
