use std::io::{ Read };
use std::fs::{ File };
use std::path::{ Path };
use std::ffi;
use std::collections::{ HashSet };
use std::error::{ Error };

pub struct BuildOptions {
	options: Vec<BuildOption>,
	string: String,
	kernel_file_names: Vec<String>,
}

impl BuildOptions {
	pub fn new(cl_options: &'static str) -> BuildOptions {
		let bo = BuildOptions {
			options: Vec::with_capacity(1 << 5),
			string: String::with_capacity(1 << 11),
			kernel_file_names: Vec::with_capacity(20),
		};

		bo.str(cl_options)
	}

	fn str(mut self, st: &'static str) -> BuildOptions {
		self.string.push_str(st);
		self
	}

	pub fn opt(mut self, name: &'static str, val: i32) -> BuildOptions {
		self.options.push(BuildOption::new(name, val));
		self
	}

	pub fn add_opt(mut self, bo: BuildOption) -> BuildOptions {
		self.options.push(bo);
		self
	}

	pub fn kern_file(mut self, file_name: String) -> BuildOptions {
		self.add_kern_file(file_name);
		self
	}

	pub fn add_kern_file(&mut self, file_name: String) {
		self.kernel_file_names.push(file_name);
	}		

	pub fn kernel_file_names(&self) -> &Vec<String> {
		&self.kernel_file_names
	}

	pub fn compiler_options(mut self) -> String {
		for option in self.options.iter() {
			if !option.is_kern_header {
				self.string.push_str(&option.to_preprocessor_option_string());
			}
		}
		self.string
	}

	pub fn kernel_str_header(&self) -> Vec<u8> {
		let mut header = String::with_capacity(300);

		header.push_str("\n");

		for option in self.options.iter() {
			if option.is_kern_header {
				header.push_str(&option.to_define_directive_string());
			}
		}

		header.push_str("\n");

		header.into_bytes()
	}

	// [FIXME] TODO: Fix up error handling.
	pub fn kernel_str(&self) -> Result<ffi::CString, String> {
		let mut kern_str: Vec<u8> = Vec::with_capacity(10000);
		let mut kern_history: HashSet<&String> = HashSet::with_capacity(20);

		let dd_string = self.kernel_str_header();
		kern_str.push_all(&dd_string);
		// print!("OCL::PARSE_KERNEL_FILES(): KERNEL FILE DIRECTIVES HEADER: \n{}", 
		// 	String::from_utf8(dd_string).ok().unwrap());

		for kfn in self.kernel_file_names().iter().rev() {
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
}



pub struct BuildOption {
	name: &'static str,
	val: String,
	is_kern_header: bool,
}

impl BuildOption {
	pub fn new(name: &'static str, val: i32) -> BuildOption {
		BuildOption {
			name: name,
			val: val.to_string(),
			is_kern_header: false,
		}
	}

	pub fn with_str_val(name: &'static str, val: String) -> BuildOption {
		BuildOption {
			name: name,
			val: val,
			is_kern_header: true,
		}
	}

	pub fn to_preprocessor_option_string(&self) -> String {
		format!(" -D{}={}", self.name, self.val)
	}

	pub fn to_define_directive_string(&self) -> String {
		format!("#define {}  {}\n", self.name, self.val)
	}
}

