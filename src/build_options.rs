use std::io::{ Read };
use std::fs::{ File };
use std::path::{ Path };
use std::ffi::{ CString, NulError };
use std::collections::{ HashSet };
use std::error::{ Error };

pub struct BuildOptions {
	// options: Vec<BuildOption>,
	options: Vec<BuildOpt>,
	// string: String,
	kernel_file_names: Vec<String>,
}

impl BuildOptions {
	pub fn new(cl_options: &'static str) -> BuildOptions {
		let bo = BuildOptions {
			options: Vec::with_capacity(32),
			// string: String::with_capacity(1 << 11),
			kernel_file_names: Vec::with_capacity(20),
		};

		bo.str(cl_options)
	}	

	pub fn opt(mut self, name: &'static str, val: i32) -> BuildOptions {
		self.options.push(BuildOpt::c_def(name, val));
		self
	}

	pub fn str(mut self, st: &'static str) -> BuildOptions {
		// self.string.push_str(st);
		self.options.push(BuildOpt::c_other(st));
		self
	}

	pub fn add_opt(mut self, bo: BuildOpt) -> BuildOptions {
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

	pub fn compiler_options(&self) -> Result<CString, NulError> {
		let mut opts: Vec<String> = Vec::with_capacity(25);

		opts.push(" ".to_string());

		for option in self.options.iter() {			
			match option {
				&BuildOpt::CmplrDefine { ref ident, ref val } => {
					opts.push(format!("-D{}={}", ident, val))
				},

				&BuildOpt::CmplrOther(ref s) => {
					opts.push(s.clone())
				},

				_ => (),	
			}
		}

		Ok(try!(CString::new(opts.join(" ").into_bytes())))
	}

	pub fn kernel_macros(&self) -> Result<Vec<CString>, NulError> {
		// let mut header = String::with_capacity(300);
		// let mut strings = Vec::with_capacity(300);
		let mut strings	= Vec::with_capacity(20);

		strings.push(try!(CString::new("\n".as_bytes())));

		for option in self.options.iter() {
			match option {
				&BuildOpt::MacroDefine { ref ident, ref val } => {
					strings.push(try!(CString::new(format!("#define {}  {}\n", ident, val).into_bytes())));
				},

				_ => (),
			};

		}			

		// strings.push_all("\n".as_strings());

		// strings.into_strings()
		Ok(strings)
	}

	// [FIXME] TODO: Fix up error handling: return proper a error type.
	pub fn kernel_strings(&self) -> Result<Vec<CString>, String> {
		// let mut kern_str: Vec<u8> = Vec::with_capacity(10000);
		let mut kernel_strings: Vec<CString> = Vec::with_capacity(40);
		let mut kern_file_history: HashSet<&String> = HashSet::with_capacity(20);

		// let dd_string = self.kernel_header_str();
		// kern_str.push_all(&dd_string);
		kernel_strings.push_all(&try!(self.kernel_macros().map_err(|e| e.to_string())));
		// print!("OCL::PARSE_KERNEL_FILES(): KERNEL FILE DIRECTIVES HEADER: \n{}", 
		// 	String::from_utf8(dd_string).ok().unwrap());

		for kfn in self.kernel_file_names().iter().rev() {
			// let file_name = format!("{}/{}/{}", env!("P"), "bismit/cl", f_n);
			// let valid_kfp = try!(valid_kernel_file_path(&kfn));
			let mut kern_str: Vec<u8> = Vec::with_capacity(10000);

			if kern_file_history.contains(kfn) { continue; }

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

			kern_file_history.insert(&kfn);

			kernel_strings.push(try!(CString::new(kern_str).map_err(|e| e.to_string())));
		}

		// Ok(CString::new(kern_str).expect("Ocl::new(): ocl::parse_kernel_files(): CString::new(): Error."))
		Ok(kernel_strings)
	}	
}


pub enum BuildOpt {
	CmplrDefine { ident: String, val: String },
	CmplrInclDir { path: String },
	CmplrOther(String),
	MacroDefine { ident: String, val: String },
	MacroRaw(String),
}

impl BuildOpt {
	pub fn c_def(ident: &'static str, val: i32) -> BuildOpt {
		BuildOpt::CmplrDefine {
			ident: ident.to_string(),
			val: val.to_string(),
		}
	}

	pub fn c_other(val: &'static str) -> BuildOpt {
		BuildOpt::CmplrOther(val.to_string())
	}

	pub fn m_def(ident: &'static str, val: String) -> BuildOpt {
		BuildOpt::MacroDefine {
			ident: ident.to_string(),
			val: val,
		}
	}

	// pub fn to_c_string(&self) -> Result<CString, NulError> {
	// 	match self {
	// 		&BuildOpt::CmplrDefine { ref ident, ref val } => {
	// 			CString::new(format!(" -D{}={}", ident, val).into_bytes())
	// 		},

	// 		&BuildOpt::CmplrOther(ref s) => {
	// 			CString::new(s.clone().into_bytes())
	// 		},		

	// 		_ => CString::new("".to_string().into_bytes()),
	// 	}
	// }

	// pub fn to_bytes(&self) -> Vec<u8> {
	// 	match self {
	// 		&BuildOpt::MacroDefine { ref ident, ref val } => {
	// 			CString::new(format!("#define {}  {}\n", ident, val).into_bytes())
	// 		},

	// 		_ => "".to_string().into_bytes(),
	// 	}
	// }
}




// pub struct BuildOption {
// 	name: &'static str,
// 	val: String,
// 	is_kern_header: bool,
// }

// impl BuildOption {
// 	pub fn new(name: &'static str, val: i32) -> BuildOption {
// 		BuildOption {
// 			name: name,
// 			val: val.to_string(),
// 			is_kern_header: false,
// 		}
// 	}

// 	pub fn with_str_val(name: &'static str, val: String) -> BuildOption {
// 		BuildOption {
// 			name: name,
// 			val: val,
// 			is_kern_header: true,
// 		}
// 	}

// 	pub fn to_preprocessor_option_string(&self) -> String {
// 		format!(" -D{}={}", self.name, self.val)
// 	}

// 	pub fn to_define_directive_string(&self) -> String {
// 		format!("#define {}  {}\n", self.name, self.val)
// 	}
// }
