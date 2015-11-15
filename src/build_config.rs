use std::io::{ Read };
use std::fs::{ File };
use std::path::{ Path };
use std::ffi::{ CString, NulError };
use std::collections::{ HashSet };
// use std::error::{ Error };

pub struct BuildConfig {
	options: Vec<BuildOpt>,
	kernel_file_names: Vec<String>,
}

impl BuildConfig {
	pub fn new() -> BuildConfig {
		BuildConfig {
			options: Vec::with_capacity(64),
			kernel_file_names: Vec::with_capacity(32),
		}
	}

	/// Command line define, i.e. '-D name=val'.
	pub fn cmplr_def(mut self, name: &'static str, val: i32) -> BuildConfig {
		self.options.push(BuildOpt::cmplr_def(name, val));
		self
	}

	pub fn cmplr_opt(mut self, st: &'static str) -> BuildConfig {
		self.options.push(BuildOpt::cmplr_opt(st));
		self
	}

	pub fn bo(mut self, bo: BuildOpt) -> BuildConfig {
		self.options.push(bo);
		self
	}

	pub fn kern_file(mut self, file_name: &'static str) -> BuildConfig {
		self.add_kern_file(file_name.to_string());
		self
	}

	pub fn add_kern_file(&mut self, file_name: String) {
		self.kernel_file_names.push(file_name);
	}		

	pub fn kernel_file_names(&self) -> &Vec<String> {
		&self.kernel_file_names
	}

	pub fn compiler_options(&self) -> Result<CString, NulError> {
		let mut opts: Vec<String> = Vec::with_capacity(64);

		opts.push(" ".to_string());

		for option in self.options.iter() {			
			match option {
				&BuildOpt::CmplrDefine { ref ident, ref val } => {
					opts.push(format!("-D{}={}", ident, val))
				},

				&BuildOpt::CmplrInclDir { ref path } => {
					opts.push(format!("-I{}", path))
				},

				&BuildOpt::CmplrOther(ref s) => {
					opts.push(s.clone())
				},

				_ => (),	
			}
		}

		CString::new(opts.join(" ").into_bytes())
	}

	pub fn kernel_includes(&self) -> Result<Vec<CString>, NulError> {
		let mut strings	= Vec::with_capacity(64);
		strings.push(try!(CString::new("\n".as_bytes())));

		for option in self.options.iter() {
			match option {
				&BuildOpt::IncludeDefine { ref ident, ref val } => {
					strings.push(try!(CString::new(format!("#define {}  {}\n", ident, val).into_bytes())));
				},

				&BuildOpt::IncludeRaw(ref text) => {
					strings.push(try!(CString::new(text.clone().into_bytes())));
				},

				_ => (),
			};

		}
		Ok(strings)
	}

	// [FIXME] TODO: Fix up error handling: return proper a error type.
	pub fn kernel_strings(&self) -> Result<Vec<CString>, String> {
		let mut kernel_strings: Vec<CString> = Vec::with_capacity(64);
		let mut kern_file_history: HashSet<&String> = HashSet::with_capacity(64);

		kernel_strings.push_all(&try!(self.kernel_includes().map_err(|e| e.to_string())));

		for kfn in self.kernel_file_names().iter().rev() {
			let mut kern_str: Vec<u8> = Vec::with_capacity(100000);

			if kern_file_history.contains(kfn) { continue; }

			kern_file_history.insert(&kfn);

			let valid_kfp = Path::new(kfn);

			// let mut kern_file = match File::open(&valid_kfp) {
			// 	Err(why) => return Err(format!("Couldn't open '{}': {}", 
			// 		kfn, Error::description(&why))),
			// 	Ok(file) => file,
			// };

			let mut kern_file = try!(File::open(&valid_kfp).map_err(|e| 
				format!("Error reading `{}`: {}", kfn, &e.to_string())));

			// match kern_file.read_to_end(&mut kern_str) {
	  //   		Err(why) => return Err(format!("Couldn't read '{}': {}", 
	  //   			kfn, Error::description(&why))),
			//     Ok(_) => (), //println!("{}OCL::BUILD(): parsing {}: {} bytes read.", MT, &file_name, bytes),
			// }

			try!(kern_file.read_to_end(&mut kern_str).map_err(|e| 
				format!("Error reading `{}`: {}", kfn, &e.to_string())));

			kern_str.shrink_to_fit();

			kernel_strings.push(try!(CString::new(kern_str).map_err(|e| e.to_string())));
		}

		Ok(kernel_strings)
	}	
}


pub enum BuildOpt {
	CmplrDefine { ident: String, val: String },
	CmplrInclDir { path: String },
	CmplrOther(String),
	IncludeDefine { ident: String, val: String },
	IncludeRaw(String),
}

impl BuildOpt {
	pub fn cmplr_def(ident: &'static str, val: i32) -> BuildOpt {
		BuildOpt::CmplrDefine {
			ident: ident.to_string(),
			val: val.to_string(),
		}
	}

	pub fn cmplr_opt(val: &'static str) -> BuildOpt {
		BuildOpt::CmplrOther(val.to_string())
	}

	pub fn include_def(ident: &'static str, val: String) -> BuildOpt {
		BuildOpt::IncludeDefine {
			ident: ident.to_string(),
			val: val,
		}
	}
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
