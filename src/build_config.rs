//! Compilation options for building an OpenCL program. Used when creating 
//! a new `Program`.

use std::io::Read;
use std::fs::File;
use std::path::Path;
use std::ffi::{CString, NulError};
use std::collections::HashSet;
use std::convert::Into;
// use std::error::{Error};

/// Compilation options for building an OpenCL program. Used when creating 
/// a new `Program`.
pub struct BuildConfig {
	options: Vec<BuildOpt>,
	kernel_file_names: Vec<String>,
	embedded_kernel_source: Vec<String>,
}

impl BuildConfig {
	/// Returns a new, empty, build configuration object.
	pub fn new() -> BuildConfig {
		BuildConfig {
			options: Vec::with_capacity(64),
			kernel_file_names: Vec::with_capacity(32),
			embedded_kernel_source: Vec::with_capacity(32),
		}
	}

	/// Adds a compiler command line definition => `-D {name}={val}` (builder-style).
	pub fn cmplr_def(mut self, name: &'static str, val: i32) -> BuildConfig {
		self.options.push(BuildOpt::cmplr_def(name, val));
		self
	}

	/// Adds a raw compiler command line option => `{co}` (builder-style).
	pub fn cmplr_opt(mut self, co: &'static str) -> BuildConfig {
		self.options.push(BuildOpt::cmplr_opt(co));
		self
	}

	/// Pushes pre-created build option to the list (builder-style).
	pub fn bo(mut self, bo: BuildOpt) -> BuildConfig {
		self.options.push(bo);
		self
	}

	/// Adds a kernel file to the list of included sources (builder-style).
	pub fn kern_file(mut self, file_name: &'static str) -> BuildConfig {
		self.add_kern_file(file_name.to_string());
		self
	}	

	/// Adds text to the included kernel source (builder-style).
	pub fn kern_embed(mut self, source: &'static str) -> BuildConfig {
		self.add_embedded_kern(source.to_string());
		self
	}

	/// Adds a kernel file to the list of included sources.
	pub fn add_kern_file(&mut self, file_name: String) {
		self.kernel_file_names.push(file_name);
	}

	/// Adds text to the included kernel source.
	pub fn add_embedded_kern(&mut self, source: String) {
		self.embedded_kernel_source.push(source);
	}

	/// Returns a list of kernel file names added for inclusion in the build.
	pub fn kernel_file_names(&self) -> &Vec<String> {
		&self.kernel_file_names
	}

	/// Returns a contatenated string of compiler command line options used when 
	/// building a `Program`.
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

	/// Parses `self.options` for options intended for inclusion at the beginning of 
	/// the final program source and returns them as a list of strings.
	///
	/// Generally used for #define directives, constants, etc. Normally called from
	/// `::kernel_strings()` but can also be called from anywhere for debugging 
	/// purposes.
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

	/// Returns the final program source code as a list of strings.
	///
	/// Order of inclusion:
	/// - includes from `::kernel_includes()`
	/// - source from files listed in `self.kernel_file_names` in reverse order
	/// - raw source from `self.embedded_kernel_source`
	/// [UNSTABLE] TODO: Fix up error handling: return an `OclResult`.
	pub fn kernel_strings(&self) -> Result<Vec<CString>, String> {
		let mut kernel_strings: Vec<CString> = Vec::with_capacity(64);
		let mut kern_file_history: HashSet<&String> = HashSet::with_capacity(64);

		kernel_strings.extend_from_slice(&try!(self.kernel_includes().map_err(|e| e.to_string())));

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

		for elem in self.embedded_kernel_source.iter() {
			kernel_strings.push(try!(CString::new(elem.clone().into_bytes()).map_err(|e| e.to_string())));
		}

		Ok(kernel_strings)
	}	
}

/// A build option intended for use either by the compiler as a command line switch
/// or for inclusion at the top of the final build source code.
///
/// A few of the often used variants have constructors for convenience.
/// [FIXME] TODO: Explain how each variant is used.
pub enum BuildOpt {
	CmplrDefine { ident: String, val: String },
	CmplrInclDir { path: String },
	CmplrOther(String),
	IncludeDefine { ident: String, val: String },
	IncludeRaw(String),
}

impl BuildOpt {
	/// Returns a `BuildOpt::CmplrDefine`.
	pub fn cmplr_def<S: Into<String>>(ident: S, val: i32) -> BuildOpt {
		BuildOpt::CmplrDefine {
			ident: ident.into(),
			val: val.to_string(),
		}
	}

	/// Returns a `BuildOpt::CmplrOther`.
	pub fn cmplr_opt<S: Into<String>>(opt: S) -> BuildOpt {
		BuildOpt::CmplrOther(opt.into())
	}

	/// Returns a `BuildOpt::IncludeDefine`.
	pub fn include_def<S: Into<String>>(ident: S, val: String) -> BuildOpt {
		BuildOpt::IncludeDefine {
			ident: ident.into(),
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
