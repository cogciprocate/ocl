//! The builder for `Program`.

use std::io::Read;
use std::fs::File;
use std::path::Path;
use std::ffi::{CString};
use std::collections::HashSet;
use std::convert::Into;

use error::{Result as OclResult, Error as OclError};
use standard::{Context, Program};

/// Compilation options for building an OpenCL program. Used when creating 
/// a new `Program`.
pub struct ProgramBuilder {
    options: Vec<BuildOpt>,
    src_file_names: Vec<String>,
    device_idxs: Vec<usize>,
    // embedded_kernel_source: Vec<String>,
}

impl ProgramBuilder {
    /// Returns a new, empty, build configuration object.
    pub fn new() -> ProgramBuilder {
        ProgramBuilder {
            options: Vec::with_capacity(64),
            src_file_names: Vec::with_capacity(32),
            device_idxs: Vec::with_capacity(8),
            // embedded_kernel_source: Vec::with_capacity(32),
        }
    }

    /// Returns a newly built Program.
    // [FIXME]: Don't map errors to strings, feed directly to OclError:
    pub fn build(&self, context: &Context) -> OclResult<Program> {
        Program::from_parts(
            try!(self.get_src_strings().map_err(|e| e.to_string())), 
            try!(self.get_compiler_options().map_err(|e| e.to_string())), 
            context.obj_raw(), 
            &context.resolve_device_idxs(&self.device_idxs))
    }

    /// Adds a build option containing a compiler command line definition.
    /// Formatted as `-D {name}={val}`.
    pub fn cmplr_def<'p>(&'p mut self, name: &'static str, val: i32) -> &'p mut ProgramBuilder {
        self.options.push(BuildOpt::cmplr_def(name, val));
        self
    }

    /// Adds a build option containing a raw compiler command line parameter. 
    /// Formatted as `{co}` (exact text).
    pub fn cmplr_opt<'p>(&'p mut self, co: &'static str) -> &'p mut ProgramBuilder {
        self.options.push(BuildOpt::cmplr_opt(co));
        self
    }

    /// Pushes pre-created build option to the list.
    pub fn bo<'p>(&'p mut self, bo: BuildOpt) -> &'p mut ProgramBuilder {
        // self.add_bo(bo);
        self.options.push(bo);
        self
    }

    /// Adds a kernel file to the list of included sources.
    pub fn src_file<'p, S: Into<String>>(&'p mut self, file_name: S) -> &'p mut ProgramBuilder {
        // self.add_src_file(file_name.to_string());
        self.src_file_names.push(file_name.into());
        self
    }   

    /// Adds text to the included kernel source.
    pub fn src<'p, S: Into<String>>(&'p mut self, src: S) -> &'p mut ProgramBuilder {
        // self.add_src(src);
        self.options.push(BuildOpt::IncludeRawEof(src.into()));
        self
    }

    /// Specify which devices this program should be built on using a vector of 
    /// zero-based device indexes.
    ///
    /// # Example
    /// If your system has 4 OpenGL devices and you want to include them all:
    /// ```
    /// let program = program::builder()
    ///     .src(source_str)
    ///     .device_idxs(vec![0, 1, 2, 3])
    ///     .build(context);
    /// ```
    pub fn device_idxs<'p>(&'p mut self, device_idxs: Vec<usize>) -> &'p mut ProgramBuilder {
        self.device_idxs.extend_from_slice(&device_idxs);
        self
    }

    // /// Adds a kernel file to the list of included sources (in place).
    // pub fn add_src_file(&mut self, file_name: String) {
    //     self.src_file_names.push(file_name);
    // }

    // /// Adds text to the included kernel source (in place).
    // pub fn add_src<S: Into<String>>(&mut self, src: S) {
    //     // self.embedded_kernel_source.push(source.into());
    //     self.options.push(BuildOpt::IncludeRawEof(src.into()));
    // }

    // /// Adds a pre-created build option to the list (in place).
    // pub fn add_bo(&mut self, bo: BuildOpt) {
    //     self.options.push(bo);
    // }

    /// Returns a list of kernel file names added for inclusion in the build.
    pub fn get_src_file_names(&self) -> &Vec<String> {
        &self.src_file_names
    }

    // Returns the list of devices indexes with which this `ProgramBuilder` is
    // configured to build on.
    pub fn get_device_idxs(&self) -> &Vec<usize> {
        &self.device_idxs
    }


    /// Parses `self.options` for options intended for inclusion at the beginning of 
    /// the final program source and returns them as a list of strings.
    ///
    /// Generally used for #define directives, constants, etc. Normally called from
    /// `::get_src_strings()` but can also be called from anywhere for debugging 
    /// purposes.
    fn get_kernel_includes(&self) -> OclResult<Vec<CString>> {
        let mut strings = Vec::with_capacity(64);
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

    /// Parses `self.options` for options intended for inclusion at the end of 
    /// the final program source and returns them as a list of strings.
    fn get_kernel_includes_eof(&self) -> OclResult<Vec<CString>> {
        let mut strings = Vec::with_capacity(64);
        strings.push(try!(CString::new("\n".as_bytes())));

        for option in self.options.iter() {
            match option {
                &BuildOpt::IncludeRawEof(ref text) => {
                    strings.push(try!(CString::new(text.clone().into_bytes())));
                },
                _ => (),
            };
        }

        Ok(strings)     
    }

    /// Returns a contatenated string of compiler command line options used when 
    /// building a `Program`.
    pub fn get_compiler_options(&self) -> OclResult<CString> {
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

        CString::new(opts.join(" ").into_bytes()).map_err(|err| OclError::from(err))
    }

    /// Returns the final program source code as a list of strings.
    ///
    /// Order of inclusion:
    /// - includes from `::get_kernel_includes()`
    /// - source from files listed in `self.src_file_names` in reverse order
    /// - raw source from `self.embedded_kernel_source`
    pub fn get_src_strings(&self) -> OclResult<Vec<CString>> {
        let mut src_strings: Vec<CString> = Vec::with_capacity(64);
        let mut src_file_history: HashSet<&String> = HashSet::with_capacity(64);

        src_strings.extend_from_slice(&try!(self.get_kernel_includes()));

        for kfn in self.get_src_file_names().iter().rev() {
            let mut src_str: Vec<u8> = Vec::with_capacity(100000);

            if src_file_history.contains(kfn) { continue; }

            src_file_history.insert(&kfn);

            let valid_kfp = Path::new(kfn);

            // let mut src_file = match File::open(&valid_kfp) {
            //  Err(why) => return Err(format!("Couldn't open '{}': {}", 
            //      kfn, Error::description(&why))),
            //  Ok(file) => file,
            // };

            let mut src_file = try!(File::open(&valid_kfp));

            // match src_file.read_to_end(&mut src_str) {
      //        Err(why) => return Err(format!("Couldn't read '{}': {}", 
      //            kfn, Error::description(&why))),
            //     Ok(_) => (), //println!("{}OCL::BUILD(): parsing {}: {} bytes read.", MT, &file_name, bytes),
            // }

            try!(src_file.read_to_end(&mut src_str));

            src_str.shrink_to_fit();

            src_strings.push(try!(CString::new(src_str)));
        }

        // for elem in self.embedded_kernel_source.iter() {
        //  src_strings.push(try!(CString::new(elem.clone().into_bytes()).map_err(|e| e.to_string())));
        // }
        src_strings.extend_from_slice(&try!(self.get_kernel_includes_eof()));

        Ok(src_strings)
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
    IncludeRawEof(String),
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
//  name: &'static str,
//  val: String,
//  is_kern_header: bool,
// }

// impl BuildOption {
//  pub fn new(name: &'static str, val: i32) -> BuildOption {
//      BuildOption {
//          name: name,
//          val: val.to_string(),
//          is_kern_header: false,
//      }
//  }

//  pub fn with_str_val(name: &'static str, val: String) -> BuildOption {
//      BuildOption {
//          name: name,
//          val: val,
//          is_kern_header: true,
//      }
//  }

//  pub fn to_preprocessor_option_string(&self) -> String {
//      format!(" -D{}={}", self.name, self.val)
//  }

//  pub fn to_define_directive_string(&self) -> String {
//      format!("#define {}  {}\n", self.name, self.val)
//  }
// }
