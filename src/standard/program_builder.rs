//! The builder for `Program`.

use std::io::Read;
use std::fs::File;
use std::path::Path;
use std::ffi::{CString};
use std::collections::HashSet;
use std::convert::Into;

use error::{Result as OclResult, Error as OclError};
// use core::{DeviceId as DeviceIdCore};
use standard::{Device, Context, Program};


/// A build option.
///
/// Strings intended for use either by the compiler as a command line switch
/// or for inclusion in the final build source code.
///
/// A few of the often used variants have constructors for convenience.
///
/// [FIXME] TODO: Explain how each variant is used.
///
/// [FIXME] TODO: Examples.
#[derive(Clone, Debug)]
pub enum BuildOpt {
    CmplrDefine { ident: String, val: String },
    CmplrInclDir { path: String },
    CmplrOther(String),
    IncludeDefine { ident: String, val: String },
    IncludeCore(String),
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


/// A builder for `Program`.
#[derive(Clone, Debug)]
pub struct ProgramBuilder {
    options: Vec<BuildOpt>,
    src_file_names: Vec<String>,
    device_idxs: Vec<usize>,
    devices: Vec<Device>,
    // embedded_kernel_source: Vec<String>,
}

impl ProgramBuilder {
    /// Returns a new, empty, build configuration object.
    pub fn new() -> ProgramBuilder {
        ProgramBuilder {
            options: Vec::with_capacity(64),
            src_file_names: Vec::with_capacity(32),
            device_idxs: Vec::with_capacity(8),
            devices: Vec::with_capacity(8),
            // embedded_kernel_source: Vec::with_capacity(32),
        }
    }

    /// Returns a newly built Program.
    ///
    /// TODO: If the context is associated with more than one device,
    /// check that at least one of those devices has been specified. An empty
    /// device list will cause an OpenCL error in that case.
    ///
    /// TODO: Check for duplicate devices in the final device list.
    pub fn build(&self, context: &Context) -> OclResult<Program> {
        let mut device_list: Vec<Device> = self.devices.iter().map(|d| d.clone()).collect();
        device_list.extend_from_slice(&context.resolve_device_idxs(&self.device_idxs));

        Program::from_parts(
            try!(self.get_src_strings().map_err(|e| e.to_string())), 
            try!(self.get_compiler_options().map_err(|e| e.to_string())), 
            context.core_as_ref(), 
            &device_list[..])
    }

    /// Adds a build option containing a compiler command line definition.
    /// Formatted as `-D {name}={val}`.
    pub fn cmplr_def<'p>(&'p mut self, name: &'static str, val: i32) -> &'p mut ProgramBuilder {
        self.options.push(BuildOpt::cmplr_def(name, val));
        self
    }

    /// Adds a build option containing a core compiler command line parameter. 
    /// Formatted as `{co}` (exact text).
    pub fn cmplr_opt<'p>(&'p mut self, co: &'static str) -> &'p mut ProgramBuilder {
        self.options.push(BuildOpt::cmplr_opt(co));
        self
    }

    /// Pushes pre-created build option to the list.
    pub fn bo<'p>(&'p mut self, bo: BuildOpt) -> &'p mut ProgramBuilder {
        self.options.push(bo);
        self
    }

    /// Adds a kernel file to the list of included sources.
    pub fn src_file<'p, S: Into<String>>(&'p mut self, file_name: S) -> &'p mut ProgramBuilder {
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
    ///
    /// If your system has 4 OpenGL devices and you want to include them all:
    /// ```
    /// let program = program::builder()
    ///     .src(source_str)
    ///     .device_idxs(vec![0, 1, 2, 3])
    ///     .build(context);
    /// ```
    /// Out of range device indexes will simply round-robin around to 0 and
    /// count up again (modulo).
    pub fn device_idxs<'p>(&'p mut self, device_idxs: Vec<usize>) -> &'p mut ProgramBuilder {
        self.device_idxs.extend_from_slice(&device_idxs);
        self
    }

    /// Specify a list of devices to build this program on. The devices must be 
    /// associated with the context passed to `::build` later on.
    pub fn devices<'p>(&'p mut self, devices: Vec<Device>) -> &'p mut ProgramBuilder {
        self.devices.extend_from_slice(&devices);
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
                &BuildOpt::IncludeCore(ref text) => {
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
    /// - core source from `self.embedded_kernel_source`
    pub fn get_src_strings(&self) -> OclResult<Vec<CString>> {
        let mut src_strings: Vec<CString> = Vec::with_capacity(64);
        let mut src_file_history: HashSet<&String> = HashSet::with_capacity(64);

        src_strings.extend_from_slice(&try!(self.get_kernel_includes()));

        for kfn in self.get_src_file_names().iter().rev() {
            let mut src_str: Vec<u8> = Vec::with_capacity(100000);

            if src_file_history.contains(kfn) { continue; }
            src_file_history.insert(&kfn);

            let valid_kfp = Path::new(kfn);
            let mut src_file = try!(File::open(&valid_kfp));

            try!(src_file.read_to_end(&mut src_str));
            src_str.shrink_to_fit();
            src_strings.push(try!(CString::new(src_str)));
        }

        src_strings.extend_from_slice(&try!(self.get_kernel_includes_eof()));

        Ok(src_strings)
    }
}
