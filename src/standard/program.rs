//! An `OpenCL` program.
use std;
use std::ops::{Deref, DerefMut};
use std::ffi::CString;
use std::io::Read;
use std::fs::File;
use std::path::PathBuf;
use std::collections::HashSet;
use std::convert::Into;

use core::error::{Result as OclResult, Error as OclError};
use core::{self, Program as ProgramCore, Context as ContextCore,
    ProgramInfo, ProgramInfoResult, ProgramBuildInfo, ProgramBuildInfoResult};
#[cfg(feature = "opencl_version_2_1")]
use core::ClVersions;
use standard::{Context, Device, DeviceSpecifier};


/// A build option used by ProgramBuilder.
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

    /// Returns a `BuildOpt::IncludeDefine`.
    pub fn include_def<S: Into<String>>(ident: S, val: String) -> BuildOpt {
        BuildOpt::IncludeDefine {
            ident: ident.into(),
            val: val,
        }
    }
}


/// A builder for `Program`.
///
// [SOMEDAY TODO]: Keep track of line number range for each string and print
// out during build failure.
//
#[derive(Clone, Debug)]
pub struct ProgramBuilder {
    options: Vec<BuildOpt>,
    src_files: Vec<PathBuf>,
    il: Option<Vec<u8>>,
    device_spec: Option<DeviceSpecifier>,
}

impl ProgramBuilder {
    /// Returns a new, empty, build configuration object.
    pub fn new() -> ProgramBuilder {
        ProgramBuilder {
            options: Vec::with_capacity(64),
            src_files: Vec::with_capacity(16),
            il: None,
            device_spec: None,
        }
    }

    /// Returns a newly built Program.
    ///
    ///
    ///
    ///
    /// [TODO]: If the context is associated with more than one device,
    /// check that at least one of those devices has been specified. An empty
    /// device list will cause an `OpenCL` error in that case.
    ///
    /// [TODO]: Check for duplicate devices in the final device list.
    #[cfg(not(feature = "opencl_version_2_1"))]
    pub fn build(self, context: &Context) -> OclResult<Program> {
        let device_list = match self.device_spec {
            Some(ref ds) => ds.to_device_list(context.platform()?)?,
            None => context.devices(),
        };

        match self.il {
            Some(_) => {
                return Err("ocl::ProgramBuilder::build: Unreachable section (IL).".into());
            },
            None => {
                Program::new(
                    try!(self.get_src_strings().map_err(|e| e.to_string())),
                    Some(&device_list[..]),
                    try!(self.get_compiler_options().map_err(|e| e.to_string())),
                    context,

                )
            },
        }
    }


    /// Returns a newly built Program.
    ///
    /// [TODO]: If the context is associated with more than one device,
    /// check that at least one of those devices has been specified. An empty
    /// device list will cause an `OpenCL` error in that case.
    ///
    /// [TODO]: Check for duplicate devices in the final device list.
    #[cfg(feature = "opencl_version_2_1")]
    pub fn build(mut self, context: &Context) -> OclResult<Program> {
        let device_list = match self.device_spec {
            Some(ref ds) => try!(ds.to_device_list(context.platform())),
            None => context.devices().to_owned(),
        };

        match self.il.take() {
            Some(il) => {
                if cfg!(feature = "opencl_version_2_1") {
                    if self.src_files.len() > 0 { return Err("ProgramBuilder::build: \
                        No source files may be set when building with IR.".into()); }

                    Program::with_il(
                        il,
                        Some(&device_list[..]),
                        self.get_compiler_options().map_err(|e| e.to_string())?,
                        context
                    )
                } else {
                    return Err("ocl::ProgramBuilder::build: Unreachable section.".into());
                }
            },
            None => {
                Program::new(
                    try!(self.get_src_strings().map_err(|e| e.to_string())),
                    Some(&device_list[..]),
                    try!(self.get_compiler_options().map_err(|e| e.to_string())),
                    context,
                )
            },
        }
    }

    /// Adds a build option containing a compiler command line definition.
    /// Formatted as `-D {name}={val}`.
    ///
    /// ## Example
    ///
    /// `...cmplr_def("MAX_ITERS", 500)...`
    ///
    pub fn cmplr_def<S: Into<String>>(mut self, name: S, val: i32) -> ProgramBuilder {
        self.options.push(BuildOpt::cmplr_def(name, val));
        self
    }

    /// Adds a build option containing a raw compiler command line parameter.
    /// Formatted as `{}` (exact text).
    ///
    /// ## Example
    ///
    /// `...cmplr_opt("-g")...`
    ///
    pub fn cmplr_opt<S: Into<String>>(mut self, co: S) -> ProgramBuilder {
        self.options.push(BuildOpt::CmplrOther(co.into()));
        self
    }

    /// Pushes pre-created build option to the list of options.
    pub fn bo(mut self, bo: BuildOpt) -> ProgramBuilder {
        self.options.push(bo);
        self
    }

    /// Adds the contents of a file to the program.
    pub fn src_file<P: Into<PathBuf>>(mut self, file_path: P) -> ProgramBuilder {
        let file_path = file_path.into();
        assert!(file_path.is_file(), "ProgramBuilder::src_file(): Source file error: \
            '{}' does not exist.", file_path.display());
        self.src_files.push(file_path);
        self
    }

    /// Adds raw text to the program source.
    pub fn src<S: Into<String>>(mut self, src: S) -> ProgramBuilder {
        self.options.push(BuildOpt::IncludeRawEof(src.into()));
        self
    }

    /// Adds SPIR-V or an implementation-defined intermediate language to this program.
    ///
    /// Any source files or source text added to this build will cause an
    /// error upon building.
    ///
    /// Use the `include_bytes!` macro to include source code from a file statically.
    ///
    /// [TODO]: Future addition: Allow IL to be loaded directly from a file
    /// in the same way that text source is.
    ///
    #[cfg(feature = "opencl_version_2_1")]
    pub fn il(mut self, il: Vec<u8>) -> ProgramBuilder {
        self.il = Some(il);
        self
    }

    /// Specifies a list of devices to build this program on. The devices must
    /// be associated with the context passed to `::build` later on.
    ///
    /// Devices may be specified in any number of ways including simply
    /// passing a device or slice of devices. See the [`impl
    /// From`][device_specifier_from] section of
    /// [`DeviceSpecifier`][device_specifier] for more information.
    ///
    ///
    /// ## Panics
    ///
    /// Devices must not have already been specified.
    ///
    /// [device_specifier_from]: enum.DeviceSpecifier.html#method.from
    /// [device_specifier]: enum.DeviceSpecifier.html
    ///
    pub fn devices<D: Into<DeviceSpecifier>>(mut self, device_spec: D)
            -> ProgramBuilder
    {
        assert!(self.device_spec.is_none(), "ocl::ProgramBuilder::devices(): Devices already specified");
        self.device_spec = Some(device_spec.into());
        self
    }

    /// Returns the devices specified to be associated the program.
    pub fn get_device_spec(&self) -> &Option<DeviceSpecifier> {
        &self.device_spec
    }

    /// Returns a concatenated string of command line options to be passed to
    /// the compiler when building this program.
    pub fn get_compiler_options(&self) -> OclResult<CString> {
        let mut opts: Vec<String> = Vec::with_capacity(64);

        for option in &self.options {
            match *option {
                BuildOpt::CmplrDefine { ref ident, ref val } => {
                    opts.push(format!("-D {}={}", ident, val))
                },

                BuildOpt::CmplrInclDir { ref path } => {
                    opts.push(format!("-I {}", path))
                },

                BuildOpt::CmplrOther(ref s) => {
                    opts.push(s.clone())
                },

                _ => (),
            }
        }

        CString::new(opts.join(" ").into_bytes()).map_err(OclError::from)
    }

    /// Returns the final program source code as a list of strings.
    ///
    /// ### Order of Inclusion
    ///
    /// 1. Macro definitions and code strings specified by a
    ///    `BuildOpt::IncludeDefine` or `BuildOpt::IncludeRaw` via `::bo`
    /// 2. Contents of files specified via `::src_file`
    /// 3. Contents of strings specified via `::src` or a
    ///   `BuildOpt::IncludeRawEof` via `::bo`
    ///
    pub fn get_src_strings(&self) -> OclResult<Vec<CString>> {
        let mut src_strings: Vec<CString> = Vec::with_capacity(64);
        let mut src_file_history: HashSet<PathBuf> = HashSet::with_capacity(64);

        src_strings.extend_from_slice(&try!(self.get_includes()));

        for srcpath in &self.src_files {
            let mut src_bytes: Vec<u8> = Vec::with_capacity(100000);

            if src_file_history.contains(srcpath) { continue; }
            src_file_history.insert(srcpath.clone());

            let mut src_file_handle = try!(File::open(srcpath));

            try!(src_file_handle.read_to_end(&mut src_bytes));
            src_bytes.shrink_to_fit();
            src_strings.push(try!(CString::new(src_bytes)));
        }

        src_strings.extend_from_slice(&try!(self.get_includes_eof()));
        src_strings.shrink_to_fit();
        Ok(src_strings)
    }

    /// Parses `self.options` for options intended for inclusion at the beginning of
    /// the final program source and returns them as a list of strings.
    ///
    /// Generally used for #define directives, constants, etc. Normally called from
    /// `::get_src_strings()`.
    fn get_includes(&self) -> OclResult<Vec<CString>> {
        let mut strings = Vec::with_capacity(64);
        strings.push(try!(CString::new("\n".as_bytes())));

        for option in &self.options {
            match *option {
                BuildOpt::IncludeDefine { ref ident, ref val } => {
                    strings.push(try!(CString::new(format!("#define {}  {}\n", ident, val)
                        .into_bytes())));
                },
                BuildOpt::IncludeRaw(ref text) => {
                    strings.push(try!(CString::new(text.clone().into_bytes())));
                },
                _ => (),
            };

        }

        strings.shrink_to_fit();
        Ok(strings)
    }

    /// Parses `self.options` for options intended for inclusion at the end of
    /// the final program source and returns them as a list of strings.
    fn get_includes_eof(&self) -> OclResult<Vec<CString>> {
        let mut strings = Vec::with_capacity(64);
        strings.push(try!(CString::new("\n".as_bytes())));

        for option in &self.options {
            if let BuildOpt::IncludeRawEof(ref text) = *option {
                strings.push(try!(CString::new(text.clone().into_bytes())));
            }
        }

        strings.shrink_to_fit();
        Ok(strings)
    }
}




/// A program from which kernels can be created from.
///
/// To use with multiple devices, create manually with `::from_parts()`.
///
/// ## Destruction
///
/// Handled automatically. Feel free to store, clone, and share among threads
/// as you please.
///
#[derive(Clone, Debug)]
pub struct Program(ProgramCore);

impl Program {
    /// Returns a new `ProgramBuilder`.
    pub fn builder() -> ProgramBuilder {
        ProgramBuilder::new()
    }

    /// Returns a new program built from pre-created build components and device
    /// list.
    ///
    /// Prefer `::builder` to create a new `Program`.
    ///
    pub fn new(src_strings: Vec<CString>, device_ids: Option<&[Device]>, cmplr_opts: CString,
            context_obj_core: &ContextCore) -> OclResult<Program>
    {
        let obj_core = try!(core::create_build_program(context_obj_core, &src_strings, &cmplr_opts,
             device_ids));

        Ok(Program(obj_core))
    }

    /// Returns a new program built from pre-created build components and device
    /// list for programs with intermediate language byte source.
    #[cfg(feature = "opencl_version_2_1")]
    pub fn with_il(il: Vec<u8>, device_ids: Option<&[Device]>, cmplr_opts: CString,
            context_obj_core: &ContextCore) -> OclResult<Program>
    {
        let device_versions = context_obj_core.device_versions()?;

        let obj_core = core::create_program_with_il(context_obj_core, &il, Some(&device_versions))?;

        core::build_program(&obj_core, device_ids, &cmplr_opts, None, None)?;

        let devices = context_obj_core.devices()?.into_iter().map(|d| d.into()).collect();

        Ok(Program {
            obj_core: obj_core,
        })
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[deprecated(since="0.13.0", note="Use `::core` instead.")]
    pub fn core_as_ref(&self) -> &ProgramCore {
        &self.0
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn core(&self) -> &ProgramCore {
        &self.0
    }

    /// Returns the list of devices associated with this program.
    #[deprecated(since="0.13.0", note="Use `::info` with `ProgramInfo::Devices` instead.")]
    pub fn devices(&self) -> Vec<Device> {
        unimplemented!();
    }

    /// Returns info about this program.
    pub fn info(&self, info_kind: ProgramInfo) -> ProgramInfoResult {
        core::get_program_info(&self.0, info_kind)
    }

    /// Returns info about this program's build.
    ///
    /// TODO: Check that device is valid.
    pub fn build_info(&self, device: Device, info_kind: ProgramBuildInfo) -> ProgramBuildInfoResult {
        core::get_program_build_info(&self.0, &device, info_kind)
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Program")
            .field("ReferenceCount", &self.info(ProgramInfo::ReferenceCount))
            .field("Context", &self.info(ProgramInfo::Context))
            .field("NumDevices", &self.info(ProgramInfo::NumDevices))
            .field("Devices", &self.info(ProgramInfo::Devices))
            .field("Source", &self.info(ProgramInfo::Source))
            .field("BinarySizes", &self.info(ProgramInfo::BinarySizes))
            .field("Binaries", &self.info(ProgramInfo::Binaries))
            .field("NumKernels", &self.info(ProgramInfo::NumKernels))
            .field("KernelNames", &self.info(ProgramInfo::KernelNames))
            .finish()
    }
}


impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl Deref for Program {
    type Target = ProgramCore;

    fn deref(&self) -> &ProgramCore {
        &self.0
    }
}

impl DerefMut for Program {
    fn deref_mut(&mut self) -> &mut ProgramCore {
        &mut self.0
    }
}
