//! An `OpenCL` program.
use std;
use std::ops::{Deref, DerefMut};
use std::ffi::CString;
use std::io::Read;
use std::fs::File;
use std::path::PathBuf;
use std::collections::HashSet;
use std::convert::Into;


use core::{self, Result as OclCoreResult, Program as ProgramCore, Context as ContextCore,
    ProgramInfo, ProgramInfoResult, ProgramBuildInfo, ProgramBuildInfoResult};
#[cfg(feature = "opencl_version_2_1")]
use core::ClVersions;
use error::{Result as OclResult, Error as OclError};
use standard::{Context, Device, DeviceSpecifier};


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
    pub fn builder<'b>() -> ProgramBuilder<'b> {
        ProgramBuilder::new()
    }

    /// Returns a new program built from pre-created build components and device
    /// list.
    ///
    /// Prefer `::builder` to create a new `Program`.
    ///
    pub fn with_source(context: &ContextCore, src_strings: &[CString],
            devices: Option<&[Device]>, cmplr_opts: &CString) -> OclResult<Program> {
        let program = core::create_program_with_source(context, src_strings)?;
        core::build_program(&program, devices, cmplr_opts, None, None)?;
        Ok(Program(program))
    }

    /// Returns a new program built from pre-created build components and device
    /// list.
    ///
    /// Prefer `::builder` to create a new `Program`.
    ///
    pub fn with_binary(context: &ContextCore, devices: &[Device],
            binaries: &[&[u8]], cmplr_opts: &CString) -> OclResult<Program> {
        let program = core::create_program_with_binary(context, devices, binaries)?;
        core::build_program(&program, Some(devices), cmplr_opts, None, None)?;
        Ok(Program(program))
    }

    /// Returns a new program built from pre-created build components and device
    /// list for programs with intermediate language byte source.
    #[cfg(feature = "opencl_version_2_1")]
    pub fn with_il(il: &[u8], devices: Option<&[Device]>, cmplr_opts: &CString,
            context: &ContextCore) -> OclResult<Program> {
        let device_versions = context.device_versions()?;
        let program = core::create_program_with_il(context, il, Some(&device_versions))?;
        core::build_program(&program, devices, cmplr_opts, None, None)?;

        Ok(Program(program))
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn as_core(&self) -> &ProgramCore {
        &self.0
    }

    /// Returns info about this program.
    pub fn info(&self, info_kind: ProgramInfo) -> OclCoreResult<ProgramInfoResult> {
        core::get_program_info(&self.0, info_kind)
    }

    /// Returns info about this program's build.
    ///
    /// * TODO: Check that device is valid.
    pub fn build_info(&self, device: Device, info_kind: ProgramBuildInfo)
            -> OclCoreResult<ProgramBuildInfoResult> {
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

impl From<ProgramCore> for Program {
    fn from(core: ProgramCore) -> Program {
        Program(core)
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


/// A build option used by ProgramBuilder.
///
/// Strings intended for use either by the compiler as a command line switch
/// or for inclusion in the final build source code.
///
/// A few of the often used variants have constructors for convenience.
///
/// * [FIXME] TODO: Explain how each variant is used.
///
/// * [FIXME] TODO: Examples.
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


/// Options for program creation.
#[allow(dead_code)]
#[derive(Clone, Debug)]
enum CreateWith<'b> {
    None,
    Source(Vec<PathBuf>),
    Binaries(&'b[&'b [u8]]),
    Il(&'b [u8]),
}


/// A builder for `Program`.
///
// * [SOMEDAY TODO]: Keep track of line number range for each string and print
// out during build failure.
//
#[must_use = "builders do nothing unless '::build' is called"]
#[derive(Clone, Debug)]
pub struct ProgramBuilder<'b> {
    options: Vec<BuildOpt>,
    with: CreateWith<'b>,
    device_spec: Option<DeviceSpecifier>,
}

impl<'b> ProgramBuilder<'b> {
    /// Returns a new, empty, build configuration object.
    pub fn new() -> ProgramBuilder<'b> {
        ProgramBuilder {
            options: Vec::with_capacity(64),
            with: CreateWith::None,
            device_spec: None,
        }
    }

    /// Adds a build option containing a compiler command line definition.
    /// Formatted as `-D {name}={val}`.
    ///
    /// ## Example
    ///
    /// `...cmplr_def("MAX_ITERS", 500)...`
    ///
    pub fn cmplr_def<'a, S: Into<String>>(&'a mut self, name: S, val: i32) -> &'a mut ProgramBuilder<'b> {
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
    pub fn cmplr_opt<'a, S: Into<String>>(&'a mut self, co: S) -> &'a mut ProgramBuilder<'b> {
        self.options.push(BuildOpt::CmplrOther(co.into()));
        self
    }

    /// Pushes pre-created build option to the list of options.
    ///
    /// If either `::il` or `::binaries` are used and raw source is added, it
    /// will be ignored.
    pub fn bo<'a>(&'a mut self, bo: BuildOpt) -> &'a mut ProgramBuilder<'b> {
        self.options.push(bo);
        self
    }

    /// Adds the contents of a file to the program.
    //
    // TODO: Deprecate
    pub fn src_file<'a, P: Into<PathBuf>>(&'a mut self, file_path: P) -> &'a mut ProgramBuilder<'b> {
        self.source_file(file_path)
    }

    /// Opens a file and adds its contents to the program source.
    pub fn source_file<'a, P: Into<PathBuf>>(&'a mut self, file_path: P) -> &'a mut ProgramBuilder<'b> {
        let file_path = file_path.into();
        assert!(file_path.is_file(), "ProgramBuilder::src_file(): Source file error: \
            '{}' does not exist.", file_path.display());
        match self.with {
            CreateWith::None => {
                let mut paths = Vec::with_capacity(8);
                paths.push(file_path);
                self.with = CreateWith::Source(paths);
            }
            CreateWith::Source(ref mut paths) => paths.push(file_path),
            _ => panic!("Source may not be used with binaries or il."),
        }
        self
    }

    /// Adds raw text to the program source.
    //
    // TODO: Deprecate
    pub fn src<'a, S: Into<String>>(&'a mut self, src: S) -> &'a mut ProgramBuilder<'b> {
        self.source(src)
    }

    /// Adds raw text to the program source.
    //
    // [TODO]: Possibly accept Into<CString>
    pub fn source<'a, S: Into<String>>(&'a mut self, src: S) -> &'a mut ProgramBuilder<'b> {
        match self.with {
            CreateWith::None => {
                self.with = CreateWith::Source(Vec::with_capacity(8));
                self.options.push(BuildOpt::IncludeRawEof(src.into()));
            }
            CreateWith::Source(_) => {
                self.options.push(BuildOpt::IncludeRawEof(src.into()));
            }
            _ => panic!("Source may not be used with binaries or il."),
        }

        self
    }

    /// Adds a binary to be loaded.
    ///
    /// There must be one binary for each device listed in `::devices`.
    pub fn binaries<'a>(&'a mut self, bins: &'b[&'b [u8]]) -> &'a mut ProgramBuilder<'b> {
        match self.with {
            CreateWith::None => self.with = CreateWith::Binaries(bins),
            CreateWith::Binaries(_) => panic!("Binaries have already been specified."),
            _ => panic!("Binaries may not be used with source or il."),
        }
        self
    }

    /// Adds

    /// Adds SPIR-V or an implementation-defined intermediate language to this program.
    ///
    /// Any source files or source text added to this build will cause an
    /// error upon building.
    ///
    /// Use the `include_bytes!` macro to include source code from a file statically.
    ///
    /// * TODO: Future addition: Allow IL to be loaded directly from a file
    /// in the same way that text source is.
    ///
    #[cfg(feature = "opencl_version_2_1")]
    pub fn il<'a>(&'a mut self, il: &'b [u8]) -> &'a mut ProgramBuilder<'b> {
        match self.with {
            CreateWith::None => self.with = CreateWith::Il(il),
            CreateWith::Il(_) => panic!("Il has already been specified."),
            _ => panic!("Il may not be used with source or binaries."),
        }
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
    pub fn devices<'a, D: Into<DeviceSpecifier>>(&'a mut self, device_spec: D)
            -> &'a mut ProgramBuilder<'b> {
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

    /// Parses `self.options` for options intended for inclusion at the beginning of
    /// the final program source and returns them as a list of strings.
    ///
    /// Generally used for #define directives, constants, etc. Normally called from
    /// `::get_src_strings()`.
    fn get_includes(&self) -> OclResult<Vec<CString>> {
        let mut strings = Vec::with_capacity(64);
        strings.push(CString::new("\n".as_bytes())?);

        for option in &self.options {
            match *option {
                BuildOpt::IncludeDefine { ref ident, ref val } => {
                    strings.push(CString::new(format!("#define {}  {}\n", ident, val)
                        .into_bytes())?);
                },
                BuildOpt::IncludeRaw(ref text) => {
                    strings.push(CString::new(text.clone().into_bytes())?);
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
        strings.push(CString::new("\n".as_bytes())?);

        for option in &self.options {
            if let BuildOpt::IncludeRawEof(ref text) = *option {
                strings.push(CString::new(text.clone().into_bytes())?);
            }
        }

        strings.shrink_to_fit();
        Ok(strings)
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

        src_strings.extend_from_slice(&self.get_includes()?);

        let src_paths = match self.with {
            CreateWith::Source(ref paths) => paths,
            _ => panic!("Cannot build program. No source specified."),
        };

        for src_path in src_paths {
            let mut src_bytes: Vec<u8> = Vec::with_capacity(100000);

            if src_file_history.contains(src_path) { continue; }
            src_file_history.insert(src_path.clone());

            let mut src_file_handle = File::open(src_path)?;

            src_file_handle.read_to_end(&mut src_bytes)?;
            src_bytes.shrink_to_fit();
            src_strings.push(CString::new(src_bytes)?);
        }

        src_strings.extend_from_slice(&self.get_includes_eof()?);
        src_strings.shrink_to_fit();
        Ok(src_strings)
    }

    /// Returns a newly built Program.
    //
    // * TODO: If the context is associated with more than one device,
    // check that at least one of those devices has been specified. An empty
    // device list will cause an `OpenCL` error in that case.
    //
    // * TODO: Check for duplicate devices in the final device list.
    #[cfg(not(feature = "opencl_version_2_1"))]
    pub fn build(&self, context: &Context) -> OclResult<Program> {
        let device_list = match self.device_spec {
            Some(ref ds) => ds.to_device_list(context.platform()?)?,
            None => context.devices(),
        };

        match self.with {
            CreateWith::Il(_) => {
                return Err("ocl::ProgramBuilder::build: Unreachable section (IL).".into());
            },
            CreateWith::Source(_) => {
                Program::with_source(
                    context,
                    &self.get_src_strings()?,
                    Some(&device_list[..]),
                    &self.get_compiler_options()?,
                ).map_err(OclError::from)
            },
            CreateWith::Binaries(bins) => {
                Program::with_binary(
                    context,
                    &device_list[..],
                    bins,
                    &self.get_compiler_options()?,
                )
            },
            CreateWith::None => return Err("Unable to build program: no source, binary, \
                or IL has been specified".into()),
        }
    }

    /// Returns a newly built Program.
    //
    // * TODO: If the context is associated with more than one device,
    // check that at least one of those devices has been specified. An empty
    // device list will cause an `OpenCL` error in that case.
    //
    // * TODO: Check for duplicate devices in the final device list.
    #[cfg(feature = "opencl_version_2_1")]
    pub fn build(&self, context: &Context) -> OclResult<Program> {
        let device_list = match self.device_spec {
            Some(ref ds) => ds.to_device_list(context.platform()?)?,
            None => context.devices().to_owned(),
        };

        match self.with {
            CreateWith::Il(il) => {
                Program::with_il(
                    il,
                    Some(&device_list[..]),
                    &self.get_compiler_options()?,
                    context
                )
            },
            CreateWith::Source(_) => {
                Program::with_source(
                    context,
                    &self.get_src_strings()?,
                    Some(&device_list[..]),
                    &self.get_compiler_options()?,
                )
            },
            CreateWith::Binaries(bins) => {
                Program::with_binary(
                    context,
                    &device_list[..],
                    bins,
                    &self.get_compiler_options()?,
                )
            },
            CreateWith::None => return Err("Unable to build program: no source, binary, \
                or IL has been specified".into()),
        }
    }
}

