//! An OpenCL program.
use std;
use std::ffi::CString;

use error::{Result as OclResult};
use core::{self, Program as ProgramCore, Context as ContextCore,
    ProgramInfo, ProgramInfoResult, ProgramBuildInfo, ProgramBuildInfoResult};
use standard::{self, ProgramBuilder, Context, Device};

/// A program, sometimes referred to as a build.
///
/// To use with multiple devices, create manually with `::from_parts()`.
///
/// # Destruction
///
/// Handled automatically. Feel free to store, clone, and share among threads
/// as you please.
///
#[derive(Clone, Debug)]
pub struct Program {
    obj_core: ProgramCore,
    devices: Vec<Device>,
}

impl Program {
    /// Returns a new `ProgramBuilder`.
    pub fn builder() -> ProgramBuilder {
        ProgramBuilder::new()
    }

    /// Returns a new program.
    pub fn new(program_builder: ProgramBuilder, context: &Context, device_idxs: Vec<usize>
            ) -> OclResult<Program> 
    {
        let device_ids = context.resolve_device_idxs(&device_idxs);

        Program::from_parts(
            try!(program_builder.get_src_strings().map_err(|e| e.to_string())), 
            try!(program_builder.get_compiler_options().map_err(|e| e.to_string())), 
            context.core_as_ref(), 
            &device_ids)
    }

    /// Returns a new program built from pre-created build components and device
    /// list.
    // [SOMEDAY TODO]: Keep track of line number range for each kernel string and print 
    // out during build failure.
    pub fn from_parts(
                src_strings: Vec<CString>, 
                cmplr_opts: CString, 
                context_obj_core: &ContextCore, 
                device_ids: &[Device],
            ) -> OclResult<Program> 
    {
        let obj_core = try!(core::create_build_program(context_obj_core, src_strings, cmplr_opts, 
             device_ids).map_err(|e| e.to_string()));

        Ok(Program {
            obj_core: obj_core,
            devices: Vec::from(device_ids),
        })
    }

    /// Returns the associated OpenCL program object.
    pub fn core_as_ref(&self) -> &ProgramCore {
        &self.obj_core
    }

    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Returns info about this program.
    pub fn info(&self, info_kind: ProgramInfo) -> ProgramInfoResult {
        match core::get_program_info(&self.obj_core, info_kind) {
            Ok(res) => res,
            Err(err) => ProgramInfoResult::Error(Box::new(err)),
        }        
    }

    /// Returns info about this program's build.
    ///
    /// TODO: Check that device is valid.
    pub fn build_info(&self, device: Device, info_kind: ProgramBuildInfo) -> ProgramBuildInfoResult {
        match core::get_program_build_info(&self.obj_core, &device, info_kind) {
            Ok(res) => res,
            Err(err) => ProgramBuildInfoResult::Error(Box::new(err)),
        }        
    }
}


impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // write!(f, "{}", &self.to_string())
        let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
            ("\n", "\n", "\n")
        } else {
            ("{ ", ", ", " }")
        };

        // ReferenceCount = cl_h::CL_PROGRAM_REFERENCE_COUNT as isize,
        // Context = cl_h::CL_PROGRAM_CONTEXT as isize,
        // NumDevices = cl_h::CL_PROGRAM_NUM_DEVICES as isize,
        // Devices = cl_h::CL_PROGRAM_DEVICES as isize,
        // Source = cl_h::CL_PROGRAM_SOURCE as isize,
        // BinarySizes = cl_h::CL_PROGRAM_BINARY_SIZES as isize,
        // Binaries = cl_h::CL_PROGRAM_BINARIES as isize,
        // NumKernels = cl_h::CL_PROGRAM_NUM_KERNELS as isize,
        // KernelNames = cl_h::CL_PROGRAM_KERNEL_NAMES as isize,

        write!(f, "[Program]: {b}\
                ReferenceCount: {}{d}\
                Context: {}{d}\
                NumDevices: {}{d}\
                Devices: {}{d}\
                Source: {}{d}\
                BinarySizes: {}{d}\
                Binaries: {}{d}\
                NumKernels: {}{d}\
                KernelNames: {}{e}\
            ",
            self.info(ProgramInfo::ReferenceCount),
            self.info(ProgramInfo::Context),
            self.info(ProgramInfo::NumDevices),
            self.info(ProgramInfo::Devices),
            self.info(ProgramInfo::Source),
            self.info(ProgramInfo::BinarySizes),
            self.info(ProgramInfo::Binaries),
            self.info(ProgramInfo::NumKernels),
            self.info(ProgramInfo::KernelNames),
            b = begin,
            d = delim,
            e = end,
        )
    }
}

