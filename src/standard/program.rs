//! An OpenCL program.
use std;
use std::ops::{Deref, DerefMut};
use std::ffi::CString;
use error::{Result as OclResult};
use core::{self, Program as ProgramCore, Context as ContextCore,
    ProgramInfo, ProgramInfoResult, ProgramBuildInfo, ProgramBuildInfoResult};
use standard::{ProgramBuilder, Context, Device};

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
            context, 
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
        let obj_core = try!(core::create_build_program(context_obj_core, &src_strings, &cmplr_opts, 
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
        &self.obj_core
    }
}

impl DerefMut for Program {
    fn deref_mut(&mut self) -> &mut ProgramCore {
        &mut self.obj_core
    }
}
