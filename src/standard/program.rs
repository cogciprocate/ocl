//! An OpenCL program.

use std::ffi::CString;

use error::{Result as OclResult};
use raw::{self, ProgramRaw, DeviceIdRaw, ContextRaw};
use standard::{ProgramBuilder, Context};

/// A program, sometimes referred to as a build.
///
/// To use with multiple devices, create manually with `::from_parts()`.
///
/// # Destruction
///
/// `::release` must be manually called by consumer (temporary).
///
/// [FIXME]: Destruction
///
pub struct Program {
    obj_raw: ProgramRaw,
    // context_obj_raw: ContextRaw,
    // device_ids: Vec<DeviceIdRaw>,
    devices_ids_raw: Vec<DeviceIdRaw>,
}


// [TODO]: ERROR HANDLING
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
            context.obj_raw(), 
            &device_ids)
    }

    /// Returns a new program built from pre-created build components and device
    /// list.
    // [SOMEDAY TODO]: Keep track of line number range for each kernel string and print 
    // out during build failure.
    pub fn from_parts(
                src_strings: Vec<CString>, 
                cmplr_opts: CString, 
                context_obj_raw: ContextRaw, 
                device_ids: &[DeviceIdRaw],
            ) -> OclResult<Program> 
    {
        let obj_raw = try!(raw::create_build_program(context_obj_raw, src_strings, cmplr_opts, 
             device_ids).map_err(|e| e.to_string()));

        Ok(Program {
            obj_raw: obj_raw,
            devices_ids_raw: Vec::from(device_ids),
        })
    }

    /// Returns the associated OpenCL program object.
    pub fn obj_raw(&self) -> ProgramRaw {
        self.obj_raw
    }

    pub fn device_ids_raw(&self) -> &[DeviceIdRaw] {
        &self.devices_ids_raw
    }

    /// Decrements the associated OpenCL program object's reference count.
    // [NOTE]: Do not move this to a Drop impl in case this Program has been cloned.
    pub fn release(&mut self) {
        raw::release_program(self.obj_raw).unwrap();
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        // raw::release_context(self.obj_raw);
        raw::release_program(self.obj_raw).unwrap();
    }
}
