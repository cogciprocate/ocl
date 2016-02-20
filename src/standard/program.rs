//! An OpenCL program.

use std::ffi::CString;

use error::{Result as OclResult};
use core::{self, Program as ProgramCore, DeviceId as DeviceIdCore, Context as ContextCore};
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
#[derive(Clone, Debug)]
pub struct Program {
    obj_core: ProgramCore,
    // context_obj_core: ContextCore,
    // device_ids: Vec<DeviceIdCore>,
    devices_ids_core: Vec<DeviceIdCore>,
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
                device_ids: &[DeviceIdCore],
            ) -> OclResult<Program> 
    {
        let obj_core = try!(core::create_build_program(context_obj_core, src_strings, cmplr_opts, 
             device_ids).map_err(|e| e.to_string()));

        Ok(Program {
            obj_core: obj_core,
            devices_ids_core: Vec::from(device_ids),
        })
    }

    /// Returns the associated OpenCL program object.
    pub fn core_as_ref(&self) -> &ProgramCore {
        &self.obj_core
    }

    pub fn devices_core_as_ref(&self) -> &[DeviceIdCore] {
        &self.devices_ids_core
    }

    // /// Decrements the associated OpenCL program object's reference count.
    // // [NOTE]: Do not move this to a Drop impl in case this Program has been cloned.
    // pub fn release(&mut self) {
    //     core::release_program(self.obj_core).unwrap();
    // }
}

// impl Drop for Program {
//     fn drop(&mut self) {
//         // println!("DROPPING PROGRAM");
//         core::release_program(self.obj_core).unwrap();
//     }
// }
