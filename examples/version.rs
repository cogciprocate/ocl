//! Parses the first device's OpenCL version and prints.
//!
//! This is more of a test than an example but whatever.
//!
//! [TODO (easy)]: Print all the devices on all the platforms.

extern crate ocl_core as core;

use std::ffi::CString;
// use core::{ContextProperties, KernelArg};

fn main() {
    use core::ClVersions;

    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    let platform_id = core::default_platform().unwrap();
    let device_ids = core::get_device_ids(&platform_id, None, None).unwrap();
    let device_id = device_ids[0];
    let context_properties = core::ContextProperties::new().platform(platform_id);
    let context = core::create_context(Some(&context_properties),
        &[device_id], None, None).unwrap();
    let src_cstring = CString::new(src).unwrap();
    let program = core::create_program_with_source(&context, &[src_cstring]).unwrap();
    core::build_program(&program, &[device_id], &CString::new("").unwrap(),
        None, None).unwrap();
    // let queue = core::create_command_queue(&context, &device_id).unwrap();
    // let dims = [2 << 20, 1, 1];

    let dv0 = core::get_device_info(&device_id, core::DeviceInfo::Version);
    println!("Pre-Parse: 'DeviceInfo::Version': {}", dv0);

    // let dv1 = core::get_device_version(&device_id).unwrap();
    let dv1 = device_id.device_versions().unwrap();
    println!("Parsed: 'core::get_device_version()': {}", dv1.first().unwrap());
}