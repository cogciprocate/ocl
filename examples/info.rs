//! Get information about various stuff.
//!
//! WORK IN PROGRESS

extern crate ocl;
use ocl::{Context, Program};
use ocl::raw::{self, PlatformInfo, DeviceInfo, ContextInfo};

static TAB: &'static str = "    ";

static SRC: &'static str = r#"
	__kernel void multiply(__global float* buffer, float coeff) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
	let context = Context::new(None, None).unwrap();
	let program = Program::builder().src(SRC).build(&context).unwrap();

	print!("\n");
	println!("############################# OpenCL Information #############################");

    // ##################################################
    // #################### PLATFORM ####################
    // ##################################################

	println!("Platform:\n\
		{t}Profile:     {}\n\
		{t}Version:     {}\n\
		{t}Name:        {}\n\
		{t}Vendor:      {}\n\
		{t}Extensions:  {}\
		",
		raw::get_platform_info(context.platform_obj_raw(), PlatformInfo::Profile).unwrap(),
		raw::get_platform_info(context.platform_obj_raw(), PlatformInfo::Version).unwrap(),
		raw::get_platform_info(context.platform_obj_raw(), PlatformInfo::Name).unwrap(),
		raw::get_platform_info(context.platform_obj_raw(), PlatformInfo::Vendor).unwrap(),
		raw::get_platform_info(context.platform_obj_raw(), PlatformInfo::Extensions).unwrap(),
		t = TAB,
	);

    // ##################################################
    // #################### DEVICES #####################
    // ##################################################

    // TODO: COMPLETE THIS SECTION

    for device in context.device_ids().iter() {
	    print!("\n");
	    println!("Device:\n\
			{t}Name:     {}\
			",
			raw::get_device_info(device.clone(), DeviceInfo::VendorId).unwrap(),
			t = TAB,
		);
	    print!("\n");
    }


    // ##################################################
    // #################### CONTEXT #####################
    // ##################################################

    print!("\n");
    println!("Context:\n\
		{t}Reference Count:  {}\n\
		{t}Devices:          {}\n\
		{t}Properties:       {}\n\
		{t}Device Count:     {}\
		",
		raw::get_context_info(context.obj_raw(), ContextInfo::ReferenceCount).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::Devices).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::Properties).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::NumDevices).unwrap(),
		t = TAB,
	);
    print!("\n");
}
