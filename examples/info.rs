//! [WORK IN PROGRESS] Get information about all the things.
//!

extern crate ocl;

use ocl::{Context, Program, Queue};
use ocl::raw::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo};

static TAB: &'static str = "    ";
static SRC: &'static str = r#"
	__kernel void multiply(__global float* buffer, float coeff) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
	let context = Context::new(None, None).unwrap();
	let queue = Queue::new(&context, None);
	let program = Program::builder().src(SRC).build(&context).unwrap();


	println!("############### OpenCL [Default Platform] [Default Device] Info ################");
	print!("\n");

    // ##################################################
    // #################### PLATFORM ####################
    // ##################################################

	println!("Platform:\n\
			{t}Profile:     {}\n\
			{t}Version:     {}\n\
			{t}Name:        {}\n\
			{t}Vendor:      {}\n\
			{t}Extensions:  {}\n\
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

    // [FIXME]: Complete this section.
    // [FIXME]: Implement `Display`/`Debug` for all variants of `DeviceInfoResult`.
    // Currently printing random utf8 stuff.

    for device in context.device_ids().iter() {
	    println!("Device:\n\
				{t}DeviceInfo::Type: {}\n\
				{t}DeviceInfo::VendorId: {}\n\
				{t}DeviceInfo::MaxComputeUnits: {}\n\
				{t}DeviceInfo::MaxWorkItemDimensions: {}\n\
				{t}DeviceInfo::MaxWorkGroupSize: {}\n\
				{t}DeviceInfo::MaxWorkItemSizes: {}\n\
				{t}DeviceInfo::PreferredVectorWidthChar: {}\n\
				{t}DeviceInfo::PreferredVectorWidthShort: {}\n\
				{t}DeviceInfo::PreferredVectorWidthInt: {}\n\
				{t}DeviceInfo::PreferredVectorWidthLong: {}\n\
				{t}DeviceInfo::PreferredVectorWidthFloat: {}\n\
				{t}DeviceInfo::PreferredVectorWidthDouble: {}\n\
				{t}DeviceInfo::MaxClockFrequency: {}\n\
				{t}DeviceInfo::AddressBits: {}\n\
				{t}DeviceInfo::MaxReadImageArgs: {}\n\
				{t}DeviceInfo::MaxWriteImageArgs: {}\n\
				{t}DeviceInfo::MaxMemAllocSize: {}\n\
				{t}DeviceInfo::Image2dMaxWidth: {}\n\
				{t}DeviceInfo::Image2dMaxHeight: {}\n\
				{t}DeviceInfo::Image3dMaxWidth: {}\n\
				{t}DeviceInfo::Image3dMaxHeight: {}\n\
				{t}DeviceInfo::Image3dMaxDepth: {}\n\
				{t}DeviceInfo::ImageSupport: {}\n\
				{t}DeviceInfo::MaxParameterSize: {}\n\
				{t}DeviceInfo::MaxSamplers: {}\n\
				{t}DeviceInfo::MemBaseAddrAlign: {}\n\
				{t}DeviceInfo::MinDataTypeAlignSize: {}\n\
				{t}DeviceInfo::SingleFpConfig: {}\n\
				{t}DeviceInfo::GlobalMemCacheType: {}\n\
				{t}DeviceInfo::GlobalMemCachelineSize: {}\n\
				{t}DeviceInfo::GlobalMemCacheSize: {}\n\
				{t}DeviceInfo::GlobalMemSize: {}\n\
				{t}DeviceInfo::MaxConstantBufferSize: {}\n\
				{t}DeviceInfo::MaxConstantArgs: {}\n\
				{t}DeviceInfo::LocalMemType: {}\n\
				{t}DeviceInfo::LocalMemSize: {}\n\
				{t}DeviceInfo::ErrorCorrectionSupport: {}\n\
				{t}DeviceInfo::ProfilingTimerResolution: {}\n\
				{t}DeviceInfo::EndianLittle: {}\n\
				{t}DeviceInfo::Available: {}\n\
				{t}DeviceInfo::CompilerAvailable: {}\n\
				{t}DeviceInfo::ExecutionCapabilities: {}\n\
				{t}DeviceInfo::QueueProperties: {}\n\
				{t}DeviceInfo::Name: {}\n\
				{t}DeviceInfo::Vendor: {}\n\
				{t}DeviceInfo::DriverVersion: {}\n\
				{t}DeviceInfo::Profile: {}\n\
				{t}DeviceInfo::Version: {}\n\
				{t}DeviceInfo::Extensions: {}\n\
				{t}DeviceInfo::Platform: {}\n\
				{t}DeviceInfo::DoubleFpConfig: {}\n\
				{t}DeviceInfo::HalfFpConfig: {}\n\
				{t}DeviceInfo::PreferredVectorWidthHalf: {}\n\
				{t}DeviceInfo::HostUnifiedMemory: {}\n\
				{t}DeviceInfo::NativeVectorWidthChar: {}\n\
				{t}DeviceInfo::NativeVectorWidthShort: {}\n\
				{t}DeviceInfo::NativeVectorWidthInt: {}\n\
				{t}DeviceInfo::NativeVectorWidthLong: {}\n\
				{t}DeviceInfo::NativeVectorWidthFloat: {}\n\
				{t}DeviceInfo::NativeVectorWidthDouble: {}\n\
				{t}DeviceInfo::NativeVectorWidthHalf: {}\n\
				{t}DeviceInfo::OpenclCVersion: {}\n\
				{t}DeviceInfo::LinkerAvailable: {}\n\
				{t}DeviceInfo::BuiltInKernels: {}\n\
				{t}DeviceInfo::ImageMaxBufferSize: {}\n\
				{t}DeviceInfo::ImageMaxArraySize: {}\n\
				{t}DeviceInfo::ParentDevice: {}\n\
				{t}DeviceInfo::PartitionMaxSubDevices: {}\n\
				{t}DeviceInfo::PartitionProperties: {}\n\
				{t}DeviceInfo::PartitionAffinityDomain: {}\n\
				{t}DeviceInfo::PartitionType: {}\n\
				{t}DeviceInfo::ReferenceCount: {}\n\
				{t}DeviceInfo::PreferredInteropUserSync: {}\n\
				{t}DeviceInfo::PrintfBufferSize: {}\n\
				{t}DeviceInfo::ImagePitchAlignment: {}\n\
				{t}DeviceInfo::ImageBaseAddressAlignment: {}\n\
			",
			raw::get_device_info(device.clone(), DeviceInfo::Type).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::VendorId).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxComputeUnits).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxWorkItemDimensions).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxWorkGroupSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxWorkItemSizes).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PreferredVectorWidthChar).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PreferredVectorWidthShort).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PreferredVectorWidthInt).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PreferredVectorWidthLong).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PreferredVectorWidthFloat).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PreferredVectorWidthDouble).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxClockFrequency).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::AddressBits).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxReadImageArgs).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxWriteImageArgs).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxMemAllocSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Image2dMaxWidth).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Image2dMaxHeight).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Image3dMaxWidth).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Image3dMaxHeight).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Image3dMaxDepth).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ImageSupport).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxParameterSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxSamplers).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MemBaseAddrAlign).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MinDataTypeAlignSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::SingleFpConfig).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::GlobalMemCacheType).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::GlobalMemCachelineSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::GlobalMemCacheSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::GlobalMemSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxConstantBufferSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::MaxConstantArgs).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::LocalMemType).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::LocalMemSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ErrorCorrectionSupport).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ProfilingTimerResolution).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::EndianLittle).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Available).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::CompilerAvailable).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ExecutionCapabilities).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::QueueProperties).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Name).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Vendor).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::DriverVersion).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Profile).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Version).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Extensions).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::Platform).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::DoubleFpConfig).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::HalfFpConfig).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PreferredVectorWidthHalf).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::HostUnifiedMemory).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::NativeVectorWidthChar).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::NativeVectorWidthShort).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::NativeVectorWidthInt).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::NativeVectorWidthLong).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::NativeVectorWidthFloat).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::NativeVectorWidthDouble).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::NativeVectorWidthHalf).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::OpenclCVersion).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::LinkerAvailable).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::BuiltInKernels).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ImageMaxBufferSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ImageMaxArraySize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ParentDevice).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PartitionMaxSubDevices).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PartitionProperties).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PartitionAffinityDomain).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PartitionType).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ReferenceCount).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PreferredInteropUserSync).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::PrintfBufferSize).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ImagePitchAlignment).unwrap(),
			raw::get_device_info(device.clone(), DeviceInfo::ImageBaseAddressAlignment).unwrap(),
			t = TAB,
		);
    }


    // ##################################################
    // #################### CONTEXT #####################
    // ##################################################

    println!("Context:\n\
			{t}Reference Count:  {}\n\
			{t}Devices:          {}\n\
			{t}Properties:       {}\n\
			{t}Device Count:     {}\n\
		",
		raw::get_context_info(context.obj_raw(), ContextInfo::ReferenceCount).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::Devices).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::Properties).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::NumDevices).unwrap(),
		t = TAB,
	);


    // ##################################################
    // ##################### QUEUE ######################
    // ##################################################

    // Context = cl_h::CL_QUEUE_CONTEXT as isize,
    // Device = cl_h::CL_QUEUE_DEVICE as isize,
    // ReferenceCount = cl_h::CL_QUEUE_REFERENCE_COUNT as isize,
    // Properties = cl_h::CL_QUEUE_PROPERTIES as isize,

	println!("Command Queue:\n\
			{t}Context:         {}\n\
			{t}Device:          {}\n\
			{t}ReferenceCount:  {}\n\
			{t}Properties:      {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Device).unwrap(),
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::ReferenceCount).unwrap(),
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Properties).unwrap(),
		t = TAB,
	);


	// ##################################################
    // ################### MEM OBJECT ###################
    // ##################################################

    // [FIXME]: Complete this section.



    // ##################################################
    // ##################### IMAGE ######################
    // ##################################################

    // [FIXME]: Complete this section.


    // ##################################################
    // #################### SAMPLER #####################
    // ##################################################

    // [FIXME]: Complete this section.


    // ##################################################
    // #################### PROGRAM #####################
    // ##################################################

    // [FIXME]: Complete this section.


    // ##################################################
    // ################# PROGRAM BUILD ##################
    // ##################################################

    // [FIXME]: Complete this section.


    // ##################################################
    // ##################### KERNEL #####################
    // ##################################################

    // [FIXME]: Complete this section.


    // ##################################################
    // ################# KERNEL ARGUMENT ################
    // ##################################################

    // [FIXME]: Complete this section.


    // ##################################################
    // ################ KERNEL WORK GROUP ###############
    // ##################################################

    // [FIXME]: Complete this section.


    // ##################################################
    // ##################### EVENT ######################
    // ##################################################

    // [FIXME]: Complete this section.


    // ##################################################
    // ################ EVENT PROFILING #################
    // ##################################################

    // [FIXME]: Complete this section.



    print!("\n");
}
