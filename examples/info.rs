//! [WORK IN PROGRESS] Get information about all the things.
//!

extern crate ocl;

use ocl::{SimpleDims, Context, Queue, Buffer, Program};
use ocl::raw::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo};

static TAB: &'static str = "    ";
static SRC: &'static str = r#"
	__kernel void multiply(__global float* buffer, float coeff) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
	let dims = SimpleDims::One(1000);

	let context = Context::new(None, None).unwrap();
	let queue = Queue::new(&context, None);
	let buffer = Buffer::<u32>::new(&dims, &queue);
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

    // pub enum MemInfo {
    //     Type = cl_h::CL_MEM_TYPE as isize,
    //     Flags = cl_h::CL_MEM_FLAGS as isize,
    //     Size = cl_h::CL_MEM_SIZE as isize,
    //     HostPtr = cl_h::CL_MEM_HOST_PTR as isize,
    //     MapCount = cl_h::CL_MEM_MAP_COUNT as isize,
    //     ReferenceCount = cl_h::CL_MEM_REFERENCE_COUNT as isize,
    //     Context = cl_h::CL_MEM_CONTEXT as isize,
    //     AssociatedMemobject = cl_h::CL_MEM_ASSOCIATED_MEMOBJECT as isize,
    //     Offset = cl_h::CL_MEM_OFFSET as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);


    // ##################################################
    // ##################### IMAGE ######################
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum ImageInfo {
    //     Format = cl_h::CL_IMAGE_FORMAT as isize,
    //     ElementSize = cl_h::CL_IMAGE_ELEMENT_SIZE as isize,
    //     RowPitch = cl_h::CL_IMAGE_ROW_PITCH as isize,
    //     SlicePitch = cl_h::CL_IMAGE_SLICE_PITCH as isize,
    //     Width = cl_h::CL_IMAGE_WIDTH as isize,
    //     Height = cl_h::CL_IMAGE_HEIGHT as isize,
    //     Depth = cl_h::CL_IMAGE_DEPTH as isize,
    //     ArraySize = cl_h::CL_IMAGE_ARRAY_SIZE as isize,
    //     Buffer = cl_h::CL_IMAGE_BUFFER as isize,
    //     NumMipLevels = cl_h::CL_IMAGE_NUM_MIP_LEVELS as isize,
    //     NumSamples = cl_h::CL_IMAGE_NUM_SAMPLES as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);

    // ##################################################
    // #################### SAMPLER #####################
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum SamplerInfo {
    //     ReferenceCount = cl_h::CL_SAMPLER_REFERENCE_COUNT as isize,
    //     Context = cl_h::CL_SAMPLER_CONTEXT as isize,
    //     NormalizedCoords = cl_h::CL_SAMPLER_NORMALIZED_COORDS as isize,
    //     AddressingMode = cl_h::CL_SAMPLER_ADDRESSING_MODE as isize,
    //     FilterMode = cl_h::CL_SAMPLER_FILTER_MODE as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);

    // ##################################################
    // #################### PROGRAM #####################
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum ProgramInfo {
    //     ReferenceCount = cl_h::CL_PROGRAM_REFERENCE_COUNT as isize,
    //     Context = cl_h::CL_PROGRAM_CONTEXT as isize,
    //     NumDevices = cl_h::CL_PROGRAM_NUM_DEVICES as isize,
    //     Devices = cl_h::CL_PROGRAM_DEVICES as isize,
    //     Source = cl_h::CL_PROGRAM_SOURCE as isize,
    //     BinarySizes = cl_h::CL_PROGRAM_BINARY_SIZES as isize,
    //     Binaries = cl_h::CL_PROGRAM_BINARIES as isize,
    //     NumKernels = cl_h::CL_PROGRAM_NUM_KERNELS as isize,
    //     KernelNames = cl_h::CL_PROGRAM_KERNEL_NAMES as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);

    // ##################################################
    // ################# PROGRAM BUILD ##################
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum ProgramBuildInfo {
    //     BuildStatus = cl_h::CL_PROGRAM_BUILD_STATUS as isize,
    //     BuildOptions = cl_h::CL_PROGRAM_BUILD_OPTIONS as isize,
    //     BuildLog = cl_h::CL_PROGRAM_BUILD_LOG as isize,
    //     BinaryType = cl_h::CL_PROGRAM_BINARY_TYPE as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);

    // ##################################################
    // ##################### KERNEL #####################
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum KernelInfo {
    //     FunctionName = cl_h::CL_KERNEL_FUNCTION_NAME as isize,
    //     NumArgs = cl_h::CL_KERNEL_NUM_ARGS as isize,
    //     ReferenceCount = cl_h::CL_KERNEL_REFERENCE_COUNT as isize,
    //     Context = cl_h::CL_KERNEL_CONTEXT as isize,
    //     Program = cl_h::CL_KERNEL_PROGRAM as isize,
    //     Attributes = cl_h::CL_KERNEL_ATTRIBUTES as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);

    // ##################################################
    // ################# KERNEL ARGUMENT ################
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum KernelArgInfo {
    //     AddressQualifier = cl_h::CL_KERNEL_ARG_ADDRESS_QUALIFIER as isize,
    //     AccessQualifier = cl_h::CL_KERNEL_ARG_ACCESS_QUALIFIER as isize,
    //     TypeName = cl_h::CL_KERNEL_ARG_TYPE_NAME as isize,
    //     TypeQualifier = cl_h::CL_KERNEL_ARG_TYPE_QUALIFIER as isize,
    //     Name = cl_h::CL_KERNEL_ARG_NAME as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);

    // ##################################################
    // ################ KERNEL WORK GROUP ###############
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum KernelWorkGroupInfo {
    //     WorkGroupSize = cl_h::CL_KERNEL_WORK_GROUP_SIZE as isize,
    //     CompileWorkGroupSize = cl_h::CL_KERNEL_COMPILE_WORK_GROUP_SIZE as isize,
    //     LocalMemSize = cl_h::CL_KERNEL_LOCAL_MEM_SIZE as isize,
    //     PreferredWorkGroupSizeMultiple = cl_h::CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE as isize,
    //     PrivateMemSize = cl_h::CL_KERNEL_PRIVATE_MEM_SIZE as isize,
    //     GlobalWorkSize = cl_h::CL_KERNEL_GLOBAL_WORK_SIZE as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);

    // ##################################################
    // ##################### EVENT ######################
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum EventInfo {
    //     CommandQueue = cl_h::CL_EVENT_COMMAND_QUEUE as isize,
    //     CommandType = cl_h::CL_EVENT_COMMAND_TYPE as isize,
    //     ReferenceCount = cl_h::CL_EVENT_REFERENCE_COUNT as isize,
    //     CommandExecutionStatus = cl_h::CL_EVENT_COMMAND_EXECUTION_STATUS as isize,
    //     Context = cl_h::CL_EVENT_CONTEXT as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);

    // ##################################################
    // ################ EVENT PROFILING #################
    // ##################################################

    // [FIXME]: Complete this section.
    // pub enum ProfilingInfo {
    //     Queued = cl_h::CL_PROFILING_COMMAND_QUEUED as isize,
    //     Submit = cl_h::CL_PROFILING_COMMAND_SUBMIT as isize,
    //     Start = cl_h::CL_PROFILING_COMMAND_START as isize,
    //     End = cl_h::CL_PROFILING_COMMAND_END as isize,
    // }

    println!("Command Queue:\n\
			{t}Context:         {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		t = TAB,
	);


    print!("\n");
}
