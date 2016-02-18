//! [WORK IN PROGRESS] Get information about all the things.
//!

extern crate ocl;

use ocl::{SimpleDims, Context, Queue, Buffer, Program, Kernel, EventList};
use ocl::raw::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo, EventInfo, ProfilingInfo};
use ocl::util;

static SRC: &'static str = r#"
	__kernel void multiply(__global float* buffer, float coeff) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
	let dims = SimpleDims::One(1000);

	let context = Context::new(None, None).unwrap();
	let queue = Queue::new(&context, None);
	let buffer = Buffer::<f32>::new(&dims, &queue);
	// let image = Image::new();
	// let sampler = Sampler::new();
	let program = Program::builder().src(SRC).build(&context).unwrap();
	let device = program.device_ids_raw()[0];
	let kernel = Kernel::new("multiply", &program, &queue, dims.work_dims()).unwrap()
        .arg_buf(&buffer)
        .arg_scl(10.0f32);
    let mut event_list = EventList::new();

    kernel.enqueue(None, Some(&mut event_list));
    let event = event_list.last().unwrap().clone();
    event_list.wait();

	println!("############### OpenCL [Default Platform] [Default Device] Info ################");
	print!("\n");

    // ##################################################
    // #################### PLATFORM ####################
    // ##################################################

	println!("Platform:\n\
			{t}Profile: {}\n\
			{t}Version: {}\n\
			{t}Name: {}\n\
			{t}Vendor: {}\n\
			{t}Extensions: {}\n\
		",
		raw::get_platform_info(context.platform_id_raw(), PlatformInfo::Profile).unwrap(),
		raw::get_platform_info(context.platform_id_raw(), PlatformInfo::Version).unwrap(),
		raw::get_platform_info(context.platform_id_raw(), PlatformInfo::Name).unwrap(),
		raw::get_platform_info(context.platform_id_raw(), PlatformInfo::Vendor).unwrap(),
		raw::get_platform_info(context.platform_id_raw(), PlatformInfo::Extensions).unwrap(),
		t = util::TAB,
	);

    // ##################################################
    // #################### DEVICES #####################
    // ##################################################

    // [FIXME]: Complete this section.
    // [FIXME]: Implement `Display`/`Debug` for all variants of `DeviceInfoResult`.
    // Currently printing random utf8 stuff.

    for &device in context.device_ids_raw().iter() {
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
			raw::get_device_info(device, DeviceInfo::Type).unwrap(),
			raw::get_device_info(device, DeviceInfo::VendorId).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxComputeUnits).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxWorkItemDimensions).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxWorkGroupSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxWorkItemSizes).unwrap(),
			raw::get_device_info(device, DeviceInfo::PreferredVectorWidthChar).unwrap(),
			raw::get_device_info(device, DeviceInfo::PreferredVectorWidthShort).unwrap(),
			raw::get_device_info(device, DeviceInfo::PreferredVectorWidthInt).unwrap(),
			raw::get_device_info(device, DeviceInfo::PreferredVectorWidthLong).unwrap(),
			raw::get_device_info(device, DeviceInfo::PreferredVectorWidthFloat).unwrap(),
			raw::get_device_info(device, DeviceInfo::PreferredVectorWidthDouble).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxClockFrequency).unwrap(),
			raw::get_device_info(device, DeviceInfo::AddressBits).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxReadImageArgs).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxWriteImageArgs).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxMemAllocSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::Image2dMaxWidth).unwrap(),
			raw::get_device_info(device, DeviceInfo::Image2dMaxHeight).unwrap(),
			raw::get_device_info(device, DeviceInfo::Image3dMaxWidth).unwrap(),
			raw::get_device_info(device, DeviceInfo::Image3dMaxHeight).unwrap(),
			raw::get_device_info(device, DeviceInfo::Image3dMaxDepth).unwrap(),
			raw::get_device_info(device, DeviceInfo::ImageSupport).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxParameterSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxSamplers).unwrap(),
			raw::get_device_info(device, DeviceInfo::MemBaseAddrAlign).unwrap(),
			raw::get_device_info(device, DeviceInfo::MinDataTypeAlignSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::SingleFpConfig).unwrap(),
			raw::get_device_info(device, DeviceInfo::GlobalMemCacheType).unwrap(),
			raw::get_device_info(device, DeviceInfo::GlobalMemCachelineSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::GlobalMemCacheSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::GlobalMemSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxConstantBufferSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::MaxConstantArgs).unwrap(),
			raw::get_device_info(device, DeviceInfo::LocalMemType).unwrap(),
			raw::get_device_info(device, DeviceInfo::LocalMemSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::ErrorCorrectionSupport).unwrap(),
			raw::get_device_info(device, DeviceInfo::ProfilingTimerResolution).unwrap(),
			raw::get_device_info(device, DeviceInfo::EndianLittle).unwrap(),
			raw::get_device_info(device, DeviceInfo::Available).unwrap(),
			raw::get_device_info(device, DeviceInfo::CompilerAvailable).unwrap(),
			raw::get_device_info(device, DeviceInfo::ExecutionCapabilities).unwrap(),
			raw::get_device_info(device, DeviceInfo::QueueProperties).unwrap(),
			raw::get_device_info(device, DeviceInfo::Name).unwrap(),
			raw::get_device_info(device, DeviceInfo::Vendor).unwrap(),
			raw::get_device_info(device, DeviceInfo::DriverVersion).unwrap(),
			raw::get_device_info(device, DeviceInfo::Profile).unwrap(),
			raw::get_device_info(device, DeviceInfo::Version).unwrap(),
			raw::get_device_info(device, DeviceInfo::Extensions).unwrap(),
			raw::get_device_info(device, DeviceInfo::Platform).unwrap(),
			raw::get_device_info(device, DeviceInfo::DoubleFpConfig).unwrap(),
			raw::get_device_info(device, DeviceInfo::HalfFpConfig).unwrap(),
			raw::get_device_info(device, DeviceInfo::PreferredVectorWidthHalf).unwrap(),
			raw::get_device_info(device, DeviceInfo::HostUnifiedMemory).unwrap(),
			raw::get_device_info(device, DeviceInfo::NativeVectorWidthChar).unwrap(),
			raw::get_device_info(device, DeviceInfo::NativeVectorWidthShort).unwrap(),
			raw::get_device_info(device, DeviceInfo::NativeVectorWidthInt).unwrap(),
			raw::get_device_info(device, DeviceInfo::NativeVectorWidthLong).unwrap(),
			raw::get_device_info(device, DeviceInfo::NativeVectorWidthFloat).unwrap(),
			raw::get_device_info(device, DeviceInfo::NativeVectorWidthDouble).unwrap(),
			raw::get_device_info(device, DeviceInfo::NativeVectorWidthHalf).unwrap(),
			raw::get_device_info(device, DeviceInfo::OpenclCVersion).unwrap(),
			raw::get_device_info(device, DeviceInfo::LinkerAvailable).unwrap(),
			raw::get_device_info(device, DeviceInfo::BuiltInKernels).unwrap(),
			raw::get_device_info(device, DeviceInfo::ImageMaxBufferSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::ImageMaxArraySize).unwrap(),
			raw::get_device_info(device, DeviceInfo::ParentDevice).unwrap(),
			raw::get_device_info(device, DeviceInfo::PartitionMaxSubDevices).unwrap(),
			raw::get_device_info(device, DeviceInfo::PartitionProperties).unwrap(),
			raw::get_device_info(device, DeviceInfo::PartitionAffinityDomain).unwrap(),
			raw::get_device_info(device, DeviceInfo::PartitionType).unwrap(),
			raw::get_device_info(device, DeviceInfo::ReferenceCount).unwrap(),
			raw::get_device_info(device, DeviceInfo::PreferredInteropUserSync).unwrap(),
			raw::get_device_info(device, DeviceInfo::PrintfBufferSize).unwrap(),
			raw::get_device_info(device, DeviceInfo::ImagePitchAlignment).unwrap(),
			raw::get_device_info(device, DeviceInfo::ImageBaseAddressAlignment).unwrap(),
			t = util::TAB,
		);
    }


    // ##################################################
    // #################### CONTEXT #####################
    // ##################################################

    println!("Context:\n\
			{t}Reference Count: {}\n\
			{t}Devices: {}\n\
			{t}Properties: {}\n\
			{t}Device Count: {}\n\
		",
		raw::get_context_info(context.obj_raw(), ContextInfo::ReferenceCount).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::Devices).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::Properties).unwrap(),
		raw::get_context_info(context.obj_raw(), ContextInfo::NumDevices).unwrap(),
		t = util::TAB,
	);


    // ##################################################
    // ##################### QUEUE ######################
    // ##################################################


	println!("Command Queue:\n\
			{t}Context: {}\n\
			{t}Device: {}\n\
			{t}ReferenceCount: {}\n\
			{t}Properties: {}\n\
		",
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Context).unwrap(),
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Device).unwrap(),
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::ReferenceCount).unwrap(),
		raw::get_command_queue_info(queue.obj_raw(), CommandQueueInfo::Properties).unwrap(),
		t = util::TAB,
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

    println!("Buffer:\n\
			{t}Type: {}\n\
	        {t}Flags: {}\n\
	        {t}Size: {}\n\
	        {t}HostPtr: {}\n\
	        {t}MapCount: {}\n\
	        {t}ReferenceCount: {}\n\
	        {t}Context: {}\n\
	        {t}AssociatedMemobject: {}\n\
	        {t}Offset: {}\n\
		",
		raw::get_mem_object_info(buffer.obj_raw(), MemInfo::Type).unwrap(),
	    raw::get_mem_object_info(buffer.obj_raw(), MemInfo::Flags).unwrap(),
        raw::get_mem_object_info(buffer.obj_raw(), MemInfo::Size).unwrap(),
        raw::get_mem_object_info(buffer.obj_raw(), MemInfo::HostPtr).unwrap(),
        raw::get_mem_object_info(buffer.obj_raw(), MemInfo::MapCount).unwrap(),
        raw::get_mem_object_info(buffer.obj_raw(), MemInfo::ReferenceCount).unwrap(),
        raw::get_mem_object_info(buffer.obj_raw(), MemInfo::Context).unwrap(),
        raw::get_mem_object_info(buffer.obj_raw(), MemInfo::AssociatedMemobject).unwrap(),
        raw::get_mem_object_info(buffer.obj_raw(), MemInfo::Offset).unwrap(),
		t = util::TAB,
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

 //    println!("[UNIMPLEMENTED] Image:\n\
	// 		{t}Format: {}\n\
 //            {t}ElementSize: {}\n\
 //            {t}RowPitch: {}\n\
 //            {t}SlicePitch: {}\n\
 //            {t}Width: {}\n\
 //            {t}Height: {}\n\
 //            {t}Depth: {}\n\
 //            {t}ArraySize: {}\n\
 //            {t}Buffer: {}\n\
 //            {t}NumMipLevels: {}\n\
 //            {t}NumSamples: {}\n\
	// 	",
	// 	raw::get_image_info(image.obj_raw(), ImageInfo::Format).unwrap(),
	//     raw::get_image_info(image.obj_raw(), ImageInfo::ElementSize).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::RowPitch).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::SlicePitch).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::Width).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::Height).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::Depth).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::ArraySize).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::Buffer).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::NumMipLevels).unwrap(),
 //        raw::get_image_info(image.obj_raw(), ImageInfo::NumSamples).unwrap(),
	// 	t = util::TAB,
	// );

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

 //    println!("[UNIMPLEMENTED] Sampler:\n\
	// 		{t}ReferenceCount: {}\n\
 //            {t}Context: {}\n\
 //            {t}NormalizedCoords: {}\n\
 //            {t}AddressingMode: {}\n\
 //            {t}FilterMode: {}\n\
	// 	",
	// 	raw::get_sampler_info(sampler.obj_raw(), SamplerInfo::ReferenceCount).unwrap(),
 //        raw::get_sampler_info(sampler.obj_raw(), SamplerInfo::Context).unwrap(),
 //        raw::get_sampler_info(sampler.obj_raw(), SamplerInfo::NormalizedCoords).unwrap(),
 //        raw::get_sampler_info(sampler.obj_raw(), SamplerInfo::AddressingMode).unwrap(),
 //        raw::get_sampler_info(sampler.obj_raw(), SamplerInfo::FilterMode).unwrap(),
	// 	t = util::TAB,
	// );

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

    println!("Program:\n\
			{t}ReferenceCount: {}\n\
            {t}Context: {}\n\
            {t}NumDevices: {}\n\
            {t}Devices: {}\n\
            {t}Source: {}\n\
            {t}BinarySizes: {}\n\
            {t}Binaries: {}\n\
            {t}NumKernels: {}\n\
            {t}KernelNames: {}\n\
		",
		raw::get_program_info(program.obj_raw(), ProgramInfo::ReferenceCount).unwrap(),
        raw::get_program_info(program.obj_raw(), ProgramInfo::Context).unwrap(),
        raw::get_program_info(program.obj_raw(), ProgramInfo::NumDevices).unwrap(),
        raw::get_program_info(program.obj_raw(), ProgramInfo::Devices).unwrap(),
        raw::get_program_info(program.obj_raw(), ProgramInfo::Source).unwrap(),
        raw::get_program_info(program.obj_raw(), ProgramInfo::BinarySizes).unwrap(),
        raw::get_program_info(program.obj_raw(), ProgramInfo::Binaries).unwrap(),
        raw::get_program_info(program.obj_raw(), ProgramInfo::NumKernels).unwrap(),
        raw::get_program_info(program.obj_raw(), ProgramInfo::KernelNames).unwrap(),
		t = util::TAB,
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

    println!("Program Build:\n\
			{t}BuildStatus: {}\n\
            {t}BuildOptions: {}\n\
            {t}BuildLog: {}\n\
            {t}BinaryType: {}\n\
		",
		raw::get_program_build_info(program.obj_raw(), device, ProgramBuildInfo::BuildStatus).unwrap(),
        raw::get_program_build_info(program.obj_raw(), device, ProgramBuildInfo::BuildOptions).unwrap(),
        raw::get_program_build_info(program.obj_raw(), device, ProgramBuildInfo::BuildLog).unwrap(),
        raw::get_program_build_info(program.obj_raw(), device, ProgramBuildInfo::BinaryType).unwrap(),
		t = util::TAB,
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

    println!("Kernel Info:\n\
			{t}FunctionName: {}\n\
            {t}NumArgs: {}\n\
            {t}ReferenceCount: {}\n\
            {t}Context: {}\n\
            {t}Program: {}\n\
            {t}Attributes: {}\n\
		",
		raw::get_kernel_info(kernel.obj_raw(), KernelInfo::FunctionName).unwrap(),
	    raw::get_kernel_info(kernel.obj_raw(), KernelInfo::NumArgs).unwrap(),
        raw::get_kernel_info(kernel.obj_raw(), KernelInfo::ReferenceCount).unwrap(),
        raw::get_kernel_info(kernel.obj_raw(), KernelInfo::Context).unwrap(),
        raw::get_kernel_info(kernel.obj_raw(), KernelInfo::Program).unwrap(),
        raw::get_kernel_info(kernel.obj_raw(), KernelInfo::Attributes).unwrap(),
		t = util::TAB,
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

    println!("KernelArgInfo:\n\
			{t}AddressQualifier: {}\n\
            {t}AccessQualifier: {}\n\
            {t}TypeName: {}\n\
            {t}TypeQualifier: {}\n\
            {t}Name: {}\n\
		",
		raw::get_kernel_arg_info(kernel.obj_raw(), 0, KernelArgInfo::AddressQualifier).unwrap(),
        raw::get_kernel_arg_info(kernel.obj_raw(), 0, KernelArgInfo::AccessQualifier).unwrap(),
        raw::get_kernel_arg_info(kernel.obj_raw(), 0, KernelArgInfo::TypeName).unwrap(),
        raw::get_kernel_arg_info(kernel.obj_raw(), 0, KernelArgInfo::TypeQualifier).unwrap(),
        raw::get_kernel_arg_info(kernel.obj_raw(), 0, KernelArgInfo::Name).unwrap(),
		t = util::TAB,
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

    println!("Kernel Work Group:\n\
			{t}WorkGroupSize: {}\n\
	    	{t}CompileWorkGroupSize: {}\n\
            {t}LocalMemSize: {}\n\
            {t}PreferredWorkGroupSizeMultiple: {}\n\
            {t}PrivateMemSize: {}\n\
            {t}GlobalWorkSize: {}\n\
		",
		raw::get_kernel_work_group_info(kernel.obj_raw(), device, 
			KernelWorkGroupInfo::WorkGroupSize).unwrap(),
	    raw::get_kernel_work_group_info(kernel.obj_raw(), device, 
	    	KernelWorkGroupInfo::CompileWorkGroupSize).unwrap(),
        raw::get_kernel_work_group_info(kernel.obj_raw(), device, 
        	KernelWorkGroupInfo::LocalMemSize).unwrap(),
        raw::get_kernel_work_group_info(kernel.obj_raw(), device, 
        	KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple).unwrap(),
        raw::get_kernel_work_group_info(kernel.obj_raw(), device, 
        	KernelWorkGroupInfo::PrivateMemSize).unwrap(),
        // raw::get_kernel_work_group_info(kernel.obj_raw(), device, 
        // 	KernelWorkGroupInfo::GlobalWorkSize).unwrap(),
    	"[KernelWorkGroupInfo::GlobalWorkSize not avaliable in this configuration]",
		t = util::TAB,
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

    println!("EventInfo:\n\
			{t}CommandQueue: {}\n\
            {t}CommandType: {}\n\
            {t}ReferenceCount: {}\n\
            {t}CommandExecutionStatus: {}\n\
            {t}Context: {}\n\
		",
		raw::get_event_info(event, EventInfo::CommandQueue).unwrap(),
        raw::get_event_info(event, EventInfo::CommandType).unwrap(),
        raw::get_event_info(event, EventInfo::ReferenceCount).unwrap(),
        raw::get_event_info(event, EventInfo::CommandExecutionStatus).unwrap(),
        raw::get_event_info(event, EventInfo::Context).unwrap(),
		t = util::TAB,
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

    println!("ProfilingInfo:\n\
			{t}Queued: {}\n\
	    	{t}Submit: {}\n\
	    	{t}Start: {}\n\
	    	{t}End: {}\n\
		",
		raw::get_event_profiling_info(event, ProfilingInfo::Queued).unwrap(),
        raw::get_event_profiling_info(event, ProfilingInfo::Submit).unwrap(),
        raw::get_event_profiling_info(event, ProfilingInfo::Start).unwrap(),
        raw::get_event_profiling_info(event, ProfilingInfo::End).unwrap(),
		t = util::TAB,
	);


    print!("\n");
}
