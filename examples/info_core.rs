//! [WORK IN PROGRESS] Get information about all the things using `core` function calls.
//!

extern crate ocl;

use ocl::{SimpleDims, Context, Queue, Buffer, Image, Program, Kernel, EventList};
use ocl::core::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ImageInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo, EventInfo, ProfilingInfo};
use ocl::util;

const INFO_FORMAT_MULTILINE: bool = true;

static SRC: &'static str = r#"
	__kernel void multiply(float coeff, __global float* buffer) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
	let dims = SimpleDims::One(1000);

	let context = Context::new_by_index_and_type(None, None).unwrap();
	let queue = Queue::new_by_device_index(&context, None);
	let buffer = Buffer::<f32>::new(&dims, &queue);
	let image = Image::builder().build(&queue).unwrap();
	// let sampler = Sampler::new();
	let program = Program::builder().src(SRC).build(&context).unwrap();
	let device = program.devices()[0].clone();
	let kernel = Kernel::new("multiply", &program, &queue, dims.clone()).unwrap()
        .arg_scl(10.0f32)
        .arg_buf(&buffer);
    let mut event_list = EventList::new();

    kernel.enqueue_with(None, None, Some(&mut event_list)).unwrap();
    let event = event_list.last_clone().unwrap();
    event_list.wait();

	println!("############### OpenCL [Default Platform] [Default Device] Info ################");
	print!("\n");

	let (begin, delim, end) = if INFO_FORMAT_MULTILINE {
        ("\n", "\n", "\n")
    } else {
        ("{ ", ", ", " }")
    };

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
		core::get_platform_info(context.platform().clone(), PlatformInfo::Profile).unwrap(),
		core::get_platform_info(context.platform().clone(), PlatformInfo::Version).unwrap(),
		core::get_platform_info(context.platform().clone(), PlatformInfo::Name).unwrap(),
		core::get_platform_info(context.platform().clone(), PlatformInfo::Vendor).unwrap(),
		core::get_platform_info(context.platform().clone(), PlatformInfo::Extensions).unwrap(),
		t = util::TAB,
	);


	//
	// CHANGE TO --->
	//

	// write!(f, "{}", &self.to_string())
        // let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
        //     ("\n", "\n", "\n")
        // } else {
        //     ("{ ", ", ", " }")
        // };

        // write!(f, "[Platform]: {b}\
        //         Profile: {}{d}\
        //         Version: {}{d}\
        //         Name: {}{d}\
        //         Vendor: {}{d}\
        //         Extensions: {}{e}\
        //     ",
        //     self.profile(),
        //     self.version(),
        //     self.name(),
        //     self.vendor(),
        //     self.extensions(),
        //     b = begin,
        //     d = delim,
        //     e = end,
        // )

    // ##################################################
    // #################### DEVICES #####################
    // ##################################################

    // [FIXME]: Complete this section.
    // [FIXME]: Implement `Display`/`Debug` for all variants of `DeviceInfoResult`.
    // Printing algorithm is highly janky (due to laziness).

    // for device in context.devices().iter() {
    for device_idx in 0..context.devices().len() {
    	let device = context.devices()[device_idx].clone();

	    println!("Device[{}]: \n\
				{t}Type: {}\n\
				{t}VendorId: {}\n\
				{t}MaxComputeUnits: {}\n\
				{t}MaxWorkItemDimensions: {}\n\
				{t}MaxWorkGroupSize: {}\n\
				{t}MaxWorkItemSizes: {}\n\
				{t}PreferredVectorWidthChar: {}\n\
				{t}PreferredVectorWidthShort: {}\n\
				{t}PreferredVectorWidthInt: {}\n\
				{t}PreferredVectorWidthLong: {}\n\
				{t}PreferredVectorWidthFloat: {}\n\
				{t}PreferredVectorWidthDouble: {}\n\
				{t}MaxClockFrequency: {}\n\
				{t}AddressBits: {}\n\
				{t}MaxReadImageArgs: {}\n\
				{t}MaxWriteImageArgs: {}\n\
				{t}MaxMemAllocSize: {}\n\
				{t}Image2dMaxWidth: {}\n\
				{t}Image2dMaxHeight: {}\n\
				{t}Image3dMaxWidth: {}\n\
				{t}Image3dMaxHeight: {}\n\
				{t}Image3dMaxDepth: {}\n\
				{t}ImageSupport: {}\n\
				{t}MaxParameterSize: {}\n\
				{t}MaxSamplers: {}\n\
				{t}MemBaseAddrAlign: {}\n\
				{t}MinDataTypeAlignSize: {}\n\
				{t}SingleFpConfig: {}\n\
				{t}GlobalMemCacheType: {}\n\
				{t}GlobalMemCachelineSize: {}\n\
				{t}GlobalMemCacheSize: {}\n\
				{t}GlobalMemSize: {}\n\
				{t}MaxConstantBufferSize: {}\n\
				{t}MaxConstantArgs: {}\n\
				{t}LocalMemType: {}\n\
				{t}LocalMemSize: {}\n\
				{t}ErrorCorrectionSupport: {}\n\
				{t}ProfilingTimerResolution: {}\n\
				{t}EndianLittle: {}\n\
				{t}Available: {}\n\
				{t}CompilerAvailable: {}\n\
				{t}ExecutionCapabilities: {}\n\
				{t}QueueProperties: {}\n\
				{t}Name: {}\n\
				{t}Vendor: {}\n\
				{t}DriverVersion: {}\n\
				{t}Profile: {}\n\
				{t}Version: {}\n\
				{t}Extensions: {}\n\
				{t}Platform: {}\n\
				{t}DoubleFpConfig: {}\n\
				{t}HalfFpConfig: {}\n\
				{t}PreferredVectorWidthHalf: {}\n\
				{t}HostUnifiedMemory: {}\n\
				{t}NativeVectorWidthChar: {}\n\
				{t}NativeVectorWidthShort: {}\n\
				{t}NativeVectorWidthInt: {}\n\
				{t}NativeVectorWidthLong: {}\n\
				{t}NativeVectorWidthFloat: {}\n\
				{t}NativeVectorWidthDouble: {}\n\
				{t}NativeVectorWidthHalf: {}\n\
				{t}OpenclCVersion: {}\n\
				{t}LinkerAvailable: {}\n\
				{t}BuiltInKernels: {}\n\
				{t}ImageMaxBufferSize: {}\n\
				{t}ImageMaxArraySize: {}\n\
				{t}ParentDevice: {}\n\
				{t}PartitionMaxSubDevices: {}\n\
				{t}PartitionProperties: {}\n\
				{t}PartitionAffinityDomain: {}\n\
				{t}PartitionType: {}\n\
				{t}ReferenceCount: {}\n\
				{t}PreferredInteropUserSync: {}\n\
				{t}PrintfBufferSize: {}\n\
				{t}ImagePitchAlignment: {}\n\
				{t}ImageBaseAddressAlignment: {}\n\
			",
			device_idx,
			core::get_device_info(&device, DeviceInfo::Type).unwrap(),
			core::get_device_info(&device, DeviceInfo::VendorId).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxComputeUnits).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxWorkItemDimensions).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxWorkGroupSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxWorkItemSizes).unwrap(),
			core::get_device_info(&device, DeviceInfo::PreferredVectorWidthChar).unwrap(),
			core::get_device_info(&device, DeviceInfo::PreferredVectorWidthShort).unwrap(),
			core::get_device_info(&device, DeviceInfo::PreferredVectorWidthInt).unwrap(),
			core::get_device_info(&device, DeviceInfo::PreferredVectorWidthLong).unwrap(),
			core::get_device_info(&device, DeviceInfo::PreferredVectorWidthFloat).unwrap(),
			core::get_device_info(&device, DeviceInfo::PreferredVectorWidthDouble).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxClockFrequency).unwrap(),
			core::get_device_info(&device, DeviceInfo::AddressBits).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxReadImageArgs).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxWriteImageArgs).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxMemAllocSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::Image2dMaxWidth).unwrap(),
			core::get_device_info(&device, DeviceInfo::Image2dMaxHeight).unwrap(),
			core::get_device_info(&device, DeviceInfo::Image3dMaxWidth).unwrap(),
			core::get_device_info(&device, DeviceInfo::Image3dMaxHeight).unwrap(),
			core::get_device_info(&device, DeviceInfo::Image3dMaxDepth).unwrap(),
			core::get_device_info(&device, DeviceInfo::ImageSupport).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxParameterSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxSamplers).unwrap(),
			core::get_device_info(&device, DeviceInfo::MemBaseAddrAlign).unwrap(),
			core::get_device_info(&device, DeviceInfo::MinDataTypeAlignSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::SingleFpConfig).unwrap(),
			core::get_device_info(&device, DeviceInfo::GlobalMemCacheType).unwrap(),
			core::get_device_info(&device, DeviceInfo::GlobalMemCachelineSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::GlobalMemCacheSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::GlobalMemSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxConstantBufferSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::MaxConstantArgs).unwrap(),
			core::get_device_info(&device, DeviceInfo::LocalMemType).unwrap(),
			core::get_device_info(&device, DeviceInfo::LocalMemSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::ErrorCorrectionSupport).unwrap(),
			core::get_device_info(&device, DeviceInfo::ProfilingTimerResolution).unwrap(),
			core::get_device_info(&device, DeviceInfo::EndianLittle).unwrap(),
			core::get_device_info(&device, DeviceInfo::Available).unwrap(),
			core::get_device_info(&device, DeviceInfo::CompilerAvailable).unwrap(),
			core::get_device_info(&device, DeviceInfo::ExecutionCapabilities).unwrap(),
			core::get_device_info(&device, DeviceInfo::QueueProperties).unwrap(),
			core::get_device_info(&device, DeviceInfo::Name).unwrap(),
			core::get_device_info(&device, DeviceInfo::Vendor).unwrap(),
			core::get_device_info(&device, DeviceInfo::DriverVersion).unwrap(),
			core::get_device_info(&device, DeviceInfo::Profile).unwrap(),
			core::get_device_info(&device, DeviceInfo::Version).unwrap(),
			core::get_device_info(&device, DeviceInfo::Extensions).unwrap(),
			core::get_device_info(&device, DeviceInfo::Platform).unwrap(),
			core::get_device_info(&device, DeviceInfo::DoubleFpConfig).unwrap(),
			core::get_device_info(&device, DeviceInfo::HalfFpConfig).unwrap(),
			core::get_device_info(&device, DeviceInfo::PreferredVectorWidthHalf).unwrap(),
			core::get_device_info(&device, DeviceInfo::HostUnifiedMemory).unwrap(),
			core::get_device_info(&device, DeviceInfo::NativeVectorWidthChar).unwrap(),
			core::get_device_info(&device, DeviceInfo::NativeVectorWidthShort).unwrap(),
			core::get_device_info(&device, DeviceInfo::NativeVectorWidthInt).unwrap(),
			core::get_device_info(&device, DeviceInfo::NativeVectorWidthLong).unwrap(),
			core::get_device_info(&device, DeviceInfo::NativeVectorWidthFloat).unwrap(),
			core::get_device_info(&device, DeviceInfo::NativeVectorWidthDouble).unwrap(),
			core::get_device_info(&device, DeviceInfo::NativeVectorWidthHalf).unwrap(),
			core::get_device_info(&device, DeviceInfo::OpenclCVersion).unwrap(),
			core::get_device_info(&device, DeviceInfo::LinkerAvailable).unwrap(),
			core::get_device_info(&device, DeviceInfo::BuiltInKernels).unwrap(),
			core::get_device_info(&device, DeviceInfo::ImageMaxBufferSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::ImageMaxArraySize).unwrap(),
			core::get_device_info(&device, DeviceInfo::ParentDevice).unwrap(),
			core::get_device_info(&device, DeviceInfo::PartitionMaxSubDevices).unwrap(),
			core::get_device_info(&device, DeviceInfo::PartitionProperties).unwrap(),
			core::get_device_info(&device, DeviceInfo::PartitionAffinityDomain).unwrap(),
			core::get_device_info(&device, DeviceInfo::PartitionType).unwrap(),
			core::get_device_info(&device, DeviceInfo::ReferenceCount).unwrap(),
			core::get_device_info(&device, DeviceInfo::PreferredInteropUserSync).unwrap(),
			core::get_device_info(&device, DeviceInfo::PrintfBufferSize).unwrap(),
			core::get_device_info(&device, DeviceInfo::ImagePitchAlignment).unwrap(),
			core::get_device_info(&device, DeviceInfo::ImageBaseAddressAlignment).unwrap(),
			t = util::TAB,
		);
    }


    //
	// CHANGE TO --->
	//


//		let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
 //            ("\n", "\n", "\n")
 //        } else {
 //            ("{ ", ", ", " }")
 //        };

 //        write!(f, "[FORMATTING NOT FULLY IMPLEMENTED][Device] {b}\
 //                DeviceInfo::Type: {}{d}\
 //                DeviceInfo::VendorId: {}{d}\
 //                DeviceInfo::MaxComputeUnits: {}{d}\
 //                DeviceInfo::MaxWorkItemDimensions: {}{d}\
 //                DeviceInfo::MaxWorkGroupSize: {}{d}\
 //                DeviceInfo::MaxWorkItemSizes: {}{d}\
 //                DeviceInfo::PreferredVectorWidthChar: {}{d}\
 //                DeviceInfo::PreferredVectorWidthShort: {}{d}\
 //                DeviceInfo::PreferredVectorWidthInt: {}{d}\
 //                DeviceInfo::PreferredVectorWidthLong: {}{d}\
 //                DeviceInfo::PreferredVectorWidthFloat: {}{d}\
 //                DeviceInfo::PreferredVectorWidthDouble: {}{d}\
 //                DeviceInfo::MaxClockFrequency: {}{d}\
 //                DeviceInfo::AddressBits: {}{d}\
 //                DeviceInfo::MaxReadImageArgs: {}{d}\
 //                DeviceInfo::MaxWriteImageArgs: {}{d}\
 //                DeviceInfo::MaxMemAllocSize: {}{d}\
 //                DeviceInfo::Image2dMaxWidth: {}{d}\
 //                DeviceInfo::Image2dMaxHeight: {}{d}\
 //                DeviceInfo::Image3dMaxWidth: {}{d}\
 //                DeviceInfo::Image3dMaxHeight: {}{d}\
 //                DeviceInfo::Image3dMaxDepth: {}{d}\
 //                DeviceInfo::ImageSupport: {}{d}\
 //                DeviceInfo::MaxParameterSize: {}{d}\
 //                DeviceInfo::MaxSamplers: {}{d}\
 //                DeviceInfo::MemBaseAddrAlign: {}{d}\
 //                DeviceInfo::MinDataTypeAlignSize: {}{d}\
 //                DeviceInfo::SingleFpConfig: {}{d}\
 //                DeviceInfo::GlobalMemCacheType: {}{d}\
 //                DeviceInfo::GlobalMemCachelineSize: {}{d}\
 //                DeviceInfo::GlobalMemCacheSize: {}{d}\
 //                DeviceInfo::GlobalMemSize: {}{d}\
 //                DeviceInfo::MaxConstantBufferSize: {}{d}\
 //                DeviceInfo::MaxConstantArgs: {}{d}\
 //                DeviceInfo::LocalMemType: {}{d}\
 //                DeviceInfo::LocalMemSize: {}{d}\
 //                DeviceInfo::ErrorCorrectionSupport: {}{d}\
 //                DeviceInfo::ProfilingTimerResolution: {}{d}\
 //                DeviceInfo::EndianLittle: {}{d}\
 //                DeviceInfo::Available: {}{d}\
 //                DeviceInfo::CompilerAvailable: {}{d}\
 //                DeviceInfo::ExecutionCapabilities: {}{d}\
 //                DeviceInfo::QueueProperties: {}{d}\
 //                DeviceInfo::Name: {}{d}\
 //                DeviceInfo::Vendor: {}{d}\
 //                DeviceInfo::DriverVersion: {}{d}\
 //                DeviceInfo::Profile: {}{d}\
 //                DeviceInfo::Version: {}{d}\
 //                DeviceInfo::Extensions: {}{d}\
 //                DeviceInfo::Platform: {}{d}\
 //                DeviceInfo::DoubleFpConfig: {}{d}\
 //                DeviceInfo::HalfFpConfig: {}{d}\
 //                DeviceInfo::PreferredVectorWidthHalf: {}{d}\
 //                DeviceInfo::HostUnifiedMemory: {}{d}\
 //                DeviceInfo::NativeVectorWidthChar: {}{d}\
 //                DeviceInfo::NativeVectorWidthShort: {}{d}\
 //                DeviceInfo::NativeVectorWidthInt: {}{d}\
 //                DeviceInfo::NativeVectorWidthLong: {}{d}\
 //                DeviceInfo::NativeVectorWidthFloat: {}{d}\
 //                DeviceInfo::NativeVectorWidthDouble: {}{d}\
 //                DeviceInfo::NativeVectorWidthHalf: {}{d}\
 //                DeviceInfo::OpenclCVersion: {}{d}\
 //                DeviceInfo::LinkerAvailable: {}{d}\
 //                DeviceInfo::BuiltInKernels: {}{d}\
 //                DeviceInfo::ImageMaxBufferSize: {}{d}\
 //                DeviceInfo::ImageMaxArraySize: {}{d}\
 //                DeviceInfo::ParentDevice: {}{d}\
 //                DeviceInfo::PartitionMaxSubDevices: {}{d}\
 //                DeviceInfo::PartitionProperties: {}{d}\
 //                DeviceInfo::PartitionAffinityDomain: {}{d}\
 //                DeviceInfo::PartitionType: {}{d}\
 //                DeviceInfo::ReferenceCount: {}{d}\
 //                DeviceInfo::PreferredInteropUserSync: {}{d}\
 //                DeviceInfo::PrintfBufferSize: {}{d}\
 //                DeviceInfo::ImagePitchAlignment: {}{d}\
 //                DeviceInfo::ImageBaseAddressAlignment: {}{e}\
 //            ",
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Type)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::VendorId)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxComputeUnits)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxWorkItemDimensions)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxWorkGroupSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxWorkItemSizes)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthChar)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthShort)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthInt)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthLong)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthFloat)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthDouble)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxClockFrequency)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::AddressBits)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxReadImageArgs)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxWriteImageArgs)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxMemAllocSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Image2dMaxWidth)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Image2dMaxHeight)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Image3dMaxWidth)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Image3dMaxHeight)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Image3dMaxDepth)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ImageSupport)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxParameterSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxSamplers)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MemBaseAddrAlign)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MinDataTypeAlignSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::SingleFpConfig)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::GlobalMemCacheType)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::GlobalMemCachelineSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::GlobalMemCacheSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::GlobalMemSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxConstantBufferSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::MaxConstantArgs)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::LocalMemType)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::LocalMemSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ErrorCorrectionSupport)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ProfilingTimerResolution)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::EndianLittle)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Available)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::CompilerAvailable)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ExecutionCapabilities)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::QueueProperties)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Name)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Vendor)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::DriverVersion)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Profile)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Version)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Extensions)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::Platform)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::DoubleFpConfig)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::HalfFpConfig)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredVectorWidthHalf)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::HostUnifiedMemory)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthChar)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthShort)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthInt)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthLong)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthFloat)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthDouble)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::NativeVectorWidthHalf)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::OpenclCVersion)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::LinkerAvailable)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::BuiltInKernels)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ImageMaxBufferSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ImageMaxArraySize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ParentDevice)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PartitionMaxSubDevices)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PartitionProperties)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PartitionAffinityDomain)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PartitionType)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ReferenceCount)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PreferredInteropUserSync)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::PrintfBufferSize)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ImagePitchAlignment)),
 //            try_to_str(core::get_device_info(&self.0, DeviceInfo::ImageBaseAddressAlignment)),
 //            b = begin,
 //            d = delim,
 //            e = end,
 //        )

    // ##################################################
    // #################### CONTEXT #####################
    // ##################################################

    println!("Context:\n\
			{t}Reference Count: {}\n\
			{t}Devices: {}\n\
			{t}Properties: {}\n\
			{t}Device Count: {}\n\
		",
		core::get_context_info(context.core_as_ref(), ContextInfo::ReferenceCount).unwrap(),
		core::get_context_info(context.core_as_ref(), ContextInfo::Devices).unwrap(),
		core::get_context_info(context.core_as_ref(), ContextInfo::Properties).unwrap(),
		core::get_context_info(context.core_as_ref(), ContextInfo::NumDevices).unwrap(),
		t = util::TAB,
	);

    //
	// CHANGE TO --->
	//

		// let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
  //           ("\n", "\n", "\n")
  //       } else {
  //           ("{ ", ", ", " }")
  //       };

  //       write!(f, "[Context]: {b}\
  //               Reference Count: {}{d}\
  //               Devices: {}{d}\
  //               Properties: {}{d}\
  //               Device Count: {}{e}\
  //           ",
  //           core::get_context_info(&self.obj_core, ContextInfo::ReferenceCount).unwrap(),
  //           core::get_context_info(&self.obj_core, ContextInfo::Devices).unwrap(),
  //           core::get_context_info(&self.obj_core, ContextInfo::Properties).unwrap(),
  //           core::get_context_info(&self.obj_core, ContextInfo::NumDevices).unwrap(),
  //           b = begin,
  //           d = delim,
  //           e = end,
  //       )


    // ##################################################
    // ##################### QUEUE ######################
    // ##################################################


	println!("Command Queue:\n\
			{t}Context: {}\n\
			{t}Device: {}\n\
			{t}ReferenceCount: {}\n\
			{t}Properties: {}\n\
		",
		core::get_command_queue_info(queue.core_as_ref(), CommandQueueInfo::Context).unwrap(),
		core::get_command_queue_info(queue.core_as_ref(), CommandQueueInfo::Device).unwrap(),
		core::get_command_queue_info(queue.core_as_ref(), CommandQueueInfo::ReferenceCount).unwrap(),
		core::get_command_queue_info(queue.core_as_ref(), CommandQueueInfo::Properties).unwrap(),
		t = util::TAB,
	);


	//
	// CHANGE TO --->
	//

	// write!(f, "{}", &self.to_string())
        // let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
        //     ("\n", "\n", "\n")
        // } else {
        //     ("{ ", ", ", " }")
        // };

        // // TemporaryPlaceholderVariant(Vec<u8>),
        // // Context(Context),
        // // Device(DeviceId),
        // // ReferenceCount(u32),
        // // Properties(CommandQueueProperties),
        // // Error(Box<OclError>),

        // write!(f, "[Queue]: {b}\
        //         Context: {}{d}\
        //         Device: {}{d}\
        //         ReferenceCount: {}{d}\
        //         Properties: {}{e}\
        //     ",
        //     self.info(CommandQueueInfo::Context),
        //     self.info(CommandQueueInfo::Device),
        //     self.info(CommandQueueInfo::ReferenceCount),
        //     self.info(CommandQueueInfo::Properties),
        //     b = begin,
        //     d = delim,
        //     e = end,
        // )


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

    println!("Buffer Memory:\n\
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
		core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Type).unwrap(),
	    core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Flags).unwrap(),
        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Size).unwrap(),
        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::HostPtr).unwrap(),
        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::MapCount).unwrap(),
        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::ReferenceCount).unwrap(),
        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Context).unwrap(),
        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::AssociatedMemobject).unwrap(),
        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Offset).unwrap(),
		t = util::TAB,
	);


	//
	// CHANGE TO --->
	//


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
	// 	core::get_image_info(image.core_as_ref(), ImageInfo::Format).unwrap(),
	//     core::get_image_info(image.core_as_ref(), ImageInfo::ElementSize).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::RowPitch).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::SlicePitch).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::Width).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::Height).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::Depth).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::ArraySize).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::Buffer).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::NumMipLevels).unwrap(),
 //        core::get_image_info(image.core_as_ref(), ImageInfo::NumSamples).unwrap(),
	// 	t = util::TAB,
	// );


		println!("Image: {b}\
                {t}ElementSize: {}{d}\
                {t}RowPitch: {}{d}\
                {t}SlicePitch: {}{d}\
                {t}Width: {}{d}\
                {t}Height: {}{d}\
                {t}Depth: {}{d}\
                {t}ArraySize: {}{d}\
                {t}Buffer: {}{d}\
                {t}NumMipLevels: {}{d}\
                {t}NumSamples: {}{e}\
            ",
            core::get_image_info(image.core_as_ref(), ImageInfo::ElementSize).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::RowPitch).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::SlicePitch).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::Width).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::Height).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::Depth).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::ArraySize).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::Buffer).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::NumMipLevels).unwrap(),
            core::get_image_info(image.core_as_ref(), ImageInfo::NumSamples).unwrap(),
            b = begin,
            d = delim,
            e = end,
            t = util::TAB,
        );

		println!("{t}Image Mem:\n\
				{t}{t}Type: {}\n\
		        {t}{t}Flags: {}\n\
		        {t}{t}Size: {}\n\
		        {t}{t}HostPtr: {}\n\
		        {t}{t}MapCount: {}\n\
		        {t}{t}ReferenceCount: {}\n\
		        {t}{t}Context: {}\n\
		        {t}{t}AssociatedMemobject: {}\n\
		        {t}{t}Offset: {}\n\
			",
			core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Type).unwrap(),
		    core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Flags).unwrap(),
	        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Size).unwrap(),
	        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::HostPtr).unwrap(),
	        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::MapCount).unwrap(),
	        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::ReferenceCount).unwrap(),
	        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Context).unwrap(),
	        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::AssociatedMemobject).unwrap(),
	        core::get_mem_object_info(buffer.core_as_ref(), MemInfo::Offset).unwrap(),
			t = util::TAB,
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

 //    println!("[UNIMPLEMENTED] Sampler:\n\
	// 		{t}ReferenceCount: {}\n\
 //            {t}Context: {}\n\
 //            {t}NormalizedCoords: {}\n\
 //            {t}AddressingMode: {}\n\
 //            {t}FilterMode: {}\n\
	// 	",
	// 	core::get_sampler_info(sampler.core_as_ref(), SamplerInfo::ReferenceCount).unwrap(),
 //        core::get_sampler_info(sampler.core_as_ref(), SamplerInfo::Context).unwrap(),
 //        core::get_sampler_info(sampler.core_as_ref(), SamplerInfo::NormalizedCoords).unwrap(),
 //        core::get_sampler_info(sampler.core_as_ref(), SamplerInfo::AddressingMode).unwrap(),
 //        core::get_sampler_info(sampler.core_as_ref(), SamplerInfo::FilterMode).unwrap(),
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
		core::get_program_info(program.core_as_ref(), ProgramInfo::ReferenceCount).unwrap(),
        core::get_program_info(program.core_as_ref(), ProgramInfo::Context).unwrap(),
        core::get_program_info(program.core_as_ref(), ProgramInfo::NumDevices).unwrap(),
        core::get_program_info(program.core_as_ref(), ProgramInfo::Devices).unwrap(),
        core::get_program_info(program.core_as_ref(), ProgramInfo::Source).unwrap(),
        core::get_program_info(program.core_as_ref(), ProgramInfo::BinarySizes).unwrap(),
        core::get_program_info(program.core_as_ref(), ProgramInfo::Binaries).unwrap(),
        core::get_program_info(program.core_as_ref(), ProgramInfo::NumKernels).unwrap(),
        core::get_program_info(program.core_as_ref(), ProgramInfo::KernelNames).unwrap(),
		t = util::TAB,
	);

	//
	// CHANGE TO --->
	//


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
		core::get_program_build_info(program.core_as_ref(), &device, ProgramBuildInfo::BuildStatus).unwrap(),
        core::get_program_build_info(program.core_as_ref(), &device, ProgramBuildInfo::BuildOptions).unwrap(),
        core::get_program_build_info(program.core_as_ref(), &device, ProgramBuildInfo::BuildLog).unwrap(),
        core::get_program_build_info(program.core_as_ref(), &device, ProgramBuildInfo::BinaryType).unwrap(),
		t = util::TAB,
	);

	//
	// CHANGE TO --->
	//

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
		core::get_kernel_info(kernel.core_as_ref(), KernelInfo::FunctionName).unwrap(),
	    core::get_kernel_info(kernel.core_as_ref(), KernelInfo::NumArgs).unwrap(),
        core::get_kernel_info(kernel.core_as_ref(), KernelInfo::ReferenceCount).unwrap(),
        core::get_kernel_info(kernel.core_as_ref(), KernelInfo::Context).unwrap(),
        core::get_kernel_info(kernel.core_as_ref(), KernelInfo::Program).unwrap(),
        core::get_kernel_info(kernel.core_as_ref(), KernelInfo::Attributes).unwrap(),
		t = util::TAB,
	);

    //
	// CHANGE TO BELOW:
	//

	// let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
	//        ("\n", "\n", "\n")
	//    } else {
	//        ("{ ", ", ", " }")
	//    };

	// write!(f, "[Kernel]: {b}\
	//        FunctionName: {}{d}\
	//        ReferenceCount: {}{d}\
	//        Context: {}{d}\
	//        Program: {}{d}\
	//        Attributes: {}{e}\
	//    ",
	//    self.info(KernelInfo::FunctionName),
	//    self.info(KernelInfo::ReferenceCount),
	//    self.info(KernelInfo::Context),
	//    self.info(KernelInfo::Program),
	//    self.info(KernelInfo::Attributes),
	//    b = begin,
	//    d = delim,
	//    e = end,
	// )

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
		core::get_kernel_arg_info(kernel.core_as_ref(), 0, KernelArgInfo::AddressQualifier).unwrap(),
        core::get_kernel_arg_info(kernel.core_as_ref(), 0, KernelArgInfo::AccessQualifier).unwrap(),
        core::get_kernel_arg_info(kernel.core_as_ref(), 0, KernelArgInfo::TypeName).unwrap(),
        core::get_kernel_arg_info(kernel.core_as_ref(), 0, KernelArgInfo::TypeQualifier).unwrap(),
        core::get_kernel_arg_info(kernel.core_as_ref(), 0, KernelArgInfo::Name).unwrap(),
		t = util::TAB,
	);

	//
	// CHANGE TO --->
	//

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
		core::get_kernel_work_group_info(kernel.core_as_ref(), &device, 
			KernelWorkGroupInfo::WorkGroupSize).unwrap(),
	    core::get_kernel_work_group_info(kernel.core_as_ref(), &device, 
	    	KernelWorkGroupInfo::CompileWorkGroupSize).unwrap(),
        core::get_kernel_work_group_info(kernel.core_as_ref(), &device, 
        	KernelWorkGroupInfo::LocalMemSize).unwrap(),
        core::get_kernel_work_group_info(kernel.core_as_ref(), &device, 
        	KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple).unwrap(),
        core::get_kernel_work_group_info(kernel.core_as_ref(), &device, 
        	KernelWorkGroupInfo::PrivateMemSize).unwrap(),
        // core::get_kernel_work_group_info(kernel.core_as_ref(), &device, 
        // 	KernelWorkGroupInfo::GlobalWorkSize).unwrap(),
    	"[KernelWorkGroupInfo::GlobalWorkSize not avaliable in this configuration]",
		t = util::TAB,
	);

	//
	// CHANGE TO --->
	//

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
		core::get_event_info(event.core_as_ref(), EventInfo::CommandQueue).unwrap(),
        core::get_event_info(event.core_as_ref(), EventInfo::CommandType).unwrap(),
        core::get_event_info(event.core_as_ref(), EventInfo::ReferenceCount).unwrap(),
        core::get_event_info(event.core_as_ref(), EventInfo::CommandExecutionStatus).unwrap(),
        core::get_event_info(event.core_as_ref(), EventInfo::Context).unwrap(),
		t = util::TAB,
	);

	//
	// CHANGE TO --->
	//

	let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
            ("\n", "\n", "\n")
        } else {
            ("{ ", ", ", " }")
        };

        write!(f, "[Event]: {b}\
                CommandQueue: {}{d}\
                CommandType: {}{d}\
                ReferenceCount: {}{d}\
                CommandExecutionStatus: {}{d}\
                Context: {}{e}\
            ",
            core::get_event_info(&self.0, EventInfo::CommandQueue).unwrap(),
            core::get_event_info(&self.0, EventInfo::CommandType).unwrap(),
            core::get_event_info(&self.0, EventInfo::ReferenceCount).unwrap(),
            core::get_event_info(&self.0, EventInfo::CommandExecutionStatus).unwrap(),
            core::get_event_info(&self.0, EventInfo::Context).unwrap(),
            b = begin,
            d = delim,
            e = end,
        )

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
		core::get_event_profiling_info(event.core_as_ref(), ProfilingInfo::Queued).unwrap(),
        core::get_event_profiling_info(event.core_as_ref(), ProfilingInfo::Submit).unwrap(),
        core::get_event_profiling_info(event.core_as_ref(), ProfilingInfo::Start).unwrap(),
        core::get_event_profiling_info(event.core_as_ref(), ProfilingInfo::End).unwrap(),
		t = util::TAB,
	);

	//
	// CHANGE TO --->
	//


	// ##################################################
    // ###################### END #######################
    // ##################################################

    print!("\n");
}
