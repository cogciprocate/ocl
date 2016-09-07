//! Get information about all the things using `core` function calls.
//!
//! [UNDERGOING SOME REDESIGN]

extern crate ocl;
#[macro_use] extern crate colorify;

// use ocl::Error as OclError;
use ocl::{Platform, Device, Context, Queue, Buffer, Image, Sampler, Program, Kernel, Event, EventList};
use ocl::core::{self, PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ImageInfo,
    SamplerInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo,
    EventInfo, ProfilingInfo};
use ocl::util;

const DIMS: [usize; 3] = [1024, 64, 16];
const INFO_FORMAT_MULTILINE: bool = true;

static SRC: &'static str = r#"
    __kernel void multiply(float coeff, __global float* buffer) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

fn main() {
    let platforms = Platform::list();
    // let platform = platforms[platforms.len() - 1];
    for platform in platforms.iter() {
        print_platform(platform.clone());
    }
}

fn print_platform(platform: Platform) {
    for device in Device::list_all(&platform).unwrap() {
        print_platform_device(platform.clone(), device);
    }
}

fn print_platform_device(platform: Platform, device: Device) {
    let device_version = device.version().unwrap();

    let context = Context::builder().platform(platform).devices(device).build().unwrap();
    let program = Program::builder()
        .devices(device)
        .src(SRC)
        .build(&context).unwrap();
    let queue = Queue::new(&context, device).unwrap();
    let buffer = Buffer::<f32>::new(&queue, None, &DIMS, None).unwrap();
    let image = Image::<u8>::builder()
        .dims(&DIMS)
        .build(&queue).unwrap();
    let sampler = Sampler::with_defaults(&context).unwrap();
        let kernel = Kernel::new("multiply", &program, &queue).unwrap()
        .gws(&DIMS)
        .arg_scl(10.0f32)
        .arg_buf(&buffer);

    let mut event_list = EventList::new();
    kernel.cmd().enew(&mut event_list).enq().unwrap();
    event_list.wait().unwrap();

    let mut event = Event::empty();
    buffer.cmd().write(&vec![0.0; DIMS[0]]).enew(&mut event).enq().unwrap();
    event.wait().unwrap();

    println!("############### OpenCL Platform-Device Full Info ################");
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
        core::get_platform_info(context.platform().unwrap(), PlatformInfo::Profile),
        core::get_platform_info(context.platform().unwrap(), PlatformInfo::Version),
        core::get_platform_info(context.platform().unwrap(), PlatformInfo::Name),
        core::get_platform_info(context.platform().unwrap(), PlatformInfo::Vendor),
        core::get_platform_info(context.platform().unwrap(), PlatformInfo::Extensions),
        t = util::colors::TAB,
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
            core::get_device_info(&device, DeviceInfo::Type),
            core::get_device_info(&device, DeviceInfo::VendorId),
            core::get_device_info(&device, DeviceInfo::MaxComputeUnits),
            core::get_device_info(&device, DeviceInfo::MaxWorkItemDimensions),
            core::get_device_info(&device, DeviceInfo::MaxWorkGroupSize),
            core::get_device_info(&device, DeviceInfo::MaxWorkItemSizes),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthChar),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthShort),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthInt),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthLong),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthFloat),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthDouble),
            core::get_device_info(&device, DeviceInfo::MaxClockFrequency),
            core::get_device_info(&device, DeviceInfo::AddressBits),
            core::get_device_info(&device, DeviceInfo::MaxReadImageArgs),
            core::get_device_info(&device, DeviceInfo::MaxWriteImageArgs),
            core::get_device_info(&device, DeviceInfo::MaxMemAllocSize),
            core::get_device_info(&device, DeviceInfo::Image2dMaxWidth),
            core::get_device_info(&device, DeviceInfo::Image2dMaxHeight),
            core::get_device_info(&device, DeviceInfo::Image3dMaxWidth),
            core::get_device_info(&device, DeviceInfo::Image3dMaxHeight),
            core::get_device_info(&device, DeviceInfo::Image3dMaxDepth),
            core::get_device_info(&device, DeviceInfo::ImageSupport),
            core::get_device_info(&device, DeviceInfo::MaxParameterSize),
            core::get_device_info(&device, DeviceInfo::MaxSamplers),
            core::get_device_info(&device, DeviceInfo::MemBaseAddrAlign),
            core::get_device_info(&device, DeviceInfo::MinDataTypeAlignSize),
            core::get_device_info(&device, DeviceInfo::SingleFpConfig),
            core::get_device_info(&device, DeviceInfo::GlobalMemCacheType),
            core::get_device_info(&device, DeviceInfo::GlobalMemCachelineSize),
            core::get_device_info(&device, DeviceInfo::GlobalMemCacheSize),
            core::get_device_info(&device, DeviceInfo::GlobalMemSize),
            core::get_device_info(&device, DeviceInfo::MaxConstantBufferSize),
            core::get_device_info(&device, DeviceInfo::MaxConstantArgs),
            core::get_device_info(&device, DeviceInfo::LocalMemType),
            core::get_device_info(&device, DeviceInfo::LocalMemSize),
            core::get_device_info(&device, DeviceInfo::ErrorCorrectionSupport),
            core::get_device_info(&device, DeviceInfo::ProfilingTimerResolution),
            core::get_device_info(&device, DeviceInfo::EndianLittle),
            core::get_device_info(&device, DeviceInfo::Available),
            core::get_device_info(&device, DeviceInfo::CompilerAvailable),
            core::get_device_info(&device, DeviceInfo::ExecutionCapabilities),
            core::get_device_info(&device, DeviceInfo::QueueProperties),
            core::get_device_info(&device, DeviceInfo::Name),
            core::get_device_info(&device, DeviceInfo::Vendor),
            core::get_device_info(&device, DeviceInfo::DriverVersion),
            core::get_device_info(&device, DeviceInfo::Profile),
            core::get_device_info(&device, DeviceInfo::Version),
            core::get_device_info(&device, DeviceInfo::Extensions),
            core::get_device_info(&device, DeviceInfo::Platform),
            core::get_device_info(&device, DeviceInfo::DoubleFpConfig),
            core::get_device_info(&device, DeviceInfo::HalfFpConfig),
            core::get_device_info(&device, DeviceInfo::PreferredVectorWidthHalf),
            core::get_device_info(&device, DeviceInfo::HostUnifiedMemory),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthChar),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthShort),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthInt),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthLong),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthFloat),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthDouble),
            core::get_device_info(&device, DeviceInfo::NativeVectorWidthHalf),
            core::get_device_info(&device, DeviceInfo::OpenclCVersion),
            core::get_device_info(&device, DeviceInfo::LinkerAvailable),
            core::get_device_info(&device, DeviceInfo::BuiltInKernels),
            core::get_device_info(&device, DeviceInfo::ImageMaxBufferSize),
            core::get_device_info(&device, DeviceInfo::ImageMaxArraySize),
            core::get_device_info(&device, DeviceInfo::ParentDevice),
            core::get_device_info(&device, DeviceInfo::PartitionMaxSubDevices),
            core::get_device_info(&device, DeviceInfo::PartitionProperties),
            core::get_device_info(&device, DeviceInfo::PartitionAffinityDomain),
            core::get_device_info(&device, DeviceInfo::PartitionType),
            core::get_device_info(&device, DeviceInfo::ReferenceCount),
            core::get_device_info(&device, DeviceInfo::PreferredInteropUserSync),
            core::get_device_info(&device, DeviceInfo::PrintfBufferSize),
            core::get_device_info(&device, DeviceInfo::ImagePitchAlignment),
            core::get_device_info(&device, DeviceInfo::ImageBaseAddressAlignment),
            t = util::colors::TAB,
        );
    }


    //
    // CHANGE TO --->
    //


//      let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
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
        core::get_context_info(&context, ContextInfo::ReferenceCount),
        core::get_context_info(&context, ContextInfo::Devices),
        core::get_context_info(&context, ContextInfo::Properties),
        core::get_context_info(&context, ContextInfo::NumDevices),
        t = util::colors::TAB,
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
  //           core::get_context_info(&self.obj_core, ContextInfo::ReferenceCount),
  //           core::get_context_info(&self.obj_core, ContextInfo::Devices),
  //           core::get_context_info(&self.obj_core, ContextInfo::Properties),
  //           core::get_context_info(&self.obj_core, ContextInfo::NumDevices),
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
        core::get_command_queue_info(&queue, CommandQueueInfo::Context),
        core::get_command_queue_info(&queue, CommandQueueInfo::Device),
        core::get_command_queue_info(&queue, CommandQueueInfo::ReferenceCount),
        core::get_command_queue_info(&queue, CommandQueueInfo::Properties),
        t = util::colors::TAB,
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
        core::get_mem_object_info(&buffer, MemInfo::Type),
        core::get_mem_object_info(&buffer, MemInfo::Flags),
        core::get_mem_object_info(&buffer, MemInfo::Size),
        core::get_mem_object_info(&buffer, MemInfo::HostPtr),
        core::get_mem_object_info(&buffer, MemInfo::MapCount),
        core::get_mem_object_info(&buffer, MemInfo::ReferenceCount),
        core::get_mem_object_info(&buffer, MemInfo::Context),
        core::get_mem_object_info(&buffer, MemInfo::AssociatedMemobject),
        core::get_mem_object_info(&buffer, MemInfo::Offset),
        t = util::colors::TAB,
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
    //      {t}Format: {}\n\
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
    //  ",
    //  core::get_image_info(&image, ImageInfo::Format).unwrap(),
    //     core::get_image_info(&image, ImageInfo::ElementSize).unwrap(),
 //        core::get_image_info(&image, ImageInfo::RowPitch).unwrap(),
 //        core::get_image_info(&image, ImageInfo::SlicePitch).unwrap(),
 //        core::get_image_info(&image, ImageInfo::Width).unwrap(),
 //        core::get_image_info(&image, ImageInfo::Height).unwrap(),
 //        core::get_image_info(&image, ImageInfo::Depth).unwrap(),
 //        core::get_image_info(&image, ImageInfo::ArraySize).unwrap(),
 //        core::get_image_info(&image, ImageInfo::Buffer).unwrap(),
 //        core::get_image_info(&image, ImageInfo::NumMipLevels).unwrap(),
 //        core::get_image_info(&image, ImageInfo::NumSamples).unwrap(),
    //  t = util::colors::TAB,
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
            core::get_image_info(&image, ImageInfo::ElementSize),
            core::get_image_info(&image, ImageInfo::RowPitch),
            core::get_image_info(&image, ImageInfo::SlicePitch),
            core::get_image_info(&image, ImageInfo::Width),
            core::get_image_info(&image, ImageInfo::Height),
            core::get_image_info(&image, ImageInfo::Depth),
            core::get_image_info(&image, ImageInfo::ArraySize),
            core::get_image_info(&image, ImageInfo::Buffer),
            core::get_image_info(&image, ImageInfo::NumMipLevels),
            core::get_image_info(&image, ImageInfo::NumSamples),
            b = begin,
            d = delim,
            e = end,
            t = util::colors::TAB,
        );

        println!("{t}Image Memory:\n\
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
            core::get_mem_object_info(&buffer, MemInfo::Type),
            core::get_mem_object_info(&buffer, MemInfo::Flags),
            core::get_mem_object_info(&buffer, MemInfo::Size),
            core::get_mem_object_info(&buffer, MemInfo::HostPtr),
            core::get_mem_object_info(&buffer, MemInfo::MapCount),
            core::get_mem_object_info(&buffer, MemInfo::ReferenceCount),
            core::get_mem_object_info(&buffer, MemInfo::Context),
            core::get_mem_object_info(&buffer, MemInfo::AssociatedMemobject),
            core::get_mem_object_info(&buffer, MemInfo::Offset),
            t = util::colors::TAB,
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

    println!("Sampler:\n\
            {t}ReferenceCount: {}\n\
            {t}Context: {}\n\
            {t}NormalizedCoords: {}\n\
            {t}AddressingMode: {}\n\
            {t}FilterMode: {}\n\
        ",
        core::get_sampler_info(&sampler, SamplerInfo::ReferenceCount),
        core::get_sampler_info(&sampler, SamplerInfo::Context),
        core::get_sampler_info(&sampler, SamplerInfo::NormalizedCoords),
        core::get_sampler_info(&sampler, SamplerInfo::AddressingMode),
        core::get_sampler_info(&sampler, SamplerInfo::FilterMode),
        t = util::colors::TAB,
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
        core::get_program_info(&program, ProgramInfo::ReferenceCount),
        core::get_program_info(&program, ProgramInfo::Context),
        core::get_program_info(&program, ProgramInfo::NumDevices),
        core::get_program_info(&program, ProgramInfo::Devices),
        core::get_program_info(&program, ProgramInfo::Source),
        core::get_program_info(&program, ProgramInfo::BinarySizes),
        //core::get_program_info(&program, ProgramInfo::Binaries),
        "n/a",
        core::get_program_info(&program, ProgramInfo::NumKernels),
        core::get_program_info(&program, ProgramInfo::KernelNames),
        t = util::colors::TAB,
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

        // // ReferenceCount = cl_h::CL_PROGRAM_REFERENCE_COUNT as isize,
        // // Context = cl_h::CL_PROGRAM_CONTEXT as isize,
        // // NumDevices = cl_h::CL_PROGRAM_NUM_DEVICES as isize,
        // // Devices = cl_h::CL_PROGRAM_DEVICES as isize,
        // // Source = cl_h::CL_PROGRAM_SOURCE as isize,
        // // BinarySizes = cl_h::CL_PROGRAM_BINARY_SIZES as isize,
        // // Binaries = cl_h::CL_PROGRAM_BINARIES as isize,
        // // NumKernels = cl_h::CL_PROGRAM_NUM_KERNELS as isize,
        // // KernelNames = cl_h::CL_PROGRAM_KERNEL_NAMES as isize,

        // write!(f, "[Program]: {b}\
        //         ReferenceCount: {}{d}\
        //         Context: {}{d}\
        //         NumDevices: {}{d}\
        //         Devices: {}{d}\
        //         Source: {}{d}\
        //         BinarySizes: {}{d}\
        //         Binaries: {}{d}\
        //         NumKernels: {}{d}\
        //         KernelNames: {}{e}\
        //     ",
        //     self.info(ProgramInfo::ReferenceCount),
        //     self.info(ProgramInfo::Context),
        //     self.info(ProgramInfo::NumDevices),
        //     self.info(ProgramInfo::Devices),
        //     self.info(ProgramInfo::Source),
        //     self.info(ProgramInfo::BinarySizes),
        //     self.info(ProgramInfo::Binaries),
        //     self.info(ProgramInfo::NumKernels),
        //     self.info(ProgramInfo::KernelNames),
        //     b = begin,
        //     d = delim,
        //     e = end,
        // )

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
            {t}BuildLog: \n\n{}\n\n\
            {t}BinaryType: {}\n\
        ",
        core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildStatus),
        core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildOptions),
        core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildLog),
        core::get_program_build_info(&program, &device, ProgramBuildInfo::BinaryType),
        t = util::colors::TAB,
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
        core::get_kernel_info(&kernel, KernelInfo::FunctionName),
        core::get_kernel_info(&kernel, KernelInfo::NumArgs),
        core::get_kernel_info(&kernel, KernelInfo::ReferenceCount),
        core::get_kernel_info(&kernel, KernelInfo::Context),
        core::get_kernel_info(&kernel, KernelInfo::Program),
        core::get_kernel_info(&kernel, KernelInfo::Attributes),
        t = util::colors::TAB,
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

    println!("Kernel Argument [0]:\n\
            {t}AddressQualifier: {}\n\
            {t}AccessQualifier: {}\n\
            {t}TypeName: {}\n\
            {t}TypeQualifier: {}\n\
            {t}Name: {}\n\
        ",
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::AddressQualifier, Some(&[device_version])),
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::AccessQualifier, Some(&[device_version])),
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::TypeName, Some(&[device_version])),
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::TypeQualifier, Some(&[device_version])),
        core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::Name, Some(&[device_version])),
        t = util::colors::TAB,
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
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::WorkGroupSize),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::CompileWorkGroupSize),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::LocalMemSize),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple),
        core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::PrivateMemSize),
        // core::get_kernel_work_group_info(&kernel, &device,
        //  KernelWorkGroupInfo::GlobalWorkSize).unwrap(),
        "[KernelWorkGroupInfo::GlobalWorkSize not avaliable in this configuration]",
        t = util::colors::TAB,
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

    println!("Event:\n\
            {t}CommandQueue: {}\n\
            {t}CommandType: {}\n\
            {t}ReferenceCount: {}\n\
            {t}CommandExecutionStatus: {}\n\
            {t}Context: {}\n\
        ",
        core::get_event_info(&event, EventInfo::CommandQueue),
        core::get_event_info(&event, EventInfo::CommandType),
        core::get_event_info(&event, EventInfo::ReferenceCount),
        core::get_event_info(&event, EventInfo::CommandExecutionStatus),
        core::get_event_info(&event, EventInfo::Context),
        t = util::colors::TAB,
    );

    //
    // CHANGE TO --->
    //

    // let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
 //            ("\n", "\n", "\n")
 //        } else {
 //            ("{ ", ", ", " }")
 //        };

 //        write!(f, "[Event]: {b}\
 //                CommandQueue: {}{d}\
 //                CommandType: {}{d}\
 //                ReferenceCount: {}{d}\
 //                CommandExecutionStatus: {}{d}\
 //                Context: {}{e}\
 //            ",
 //            core::get_event_info(&self.0, EventInfo::CommandQueue),
 //            core::get_event_info(&self.0, EventInfo::CommandType),
 //            core::get_event_info(&self.0, EventInfo::ReferenceCount),
 //            core::get_event_info(&self.0, EventInfo::CommandExecutionStatus),
 //            core::get_event_info(&self.0, EventInfo::Context),
 //            b = begin,
 //            d = delim,
 //            e = end,
 //        )

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

    println!("Event Profiling:\n\
            {t}Queued: {}\n\
            {t}Submit: {}\n\
            {t}Start: {}\n\
            {t}End: {}\n\
        ",
        core::get_event_profiling_info(&event, ProfilingInfo::Queued),
        core::get_event_profiling_info(&event, ProfilingInfo::Submit),
        core::get_event_profiling_info(&event, ProfilingInfo::Start),
        core::get_event_profiling_info(&event, ProfilingInfo::End),
        t = util::colors::TAB,
    );

    //
    // CHANGE TO --->
    //


    // ##################################################
    // ###################### END #######################
    // ##################################################

    print!("\n");
}
