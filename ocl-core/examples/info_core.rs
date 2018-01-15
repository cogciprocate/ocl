//! Get information about all the things using `core` function calls.
//!
//! Set `INFO_FORMAT_MULTILINE` to `false` for compact printing.

extern crate ocl_core as core;

use std::ffi::CString;
use core::error::Result as OclCoreResult;
use core::{PlatformInfo, DeviceInfo, ContextInfo, CommandQueueInfo, MemInfo, ImageInfo,
    SamplerInfo, ProgramInfo, ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo,
    EventInfo, ProfilingInfo, ContextProperties, PlatformId, DeviceId, ImageFormat,
    ImageDescriptor, MemObjectType, AddressingMode, FilterMode, Event, ContextInfoResult,
    KernelArg};

const DIMS: [usize; 3] = [1024, 64, 16];
const INFO_FORMAT_MULTILINE: bool = true;

static SRC: &'static str = r#"
    __kernel void multiply(float coeff, __global float* buffer) {
        buffer[get_global_id(0)] *= coeff;
    }
"#;

/// Convert the info or error to a string for printing:
macro_rules! to_string {
    ( $ expr : expr ) => {
        match $expr {
            Ok(info) => info.to_string(),
            Err(err) => err.to_string(),
        }
    };
}

fn print_platform_device(plat_idx: usize, platform: PlatformId, device_idx: usize,
        device: DeviceId) -> OclCoreResult<()> {
    let context_properties = ContextProperties::new().platform(platform);
    let context = core::create_context(Some(&context_properties), &[device], None, None)?;

    let program = core::create_build_program(&context, &[CString::new(SRC).unwrap()],
        None::<&[()]>, &CString::new("").unwrap())?;

    let queue = core::create_command_queue(&context, &device,
        Some(core::QUEUE_PROFILING_ENABLE))?;
    let len = DIMS[0] * DIMS[1] * DIMS[2];
    let buffer = unsafe { core::create_buffer::<_, f32>(&context, core::MEM_READ_WRITE, len, None)? };

    let image_descriptor = ImageDescriptor::new(MemObjectType::Image1d,
        DIMS[0], DIMS[1], DIMS[2], 0, 0, 0, None);

    let image = unsafe { core::create_image::<_, u8>(&context, core::MEM_READ_WRITE,
        &ImageFormat::new_rgba(), &image_descriptor, None, None)? };

    let sampler = core::create_sampler(&context, false, AddressingMode::None, FilterMode::Nearest)?;
    let kernel = core::create_kernel(&program, "multiply")?;
    core::set_kernel_arg(&kernel, 0, KernelArg::Scalar(10.0f32))?;
    core::set_kernel_arg::<usize>(&kernel, 1, KernelArg::Mem(&buffer))?;

    unsafe {
        core::enqueue_kernel(&queue, &kernel, DIMS.len() as u32, None, &DIMS, None,
            None::<Event>, None::<&mut Event>)?;
    }
    core::finish(&queue)?;

    let mut event = Event::null();
    unsafe {
        core::enqueue_write_buffer(&queue, &buffer, true, 0, &vec![0.0; DIMS[0]],
            None::<Event>, Some(&mut event))?;
    }
    core::finish(&queue)?;

    let device_version = device.version()?;

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

    println!("Platform [{}]:{b}\
            Profile: {}{d}\
            Version: {}{d}\
            Name: {}{d}\
            Vendor: {}{d}\
            Extensions: {}{e}\
        ",
        plat_idx,
        to_string!(core::get_platform_info(platform, PlatformInfo::Profile)),
        to_string!(core::get_platform_info(platform, PlatformInfo::Version)),
        to_string!(core::get_platform_info(platform, PlatformInfo::Name)),
        to_string!(core::get_platform_info(platform, PlatformInfo::Vendor)),
        to_string!(core::get_platform_info(platform, PlatformInfo::Extensions)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // #################### DEVICES #####################
    // ##################################################

    let devices = match core::get_context_info(&context, ContextInfo::Devices)? {
        ContextInfoResult::Devices(devices) => devices,
        _ => unreachable!()
    };
    debug_assert!(devices.len() == 1);

    println!("Device [{}]: {b}\
            Type: {}{d}\
            VendorId: {}{d}\
            MaxComputeUnits: {}{d}\
            MaxWorkItemDimensions: {}{d}\
            MaxWorkGroupSize: {}{d}\
            MaxWorkItemSizes: {}{d}\
            PreferredVectorWidthChar: {}{d}\
            PreferredVectorWidthShort: {}{d}\
            PreferredVectorWidthInt: {}{d}\
            PreferredVectorWidthLong: {}{d}\
            PreferredVectorWidthFloat: {}{d}\
            PreferredVectorWidthDouble: {}{d}\
            MaxClockFrequency: {}{d}\
            AddressBits: {}{d}\
            MaxReadImageArgs: {}{d}\
            MaxWriteImageArgs: {}{d}\
            MaxMemAllocSize: {}{d}\
            Image2dMaxWidth: {}{d}\
            Image2dMaxHeight: {}{d}\
            Image3dMaxWidth: {}{d}\
            Image3dMaxHeight: {}{d}\
            Image3dMaxDepth: {}{d}\
            ImageSupport: {}{d}\
            MaxParameterSize: {}{d}\
            MaxSamplers: {}{d}\
            MemBaseAddrAlign: {}{d}\
            MinDataTypeAlignSize: {}{d}\
            SingleFpConfig: {}{d}\
            GlobalMemCacheType: {}{d}\
            GlobalMemCachelineSize: {}{d}\
            GlobalMemCacheSize: {}{d}\
            GlobalMemSize: {}{d}\
            MaxConstantBufferSize: {}{d}\
            MaxConstantArgs: {}{d}\
            LocalMemType: {}{d}\
            LocalMemSize: {}{d}\
            ErrorCorrectionSupport: {}{d}\
            ProfilingTimerResolution: {}{d}\
            EndianLittle: {}{d}\
            Available: {}{d}\
            CompilerAvailable: {}{d}\
            ExecutionCapabilities: {}{d}\
            QueueProperties: {}{d}\
            Name: {}{d}\
            Vendor: {}{d}\
            DriverVersion: {}{d}\
            Profile: {}{d}\
            Version: {}{d}\
            Extensions: {}{d}\
            Platform: {}{d}\
            DoubleFpConfig: {}{d}\
            HalfFpConfig: {}{d}\
            PreferredVectorWidthHalf: {}{d}\
            HostUnifiedMemory: {}{d}\
            NativeVectorWidthChar: {}{d}\
            NativeVectorWidthShort: {}{d}\
            NativeVectorWidthInt: {}{d}\
            NativeVectorWidthLong: {}{d}\
            NativeVectorWidthFloat: {}{d}\
            NativeVectorWidthDouble: {}{d}\
            NativeVectorWidthHalf: {}{d}\
            OpenclCVersion: {}{d}\
            LinkerAvailable: {}{d}\
            BuiltInKernels: {}{d}\
            ImageMaxBufferSize: {}{d}\
            ImageMaxArraySize: {}{d}\
            ParentDevice: {}{d}\
            PartitionMaxSubDevices: {}{d}\
            PartitionProperties: {}{d}\
            PartitionAffinityDomain: {}{d}\
            PartitionType: {}{d}\
            ReferenceCount: {}{d}\
            PreferredInteropUserSync: {}{d}\
            PrintfBufferSize: {}{d}\
            ImagePitchAlignment: {}{d}\
            ImageBaseAddressAlignment: {}{e}\
        ",
        device_idx,
        to_string!(core::get_device_info(&device, DeviceInfo::Type)),
        to_string!(core::get_device_info(&device, DeviceInfo::VendorId)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxComputeUnits)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxWorkItemDimensions)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxWorkGroupSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxWorkItemSizes)),
        to_string!(core::get_device_info(&device, DeviceInfo::PreferredVectorWidthChar)),
        to_string!(core::get_device_info(&device, DeviceInfo::PreferredVectorWidthShort)),
        to_string!(core::get_device_info(&device, DeviceInfo::PreferredVectorWidthInt)),
        to_string!(core::get_device_info(&device, DeviceInfo::PreferredVectorWidthLong)),
        to_string!(core::get_device_info(&device, DeviceInfo::PreferredVectorWidthFloat)),
        to_string!(core::get_device_info(&device, DeviceInfo::PreferredVectorWidthDouble)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxClockFrequency)),
        to_string!(core::get_device_info(&device, DeviceInfo::AddressBits)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxReadImageArgs)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxWriteImageArgs)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxMemAllocSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::Image2dMaxWidth)),
        to_string!(core::get_device_info(&device, DeviceInfo::Image2dMaxHeight)),
        to_string!(core::get_device_info(&device, DeviceInfo::Image3dMaxWidth)),
        to_string!(core::get_device_info(&device, DeviceInfo::Image3dMaxHeight)),
        to_string!(core::get_device_info(&device, DeviceInfo::Image3dMaxDepth)),
        to_string!(core::get_device_info(&device, DeviceInfo::ImageSupport)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxParameterSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxSamplers)),
        to_string!(core::get_device_info(&device, DeviceInfo::MemBaseAddrAlign)),
        to_string!(core::get_device_info(&device, DeviceInfo::MinDataTypeAlignSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::SingleFpConfig)),
        to_string!(core::get_device_info(&device, DeviceInfo::GlobalMemCacheType)),
        to_string!(core::get_device_info(&device, DeviceInfo::GlobalMemCachelineSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::GlobalMemCacheSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::GlobalMemSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxConstantBufferSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::MaxConstantArgs)),
        to_string!(core::get_device_info(&device, DeviceInfo::LocalMemType)),
        to_string!(core::get_device_info(&device, DeviceInfo::LocalMemSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::ErrorCorrectionSupport)),
        to_string!(core::get_device_info(&device, DeviceInfo::ProfilingTimerResolution)),
        to_string!(core::get_device_info(&device, DeviceInfo::EndianLittle)),
        to_string!(core::get_device_info(&device, DeviceInfo::Available)),
        to_string!(core::get_device_info(&device, DeviceInfo::CompilerAvailable)),
        to_string!(core::get_device_info(&device, DeviceInfo::ExecutionCapabilities)),
        to_string!(core::get_device_info(&device, DeviceInfo::QueueProperties)),
        to_string!(core::get_device_info(&device, DeviceInfo::Name)),
        to_string!(core::get_device_info(&device, DeviceInfo::Vendor)),
        to_string!(core::get_device_info(&device, DeviceInfo::DriverVersion)),
        to_string!(core::get_device_info(&device, DeviceInfo::Profile)),
        to_string!(core::get_device_info(&device, DeviceInfo::Version)),
        to_string!(core::get_device_info(&device, DeviceInfo::Extensions)),
        to_string!(core::get_device_info(&device, DeviceInfo::Platform)),
        to_string!(core::get_device_info(&device, DeviceInfo::DoubleFpConfig)),
        to_string!(core::get_device_info(&device, DeviceInfo::HalfFpConfig)),
        to_string!(core::get_device_info(&device, DeviceInfo::PreferredVectorWidthHalf)),
        to_string!(core::get_device_info(&device, DeviceInfo::HostUnifiedMemory)),
        to_string!(core::get_device_info(&device, DeviceInfo::NativeVectorWidthChar)),
        to_string!(core::get_device_info(&device, DeviceInfo::NativeVectorWidthShort)),
        to_string!(core::get_device_info(&device, DeviceInfo::NativeVectorWidthInt)),
        to_string!(core::get_device_info(&device, DeviceInfo::NativeVectorWidthLong)),
        to_string!(core::get_device_info(&device, DeviceInfo::NativeVectorWidthFloat)),
        to_string!(core::get_device_info(&device, DeviceInfo::NativeVectorWidthDouble)),
        to_string!(core::get_device_info(&device, DeviceInfo::NativeVectorWidthHalf)),
        to_string!(core::get_device_info(&device, DeviceInfo::OpenclCVersion)),
        to_string!(core::get_device_info(&device, DeviceInfo::LinkerAvailable)),
        to_string!(core::get_device_info(&device, DeviceInfo::BuiltInKernels)),
        to_string!(core::get_device_info(&device, DeviceInfo::ImageMaxBufferSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::ImageMaxArraySize)),
        to_string!(core::get_device_info(&device, DeviceInfo::ParentDevice)),
        to_string!(core::get_device_info(&device, DeviceInfo::PartitionMaxSubDevices)),
        to_string!(core::get_device_info(&device, DeviceInfo::PartitionProperties)),
        to_string!(core::get_device_info(&device, DeviceInfo::PartitionAffinityDomain)),
        to_string!(core::get_device_info(&device, DeviceInfo::PartitionType)),
        to_string!(core::get_device_info(&device, DeviceInfo::ReferenceCount)),
        to_string!(core::get_device_info(&device, DeviceInfo::PreferredInteropUserSync)),
        to_string!(core::get_device_info(&device, DeviceInfo::PrintfBufferSize)),
        to_string!(core::get_device_info(&device, DeviceInfo::ImagePitchAlignment)),
        to_string!(core::get_device_info(&device, DeviceInfo::ImageBaseAddressAlignment)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // #################### CONTEXT #####################
    // ##################################################

    println!("Context:{b}\
            Reference Count: {}{d}\
            Devices: {}{d}\
            Properties: {}{d}\
            Device Count: {}{e}\
        ",
        to_string!(core::get_context_info(&context, ContextInfo::ReferenceCount)),
        to_string!(core::get_context_info(&context, ContextInfo::Devices)),
        to_string!(core::get_context_info(&context, ContextInfo::Properties)),
        to_string!(core::get_context_info(&context, ContextInfo::NumDevices)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ##################### QUEUE ######################
    // ##################################################

    println!("Command Queue:{b}\
            Context: {}{d}\
            Device: {}{d}\
            ReferenceCount: {}{d}\
            Properties: {}{e}\
        ",
        to_string!(core::get_command_queue_info(&queue, CommandQueueInfo::Context)),
        to_string!(core::get_command_queue_info(&queue, CommandQueueInfo::Device)),
        to_string!(core::get_command_queue_info(&queue, CommandQueueInfo::ReferenceCount)),
        to_string!(core::get_command_queue_info(&queue, CommandQueueInfo::Properties)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ################### MEM OBJECT ###################
    // ##################################################

    println!("Buffer Memory:{b}\
            Type: {}{d}\
            Flags: {}{d}\
            Size: {}{d}\
            HostPtr: {}{d}\
            MapCount: {}{d}\
            ReferenceCount: {}{d}\
            Context: {}{d}\
            AssociatedMemobject: {}{d}\
            Offset: {}{e}\
        ",
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Type)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Flags)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Size)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::HostPtr)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::MapCount)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::ReferenceCount)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Context)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::AssociatedMemobject)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Offset)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ##################### IMAGE ######################
    // ##################################################

    println!("Image: {b}\
            ElementSize: {}{d}\
            RowPitch: {}{d}\
            SlicePitch: {}{d}\
            Width: {}{d}\
            Height: {}{d}\
            Depth: {}{d}\
            ArraySize: {}{d}\
            Buffer: {}{d}\
            NumMipLevels: {}{d}\
            NumSamples: {}{e}\
        ",
        to_string!(core::get_image_info(&image, ImageInfo::ElementSize)),
        to_string!(core::get_image_info(&image, ImageInfo::RowPitch)),
        to_string!(core::get_image_info(&image, ImageInfo::SlicePitch)),
        to_string!(core::get_image_info(&image, ImageInfo::Width)),
        to_string!(core::get_image_info(&image, ImageInfo::Height)),
        to_string!(core::get_image_info(&image, ImageInfo::Depth)),
        to_string!(core::get_image_info(&image, ImageInfo::ArraySize)),
        to_string!(core::get_image_info(&image, ImageInfo::Buffer)),
        to_string!(core::get_image_info(&image, ImageInfo::NumMipLevels)),
        to_string!(core::get_image_info(&image, ImageInfo::NumSamples)),
        b = begin, d = delim, e = end,
    );

    println!("Image Memory:{b}\
            Type: {}{d}\
            Flags: {}{d}\
            Size: {}{d}\
            HostPtr: {}{d}\
            MapCount: {}{d}\
            ReferenceCount: {}{d}\
            Context: {}{d}\
            AssociatedMemobject: {}{d}\
            Offset: {}{e}\
        ",
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Type)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Flags)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Size)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::HostPtr)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::MapCount)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::ReferenceCount)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Context)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::AssociatedMemobject)),
        to_string!(core::get_mem_object_info(&buffer, MemInfo::Offset)),
        b = begin, d = delim, e = end,
    );

    // ##################################################
    // #################### SAMPLER #####################
    // ##################################################


    println!("Sampler:{b}\
            ReferenceCount: {}{d}\
            Context: {}{d}\
            NormalizedCoords: {}{d}\
            AddressingMode: {}{d}\
            FilterMode: {}{e}\
        ",
        to_string!(core::get_sampler_info(&sampler, SamplerInfo::ReferenceCount)),
        to_string!(core::get_sampler_info(&sampler, SamplerInfo::Context)),
        to_string!(core::get_sampler_info(&sampler, SamplerInfo::NormalizedCoords)),
        to_string!(core::get_sampler_info(&sampler, SamplerInfo::AddressingMode)),
        to_string!(core::get_sampler_info(&sampler, SamplerInfo::FilterMode)),
        b = begin, d = delim, e = end,
    );

    // ##################################################
    // #################### PROGRAM #####################
    // ##################################################

    println!("Program:{b}\
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
        to_string!(core::get_program_info(&program, ProgramInfo::ReferenceCount)),
        to_string!(core::get_program_info(&program, ProgramInfo::Context)),
        to_string!(core::get_program_info(&program, ProgramInfo::NumDevices)),
        to_string!(core::get_program_info(&program, ProgramInfo::Devices)),
        to_string!(core::get_program_info(&program, ProgramInfo::Source)),
        to_string!(core::get_program_info(&program, ProgramInfo::BinarySizes)),
        "{{unprintable}}",
        to_string!(core::get_program_info(&program, ProgramInfo::NumKernels)),
        to_string!(core::get_program_info(&program, ProgramInfo::KernelNames)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ################# PROGRAM BUILD ##################
    // ##################################################

    println!("Program Build:{b}\
            BuildStatus: {}{d}\
            BuildOptions: {}{d}\
            BuildLog: \n\n{}{d}\n\
            BinaryType: {}{e}\
        ",
        to_string!(core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildStatus)),
        to_string!(core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildOptions)),
        to_string!(core::get_program_build_info(&program, &device, ProgramBuildInfo::BuildLog)),
        to_string!(core::get_program_build_info(&program, &device, ProgramBuildInfo::BinaryType)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ##################### KERNEL #####################
    // ##################################################

    println!("Kernel Info:{b}\
            FunctionName: {}{d}\
            NumArgs: {}{d}\
            ReferenceCount: {}{d}\
            Context: {}{d}\
            Program: {}{d}\
            Attributes: {}{e}\
        ",
        to_string!(core::get_kernel_info(&kernel, KernelInfo::FunctionName)),
        to_string!(core::get_kernel_info(&kernel, KernelInfo::NumArgs)),
        to_string!(core::get_kernel_info(&kernel, KernelInfo::ReferenceCount)),
        to_string!(core::get_kernel_info(&kernel, KernelInfo::Context)),
        to_string!(core::get_kernel_info(&kernel, KernelInfo::Program)),
        to_string!(core::get_kernel_info(&kernel, KernelInfo::Attributes)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ################# KERNEL ARGUMENT ################
    // ##################################################

    println!("Kernel Argument [0]:{b}\
            AddressQualifier: {}{d}\
            AccessQualifier: {}{d}\
            TypeName: {}{d}\
            TypeQualifier: {}{d}\
            Name: {}{e}\
        ",
        to_string!(core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::AddressQualifier, Some(&[device_version]))),
        to_string!(core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::AccessQualifier, Some(&[device_version]))),
        to_string!(core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::TypeName, Some(&[device_version]))),
        to_string!(core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::TypeQualifier, Some(&[device_version]))),
        to_string!(core::get_kernel_arg_info(&kernel, 0, KernelArgInfo::Name, Some(&[device_version]))),
        b = begin, d = delim, e = end,
    );

    // ##################################################
    // ################ KERNEL WORK GROUP ###############
    // ##################################################

    println!("Kernel Work Group:{b}\
            WorkGroupSize: {}{d}\
            CompileWorkGroupSize: {}{d}\
            LocalMemSize: {}{d}\
            PreferredWorkGroupSizeMultiple: {}{d}\
            PrivateMemSize: {}{d}\
            GlobalWorkSize: {}{e}\
        ",
        to_string!(core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::WorkGroupSize)),
        to_string!(core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::CompileWorkGroupSize)),
        to_string!(core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::LocalMemSize)),
        to_string!(core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::PreferredWorkGroupSizeMultiple)),
        to_string!(core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::PrivateMemSize)),
        to_string!(core::get_kernel_work_group_info(&kernel, &device, KernelWorkGroupInfo::GlobalWorkSize)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ##################### EVENT ######################
    // ##################################################

    println!("Event:{b}\
            CommandQueue: {}{d}\
            CommandType: {}{d}\
            ReferenceCount: {}{d}\
            CommandExecutionStatus: {}{d}\
            Context: {}{e}\
        ",
        to_string!(core::get_event_info(&event, EventInfo::CommandQueue)),
        to_string!(core::get_event_info(&event, EventInfo::CommandType)),
        to_string!(core::get_event_info(&event, EventInfo::ReferenceCount)),
        to_string!(core::get_event_info(&event, EventInfo::CommandExecutionStatus)),
        to_string!(core::get_event_info(&event, EventInfo::Context)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ################ EVENT PROFILING #################
    // ##################################################

    println!("Event Profiling:{b}\
            Queued: {}{d}\
            Submit: {}{d}\
            Start: {}{d}\
            End: {}{e}\
        ",
        to_string!(core::get_event_profiling_info(&event, ProfilingInfo::Queued)),
        to_string!(core::get_event_profiling_info(&event, ProfilingInfo::Submit)),
        to_string!(core::get_event_profiling_info(&event, ProfilingInfo::Start)),
        to_string!(core::get_event_profiling_info(&event, ProfilingInfo::End)),
        b = begin, d = delim, e = end,
    );


    // ##################################################
    // ###################### END #######################
    // ##################################################

    print!("\n");
    Ok(())
}

fn print_platform(plat_idx: usize, platform: PlatformId) -> OclCoreResult<()> {
    let devices = core::get_device_ids(&platform, None, None)?;
    for (device_idx, &device) in devices.iter().enumerate() {
        print_platform_device(plat_idx, platform, device_idx, device)?;
    }
    Ok(())
}

fn info_core() -> OclCoreResult<()> {
    let platforms = core::get_platform_ids()?;
    for (plat_idx, &platform) in platforms.iter().enumerate() {
        print_platform(plat_idx, platform)?;
    }
    Ok(())
}

pub fn main() {
    match info_core() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}