use ocl::{Device, DeviceType, CommandQueueProperties};
use ocl::enums::{DeviceMemCacheType, DevicePartitionProperty, DeviceLocalMemType};
use ocl::flags::{DeviceAffinityDomain, DeviceExecCapabilities, DeviceFpConfig};
use ocl::error::{Result as OclResult};
use ocl::core::{DeviceId, PlatformId, OpenclVersion, DeviceInfo, DeviceInfoResult};

/// Provides a more convenient and safe interface to access less commonly used device information.
/// The methods return the appropriate type for the given device info, rather than a
/// `DeviceInfoResult` which must then be manually converted. This saves a significant amount of
/// boilerplate when querying multiple different types of device information or using ones which
/// don't already have a method on `Device`.
/// 
/// # Examples
/// Instead of using `device.info` like this...
/// ```
/// let compute_units = match device.info(DeviceInfo::MaxComputeUnits)? {
///     DeviceInfoResult::MaxComputeUnits(c) => c,
///     _ => panic!("...")
/// };
/// ```
/// ...you can use the trait and then call the method for whichever device information you need:
/// ```
/// use ocl-extras::full_device_info::FullDeviceInfo;
/// let compute_units = device.max_compute_units()?;
/// ```
pub trait FullDeviceInfo {
	fn device_type(&self) -> OclResult<DeviceType>;
	fn vendor_id(&self) -> OclResult<u32>;
	fn max_compute_units(&self) -> OclResult<u32>;
	fn max_work_item_dimensions(&self) -> OclResult<u32>;
	fn max_work_group_size(&self) -> OclResult<usize>;
	fn max_work_item_sizes(&self) -> OclResult<Vec<usize>>;
	fn preferred_vector_width_char(&self) -> OclResult<u32>;
	fn preferred_vector_width_short(&self) -> OclResult<u32>;
	fn preferred_vector_width_int(&self) -> OclResult<u32>;
	fn preferred_vector_width_long(&self) -> OclResult<u32>;
	fn preferred_vector_width_float(&self) -> OclResult<u32>;
	fn preferred_vector_width_double(&self) -> OclResult<u32>;
	fn max_clock_frequency(&self) -> OclResult<u32>;
	fn address_bits(&self) -> OclResult<u32>;
	fn max_read_image_args(&self) -> OclResult<u32>;
	fn max_write_image_args(&self) -> OclResult<u32>;
	fn max_mem_alloc_size(&self) -> OclResult<u64>;
	fn image2d_max_width(&self) -> OclResult<usize>;
	fn image2d_max_height(&self) -> OclResult<usize>;
	fn image3d_max_width(&self) -> OclResult<usize>;
	fn image3d_max_height(&self) -> OclResult<usize>;
	fn image3d_max_depth(&self) -> OclResult<usize>;
	fn image_support(&self) -> OclResult<bool>;
	fn max_parameter_size(&self) -> OclResult<usize>;
	fn max_samplers(&self) -> OclResult<u32>;
	fn mem_base_addr_align(&self) -> OclResult<u32>;
	fn min_data_type_align_size(&self) -> OclResult<u32>;
	fn single_fp_config(&self) -> OclResult<DeviceFpConfig>;
	fn global_mem_cache_type(&self) -> OclResult<DeviceMemCacheType>;
	fn global_mem_cacheline_size(&self) -> OclResult<u32>;
	fn global_mem_cache_size(&self) -> OclResult<u64>;
	fn global_mem_size(&self) -> OclResult<u64>;
	fn max_constant_buffer_size(&self) -> OclResult<u64>;
	fn max_constant_args(&self) -> OclResult<u32>;
	fn local_mem_type(&self) -> OclResult<DeviceLocalMemType>;
	fn local_mem_size(&self) -> OclResult<u64>;
	fn error_correction_support(&self) -> OclResult<bool>;
	fn profiling_timer_resolution(&self) -> OclResult<usize>;
	fn endian_little(&self) -> OclResult<bool>;
	fn available(&self) -> OclResult<bool>;
	fn compiler_available(&self) -> OclResult<bool>;
	fn execution_capabilities(&self) -> OclResult<DeviceExecCapabilities>;
	fn queue_properties(&self) -> OclResult<CommandQueueProperties>;
	fn name(&self) -> OclResult<String>;
	fn vendor(&self) -> OclResult<String>;
	fn driver_version(&self) -> OclResult<String>;
	fn profile(&self) -> OclResult<String>;
	fn version(&self) -> OclResult<OpenclVersion>;
	fn extensions(&self) -> OclResult<String>;
	fn platform(&self) -> OclResult<PlatformId>;
	fn double_fp_config(&self) -> OclResult<DeviceFpConfig>;
	fn half_fp_config(&self) -> OclResult<DeviceFpConfig>;
	fn preferred_vector_width_half(&self) -> OclResult<u32>;
	fn host_unified_memory(&self) -> OclResult<bool>;
	fn native_vector_width_char(&self) -> OclResult<u32>;
	fn native_vector_width_short(&self) -> OclResult<u32>;
	fn native_vector_width_int(&self) -> OclResult<u32>;
	fn native_vector_width_long(&self) -> OclResult<u32>;
	fn native_vector_width_float(&self) -> OclResult<u32>;
	fn native_vector_width_double(&self) -> OclResult<u32>;
	fn native_vector_width_half(&self) -> OclResult<u32>;
	fn opencl_c_version(&self) -> OclResult<String>;
	fn linker_available(&self) -> OclResult<bool>;
	fn built_in_kernels(&self) -> OclResult<String>;
	fn image_max_buffer_size(&self) -> OclResult<usize>;
	fn image_max_array_size(&self) -> OclResult<usize>;
	fn parent_device(&self) -> OclResult<Option<DeviceId>>;
	fn partition_max_sub_devices(&self) -> OclResult<u32>;
	fn partition_properties(&self) -> OclResult<Vec<DevicePartitionProperty>>;
	fn partition_affinity_domain(&self) -> OclResult<DeviceAffinityDomain>;
	fn partition_type(&self) -> OclResult<Vec<DevicePartitionProperty>>;
	fn reference_count(&self) -> OclResult<u32>;
	fn preferred_interop_user_sync(&self) -> OclResult<bool>;
	fn printf_buffer_size(&self) -> OclResult<usize>;
	fn image_pitch_alignment(&self) -> OclResult<u32>;
	fn image_base_address_alignment(&self) -> OclResult<u32>;
}

macro_rules! dev_info_fn {
    (fn $name:ident() $info:ident -> $ret:ty) => {
        fn $name(&self) -> OclResult<$ret> {
            match self.info(DeviceInfo::$info)? {
                DeviceInfoResult::$info(v) => Ok(v),
                _ => panic!("Unexpected DeviceInfoResult variant")
            }
        }
    };
}

impl FullDeviceInfo for Device {
	dev_info_fn! { fn device_type() Type -> DeviceType }
	dev_info_fn! { fn vendor_id() VendorId -> u32 }
	dev_info_fn! { fn max_compute_units() MaxComputeUnits -> u32 }
	dev_info_fn! { fn max_work_item_dimensions() MaxWorkItemDimensions -> u32 }
	dev_info_fn! { fn max_work_group_size() MaxWorkGroupSize -> usize }
	dev_info_fn! { fn max_work_item_sizes() MaxWorkItemSizes -> Vec<usize> }
	dev_info_fn! { fn preferred_vector_width_char() PreferredVectorWidthChar -> u32 }
	dev_info_fn! { fn preferred_vector_width_short() PreferredVectorWidthShort -> u32 }
	dev_info_fn! { fn preferred_vector_width_int() PreferredVectorWidthInt -> u32 }
	dev_info_fn! { fn preferred_vector_width_long() PreferredVectorWidthLong -> u32 }
	dev_info_fn! { fn preferred_vector_width_float() PreferredVectorWidthFloat -> u32 }
	dev_info_fn! { fn preferred_vector_width_double() PreferredVectorWidthDouble -> u32 }
	dev_info_fn! { fn max_clock_frequency() MaxClockFrequency -> u32 }
	dev_info_fn! { fn address_bits() AddressBits -> u32 }
	dev_info_fn! { fn max_read_image_args() MaxReadImageArgs -> u32 }
	dev_info_fn! { fn max_write_image_args() MaxWriteImageArgs -> u32 }
	dev_info_fn! { fn max_mem_alloc_size() MaxMemAllocSize -> u64 }
	dev_info_fn! { fn image2d_max_width() Image2dMaxWidth -> usize }
	dev_info_fn! { fn image2d_max_height() Image2dMaxHeight -> usize }
	dev_info_fn! { fn image3d_max_width() Image3dMaxWidth -> usize }
	dev_info_fn! { fn image3d_max_height() Image3dMaxHeight -> usize }
	dev_info_fn! { fn image3d_max_depth() Image3dMaxDepth -> usize }
	dev_info_fn! { fn image_support() ImageSupport -> bool }
	dev_info_fn! { fn max_parameter_size() MaxParameterSize -> usize }
	dev_info_fn! { fn max_samplers() MaxSamplers -> u32 }
	dev_info_fn! { fn mem_base_addr_align() MemBaseAddrAlign -> u32 }
	dev_info_fn! { fn min_data_type_align_size() MinDataTypeAlignSize -> u32 }
	dev_info_fn! { fn single_fp_config() SingleFpConfig -> DeviceFpConfig }
	dev_info_fn! { fn global_mem_cache_type() GlobalMemCacheType -> DeviceMemCacheType }
	dev_info_fn! { fn global_mem_cacheline_size() GlobalMemCachelineSize -> u32 }
	dev_info_fn! { fn global_mem_cache_size() GlobalMemCacheSize -> u64 }
	dev_info_fn! { fn global_mem_size() GlobalMemSize -> u64 }
	dev_info_fn! { fn max_constant_buffer_size() MaxConstantBufferSize -> u64 }
	dev_info_fn! { fn max_constant_args() MaxConstantArgs -> u32 }
	dev_info_fn! { fn local_mem_type() LocalMemType -> DeviceLocalMemType }
	dev_info_fn! { fn local_mem_size() LocalMemSize -> u64 }
	dev_info_fn! { fn error_correction_support() ErrorCorrectionSupport -> bool }
	dev_info_fn! { fn profiling_timer_resolution() ProfilingTimerResolution -> usize }
	dev_info_fn! { fn endian_little() EndianLittle -> bool }
	dev_info_fn! { fn available() Available -> bool }
	dev_info_fn! { fn compiler_available() CompilerAvailable -> bool }
	dev_info_fn! { fn execution_capabilities() ExecutionCapabilities -> DeviceExecCapabilities }
	dev_info_fn! { fn queue_properties() QueueProperties -> CommandQueueProperties }
	dev_info_fn! { fn name() Name -> String }
	dev_info_fn! { fn vendor() Vendor -> String }
	dev_info_fn! { fn driver_version() DriverVersion -> String }
	dev_info_fn! { fn profile() Profile -> String }
	dev_info_fn! { fn version() Version -> OpenclVersion }
	dev_info_fn! { fn extensions() Extensions -> String }
	dev_info_fn! { fn platform() Platform -> PlatformId }
	dev_info_fn! { fn double_fp_config() DoubleFpConfig -> DeviceFpConfig }
	dev_info_fn! { fn half_fp_config() HalfFpConfig -> DeviceFpConfig }
	dev_info_fn! { fn preferred_vector_width_half() PreferredVectorWidthHalf -> u32 }
	dev_info_fn! { fn host_unified_memory() HostUnifiedMemory -> bool }
	dev_info_fn! { fn native_vector_width_char() NativeVectorWidthChar -> u32 }
	dev_info_fn! { fn native_vector_width_short() NativeVectorWidthShort -> u32 }
	dev_info_fn! { fn native_vector_width_int() NativeVectorWidthInt -> u32 }
	dev_info_fn! { fn native_vector_width_long() NativeVectorWidthLong -> u32 }
	dev_info_fn! { fn native_vector_width_float() NativeVectorWidthFloat -> u32 }
	dev_info_fn! { fn native_vector_width_double() NativeVectorWidthDouble -> u32 }
	dev_info_fn! { fn native_vector_width_half() NativeVectorWidthHalf -> u32 }
	dev_info_fn! { fn opencl_c_version() OpenclCVersion -> String }
	dev_info_fn! { fn linker_available() LinkerAvailable -> bool }
	dev_info_fn! { fn built_in_kernels() BuiltInKernels -> String }
	dev_info_fn! { fn image_max_buffer_size() ImageMaxBufferSize -> usize }
	dev_info_fn! { fn image_max_array_size() ImageMaxArraySize -> usize }
	dev_info_fn! { fn parent_device() ParentDevice -> Option<DeviceId> }
	dev_info_fn! { fn partition_max_sub_devices() PartitionMaxSubDevices -> u32 }
	dev_info_fn! { fn partition_properties() PartitionProperties -> Vec<DevicePartitionProperty> }
	dev_info_fn! { fn partition_affinity_domain() PartitionAffinityDomain -> DeviceAffinityDomain }
	dev_info_fn! { fn partition_type() PartitionType -> Vec<DevicePartitionProperty> }
	dev_info_fn! { fn reference_count() ReferenceCount -> u32 }
	dev_info_fn! { fn preferred_interop_user_sync() PreferredInteropUserSync -> bool }
	dev_info_fn! { fn printf_buffer_size() PrintfBufferSize -> usize }
	dev_info_fn! { fn image_pitch_alignment() ImagePitchAlignment -> u32 }
	dev_info_fn! { fn image_base_address_alignment() ImageBaseAddressAlignment -> u32 }
}