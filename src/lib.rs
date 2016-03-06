//! # [![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl) | [GitHub](https://github.com/cogciprocate/ocl)
//!
//! Rust implementation of OpenCL&trade;.
//!
//! This documentation is sometimes built from the [`dev` branch](https://github.com/cogciprocate/ocl/tree/dev) and may occasionally differ slightly from what is on crates.io and the master branch. 
//!
//! Documentation is very much a work in progress, as is the library itself. Please help by filing an [issue](https://github.com/cogciprocate/ocl/issues) about unclear and/or incomplete documentation and it will be addressed.
//!
//! An explanation of how dimensions and sizes of buffers and work queues are intended 
//! to be used will be coming as soon as a few more things are ironed out. Until then please see the [examples].
//!
//!
//! ## Low Level Interfaces
//!
//! For lower level interfaces and to use OpenCL features that have not yet been implemented on the top-level interface types, see the [`core`] and [`cl_h`] modules.
//!
//!
//! ## Help Wanted
//!
//! Please help complete any functionality you may need by filing an 
//! [issue] or creating a [pull request](https://github.com/cogciprocate/ocl/pulls).
//!
//! <br/>
//! *“OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.”*

//!
//! [issue]: https://github.com/cogciprocate/ocl/issues
//! [`core`]: http://docs.cogciprocate.com/ocl/core/index.html
//! [`cl_h`]: http://docs.cogciprocate.com/ocl/cl_h/index.html
//! [`Result`]: http://docs.cogciprocate.com/ocl/type.Result.html
//! [examples]: https://github.com/cogciprocate/ocl/tree/master/examples

// #![warn(missing_docs)]
#![feature(zero_one)]

// For some reason have to have this to supress the warning (TODO: figure out
// how to conditionally allow [feature(time2)] for cfg(test) only).
#![allow(unused_features)]
#![feature(time2)]

#[macro_use] extern crate enum_primitive;
#[macro_use] extern crate bitflags;
extern crate libc;
extern crate num;
extern crate rand;

#[macro_use] pub mod util;
#[cfg(test)] mod tests;
mod standard;
mod error;
pub mod core;
pub mod cl_h;

pub use standard::{Platform, Device, Context, Program, Queue, Kernel, Buffer, Image, Event, 
	EventList, Sampler, SpatialDims, ProQue};
pub use self::error::{Error, Result};

pub mod traits {
	//! Commonly used traits.
	pub use standard::{WorkDims, MemDims};
	pub use core::OclPrm;
}

pub mod build {
	//! Builders and associated settings-related types.

	pub use standard::{ContextBuilder, BuildOpt, ProgramBuilder, ImageBuilder, ProQueBuilder,
		BufferCmd, DeviceSpecifier};
	pub use core::{ImageFormat, ImageDescriptor, ContextProperties};
	#[cfg(not(release))] pub use standard::BufferTest;
}

pub mod flags {
	//! Bitflags for various parameter types.

	pub use core::{
		// cl_device_type - bitfield 
		DeviceType, DEVICE_TYPE_DEFAULT, DEVICE_TYPE_CPU, DEVICE_TYPE_GPU, DEVICE_TYPE_ACCELERATOR,
			DEVICE_TYPE_CUSTOM, DEVICE_TYPE_ALL,
		// cl_device_fp_config - bitfield
		DeviceFpConfig, FP_DENORM, FP_INF_NAN, FP_ROUND_TO_NEAREST, FP_ROUND_TO_ZERO, 
			FP_ROUND_TO_INF, FP_FMA, FP_SOFT_FLOAT, FP_CORRECTLY_ROUNDED_DIVIDE_SQRT,
		// cl_device_exec_capabilities - bitfield
		DeviceExecCapabilities, EXEC_KERNEL, EXEC_NATIVE_KERNEL,
		// cl_command_queue_properties - bitfield
		CommandQueueProperties, QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, QUEUE_PROFILING_ENABLE,
		// cl_device_affinity_domain
		DeviceAffinityDomain, DEVICE_AFFINITY_DOMAIN_NUMA, DEVICE_AFFINITY_DOMAIN_L4_CACHE, 
			DEVICE_AFFINITY_DOMAIN_L3_CACHE, DEVICE_AFFINITY_DOMAIN_L2_CACHE, 
			DEVICE_AFFINITY_DOMAIN_L1_CACHE, DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
		// cl_mem_flags - bitfield
		MemFlags, MEM_READ_WRITE, MEM_WRITE_ONLY, MEM_READ_ONLY, MEM_USE_HOST_PTR, 
			MEM_ALLOC_HOST_PTR, MEM_COPY_HOST_PTR, MEM_HOST_WRITE_ONLY, MEM_HOST_READ_ONLY, 
			MEM_HOST_NO_ACCESS,
		// cl_mem_migration_flags - bitfield
		MemMigrationFlags, MIGRATE_MEM_OBJECT_HOST, MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
		// cl_map_flags - bitfield
		MapFlags, MAP_READ, MAP_WRITE, MAP_WRITE_INVALIDATE_REGION,
		// cl_program_binary_type
		ProgramBinaryType, PROGRAM_BINARY_TYPE_NONE, PROGRAM_BINARY_TYPE_COMPILED_OBJECT, 
			PROGRAM_BINARY_TYPE_LIBRARY, PROGRAM_BINARY_TYPE_EXECUTABLE,
		// cl_kernel_arg_type_qualifer 
		KernelArgTypeQualifier, KERNEL_ARG_TYPE_NONE, KERNEL_ARG_TYPE_CONST, 
			KERNEL_ARG_TYPE_RESTRICT, KERNEL_ARG_TYPE_VOLATILE,	
	};
}

pub mod enums {
	//! Enumerators for settings and information requests.

	pub use standard::{DeviceSpecifier, BufferCmdKind, BufferCmdDataShape};

	// API enums.
	pub use core::{ImageChannelOrder, ImageChannelDataType, Cbool, Polling, PlatformInfo,
		DeviceInfo, DeviceMemCacheType, DeviceLocalMemType, ContextInfo,
		ContextInfoOrPropertiesPointerType, PartitionProperty, CommandQueueInfo, ChannelType, 
		MemObjectType, MemInfo, ImageInfo, AddressingMode, FilterMode, SamplerInfo, ProgramInfo,
		ProgramBuildInfo, BuildStatus, KernelInfo, KernelArgInfo, KernelArgAddressQualifier, 
		KernelArgAccessQualifier, KernelWorkGroupInfo, EventInfo, CommandType, 
		CommandExecutionStatus, BufferCreateType, ProfilingInfo};
	// Custom enums.
	pub use core::{KernelArg, ContextProperty, PlatformInfoResult, DeviceInfoResult, 
		ContextInfoResult, CommandQueueInfoResult, MemInfoResult, ImageInfoResult, 
		SamplerInfoResult, ProgramInfoResult, ProgramBuildInfoResult, KernelInfoResult, 
		KernelArgInfoResult, KernelWorkGroupInfoResult, EventInfoResult, ProfilingInfoResult};
}
