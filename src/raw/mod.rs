//! 'Raw' functions, enums, and bitflags for the OpenCL C FFI.
//!
//! The thin layer between the FFI interfaces and the ocl types.
//!
//! Allows access to OpenCL FFI functions with a minimal layer of abstraction providing safety and convenience. Using functions in this module is only recommended for use when functionality has not yet been implemented on the 'standard' ocl interfaces although the 'raw' and 'standard' interfaces are all completely interoperable (and generally feature-equivalent).
//! 
//! Object pointers can generally be shared between threads except for kernel. 
//! See [clSetKernelArg documentation](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clSetKernelArg.html)
//!
//! ## Even Lower Level: `cl_h`
//!
//! *Not as raw as...*
//!
//! If there's still something missing, or for some reason you need direct FFI access, use the functions in the `cl_h` module.
//!
//! # Performance
//!
//! Performance between all three levels of interface, `cl_h`, `raw`, and the standard types, is virtually identical (if not, file an issue).
//!
//! ## Safety
//!
//! At the time of writing, some functions still *may* break Rust's usual safety promises and have not been comprehensively tested or evaluated. Please file an [issue](https://github.com/cogciprocate/ocl/issues) if you discover something!
//!
//! ## Panics
//!
//! [NOT UP TO DATE: more and more functions are returning results] All functions will panic upon OpenCL error. This will be changing over time. Certain errors will eventually be returned as an `Error` type instead.
//!
//! ### Official Documentation
//!
//! [OpenCL 1.2 SDK Reference: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/)
//!
//! ### Help Wanted
//!
//! Please help complete coverage of any FFI functions you may need by filing an [issue](https://github.com/cogciprocate/ocl/issues) or creating a [pull request](https://github.com/cogciprocate/ocl/pulls).

mod function;
mod cl;
mod custom;

// use cl_h;
// [FIXME]: Import things from `raw::function` individually.
pub use self::function::*;
pub use self::cl::abs::{PlatformIdRaw, DeviceIdRaw, ContextRaw, CommandQueueRaw, MemRaw, 
	ProgramRaw, KernelRaw, EventRaw, SamplerRaw};
pub use self::cl::enum_orgy::{MemObjectType, ContextInfo};
pub use self::cl::image::{ImageFormat, ImageChannelOrder, ImageChannelDataType, ImageDescriptor};
pub use self::custom::enums::{KernelArg};

//=============================================================================
//================================ CONSTANTS ==================================
//=============================================================================

// pub const DEFAULT_DEVICE_TYPE: cl_h::cl_device_type = cl_h::CL_DEVICE_TYPE_DEFAULT;

pub const DEVICES_MAX: u32 = 16;
pub const DEFAULT_PLATFORM_IDX: usize = 0;
pub const DEFAULT_DEVICE_IDX: usize = 0;

//=============================================================================
//================================ BITFIELDS ==================================
//=============================================================================

/// cl_device_type - bitfield 
bitflags! {
    pub flags DeviceType: u64 {
		const DEVICE_TYPE_DEFAULT = 1 << 0,
		const DEVICE_TYPE_CPU = 1 << 1,
		const DEVICE_TYPE_GPU = 1 << 2,
		const DEVICE_TYPE_ACCELERATOR = 1 << 3,
		const DEVICE_TYPE_CUSTOM = 1 << 4,
		const DEVICE_TYPE_ALL = 0xFFFFFFFF,
    }
}

/// cl_device_fp_config - bitfield
bitflags! {
    pub flags DeviceFpConfig: u64 {
		const FP_DENORM = 1 << 0,
		const FP_INF_NAN = 1 << 1,
		const FP_ROUND_TO_NEAREST = 1 << 2,
		const FP_ROUND_TO_ZERO = 1 << 3,
		const FP_ROUND_TO_INF = 1 << 4,
		const FP_FMA = 1 << 5,
		const FP_SOFT_FLOAT = 1 << 6,
		const FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = 1 << 7,
    }
}

/// cl_device_exec_capabilities - bitfield
bitflags! {
    pub flags DeviceExecCapabilities: u64 {
		const EXEC_KERNEL = 1 << 0,
		const EXEC_NATIVE_KERNEL = 1 << 1,
    }
}

/// cl_command_queue_properties - bitfield
bitflags! {
    pub flags CommandQueueProperties: u64 {
		const QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1 << 0,
		const QUEUE_PROFILING_ENABLE = 1 << 1,
    }
}

/// cl_device_affinity_domain
bitflags! {
    pub flags DeviceAffinityDomain: u64 {
		const DEVICE_AFFINITY_DOMAIN_NUMA = 1 << 0,
		const DEVICE_AFFINITY_DOMAIN_L4_CACHE = 1 << 1,
		const DEVICE_AFFINITY_DOMAIN_L3_CACHE = 1 << 2,
		const DEVICE_AFFINITY_DOMAIN_L2_CACHE = 1 << 3,
		const DEVICE_AFFINITY_DOMAIN_L1_CACHE = 1 << 4,
		const DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE = 1 << 5,
    }
}

/// cl_mem_flags - bitfield
bitflags! {
    pub flags MemFlags: u64 {
		const MEM_READ_WRITE = 1 << 0,
		const MEM_WRITE_ONLY = 1 << 1,
		const MEM_READ_ONLY = 1 << 2,
		const MEM_USE_HOST_PTR = 1 << 3,
		const MEM_ALLOC_HOST_PTR = 1 << 4,
		const MEM_COPY_HOST_PTR = 1 << 5,
		// RESERVED            1<< 6,
		const MEM_HOST_WRITE_ONLY = 1 << 7,
		const MEM_HOST_READ_ONLY = 1 << 8,
		const MEM_HOST_NO_ACCESS = 1 << 9,
    }
}

/// cl_mem_migration_flags - bitfield
bitflags! {
    pub flags MemMigrationFlags: u64 {
		const MIGRATE_MEM_OBJECT_HOST = 1 << 0,
		const MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED = 1 << 1,
    }
}

/// cl_map_flags - bitfield
bitflags! {
    pub flags MapFlags: u64 {
		const MAP_READ = 1 << 0,
		const MAP_WRITE = 1 << 1,
		const MAP_WRITE_INVALIDATE_REGION = 1 << 2,
    }
}

/// cl_program_binary_type
bitflags! {
    pub flags ProgramBinaryType: u64 {
		const PROGRAM_BINARY_TYPE_NONE = 0x0,
		const PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 0x1,
		const PROGRAM_BINARY_TYPE_LIBRARY = 0x2,
		const PROGRAM_BINARY_TYPE_EXECUTABLE = 0x4,
    }
}

/// cl_kernel_arg_type_qualifer 
bitflags! {
    pub flags KernelArgTypeQualifier: u64 {
		const KERNEL_ARG_TYPE_NONE = 0,
		const KERNEL_ARG_TYPE_CONST = 1 << 0,
		const KERNEL_ARG_TYPE_RESTRICT = 1 << 1,
		const KERNEL_ARG_TYPE_VOLATILE = 1 << 2,
    }
}
