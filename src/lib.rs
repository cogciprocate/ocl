//! # [![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl) | [GitHub](https://github.com/cogciprocate/ocl)
//!
//! Rust implementation of OpenCL&trade;.
//!
//! This documentation is generally built from the [`dev` branch](https://github.com/cogciprocate/ocl/tree/dev) and may be newer than what is on crates.io and the master branch. 
//!
//! Documentation is very much a work in progress, as is the library itself. Please help by filing an [issue](https://github.com/cogciprocate/ocl/issues) about unclear and/or incomplete documentation and it will be addressed.
//!
//! An explanation of how dimensions and sizes of buffers and work queues are intended 
//! to be used will be coming as soon as a few more things are ironed out.
//!
//! ## Low Level Interfaces
//!
//! For lower level interfaces and to use OpenCL features that have not yet been implemented, see the [`core`] and [`cl_h`] modules.
//!
//! ## Library Wide Panics
//!
//! Many operations will panic upon any OpenCL error [UPDATE: Error handling is in a state of transition, some things panic, some return results].
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

#[cfg(test)] mod tests;
mod standard;
mod error;
pub mod core;
pub mod cl_h;
pub mod util;

pub use core::{OclNum, ImageFormat, ImageDescriptor, ImageChannelOrder, ImageChannelDataType, MemFlags, MemObjectType, MEM_READ_WRITE, MEM_WRITE_ONLY, MEM_READ_ONLY, MEM_USE_HOST_PTR, MEM_ALLOC_HOST_PTR, MEM_COPY_HOST_PTR, MEM_HOST_WRITE_ONLY, MEM_HOST_READ_ONLY, MEM_HOST_NO_ACCESS, PlatformInfo, DeviceInfo, ContextInfo, ProgramInfo, CommandQueueInfo, KernelInfo, MemInfo, ImageInfo, EventInfo};
#[cfg(not(release))] pub use standard::BufferTest;
pub use standard::{Platform, Device, ContextBuilder, Context, BuildOpt, ProgramBuilder, Program, Queue, Kernel, Buffer, Image, ProQueBuilder, ProQue, SimpleDims, WorkDims, BufferDims, Event, EventList, DeviceSpecifier};
pub use self::error::{Error, Result};

