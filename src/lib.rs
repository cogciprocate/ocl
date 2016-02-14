//! # [![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl) | [GitHub](https://github.com/cogciprocate/ocl)
//!
//! Rust implementation of OpenCL.
//!
//!
//! This documentation is very much a work in progress, as is the library itself. Please help by filing an [issue](https://github.com/cogciprocate/ocl/issues) about unclear and/or incomplete documentation and it will be addressed.
//!
//! An explanation of how dimensions and sizes of buffers and work queues are intended 
//! to be used will be coming as soon as a few more things are ironed out.
//!
//! ## Low Level Interfaces
//!
//! For even lower level interfaces see the `raw` and `cl_h` modules.
//!
//! ## Library Wide Panics
//!
//! All operations will panic upon any OpenCL error. Some work needs to be done
//! evaluating which errors are easily uncovered during development
//! and are therefore better off 
//! continuing to panic such as invalid kernel code, and which errors are 
//! more run-timeish and should be returned in a `Result`.
//!
//! ## Help Wanted
//!
//! Please help complete any functionality you may need by filing an 
//! [issue] or creating a [pull request](https://github.com/cogciprocate/ocl/pulls).
//!
//! [issue]: https://github.com/cogciprocate/ocl/issues

// #![warn(missing_docs)]
#![feature(zero_one)]

#[macro_use] extern crate enum_primitive;
#[macro_use] extern crate bitflags;
extern crate libc;
extern crate num;
extern crate rand;

#[cfg(test)] mod tests;
mod standard;
mod error;
pub mod raw;
pub mod cl_h;
pub mod util;

#[cfg(not(release))] pub use standard::BufferTest;
pub use standard::{Context, ProgramBuilder, BuildOpt, Program, Queue, Kernel, Buffer, Image,
    ProQueBuilder, ProQue, SimpleDims, WorkDims, OclNum, BufferDims, EventList};
pub use self::error::{Error, Result};

