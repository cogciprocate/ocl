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
//! All functions will panic upon OpenCL error. This will be changing over time. Certain errors will eventually be returned as an `Error` type instead.
//!
//! ### Official Documentation
//!
//! [OpenCL 1.2 SDK Reference: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/)
//!
//! ### Help Wanted
//!
//! Please help complete coverage of any FFI functions you may need by filing an [issue](https://github.com/cogciprocate/ocl/issues) or creating a [pull request](https://github.com/cogciprocate/ocl/pulls).

mod functions;
pub mod enums;

// [FIXME]: Import everything individually.
pub use self::functions::*;
