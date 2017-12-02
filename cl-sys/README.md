OpenCL C FFI Bindings.

### [Documentation](https://docs.rs/cl-sys)

[![](http://meritbadge.herokuapp.com/cl_sys)](https://crates.io/crates/cl_sys)
[![](https://docs.rs/cl-sys/badge.svg)](https://docs.rs/cl-sys)

For a high level, easier to use, and far less verbose OpenCL interface (that
compiles to virtually the same thing) see the
[ocl](https://github.com/cogciprocate/ocl) crate.

Example usage exists within the
[ocl-core](https://github.com/cogciprocate/ocl/tree/master/ocl-core) repo (see:
https://github.com/cogciprocate/ocl/blob/master/ocl-core/src/functions.rs).

If you have need of any unimplemented functionality [please file an
issue](https://github.com/cogciprocate/ocl/issues) and request it.


#### Troubleshooting

Compiling on Windows (particularly MSVC) takes a bit of effort. Better
documentation is needed (please contribute!). If you have trouble please file
an [issue](https://github.com/cogciprocate/ocl/issues) and let us know about
the problem so we can improve our documentation.

Your device drivers should include OpenCL drivers. If not, download the
appropriate SDK from one of the following links:
[AMD](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/),
[NVIDIA](https://developer.nvidia.com/opencl),
[Intel](https://software.intel.com/en-us/intel-opencl)


#### License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.


#### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.


<br/>*“OpenCL and the OpenCL logo are trademarks of Apple Inc. used by
permission by Khronos.”* <br/>*“Vulkan and the Vulkan logo are trademarks of
the Khronos Group Inc.”*