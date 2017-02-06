ocl
===

#### [Documentation](https://docs.rs/ocl) | [Change Log](https://github.com/cogciprocate/ocl/blob/master/RELEASES.md)

[![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl) [![](https://docs.rs/ocl/badge.svg)](https://docs.rs/ocl)


Pure OpenCL&trade; bindings and interfaces for
[Rust](https://www.rust-lang.org/). Makes easy to use the most common features
of OpenCL. All interfaces are virtually zero-cost and perform on a par with
native (C/C++) bindings.


## Goals

To provide:
- A simple and intuitive interface with OpenCL devices
- The full functionality of the OpenCL API
- An absolute minimum of boilerplate
- Zero or virtually zero performance overhead
- Thread-safe and automatic management of API pointers and resources


## Usage

Ensure that an OpenCL library is installed for your platform and that `clinfo`
or some other diagnostic command will run. Add the following to your project's
`Cargo.toml`:

```rust
[dependencies] 
ocl = "0.12"
```

And add the following to your crate root (lib.rs or main.rs):
```rust
extern crate ocl;
```


## Example 

From [`examples/trivial.rs`]:
```rust
extern crate ocl;
use ocl::ProQue;

fn main() {
    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims([2 << 20])
        .build().unwrap();

    let buffer = pro_que.create_buffer::<f32>().unwrap();

    let kernel = pro_que.create_kernel("add").unwrap()
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    kernel.enq().unwrap();

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq().unwrap();

    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
}
```

See the the remainder of [`examples/trivial.rs`] for more information about
how this library leverages Rust's zero-cost abstractions to provide the full
power of the C API in a simple package.

Also see the new [alloc_async] example for an assortment of more advanced
uses.


## Development Status

More feedback needed before we can start to stabilize. File an issue just to
let us know what you think or if you have ideas on what could be improved and
smoothed out (nitpicks welcome and appreciated).


#### Diving Deeper

Already familiar with the standard OpenCL core API? See the [`ocl-core`] crate
for access to the complete feature set in the conventional API style with
Rust's safety and convenience.


##### Version Support

OpenCL versions 1.1 and above are supported. OpenCL version 1.0 is **not**
supported due to its inherent thread unsafety.


##### What About Vulkan&trade;?

The OpenCL API already posesses all of the new attributes of the Vulkan API
such as low-overhead, high performance, and unfettered hardware access. For all
practical purposes, Vulkan is simply a graphics-focused superset of OpenCL's
features (sorta kinda). OpenCL 2.1+ and Vulkan kernels/shaders now both
compile into SPIR-V making the device side of things the same. I wouldn't be
suprised if most driver vendors implement the two host APIs identically.

In the future it's possible the two may completely merge (or that Vulkan will
absorb OpenCL). Whatever happens, nothing will change as far as the front end
of this library is concerned. This library will maintain its focus on the
compute side of things. For the graphics side, see the excellent OpenGL
library, [glium], and its younger sibling, [vulkano].


##### Help

If troubleshooting your vendor's drivers:

1. Clone this repo: `git clone https://github.com/cogciprocate/ocl.git`.
2. Change to the newly created directory: `cd ocl`.
3. Run some of the info examples: `cargo run --example info` and/or `cargo run
   --example info_core`.
4. Make sure your platform(s) and device(s) are printed out. If so, you're
   probably good to go.

*Other things to try (linux):* check that `/usr/lib/libOpenCL.so.1` exists. Go
ahead and link `/usr/lib/libOpenCL.so -> libOpenCL.so.1` just in case it's not
already done (some vendors don't create this link). 

If you're still having trouble getting your GPU to work, Intel and AMD also
have OpenCL libraries for your CPU: [amd-app-sdk], [intel-win64],
[intel-linux64-redhat-suse], [intel-linux64-ubuntu]. [amd-app-sdk] works well
on both Intel and AMD processors and is great if you don't mind installing all
of the extra tools (which are pretty decent by the way).

A short HOWTO for getting OpenCL drivers installed properly and working with
Rust-Windows-GNU (MSVC too eventually) is in the works. For now just be sure
to put a copy of the platform-agnostic ICD loader, OpenCL.dll, usually found
somewhere within the `C:\Windows` folder tree, into the Rust library folder
(defaults to `C:\Program Files\{Rust
folder}\lib\rustlib\x86_64-pc-windows-gnu\lib`) and make sure your platform
drivers are installed correctly (there's a registry setting + they must be in
the PATH).

Due to buggy and/or intentionally crippled drivers, functionality involving
multiple host threads, multiple devices, or asynchronous callbacks may not
work on NVIDIA hardware. Until NVIDIA's implementation is corrected, tests and
examples involving multiple asynchronous tasks may occasionally fail or
produce erroneous results (you are suffering from this if the `events.rs` test
fails when you run `cargo test`). It's recommended that you use Intel or AMD
CPU drivers in the meanwhile and switch if/when NVIDIA ever gets their act
together.

Please ask questions and provide feedback by opening an
[issue].

##### License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

##### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

<br/>*“OpenCL and the OpenCL logo are trademarks of Apple Inc. used by
permission by Khronos.”* <br/>*“Vulkan and the Vulkan logo are trademarks of
the Khronos Group Inc.”*

[alloc_async]: https://github.com/cogciprocate/ocl/blob/master/examples/alloc_async.rs
[OpenCL libraries for your CPU]: https://software.intel.com/en-us/intel-opencl/download
[AMD]: https://software.intel.com/en-us/intel-opencl/download
[`ocl-core`]: https://github.com/cogciprocate/ocl-core
[issue]: https://github.com/cogciprocate/ocl_rust/issues
[provide feedback]: https://github.com/cogciprocate/ocl_rust/issues
[`examples`]: https://github.com/cogciprocate/ocl/tree/master/examples
[`examples/trivial.rs`]: https://github.com/cogciprocate/ocl/blob/master/examples/trivial.rs#L27
[glium]: https://github.com/tomaka/glium
[vulkano]: https://github.com/tomaka/vulkano/tree/master/vulkano
[intel-win64]: https://software.intel.com/en-us/articles/opencl-drivers#win64
[intel-linux64-redhat-suse]: https://software.intel.com/en-us/articles/opencl-drivers#lin64
[intel-linux64-ubuntu]: https://software.intel.com/en-us/articles/opencl-drivers#ubuntu64
[amd-app-sdk]: http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/