ocl
===

#### Documentation: [Release](https://docs.rs/ocl)/[Master](http://docs.cogciprocate.com/ocl/ocl/) | [Change Log](https://github.com/cogciprocate/ocl/blob/master/RELEASES.md)

[![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl) [![](https://docs.rs/ocl/badge.svg)](https://docs.rs/ocl)
[![Supported platforms](https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux%20%7C%20bsd-orange.svg)](https://en.wikipedia.org/wiki/Cross-platform)


Pure OpenCL&trade; bindings and interfaces for
[Rust](https://www.rust-lang.org/).

## Goals

To provide:
- A simple and intuitive interface to OpenCL devices
- The full functionality and power of the OpenCL API
- An absolute minimum of boilerplate
- Zero or virtually zero performance overhead
- Thread-safe and automatic management of API pointers and resources

## Usage

Ensure that an OpenCL library is installed for your platform and that `clinfo`
or some other diagnostic command will run. Add the following to your project's
`Cargo.toml`:

```toml
[dependencies]
ocl = "0.15"
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
        .dims(1 << 20)
        .build().unwrap();

    let buffer = pro_que.create_buffer::<f32>().unwrap();

    let kernel = pro_que.create_kernel("add").unwrap()
        .arg_buf(&buffer)
        .arg_scl(10.0f32);

    unsafe { kernel.enq().unwrap(); }

    let mut vec = vec![0.0f32; buffer.len()];
    buffer.read(&mut vec).enq().unwrap();

    println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
}
```

See the the remainder of [`examples/trivial.rs`] for more information about
how this library leverages Rust's zero-cost abstractions to provide the full
power and performance of the C API in a simple package.

##### Introduction to OpenCL

For a quick but thorough primer on the basics of OpenCL, please see [Matthew
Scarpino's excellent article, 'A Gentle Introduction to OpenCL' at
drdobbs.com](http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854)
(his
[book](https://www.amazon.com/OpenCL-Action-Accelerate-Graphics-Computations/dp/1617290173/ref=sr_1_2?ie=UTF8&qid=1500745843&sr=8-2&keywords=opencl)
is great too).

##### Diving Deeper

Already familiar with the standard OpenCL core API? See the [`ocl-core`] crate
for access to the complete feature set in the conventional API style with
Rust's safety and convenience.

##### Version Support

OpenCL versions 1.1 and above are supported. OpenCL version 1.0 is **not**
supported due to its inherent thread unsafety.

##### Vulkan&trade; and the Future

The OpenCL API already posesses all of the new attributes of the Vulkan API
such as low-overhead, high performance, and unfettered hardware access. For all
practical purposes, Vulkan is simply a graphics-focused superset of OpenCL's
features (sorta kinda). OpenCL 2.1+ and Vulkan kernels/shaders now both
compile into SPIR-V making the device side of things the same. I wouldn't be
suprised if most driver vendors implement the two host APIs identically.

In the future it's possible the two may completely merge (or that Vulkan will
absorb OpenCL). Whatever happens, nothing will change as far as the front end
of this library is concerned. This library will maintain its focus on the
compute side of things. For the graphics side, see the [voodoo] library.

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

[`async_cycles.rs`]: https://github.com/cogciprocate/ocl/blob/master/examples/async_cycles.rs
[`async_menagerie.rs`]: https://github.com/cogciprocate/ocl/blob/master/examples/async_menagerie.rs
[`async_process.rs`]: https://github.com/cogciprocate/ocl/blob/master/examples/async_process.rs
[OpenCL libraries for your CPU]: https://software.intel.com/en-us/intel-opencl/download
[AMD]: https://software.intel.com/en-us/intel-opencl/download
[`ocl-core`]: https://github.com/cogciprocate/ocl-core
[issue]: https://github.com/cogciprocate/ocl_rust/issues
[provide feedback]: https://github.com/cogciprocate/ocl_rust/issues
[`examples`]: https://github.com/cogciprocate/ocl/tree/master/examples
[`examples/trivial.rs`]: https://github.com/cogciprocate/ocl/blob/master/examples/trivial.rs
[voodoo]: https://github.com/cogciprocate/voodoo
[intel-win64]: https://software.intel.com/en-us/articles/opencl-drivers#win64
[intel-linux64-redhat-suse]: https://software.intel.com/en-us/articles/opencl-drivers#lin64
[intel-linux64-ubuntu]: https://software.intel.com/en-us/articles/opencl-drivers#ubuntu64
[amd-app-sdk]: http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
