ocl
===

#### [Documentation](http://doc.cogciprocate.com/ocl/ocl/) | [Change Log](https://github.com/cogciprocate/ocl/blob/master/RELEASES.md)

[![](http://meritbadge.herokuapp.com/ocl)](https://crates.io/crates/ocl)


Pure OpenCL&trade; bindings and interfaces for Rust. Makes easy to use the
most common features of OpenCL. All interfaces are virtually zero-cost and
perform on a par with the C++ bindings.

Interfaces are still mildly unstable. Changes are now being documented in
[RELEASES.md](https://github.com/cogciprocate/ocl/blob/master/RELEASES.md).


## Goals

To provide:
- A simple and intuitive interface with OpenCL devices
- The full functionality of the OpenCL API
- An absolute minimum of boilerplate
- Zero or virtually zero performance overhead
- Thread-safe and automatic management of API pointers and resources


## Installation

Ensure that an OpenCL library is installed for your platform and that `clinfo`
or some other diagnostic command will run. Add the following to your project's
`Cargo.toml`:

```rust
[dependencies] 
ocl = "0.8"
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
///////////// See the original file for more /////////////
```

See the the remainder of [`examples/trivial.rs`] for much more.


#### Diving Deeper

Already familiar with the standard OpenCL core API? See the [`core`] module
for access to the complete feature set in the conventional API style with
Rust's safety and convenience.


##### Version Support

1.1 support is intact but intentionally disabled for simplicity. If this
support is needed, please file an [issue] and it will be reenabled. Automatic
best-version support for versions going all the way back to 1.0 will
eventually be added.


##### What About Vulkan&trade;?

The OpenCL API already posesses all of the new attributes of the Vulkan API
such as low-overhead, high performance, and unfettered hardware access. For all
practical purposes, Vulkan is simply a graphics-focused superset of OpenCL's
features (sorta kinda). OpenCL 2.1+ and Vulkan kernels/shaders now both
compile into SPIR-V making the device side of things the same. I wouldn't be
suprised if most driver vendors implement the two host APIs identically.

In the future it's possible the two may completely merge (or that Vulkan will
absorb OpenCL). Whatever happens, not much will change as far as the front end
of this library is concerned (though the `core` module functions / types could
get some very minor renaming, etc. but it wouldn't be for a very long time...
version 2.0...). This library will maintain it's focus on the compute side of
things. For the graphics side, see the excellent OpenGL library, [glium], and
its younger sibling, [vulkano].


##### Help

Try `cargo run --example info` or `cargo run --example info_core` and see what
happens.

*If troubleshooting your OpenCL drivers:* check that `/usr/lib/libOpenCL.so.1`
exists. Go ahead and link `/usr/lib/libOpenCL.so -> libOpenCL.so.1` just in
case it's not already done (AMD drivers sometimes don't create this link).
Intel also has [OpenCL libraries for your CPU] if you're having trouble
getting your GPU to work (AMD used to have some for CPUs too, can't find them
anymore).

Due to buggy and/or intentionally crippled drivers, functionality involving
multiple host threads, multiple devices, or asynchronous callbacks may not
work on NVIDIA hardware. Until NVIDIA's implementation is corrected, tests and
examples involving multiple asynchronous tasks may occasionally fail on that
platform.

Please ask questions and provide feedback by opening an
[issue].

<br/>

*“OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission
by Khronos.”* *“Vulkan and the Vulkan logo are trademarks of the Khronos Group Inc.”*

[OpenCL libraries for your CPU]: https://software.intel.com/en-us/intel-opencl/download
[AMD]: https://software.intel.com/en-us/intel-opencl/download
[`core`]: http://docs.cogciprocate.com/ocl/ocl/core/index.html
[issue]: https://github.com/cogciprocate/ocl_rust/issues
[provide feedback]: https://github.com/cogciprocate/ocl_rust/issues
[`examples`]: https://github.com/cogciprocate/ocl/tree/master/examples
[`examples/trivial.rs`]: https://github.com/cogciprocate/ocl/blob/master/examples/trivial.rs#L27
[glium]: https://github.com/tomaka/glium
[vulkano]: https://github.com/tomaka/vulkano/tree/master/vulkano
