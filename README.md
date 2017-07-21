# OpenCL+OpenGL Interop for Rust
Creates the `ContextProperties` that you pass to the ContextBuilder to set up the CL context.
Uses the [`ocl`](https://github.com/cogciprocate/ocl) OpenCL binding.
MacOS support is untested, but should work.
OpenGL ES isn't supported, but will be added soon.

## Usage
Add the following to your Cargo.toml, assuming you are using cargo.
```toml
[dependencies]
ocl-interop = "*"
```
Then, when you need the context
```rust
// .. other code, including picking a platform ..
let mut context = ocl-gl::get_properties_list().platform(platform).build().unwrap();
// .. go on and use it ..
```

