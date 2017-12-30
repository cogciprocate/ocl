# OpenCL+OpenGL Interop for Rust
Creates a Context with GL interoperability enabled.
DirectX Interop is theoretically possible, but currently not implemented.
MacOS & OpenGL ES support is untested, but should work.

## Usage
Add the following to your Cargo dependancies
```toml
ocl-interop="0.1.3"
```
Then, when you need the context
```rust
// Make sure the gl context is active
//Create an OpenCL context with the GL interop enabled
let context=get_context().expect("Cannot create OpenCL Context");
// use it!
```
