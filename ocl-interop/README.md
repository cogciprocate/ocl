# OpenCL + OpenGL Interoperability for Rust
Creates a Context with OpenGL interoperability enabled. DirectX interop is
theoretically possible, but not currently implemented. MacOS & OpenGL ES
support is untested, but should work.

## Usage

Your preferred OpenGL library should be set up and working.

Add the following to your `Cargo.toml`:

```toml
ocl-interop = "0.1"
```

Add the following to your crate root (lib.rs or main.rs):

```
extern crate ocl_interop;
```

Then, when you need the context:

```rust
// First, create an OpenGL context and make sure it is active...

// Next, Create an OpenCL context with the interop enabled: (NOTE:
// `::get_context` will return the first available GPU device on your that
// supports OpenGL interop on your system -- you may need to choose a device
// and create the context manually instead if this does not work):
let context = ocl_interop::get_context()?;

// Later, after creating an OpenGL buffer...

// Create an OpenCL buffer from an OpenGL buffer:
let cl_buffer = ocl::Buffer::<f32>::from_gl_buffer(&queue, None, gl_buffer)?;

// Acquire the buffer, making it usable:
cl_buffer.cmd().gl_acquire().enq()?;

// Use the buffer...

// Release the acquisition:
cl_buffer.cmd().gl_release().enq()?;

```


