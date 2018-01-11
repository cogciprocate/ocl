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
// Create an OpenGL context, make sure it is active, then create an OpenCL
// context with the interop enabled:
let context = ocl_interop::get_context()?;

// use it!

```
