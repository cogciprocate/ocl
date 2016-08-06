Version 0.11.0 (UNRELEASED)
===========================

The `core` and `ffi` modules have been moved into new crates, [ocl-core] and
[cl-sys] respectively.

The version control system has been implemented. Functions added since OpenCL
1.1 now take an additional argument to ensure that the device or platform
supports that feature. See the [ocl-core crate documentation] for more
information.

Breaking Changes
----------------
* `Context::platform` now returns a `Option<&Platform>` instead of
  `Option<Platform>` to make it consistent with other methods of its kind.
* `DeviceSpecifier::to_device_list` now accepts a `Option<&Platform>`.


[ocl-core crate documentation]: http://docs.cogciprocate.com/ocl_core/ocl_core
[ocl-core]: https://github.com/cogciprocate/ocl-core
[cl-sys]: https://github.com/cogciprocate/cl-sys


Version 0.10.0 (2016-06-11)
===========================

New Features
------------
* Vector types have been added making their use more intuitive.
* MSVC support is working and should be much easier to get running (more
  simplification to linking libraries coming).
* Preliminary OpenGL interop support:
  - OpenGL context handles are accepted as properties during `Context` creation.
    - Note: The new methods involved in this may soon be renamed.
  - Buffers can be created from a GL object.

Breaking Changes
----------------
* The command builders for kernel, buffer, and image now accept either an
  `Event` or `EventList` argument when setting the wait list using `::ewait`
  or `::ewait_opt`. Awkward type ascriptions when not passing a wait list can
  now be removed due to the use of a trait object argument type.
* 'fill' methods now accept a vector type.
* `Kernel::arg_vec` now takes a vector type.



Version 0.9.0 (2016-03-28)
==========================

Approaching Stability
---------------------

Ocl is now confirmed to work on Rust-Windows-GNU (MSYS2/MinGW-w64). AMD and
Intel drivers appear to work flawlessly. NVIDIA drivers have glitches here and
there (particularly to do with concurrency) but are mostly functional.
Windows-MSVC testing is in progress. A short HOWTO for getting OpenCL drivers
installed properly and working with Rust is in the works. For now just be sure
to put a copy of the ICD loader, OpenCL.dll, usually found somewhere within
the `C:\Windows` folder tree, into the Rust library folder (defaults to
`C:\Program Files\{Rust folder}\lib\rustlib\x86_64 -pc-windows-gnu\lib`) and
make sure your platform drivers are installed correctly (there's a registry
setting + they must be in the PATH). See [README.md] for links to drivers.

Still no verification on the status of OS X but there is no reason it
shouldn't work fine.


Breaking Changes
----------------
* [`ImageBuilder`] has had its `row_pitch` and `slice_pitch` methods renamed
  to `row_pitch_bytes` and `slice_pitch_bytes` to indicate the units they
  expect their arguments expressed in.
* `core::unload_platform_compiler` has been removed due to platform
  incompatability with some drivers.
* The `KernelArg::Scalar` variant now contains a primitive rather than a
  primitive reference.
* `::src_file_name` and `::get_file_names` have been removed from
  `ProgramBuilder`. Use `::src_file` to set a source file `Path` or `PathBuf`
  to include in the build.
* The `BuildOpt::cmplr_opt` method has been removed.
* The `build` module has been renamed to `builders`.

[README.md]: https://github.com/cogciprocate/ocl/blob/master/README.md


Version 0.8.0 (2016-03-09)
==========================

Breaking Changes
----------------
* [`Buffer`] has undergone a major redesign:
  * The optional built-in vector has been removed. Reads and writes must now
    all be done using a separate data container.
  * All of the methods for reading and writing have been removed and
    replaced by the new command builder system, accessible with `::cmd`
    (more documentation to come).
  * All of the traits pertaining to the internal vector, such as Index, have
    been removed.
  * All of the constructors have been removed and replaced by a single
    method, `::new`.
  * Many of the convenience methods for initializing buffers with randomly
    scrambled or shuffled data now take the form of functions as,
    `::scrambled_vec` and `::shuffled_vec` in [`ocl::util`].
  * `::set_queue` has been renamed `::set_default_queue`.
  * `::wait` has been removed. Queue related methods are now accessed on the
    queue itself using `::queue`.
  * Length is no longer padded out to the next workgroup size. It is up to
    the consumer to pad the sizes of buffers (the new kernel method,
    `::wg_info` can help determine optimal sizes).
* [`Image`] has also had its read/write methods removed and replaced with a
  command builder accessible using `::cmd` (more documentation to come).
* `Image::set_queue` has been renamed `::set_default_queue`.
* [`Kernel`] has had its various `::enqueue_***` methods removed and replaced
    with, you guessed it, a command builder (`::cmd`).
* `Kernel::new` no longer accepts a global work size as an argument. Instead
    use the new builder-style method, `::gws` after creating.
* `Kernel::set_queue` has been renamed `::set_default_queue`.
* `Queue::new_by_device_index` has been removed.
* The `device` argument for `Queue::new` is no longer optional.
* `ProgramBuilder::src_file` now takes a `Path` instead of a string.
* `ProQue::create_kernel_with_dims` has been removed.
* `ProQue::device_idx` has been replaced by `::device`.
* `Context::new_by_index_and_type` has been removed.
* `core::set_kernel_arg` and `::enqueue_kernel` no longer have an argument for
  the kernel function name. Instead it is queried when needed using
  `::get_kernel_info`.
* `SimpleDims` has been renamed `SpatialDims` and many of its methods now
  return `Result` types.
* `OclNum` has been renamed `OclPrm`


New Features
------------
* Command builders for [`Kernel`], [`Buffer`], and [`Image`] can now be used
  by calling `::cmd` on each.
* Rectangular reads, writes, and copies are now wired up and have been tested.
* Most of the remaining functions in the `core` module have been implemented.
  Coverage is now about 98%.
* [`Sampler`] has been added along with the appropriate methods on [`Kernel`]
  to accept samplers as arguments. See `examples/image.rs` for usage.
* Dimensions for images, buffers, kernels, and everything else can now be
  specified by using a tuple OR array with 1, 2, or, 3 components (i.e. `[80,
  150]`, `(5, 1, 7)` or just `[250]`).


[`Sampler`]: http://doc.cogciprocate.com/ocl/ocl/struct.Sampler.html
[`ocl::util`]: http://docs.cogciprocate.com/ocl/ocl/util/index.html


Version 0.7.0 (2016-02-27)
==========================

Breaking Changes
----------------
* `Kernel::enqueue` is now called `Kernel::enqueue_with` and has an additional
  parameter to set an alternative command queue. A new method with the old
  name is now a convenience shortcut for `.enqueue_with(None, None, None)`.
* `ProQue::create_kernel` has been renamed `ProQue::create_kernel_with_dims`.
  A new method with the old name, is now a shortcut for kernel creation using
  pre-assigned dimensions` (this naming is likely temporary).
* The kernel created using `ProQue::create_kernel` is no longer wrapped in a
  result and instead panics if there is a problem. If you require a
  non-panicing way to create a kernel use `Kernel::new`.
* `Context::new` has been redesigned. It is now recommended to use
  `Context::builder` or its equivalent, `ContextBuilder::new' to create a
  `Context`.
* `Queue::new` now takes a `Device` as its second argument. Use the new
  `Context::get_device_by_index` to achieve the same result.
* All 'standard' types refer to `Device` and `Platform` instead of
  `core::DeviceId` and `core::PlatformId` in method signatures.
* `Buffer::read_async` and `::write_async` have been renamed `::enqueue_read`
  and `::enqueue_write` and have an additional parameter to set blocking.
* `Buffer::fill_vec_async` and `::flush_vec_async` have been renamed
  `::enqueue_fill_vec` and `::enqueue_flush_vec` and have an additional
  parameter to set blocking.

New Features
------------
* Images! I can see! ... oh shut up.
* [`Image`] and [`ImageBuilder`] have been added. Please see their
  documentation along with [`examples/image.rs`].


[`ImageBuilder`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.ImageBuilder.html
[`examples/image.rs`]: https://github.com/cogciprocate/ocl/blob/master/examples/image.rs


Version 0.6.0 (2016-02-20)
==========================

Breaking Changes
----------------
* Some methods and functions now return results where before they would unwind.
* The `::release` method has been removed from those types which still had it.
  All types now automatically release their resources properly.
* `Buffer::fill_vec` and `Buffer::flush_vec` no longer return results and
  instead panic in the event of an error.
* All of the `Buffer` creation methods (such as `::new` and `::with_vec`) now
  take a reference to a `BufferDims` type for the `dims` argument instead
  moving it.
* The `raw` module has been renamed to `core` for clarity.
* Functions in the `core` module now take references to `*Raw` types instead of
  copying them.
* `*Raw` types no longer implement `Copy`.
* Many of the method names dealing with references to `core` objects have been
  renamed.

New Features
------------
* `core` has a considerable number of newly implemented (and unimplemented
  placeholder) functions.
* Many 'info' functions and types have been added. See the example, `info.rs`,
  for details on how to use them.
* All types are now completely safe to clone (where appropriate) and share
  between threads (with the exception of `Kernel`, for good reason) and are
  reference counted automatically in coordination with the API to ensure safe
  and leak-free destruction.


[0.6doc]: http://doc.cogciprocate.com/ocl/ocl/
[`raw`]: http://docs.cogciprocate.com/ocl/ocl/raw/index.html


Version 0.5.0 (2016-02-15)
==========================

Lots of changes, breaking and otherwise:

A new type, [`Image`] has been added for processing images. It is still very
much a work in progress.

The new [`raw`] api allows access to OpenCL&trade; FFI functions with only a
thin layer of abstraction providing safety and convenience. Using functions in
this module is only recommended for use when functionality has not yet been
implemented on the 'standard' ocl interfaces.

Breaking Changes
----------------
* [`Buffer`] has had several methods dealing with reading and writing renamed
  and two new ones created.
   * `::flush_vec` and `::fill_vec` have been renamed to `::flush_vec_async`
     and `::fill_vec_async`.
   * `::flush_vec_wait` and `::fill_vec_wait` have been renamed to
     `::flush_vec` and `::fill_vec`.
   * `::read` and `::write` have been renamed `::read_async` and
     `::write_async`.
   * Blocking versions of read and write have been created called, you guessed
     it, `::read` and `::write`.
  The more straightforward, blocking versions of these methods now have the
  simplest names wheras the more complicated, non-blocking versions have the
  `_async` suffix.
* [`Buffer`] non-blocking read methods (*_async) are now unsafe pending review.
* [`Buffer`] reading and writing methods now return a `Result<()>`.
* The `Buffer::with_vec_shuffled` and `Buffer::with_vec_scrambled` methods
  now accept a 2-tuple as the first argument instead of two separate values for
  the first two arguments.
* `ProQue::build` has been renamed `ProQue::build_program`.
* `BuildOptions` has been renamed to [`ProgramBuilder`] and has been
  redesigned:
   * A new [`ProgramBuilder`] can be created with `ProgramBuilder::new` or
     `Program::builder`.
   * The `::build` method has been added, consuming the builder and returning
     a new [`Program`].
   * Methods dealing with kernel source code have been renamed for clarity.
   * Extraneous methods have been removed.
* The following methods on [`Kernel`] have been renamed reflecting `Envoy`
  having been recently renamed to [`Buffer`] in v0.4.0:
  * `::arg_env` to `::arg_buf`
  * `::arg_env_named` to `::arg_buf_named`
  * `::set_arg_env_named` to `::set_arg_buf_named`
* Several non-essential methods on [`Kernel`] have been depricated.
* `Kernel::new` and its equivalent, `ProQue::create_kernel`, now return a
  `Result<Kernel>` instead of just [`Kernel`].
* `Kernel::set_arg_buf_named` and `Kernel::set_arg_buf_named` now require an
  `Option` wrapper.
* [`SimpleDims`] has had its variants renamed.
* `WorkSize` has been renamed to [`WorkDims`] and has had its variants renamed.
* `Context::new` now takes a [`DeviceType`] instead of a u32.


New Types
---------
* [`ProQueBuilder`] is now the most boilerplate-free way to create an OpenCL
  context, program, and queue. Create one by calling [`ProQue::builder`].
  See [`basics.rs` for an example][0.5ba] and [documentation][0.5doc] for more info.
* [`Image`] is still a newborn.


[0.5doc]: http://doc.cogciprocate.com/ocl/ocl/
[0.5ba]: https://github.com/cogciprocate/ocl/blob/master/examples/basics.rs
[`Buffer`]: http://docs.cogciprocate.com/ocl/ocl/struct.Buffer.html
[`Image`]: http://docs.cogciprocate.com/ocl/ocl/struct.Image.html
[`raw`]: http://docs.cogciprocate.com/ocl/ocl/raw/index.html
[`ProQueBuilder`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.ProQueBuilder.html
[`ProQue`]: http://docs.cogciprocate.com/ocl/ocl/struct.ProQue.html
['ProQue::builder']: http://docs.cogciprocate.com/ocl/ocl/struct.ProQue.html#method.builder
[`ProgramBuilder`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.ProgramBuilder.html
[`Program`]: http://docs.cogciprocate.com/ocl/ocl/struct.Program.html
[`Kernel`]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html
[`SimpleDims`]: http://doc.cogciprocate.com/ocl/ocl/enum.SimpleDims.html
[`WorkDims`]: http://doc.cogciprocate.com/ocl/ocl/enum.WorkDims.html
[`DeviceType`]: http://doc.cogciprocate.com/ocl/ocl/raw/struct.DeviceType.html

<br/><br/>
*“OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.”*
