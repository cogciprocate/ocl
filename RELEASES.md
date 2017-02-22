Version 0.13.0 (UNRELEASED)
===========================

The futures have arrived! [futures-rs] has begun to find its way into ocl.
This makes doing things embedding (slipstreaming?) host processing work into
the natural chain of enqueued commands, something that was previously
cumbersome to implement, very easy and intuitive now. See the `FutureMemMap`
and `MemMap` types and the new examples. [FIXME] Add links at release

This library will be approaching forward-compatible stabilization over the
next year for all top level types. Before that point can be reached, we'll
have to break a few eggs. This release brings consistency and simplification
changes to a few important functions, notably `Buffer::new` and `Kernel::new`.
See the breaking changes section below for details.

[FIXME] Add doc links at release

* Asynchrony and Futures...
  * [FIXME] Buffer mapping...
  * [FIXME] `BufferCmd`, `ImageCmd`, and `KernelCmd`
  * [FIXME] have received some streamlining and optimizing to events.
  * [FIXME] comment on changes to the types that `::enew` and `::ewait` accept.
  * [FIXME] refer to the breaking changes below
  * [FIXME] ::read, ::write, ::map
    * read no longer unsafe
    * how to use futures, etc.

* `SubBuffer` has been added and represents a subregion of a `Buffer`. It can
  be used just as you would `Buffer`. Use `SubBuffer::new` or
  `Buffer::create_sub_buffer` to create one.
* `Kernel` buffer and image related functions (such as `arg_buf`) can now
  interchangeably accept either `Buffer<T>`, `SubBuffer<T>`, or `Image<T>`
  types.
* `Kernel::set_arg` and `Kernel::named_arg_idx` have been added allowing the
  ability to set a kernel argument by index and retrieve the index of a named
  argument. Argument indexes always correspond exactly to the order arguments
  are declared.
* Command queue properties can now be specified when creating a `Queue` or
  `ProQue` allowing out of order execution and profiling to be enabled.
  Profiling had previously been enabled by default but now must be explicitly
  enabled by setting the `QUEUE_PROFILING_ENABLE` flag.

Breaking Changes
----------------
* `Buffer::new` continues to be unstable and is not reccommended for use
  directly. Instead use the new [`BufferBuilder`] by calling
  [`Buffer::builder()`].
  * Before: 

    ```
    Buffer::new(queue, Some(flags), dims, Some(&data))
    ```
  * Now: 

    ```
    Buffer::builder()
      .queue(queue)
      .flags(flags)
      .dims(dims)
      .host_data(&data)
      .build()
    ```
* [`Kernel::new`] no longer accepts a queue as a third argument. Instead use the
  [`::queue`][kernel_queue] (builder-style) or
  [`::set_default_queue`][kernel_set_default_queue] methods. For example:
  * Before: 

    ```
    Kernel::new("kernel_name", &program, queue).unwrap()
    ```
  * Now: 

    ```
    Kernel::new("kernel_name", &program).unwrap().queue(queue)
    ```
    [FIXME] Add/update links
* `BufferCmd`, `ImageCmd`, and `KernelCmd` now [FIXME: complete]  
  * [FIXME] `::copy` signature change (`offset` and `len` (size))
  * [FIXME] `::enew`, `::enew_opt`, `::ewait`,  and `::ewait_opt` signature
    changes (trait obj -> enum)
* `Buffer::is_empty` has been removed.
* `Buffer::from_gl_buffer`, `Buffer::set_default_queue`,
  `ImageBuilder::build`, `ImageBuilder::build_with_data`, `Image::new`,
  `Image::from_gl_texture`, `Image::from_gl_renderbuffer`,
  `Image::set_default_queue`, `Kernel::new`, `Kernel::set_default_queue` now
  take an owned `Queue` instead of a `&Queue` (clone it yourself).
* All row pitch and slice pitch arguments (used with image and rectangular
  buffer operations) must now be expressed in bytes.
* `Kernel::set_default_queue` no longer result-wraps its `&'mut Kernel` return
  value.
* `Kernel` named argument declaration functions such as `::arg_buf_named` or
  `::set_arg_img_named` called with a `None` variant must now specify the full
  type of the image, buffer, or sub-buffer which will be used for that
  argument. Where before you might have used: 
    ```.arg_buf_named::<f32>("buf", None)```
  you must now use: 
    ```.arg_buf_named("buf", None::<Buffer<f32>>)``` 
  or 
    ```.arg_buf_named::<f32, Buffer<f32>>("buf", None)```.
* `Queue::new` now takes a third argument: `properties` (details below in
  ocl-core section).
* `Queue::finish` now returns a result instead of unwrapping.
* `Program::new` has had its arguments rearranged for consistency.
* `Event::wait` and `EventList::wait` have both been renamed to `::wait_for`.
* [FIXME: elaborate] `core_as_ref` & `core_as_mut` rename

### Breaking Changes to `ocl-core`
* Passing event wait list and new event references has been completely
  overhauled. Previously, references of this type had to be converted into the
  trait objects `ClWaitList` and `ClEventPtrNew`. This was convenient (outside
  of the occasional awkward conversion) and obviated the need to type annotate
  every time you passed `None` to an `enqueue...` function. Now that futures
  have come to the library though, every ounce of performance must be wrung
  out how events are processed and handled. This means that events and event
  lists are now treated as normal generics, not trait objects, and are
  therefore slightly more efficient, at the cost of it being less convenient
  to call functions when you are *not* using them. This leads to the following
  breaking changes:
  * The `ClWaitList` trait has been renamed to `ClWaitListPtr`
  * The `ClEventPtrNew` trait has been renamed to `ClNullEventPtr`
  * All `core::enqueue...` functions (and a few others) may now require
    additional type annotations when `None` is passed as either the event wait
    list or new event reference. Passing a `None::<EventList>` will suffice
    for either parameter (because it can serve either role) and is the easiest
    way to shut the compiler up. If you were previously annotating a type
    using the 'turbo-fish' syntax ('::<...>') on one of these functions, you
    will also have to include two additional type annotations. It may be more
    convenient to do all the annotation in one spot in that case and remove it
    from the `None`s.
* All row pitch and slice pitch arguments (for image and rectangular buffer
  operations) must now be expressed in bytes.

* `EventList::pop` now returns an `Option<Event>` instead of an
  `Option<Result<Event>>`.
* `::create_command_queue` now takes a third argument: `properties`, an
  optional bitfield described in the [clCreateCommandQueue SDK
  Documentation]. Valid options include
  `QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` and `QUEUE_PROFILING_ENABLE`.
* `::build_program` the `devices` argument is now optional. Not specifying any
  devices will cause the program to be built for all available devices within
  the provided context.


  [FIXME]
  * `enqueue_map_buffer`, `enqueue_map_image`, `enqueue_unmap_mem_object`

Other Changes
-------------
* `EventList::clear` has been added.
* `EventList` auto-clearing has been experimentally re-enabled.


[FIXME] Update links

[`Buffer::builder()`]: http://docs.cogciprocate.com/ocl/ocl/struct.Buffer.html#method.builder
[`BufferBuilder`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.BufferBuilder.html
[`Kernel::new`]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.new
[kernel_queue]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.queue
[kernel_set_default_queue]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.set_default_queue
[futures-rs]: https://github.com/alexcrichton/futures-rs
[clCreateCommandQueue SDK Documentation]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateCommandQueue.html


Version 0.12.0 (2017-01-14)
===========================

Breaking Changes
----------------
* `Buffer::new` has undergone small signature changes.
  * The `queue` argument now accepts an owned rather than a borrowed `Queue`.
    `Buffer` now stores it's own `ocl::Queue` (changed from a
    `core::CommandQueue`, a result of the addition of the 'version control'
    system). It is now left to the caller to clone the queue when necessary.
  * The `dims` argument is now constrained by `Into<SpatialDims>` rather than
    `MemLen` for consistency with other arguments of it's kind.
* `Buffer` now has a `::dims` method which returns a `SpatialDims` reference.
* `Device::list`, `::list_all`, `::list_select`, and `::list_select_wrap` now
  wrap their return value in an `ocl::Result`.
* `Device::max_wg_size` now returns an `ocl::Result` instead of panicing.
* `ProQue::max_wg_size` now returns an `ocl::Result` instead of panicing.
* `EventList::push` and `EventList::pop` have been added.
* ocl-core: 
  * `::create_context` and `::create_context_from_type` have had their
    signatures changed. The `properties` argument is now an
    `Option<&ContextProperties>`.


Version 0.11.0 (2016-08-29)
===========================

The `core` and `ffi` modules have been moved out into new crates, [ocl-core]
and [cl-sys] respectively. They continue to be exported with the same names.
This document will continue to contain information pertaining to all three
libraries ([ocl], [ocl-core], and [cl-sys]). Issues will likewise be centrally
handled from the [ocl] repo page.

The version control system has been implemented. Functions added since OpenCL
1.1 now take an additional argument to ensure that the device or platform
supports that feature. See the [ocl-core crate documentation] for more
information.

Breaking Changes
----------------
* `Context::platform` now returns a `Option<&Platform>` instead of
  `Option<Platform>` to make it consistent with other methods of its kind.
* [ocl-core] `DeviceSpecifier::to_device_list` now accepts a
  `Option<&Platform>`.
* [ocl-core] Certain functions now require an additional argument for version
  control purposes (see the [ocl-core crate documentation]).
* [cl-sys] Types/functions/constants from the `cl_h` module are now
  re-exported from the root crate. `cl_h` is now private.
* [cl-sys] Functions from the OpenCL 2.0 & 2.1 specifications have been added.
  This may cause problems on older devices. Please file an [issue] if it does.


[ocl-core crate documentation]: http://docs.cogciprocate.com/ocl_core/ocl_core
[ocl]: https://github.com/cogciprocate/ocl
[ocl-core]: https://github.com/cogciprocate/ocl-core
[cl-sys]: https://github.com/cogciprocate/cl-sys
[issue]: https://github.com/cogciprocate/ocl/issues


Version 0.10.0 (2016-06-11)
===========================

New Features
------------
* Vector types have been added making their use more intuitive.
* MSVC support is working and should be much easier to get running (more
  simplification to linking libraries coming).
* Preliminary OpenGL interop support:
  * OpenGL context handles are accepted as properties during `Context` creation.
    - Note: The new methods involved in this may soon be renamed.
  * Buffers can be created from a GL object.

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
