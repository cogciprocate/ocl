Version 0.19.0 (UNRELEASED)
===========================

* Clean up returned error types.
* Add raw info functions for devices (`ocl::Device::info_raw` and
  `ocl-core::get_device_info_raw`). These can be used to query device
  information provided by non-standard OpenCL extensions.


Version 0.18.0 (2018-04-02)
===========================

[`Kernel`] has received a much-needed makeover. Creating a kernel is not
exactly like creating other types of objects due to the fact that kernel
arguments are normally specified after actual kernel creation. Because of
this, the old design became a quirky mix of builder-style and normal methods
intended to be as versatile as possible.

To further complicate matters, the issue of if and when `Kernel` could be
cloned and/or sent between threads was an unsolved problem, potentially
allowing unsafety (race conditions) to occur when setting arguments from
multiple threads.

The original design has been thrown out and a new streamlined design has been
adopted. Many rough spots have been smoothed out and potential invalid uses
have been eliminated.

* `KernelBuilder` has been added. All kernels must now be created using a builder.
* `KernelBuilder::arg`, `KernelBuilder::arg_named`, and `Kernel::set_arg` have
  been been added in order to streamline setting arguments. Each of the new
  methods will all automatically detect whether a `Buffer`, `Image`, scalar,
  or vector value is being passed. (Type annotation is required when `None` is
  passed).
  * `Kernel::set_arg` will accept either a name (`&'static str`) or numeric
    index. If passing a name, the argument must have been declared using
    `KernelBuilder::arg_named`.
* `ProQue::buffer_builder` has been added and can be used to obtain a
  `BufferBuilder` with pre-configured length and default queue.
* `ProQue::kernel_builder` has also been added (see below).
* `ProgramBuilder::binaries` and `Program::with_binary` have been added.
* The new `KernelError` type has been added.

Breaking Changes
----------------
* [`Kernel`][kernel-0.18.0] has been completely redesigned:
  * `Kernel::new` has been removed. Create a kernel with `Kernel::builder` or
    `KernelBuilder::new`.
  * All setup parameters such as the kernel name, program, work sizes, and
    arguments are now configured using the builder. All of the methods
    associated with construction have been removed from `Kernel`.
  * Most `arg_...` and `arg_..._named` methods have been deprecated in favor
    of `::arg` or `arg_named`.
  * Where before, `::arg_scl_named` and `arg_vec_named` accepted `None`
    values, now scalar and vector values set using `::arg_named` are not
    optional. Use zero instead (e.g. `.arg_scl_named::<i32>("rnd", None)` -->
    `arg_named("rnd", 0i32)`).
  * `Kernel` no longer implements `Clone`. Instead, the `KernelBuilder` can be
    cloned or re-used (and sent between threads) to create multiple identical
    or near-identical kernels.
  * `::gwo`, `::gws`, and `::lws` have been deprecated and should be replaced
    by `::global_work_offset`, `::global_work_size`, and `::local_work_size`
    respectively.
  * Most `set_arg_***_named` methods have been deprecated in favor of
    `::set_arg`.
* `KernelCmd::gwo`, `::gws`, and `::lws` have been deprecated and should be
  replaced by `::global_work_offset`, `::global_work_size`, and
  `::local_work_size` respectively.
* `ProQue::builder` now has a lifetime.
* `ProQue::create_kernel` has been removed and replaced with
  `ProQue::kernel_builder` which can be used to return a new `KernelBuilder`
  with pre-configured name, program, default queue, and global work size.
* `Buffer::new` and `Image::new` are now unsafe.
* `BufferBuilder::host_data` and `ImageBuilder::host_data` have been removed.
  Use `::copy_host_slice` or `::use_host_slice` instead.
* `ProgramBuilder` is now non-consuming and has a lifetime.
* `ProgramBuilder::il` now accepts a slice instead of a `Vec`.
* `Program::new` has been renamed to `Program::with_source`. Source strings
  and compiler options are now passed as slices.
* `Program::with_il` now accepts intermediate language byte source and
  compiler options as slices.

* (ocl-core) `KernelArg` has been removed and replaced with `ArgVal`.
  Conversion is relatively straightforward, enum variants map to 'constructor'
  methods. Where before you may have created a `KernelArg` using
  `KernelArg::Mem(&mem)`, you will now create an `ArgVal` with
  `ArgVal::mem(&mem)`. Scalar and vector types must now be specified as
  references: `KernelArg::Scalar(10.0f32)` --> `ArgVal::scalar(&10.0f32)`.
* (ocl-core) `set_kernel_arg` has had its argument type changed to the new
  `ArgVal` type. As `ArgVal` has no associated type, `set_kernel_arg` no
  longer has a type parameter.
  * Example:
    ```
    let kernel = core::create_kernel(&program, "multiply")?;
    core::set_kernel_arg(&kernel, 0, ArgVal::scalar(&10.0f32))?;
    core::set_kernel_arg(&kernel, 1, ArgVal::mem(&buffer))?;
    ```


### Quick Update Instructions

1) Replace uses of `ProQue::create_kernel` with `::kernel_builder` (example
  regex: `ProQue::create_kernel\(([^\)]+)\)` --> `ProQue::kernel_builder(\1)`).
2) Replace uses of `Kernel::new` with `Kernel::builder`, `::name`, and
  `::program` (example regex: `Kernel::new\(([^,]+, [^\)]+)\)` -->
  `Kernel::builder().name(\1).program(\2)`).
3) Add `.build()` to the end of the list of kernel builder parameters.
4) Move error handling (`?`) to the end, after `.build()`.
5) Rename various deprecated `::arg...` methods.

Other things to check:

* If `None` was being passed to `::arg_scl_named` or `::arg_vec_named`,
  replace with a zero (e.g. `.arg_scl_named::<i32>("rnd", None)` -->
  `arg_named("rnd", &0i32)`).
* If you were previously cloning the kernel, you will now instead need to use
  the `KernelBuilder` to create multiple copies. Please note that before,
  clones of kernels would share argument values. This will no longer be the
  case, each copy of a kernel will have independent argument values.
  * If you were relying on shared argument values or if for some other reason
    the new design does not work for you, you can wrap `Kernel` with an
    `Rc<RefCell<_>>` or `Arc<Mutex<_>>`.

[kernel-0.18.0]: https://docs.rs/ocl/0.18.0/ocl/struct.Kernel.html


Version 0.17.0 (2018-02-20)
===========================

* The [ocl-interop] crate has been added to the project. This crate provides
  OpenCL <-> OpenGL interoperability. See the [README][ocl-interop] for more.

Breaking Changes
----------------
* Error handling has been completely revamed and now uses the
  [failure](https://github.com/withoutboats/failure) crate. Breakages are
  unlikely and will only occur if your crate depended on certain, now removed,
  features. If you experience any breakages which are not obvious how to fix,
  please file an issue so that more instruction can be added here.
  * The top level `ocl` crate now has it's own `Error` type, distinct from
    `ocl_core::Error`.
  * `ocl::async::Error` has been removed.
* `Platform::first` has had its `ignore_env_var` argument removed. If you
  previously called `Platform::first(false)` (to respect the
  `OCL_DEFAULT_PLATFORM_IDX` environment variable) you will now want to use
  `Platform::default` instead. If you previously called
  `platform.first(true)` you will now simply use `platform.first()`.

* `Buffer` now uses a linear `usize` rather than a multi-dimensional
  `SpatialDims` to store its size.
  * `Buffer::new` has had its `dims` argument renamed to `len`.
  * `Buffer::dims` has been renamed to `::len`.
  * `BufferBuilder::dims` has been renamed to `::len`.

* Many types have had their `::core` methods renamed to `::as_core`.
* `EventList` and `EventArray` have had their `::push_some` methods removed.
* `RwVec::len` has been renamed to `::len_stale` to clarify that the value
  returned is potentially out of date.
* (ocl-core) The `::scrambled_vec`, `::shuffled_vec`, and `shuffle` functions
  have been moved to the `ocl-extras` crate. `rand` has been removed as a
  dependency.


[ocl-interop]: https://github.com/cogciprocate/ocl/tree/master/ocl-interop


Version 0.16.0 (2017-12-02)
===========================

* The `ocl-core` and `cl-sys` repositories have been merged into the main
  `ocl` repository.
* Various 'future guards' (such as `FutureReadGuard`) now behave more
  gracefully when dropped before being polled.


Breaking Safety Fixes
---------------------
`Kernel::enq` and `KernelCmd::enq` are now unsafe functions. Even though the
API call itself is safe, all kernel code is inherently untrusted and
potentially dangerous. This change is long overdue.

This change will break virtually all existing code, though the fix is trivial:
simply wrap all calls to `.enq()` in an unsafe block.

Before:
```
kernel.enq().unwrap();
```

Now:
```
unsafe { kernel.enq().unwrap(); }
```

Breaking Changes
----------------
* `Kernel::enq` and `KernelCmd::enq` are now marked `unsafe`.
* `BufferMapCmd::enq` and `BufferMapCmd::enq_async` are now marked `unsafe`.
* `Buffer::from_gl_buffer` has had its `dims` argument removed. Its size is
  now determined from the size of the OpenGL memory object.
* `BufferWriteCmd::enq_async` now returns a `FutureReadGuard` instead of a
  `FutureWriteGuard`. A `FutureWriteGuard` can now be obtained by calling
  `BufferWriteCmd::enq_async_then_write`.
* `Buffer::flags` now returns a result.
* The `FutureReadGuard` and `FutureWriteGuard` type aliases have been added.
  * `FutureReader<T>` and `FutureWriter<T>` are equivalent and should be
    translated to `FutureReadGuard<Vec<T>>` and `FutureWriteGuard<Vec<T>>`.
* The `FutureRwGuard` type alias has been removed.
* `FutureMemMap` has had its `::set_wait_list` method renamed to
  `::set_lock_wait_events`.
* `FutureReadGuard`/`FutureWriteGuard`: `::set_wait_list` has likewise been
  renamed to `::set_lock_wait_events` and `::set_command_completion_event` has
  been renamed to `::set_command_wait_event`. Other similar methods have been
  renamed accordingly.
* `FutureMemMap::set_unmap_wait_list` has been renamed to
  `::set_unmap_wait_events` and `::create_unmap_completion_event` has been
  renamed to `::create_unmap_event`. Other similar methods have been renamed
  accordingly.
* (ocl-core) `::enqueue_write_buffer`, `::enqueue_write_buffer_rect`,
  `::enqueue_write_image`, `enqueue_kernel`, and `enqueue_task` are now
  correctly marked `unsafe`.


Version 0.15.0 (2017-08-31)
===========================

* Small changes to `BufferBuilder` and `Kernel` have been made to maintain
  OpenCL 1.1 compatability.
* The [`Platform::first`] method has been added which, unlike
  `Platform::default`, returns an error instead of panicking when no platform
  is available.
* `ContextBuilder::new`, `ProQueBuilder::build`, and some other methods which
  attempt to use the first available platform no longer panic when none is
  available.
* (ocl-core) `enqueue_acquire_gl_buffer` and `enqueue_release_gl_buffer` have
  been deprecated in favor of the new [`enqueue_acquire_gl_objects`] and
  [`enqueue_release_gl_objects`] functions and will be removed in a future
  version.
* (ocl-core)(WIP) The [`get_gl_context_info_khr`] function along with the
  [`GlContextInfo`] and [`GlContextInfoResult`] types have been added but are
  still a work in progress. These can be used to determine which
  OpenCL-accessible device is being used by an existing OpenGL context (or to
  list all associable devices). Please see [documentation notes and
  code][`::get_gl_context_info_khr`] if you can offer any help to get this
  working.

Breaking Changes
----------------
* `Buffer::new` has had the `fill_val` argument removed and continues to be
  unstable. Automatically filling a buffer after creation can be done using
  the appropriate `BufferBuilder` methods
  ([`fill_val`][`BufferBuilder::fill_val`] and
  [`fill_event`][`BufferBuilder::fill_event`], see change below).
* [`BufferBuilder::fill_val`] now only accepts a single argument, the value.
  Setting an associated event may now optionally be done using the new
  [`fill_event`][`BufferBuilder::fill_event`]. Not setting an event continues
  to simply block the current thread until the fill is complete (just after
  buffer creation).
* `ContextBuilder::gl_context` now accepts a `*mut c_void` instead of an
  integer for its argument.
* (ocl-core) Error handling has been changed and allows error chaining.
  Matching against error variants now must be done using the [`ErrorKind`]
  type returned by the [`Error::kind`] method.
* (ocl-core) The `ContextPropertyValue::GlContextKhr` variant is now holds the
  `*mut c_void` type.


[`get_gl_context_info_khr`]: http://docs.cogciprocate.com/ocl_core/ocl_core/fn.get_gl_context_info_khr.html
[`GlContextInfo`]: http://docs.cogciprocate.com/ocl_core/ocl_core/enum.GlContextInfo.html
[`GlContextInfoResult`]: http://docs.cogciprocate.com/ocl_core/ocl_core/types/enums/enum.GlContextInfoResult.html
[`::get_gl_context_info_khr`]: https://github.com/cogciprocate/ocl-core/blob/master/src/functions.rs#L776
[`enqueue_acquire_gl_objects`]: http://doc.cogciprocate.com/ocl_core/ocl_core/fn.enqueue_acquire_gl_objects.html
[`enqueue_release_gl_objects`]: http://doc.cogciprocate.com/ocl_core/ocl_core/fn.enqueue_release_gl_objects.html

[`BufferBuilder::fill_val`]: [http://doc.cogciprocate.com/ocl/ocl/builders/struct.BufferBuilder.html#method.fill_val
[`BufferBuilder::fill_event`]: http://doc.cogciprocate.com/ocl/ocl/builders/struct.BufferBuilder.html#method.fill_event
[`ErrorKind`]: http://doc.cogciprocate.com/ocl_core/ocl_core/error/enum.ErrorKind.html
[`Error::kind`]: http://doc.cogciprocate.com/ocl_core/ocl_core/error/struct.Error.html#method.kind


Version 0.14.0 (2017-05-31)
===========================

* `SpatialDims` now derives `PartialEq` and `Eq`.

Breaking Changes
----------------
* Supported image format returned from various functions such as
  `core::get_supported_image_formats` or `Image::supported_formats` are now
  individually result-wrapped allowing the use of these functions on platforms
  that return unsupported/unknown formats (Apple).


Version 0.13.1 (2017-04-20)
===========================

Bug Fixes
---------
* Fix certain platform-specific issues.
* Remove a `println` when building a `ProQue` (oops).


Version 0.13.0 (2017-04-11)
===========================

The futures have arrived! The [`futures`] crate has begun to find its way into
ocl. This makes doing things like embedding/inserting host processing work
into the sequence of enqueued commands easy and intuitive. See the new
[`MemMap`], [`RwVec`], [`FutureMemMap`], and [`FutureRwGuard`] types.

We will be approaching stabilization over the next year for all top level
types. Before that point can be reached, we'll have to break a few eggs. This
release brings consistency and simplification changes to a few important
functions, notably [`Buffer::new`] and [`Kernel::new`]. See the breaking
changes section below for details.

* Asynchrony and Futures:
  * Buffers can now be mapped (and unmapped) safely both synchronously
    (thread-blocking) and asynchronously (using futures) using
    [`Buffer::map`].
  * Calling `::read`, `::write`, or `::map` on a [`Buffer`], [`Image`],
    [`BufferCmd`] or [`ImageCmd`] will now return a specialized
    [`BufferReadCmd`], [`BufferWriteCmd`], or [`BufferMapCmd`] command
    builder. These three new special command builders provide an `::enq_async`
    method in addition to the usual `::enq` method. For these I/O commands
    only, `::enq` will now always block the current thread until completion.
    `::enq_async` will instead return a future representing the completion of
    the command and will also allow safe access to buffer data after the
    command has completed. This means that host-side processing can be easily
    inserted into the stream of OpenCL commands seamlessly using
    [futures][futures-doc] interfaces.
* Sub-buffers can now be safely created from a [`Buffer`] using
  [`Buffer::create_sub_buffer`].
* Default queues on kernels, buffers, and images are no longer mandatory. See
  the breaking changes section below for details on how this impacts existing
  code.
* Command queue properties can now be specified when creating a [`Queue`] or
  [`ProQue`] allowing out of order execution and profiling to be enabled.
  Profiling had previously been enabled by default but now must be explicitly
  enabled by setting the [`QUEUE_PROFILING_ENABLE`] flag.
* [`EventList`] has undergone tweaking and is now a 'smart' list, being stack
  allocated by default and heap allocated when its length exceeds 8 events.
* [`EventArray`], a stack allocated array of events, has been added.
  [`EventArray`] is automatically used internally by [`EventList`] when
  necessary but can also be used on its own.
* When setting a kernel argument, the type associated with the argument is now
  checked against the type specified in the kernel's source code. This check
  can be opted out of by using the new [`Kernel::set_arg_unchecked`] function
  described below.
* [`Kernel::set_arg_unchecked`] and [`Kernel::named_arg_idx`] have been added
  allowing the ability to set a kernel argument by index and retrieve the
  index of a named argument. Argument indexes always correspond exactly to the
  order arguments are declared within the program source code for a kernel.
* [`Kernel`] buffer and image related functions (such as `arg_buf`) can now
  interchangeably accept either [`Buffer`] or [`Image`].
* [`BufferCmd`], [`ImageCmd`], [`KernelCmd`], et al. have received
  streamlining and optimizations with regards to events.
* Complete rename, redesign, and macro-based reimplementation of all vector
  types. Vector types now implement all of the same operations as scalar types
  and use wrapping arithmetic (see breaking changes for more). There is also
  now a scalar version for each type using all of the same (wrapping)
  operations and naming conventions (ex.: Double, Int, Uchar). Future
  optimization and/or interaction with the [`ndarray`] crate may be added if
  requested (file an issue and let us hear your thoughts).

Breaking Changes
----------------
* [`Buffer::new`] continues to be unstable and is not recommended for use
  directly. Instead use the new [`BufferBuilder`] by calling
  [`Buffer::builder`] or [`BufferBuilder::new`].
  * Before:

    ```Buffer::new(queue, Some(flags), dims, Some(&data)) ```

  * Now:

    ```
    Buffer::builder()
      .queue(queue)
      .flags(flags)
      .dims(dims)
      .host_data(&data)
      .build()
    ```

* [`Kernel::new`] no longer accepts a queue as a third argument (and can now
  be considered stabilized). Instead use the [`::queue`][kernel_queue]
  (builder-style) or [`::set_default_queue`][kernel_set_default_queue]
  methods. For example:
  * Before:

    ```Kernel::new("kernel_name", &program, queue)?```

  * Now:

    ```Kernel::new("kernel_name", &program)?.queue(queue)```

* [`BufferCmd`], [`ImageCmd`], and [`KernelCmd`] have undergone changes:
  * `::copy` signature change `offset` and `len` (size) are now optional.
    Offset will default to zero and length will default to the entire length
    of the buffer.
  * `::enew`, `::enew_opt`, `::ewait`,  and `::ewait_opt` for command builders
    have had a signature change and now use generics. This may affect certain
    types of casts.
* `Buffer::is_empty` has been removed.
* `Buffer::from_gl_buffer`, `Buffer::set_default_queue`,
  `ImageBuilder::build`, `ImageBuilder::build_with_data`, `Image::new`,
  `Image::from_gl_texture`, `Image::from_gl_renderbuffer`,
  `Image::set_default_queue`, `Kernel::set_default_queue` now take an owned
  [`Queue`] instead of a `&Queue` (clone it yourself).
* All row pitch and slice pitch arguments (for image and rectangular buffer
  enqueue operations) must now be expressed in bytes.
* [`Kernel::set_default_queue`] no longer result-wraps its `&'mut Kernel` return
  value.
* [`Kernel`] named argument declaration functions such as `::arg_buf_named` or
  `::set_arg_img_named` called with a `None` variant must now specify the full
  type of the image, buffer, or sub-buffer which will be used for that
  argument. Where before you might have used:

    ```.arg_buf_named::<f32>("buf", None)```

  you must now use:

    ```.arg_buf_named("buf", None::<Buffer<f32>>)```

  or:

    ```.arg_buf_named::<f32, Buffer<f32>>("buf", None)```

* [`Queue::new`] now takes a third argument, `properties` (details below in
  ocl-core section).
* [`Queue::finish`] now returns a result instead of unwrapping.
* [`Program::new`] has had its arguments rearranged for consistency.
* `Event::wait` and `EventList::wait` have both been renamed to
  `::wait_for` to avoid conflicts with the [`Future`] trait. The new futures
  versions of `::wait` do exactly the same thing however.
* `::core_as_ref` and `::core_as_mut` for several types have been renamed to
  `::core` and `::core_mut`.
* [Vector types] have been redesigned and moved into their own sub-crate:
  * The `Cl` prefix for each type has been removed.
  * Tuple struct notation for creation has been removed. Use `::new` or
    `::from`.
  * All arithmetic operations are fully implemented and are wrapping.
  * Bitwise and shift operators have been added for integer types.
  * Now located within the [`ocl::prm`] module.

### Breaking changes specific to [`ocl-core`]
* Passing event wait list and new event references has been completely
  overhauled. Previously, references of this type had to be converted into the
  trait objects `ClWaitList` and `ClEventPtrNew`. This was convenient
  (outside of the occasional awkward conversion) and obviated the need to type
  annotate every time you passed `None` to an `enqueue_...` function. Now that
  futures have come to the library though, every last ounce of performance
  must be wrung out of event processing. This means that events and event
  lists are now treated as normal generics, not trait objects, and are
  therefore optimally efficient, at the cost of it being less convenient to
  call functions when you are *not* using them.
  * The `ClWaitList` trait has been renamed to [`ClWaitListPtr`]
  * The `ClEventPtrNew` trait has been renamed to [`ClNullEventPtr`]
  * All `core::enqueue_...` functions (and a few others) may now require
    additional type annotations when `None` is passed as either the event wait
    list or new event reference. Passing a `None::<Event>` will suffice
    for either parameter (because it can serve either role) and is the easiest
    way to shut the compiler up. If you were previously annotating a type
    using the 'turbo-fish' syntax ('::<...>') on one of these functions, you
    will also have to include two additional type annotations. It may be more
    convenient to do all the annotation in one spot in that case and remove it
    from the `None`s.
* All row pitch and slice pitch arguments (for image and rectangular buffer
  operations) must now be expressed in bytes.
* [`EventList::pop`] now returns an `Option<Event>` instead of an
  `Option<Result<Event>>`.
* [`::create_command_queue`] now takes a third argument: `properties`, an
  optional bitfield described in the [clCreateCommandQueue SDK
  Documentation]. Valid options include
  [`QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`] and [`QUEUE_PROFILING_ENABLE`].
* [`::build_program`] the `devices` argument is now optional. Not specifying any
  devices will cause the program to be built for all available devices within
  the provided context.
* [`::enqueue_map_buffer`], [`::enqueue_map_image`], and
  [`::enqueue_unmap_mem_object`] have had signature changes.

FIXME: Change doc link root to `https://docs.rs/ocl/0.13/` at release.

[`MemMap`]: http://docs.cogciprocate.com/ocl/ocl/struct.MemMap.html
[`RwVec`]: http://docs.cogciprocate.com/ocl/ocl/struct.RwVec.html
[`FutureMemMap`]: http://docs.cogciprocate.com/ocl/ocl/struct.FutureMemMap.html
[`FutureRwGuard`]: http://docs.cogciprocate.com/ocl/ocl/struct.FutureRwGuard.html
[`Buffer`]: http://docs.cogciprocate.com/ocl/ocl/struct.Buffer.html
[`Buffer::new`]: http://docs.cogciprocate.com/ocl/ocl/struct.Buffer.html#method.new
[`Buffer::map`]: http://docs.cogciprocate.com/ocl/ocl/struct.Buffer.html#method.map
[`Buffer::create_sub_buffer`]: http://docs.cogciprocate.com/ocl/ocl/struct.Buffer.html#method.create_sub_buffer
[`Buffer::builder`]: http://docs.cogciprocate.com/ocl/ocl/struct.Buffer.html#method.builder
[`BufferCmd`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.BufferCmd.html
[`BufferReadCmd`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.BufferReadCmd.html
[`BufferWriteCmd`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.BufferWriteCmd.html
[`BufferMapCmd`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.BufferMapCmd.html
[`BufferBuilder`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.BufferBuilder.html
[`BufferBuilder::new`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.BufferBuilder.html#method.new
[`Kernel`]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html
[`Kernel::new`]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.new
[`Kernel::set_arg_unchecked`]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.set_arg_unchecked
[`Kernel::named_arg_idx`]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.named_arg_idx
[`Kernel::set_default_queue`]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.set_default_queue
[kernel_queue]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.queue
[kernel_set_default_queue]: http://docs.cogciprocate.com/ocl/ocl/struct.Kernel.html#method.set_default_queue
[`KernelCmd`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.KernelCmd.html
[`Image`]: http://docs.cogciprocate.com/ocl/ocl/struct.Image.html
[`ImageCmd`]: http://docs.cogciprocate.com/ocl/ocl/builders/struct.ImageCmd.html
[`Queue`]: http://docs.cogciprocate.com/ocl/ocl/struct.Queue.html
[`Queue::new`]: http://docs.cogciprocate.com/ocl/ocl/struct.Queue.html#method.new
[`Queue::finish`]: http://docs.cogciprocate.com/ocl/ocl/struct.Queue.html#method.finish
[`ProQue`]: http://docs.cogciprocate.com/ocl/ocl/struct.ProQue.html
[`QUEUE_PROFILING_ENABLE`]: http://docs.cogciprocate.com/ocl/ocl/flags/constant.QUEUE_PROFILING_ENABLE.html
[`QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`]: http://docs.cogciprocate.com/ocl/ocl/flags/constant.QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE.html
[`EventList`]: http://docs.cogciprocate.com/ocl/ocl/struct.EventList.html
[`EventList::pop`]: http://docs.cogciprocate.com/ocl/ocl/struct.EventList.html#method.pop
[`EventArray`]: http://docs.cogciprocate.com/ocl/ocl/struct.EventArray.html
[`Program::new`]: http://docs.cogciprocate.com/ocl/ocl/struct.Program.html#method.new
[`ocl-core`]: https://github.com/cogciprocate/ocl-core
[`ClWaitListPtr`]: http://docs.cogciprocate.com/ocl/ocl_core/types/abs/trait.ClWaitListPtr.html
[`ClNullEventPtr`]: docs.cogciprocate.com/ocl/ocl_core/types/abs/trait.ClNullEventPtr.html
[`::create_command_queue`]: http://docs.cogciprocate.com/ocl/ocl_core/fn.create_command_queue.html
[`::build_program`]: http://docs.cogciprocate.com/ocl/ocl_core/fn.build_program.html
[`::enqueue_map_buffer`]: http://docs.cogciprocate.com/ocl/ocl_core/fn.enqueue_map_buffer.html
[`::enqueue_map_image`]: http://docs.cogciprocate.com/ocl/ocl_core/fn.enqueue_map_image.html
[`::enqueue_unmap_mem_object`]: http://docs.cogciprocate.com/ocl/ocl_core/fn.enqueue_unmap_mem_object.html
[`futures`]: https://github.com/alexcrichton/futures-rs
[futures-doc]: https://docs.rs/futures
[`Futures`]: https://docs.rs/futures/0.1
[future_trait]: https://docs.rs/futures/0.1.13/futures/future/trait.Future.html
[`ndarray`]: https://github.com/bluss/rust-ndarray
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

The `core` and `ffi` modules have been moved out into new crates,
[ocl-core][0.11.0.ocl-core] and [cl-sys][0.11.0.cl-sys] respectively. They
continue to be exported with the same names. This document will continue to
contain information pertaining to all three libraries ([ocl],
[ocl-core][0.11.0.ocl-core], and [cl-sys][0.11.0.cl-sys]). Issues will
likewise be centrally handled from the [ocl] repo page.

The version control system has been implemented. Functions added since OpenCL
1.1 now take an additional argument to ensure that the device or platform
supports that feature. See the [ocl-core crate documentation] for more
information.

Breaking Changes
----------------
* `Context::platform` now returns a `Option<&Platform>` instead of
  `Option<Platform>` to make it consistent with other methods of its kind.
* (ocl-core) `DeviceSpecifier::to_device_list` now accepts a
  `Option<&Platform>`.
* (ocl-core) Certain functions now require an additional argument for version
  control purposes (see the [ocl-core crate documentation]).
* [cl-sys] Types/functions/constants from the `cl_h` module are now
  re-exported from the root crate. `cl_h` is now private.
* [cl-sys] Functions from the OpenCL 2.0 & 2.1 specifications have been added.
  This may cause problems on older devices. Please file an [issue] if it does.


[ocl-core crate documentation]: http://docs.cogciprocate.com/ocl_core/ocl_core
[ocl]: https://github.com/cogciprocate/ocl
[0.11.0.ocl-core]: https://github.com/cogciprocate/ocl-core
[0.11.0.cl-sys]: https://github.com/cogciprocate/cl-sys
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
