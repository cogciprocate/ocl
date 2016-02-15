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


[0.5doc]: http://doc.cogciprocate.com/ocl/
[0.5ba]: https://github.com/cogciprocate/ocl/blob/master/examples/basics.rs
[`Buffer`]: http://docs.cogciprocate.com/ocl/struct.Buffer.html
[`Image`]: http://docs.cogciprocate.com/ocl/struct.Image.html
[`raw`]: http://docs.cogciprocate.com/ocl/raw/index.html
[`ProQueBuilder`]: http://docs.cogciprocate.com/ocl/struct.ProQueBuilder.html
[`ProQue`]: http://docs.cogciprocate.com/ocl/struct.ProQue.html
['ProQue::builder']: http://docs.cogciprocate.com/ocl/struct.ProQue.html#method.builder
[`ProgramBuilder`]: http://docs.cogciprocate.com/ocl/struct.ProgramBuilder.html
[`Program`]: http://docs.cogciprocate.com/ocl/struct.Program.html
[`Kernel`]: http://docs.cogciprocate.com/ocl/struct.Kernel.html
[`SimpleDims`]: http://doc.cogciprocate.com/ocl/enum.SimpleDims.html
[`WorkDims`]: http://doc.cogciprocate.com/ocl/enum.WorkDims.html
[`DeviceType`]: http://doc.cogciprocate.com/ocl/raw/struct.DeviceType.html

<br/><br/>
*“OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.”*
