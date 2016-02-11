Version 0.5.0 (2016-02-09)
==========================

Breaking Changes
----------------
* `Buffer` has had several methods dealing with reading and writing renamed
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
* `ProQue::build` has been renamed [`ProQue::build_program`].
* The [`Buffer::with_vec_shuffled`] and [`Buffer::with_vec_scrambled`] methods 
  now accept a 2-tuple as the first argument instead of two separate values for 
  the first two arguments.
* `BuildOptions` has been renamed to [`ProgramBuilder`] and has been 
  redesigned:
   * A new `ProgramBuilder` can be created with `ProgramBuilder::new` or 
     `Program::builder`.
   * The `::build` method has been added, consuming the builder and returning
     a new [`Program`].
   * Methods dealing with kernel source code renamed from `kern` to `src` 
     for clarity and simplicity.
   * Extraneous methods have been removed.
* The following methods on `Kernel` have been renamed reflecting `Envoy` having 
  been recently renamed to `Buffer' in v0.4.0:
  * `::arg_env` to `::arg_buf`
  * `::arg_env_named` to `::arg_buf_named`
  * `::set_arg_env_named` to `::set_arg_buf_named`
* `SimpleDims` has had its variants renamed.


New Types
---------
* [`ProQueBuilder`] is now the most boilerplate-free way to create an OpenCL
  context, program, and queue. Create one by calling ['ProQue::builder'].
  See [`basics.rs` for an example][0.5ba] and [documentation][0.5doc] for more info.


[0.5doc]: http://doc.cogciprocate.com/ocl/
[0.5ba]: https://github.com/cogciprocate/ocl/blob/master/examples/basics.rs
[`ProQue::build_program`]: http://doc.cogciprocate.com/ocl/
[`Buffer::with_vec_shuffled`]: http://doc.cogciprocate.com/ocl/
[`Buffer::with_vec_scrambled`]: http://doc.cogciprocate.com/ocl/
[`ProQueBuilder`]: http://doc.cogciprocate.com/ocl/
[`ProQue::builder`]: http://doc.cogciprocate.com/ocl/
[`ProgramBuilder`]: http://doc.cogciprocate.com/ocl/
[`Program`]: http://doc.cogciprocate.com/ocl/
