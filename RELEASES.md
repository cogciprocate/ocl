Version 0.5.0 (2016-02-08)
==========================

Breaking Changes
----------------
- `ProQue::build` has been renamed [`ProQue::build_program`].
- The [`Buffer::with_vec_shuffled`] and [`Buffer::with_vec_scrambled`] methods 
  now accept a 2-tuple as the first argument instead of two separate values for 
  the first two arguments.
- `BuildOptions` has been renamed to [`ProgramBuilder`] and has been 
  redesigned:
   - A new `ProgramBuilder` can be created with `ProgramBuilder::new` or 
     `Program::builder`.
   - The `::build` method has been added, consuming the builder and returning
     a new [`Program`].
   - Methods dealing with kernel source code renamed from `kern` to `src` 
     for clarity and simplicity.

New Types
---------
- [`ProQueBuilder`] is now the most boilerplate-free way to create an OpenCL
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
