Version 0.5.0 (2016-02-08)
==========================

Breaking Changes
----------------
- `ProQue::build` has been renamed [`ProQue::build_program`].
- The [`Buffer::with_vec_shuffled`] and [`Buffer::with_vec_scrambled`] methods 
  now accept a 2-tuple as the first argument instead of separate values as the 
  first two arguments.
- `BuildOptions` has been renamed to [`ProgramBuilder`] and has been 
  redesigned:
   - A new `ProgramBuilder` can be created with `ProgramBuilder::new()` or 
     `Program::builder()`.
   - The `::build` method has been added, consuming the builder and returning
     a new `Program`.
   - Methods dealing with kernel source code renamed from `kern` to `src` 
     for clarity and simplicity.

Builders
--------
- ProQue now has a builder type: [`ProQueBuilder`] created by calling 
  [`ProQue::builder()`].


[`ProQue::build_program`]: 
[`Buffer::with_vec_shuffled`]:
[`Buffer::with_vec_scrambled`]:
[`ProQueBuilder`]:
[`ProQue::builder()`]:
[`ProgramBuilder`]:
