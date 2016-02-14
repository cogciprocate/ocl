//! Custom enumerators not specifically based on OpenCL C-style enums.
use libc::{size_t, c_void};
use raw::{MemRaw, SamplerRaw};

/// Kernel argument option type.
///
/// The type argument `T` is ignored for `Mem`, `Sampler`, and `Other` 
/// (just put `usize` or anything).
///
pub enum KernelArg<'a, T: 'a> {
    /// Type `T` is ignored.
    Mem(MemRaw),
    /// Type `T` is ignored.
    Sampler(SamplerRaw),
    Scalar(&'a T),
    Vector(&'a [T]),
    /// Length in multiples of T (not bytes).
    Local(usize),
    /// `size`: size in bytes. Type `T` is ignored.
    Other { size: size_t, value: *const c_void },
}
