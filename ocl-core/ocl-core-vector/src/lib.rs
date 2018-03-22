//! OpenCL scalar and vector primitive types.
//!
//! Primitives may have subtly different behaviour within Rust as they do
//! within kernels. Wrapping is one example of this. Scalar integers
//! within Rust may do overflow checks where in the kernel they do not.
//! Therefore two slightly different implementations of the scalar types
//! are provided in addition to a corresponding vector type for each.
//!
//! The `cl_...` (`cl_uchar`, `cl_int`, `cl_float`, etc.) types found in the
//! main `ocl-core` library are simple aliases of the Rust built-in primitive
//! types and therefore always behave exactly the same way. The
//! uppercase-named types (`Uchar`, `Int`, `Float`, etc.) are designed to
//! behave identically to their corresponding types within kernels.
//!
//! Please file an issue if any of the uppercase-named kernel-mimicking
//! types deviate from what they should (as they are reasonably new this
//! is definitely something to watch out for).
//!
//! Vector type fields can be accessed using index operations i.e. [0],
//! [1], [2] ... etc. Plans for other ways of accessing fields (such as
//! `.x()`, `.y()`, `.s0()`, `.s15()`, etc.) may be considered. Create an
//! issue if you have an opinion on the matter.
//!
//! [NOTE]: This module may be renamed.

extern crate num_traits;

mod vectors;

pub use self::vectors::{
    Char, Char2, Char3, Char4, Char8, Char16,
    Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
    Short, Short2, Short3, Short4, Short8, Short16,
    Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
    Int, Int2, Int3, Int4, Int8, Int16,
    Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
    Long, Long2, Long3, Long4, Long8, Long16,
    Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
    Float, Float2, Float3, Float4, Float8, Float16,
    Double, Double2, Double3, Double4, Double8, Double16,
};