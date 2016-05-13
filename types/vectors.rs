//! OpenCL Vector Types
//!
//! [UNSTABLE]: These MAY be integrated / replaced with whatever becomes the
//! official rust SIMD library. As such, additional methods such as .xyzw,
//! .s0, .s1, .lo, etc. are not currently planned for addition.
//!
//! These types will have the same memory layout no matter which `repr` is
//! used. The default (`repr(rust)`) will be used until SIMD support is
//! stable and its use can be evaluated.
//!
//! If anyone wants to implement `Index` on all of these types, please feel
//! free to do so. Same goes for .x, .y, .s0, .sA, .lo, etc. 

#![allow(non_camel_case_types)]

#[derive(Debug, Clone, Copy)]
pub struct cl_char2(pub i8, pub i8);

#[derive(Debug, Clone, Copy)]
pub struct cl_char3(i8, i8, i8, i8);

#[derive(Debug, Clone, Copy)]
pub struct cl_char4(i8, i8, i8, i8);

#[derive(Debug, Clone, Copy)]
pub struct cl_char8(i8, i8, i8, i8, i8, i8, i8, i8);

#[derive(Debug, Clone, Copy)]
pub struct cl_char16(i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8);


#[derive(Debug, Clone, Copy)]
pub struct cl_uchar2(u8, u8);

#[derive(Debug, Clone, Copy)]
pub struct cl_uchar3(u8, u8, u8, u8);

#[derive(Debug, Clone, Copy)]
pub struct cl_uchar4(u8, u8, u8, u8);

#[derive(Debug, Clone, Copy)]
pub struct cl_uchar8(u8, u8, u8, u8, u8, u8, u8, u8);

#[derive(Debug, Clone, Copy)]
pub struct cl_uchar16(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);


#[derive(Debug, Clone, Copy)]
pub struct cl_short2(i16, i16);

#[derive(Debug, Clone, Copy)]
pub struct cl_short3(i16, i16, i16, i16);

#[derive(Debug, Clone, Copy)]
pub struct cl_short4(i16, i16, i16, i16);

#[derive(Debug, Clone, Copy)]
pub struct cl_short8(i16, i16, i16, i16, i16, i16, i16, i16);

#[derive(Debug, Clone, Copy)]
pub struct cl_short16(i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16, i16);


#[derive(Debug, Clone, Copy)]
pub struct cl_ushort2(u16, u16);

#[derive(Debug, Clone, Copy)]
pub struct cl_ushort3(u16, u16, u16, u16);

#[derive(Debug, Clone, Copy)]
pub struct cl_ushort4(u16, u16, u16, u16);

#[derive(Debug, Clone, Copy)]
pub struct cl_ushort8(u16, u16, u16, u16, u16, u16, u16, u16);

#[derive(Debug, Clone, Copy)]
pub struct cl_ushort16(u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16, u16);


#[derive(Debug, Clone, Copy)]
pub struct cl_int2(i32, i32);

#[derive(Debug, Clone, Copy)]
pub struct cl_int3(i32, i32, i32, i32);

#[derive(Debug, Clone, Copy)]
pub struct cl_int4(i32, i32, i32, i32);

#[derive(Debug, Clone, Copy)]
pub struct cl_int8(i32, i32, i32, i32, i32, i32, i32, i32);

#[derive(Debug, Clone, Copy)]
pub struct cl_int16(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32);


#[derive(Debug, Clone, Copy)]
pub struct cl_uint2(u32, u32);

#[derive(Debug, Clone, Copy)]
pub struct cl_uint3(u32, u32, u32, u32);

#[derive(Debug, Clone, Copy)]
pub struct cl_uint4(u32, u32, u32, u32);

#[derive(Debug, Clone, Copy)]
pub struct cl_uint8(u32, u32, u32, u32, u32, u32, u32, u32);

#[derive(Debug, Clone, Copy)]
pub struct cl_uint16(u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32);


#[derive(Debug, Clone, Copy)]
pub struct cl_long1(i64);

#[derive(Debug, Clone, Copy)]
pub struct cl_long2(i64, i64);

#[derive(Debug, Clone, Copy)]
pub struct cl_long3(i64, i64, i64, i64);

#[derive(Debug, Clone, Copy)]
pub struct cl_long4(i64, i64, i64, i64);

#[derive(Debug, Clone, Copy)]
pub struct cl_long8(i64, i64, i64, i64, i64, i64, i64, i64);

#[derive(Debug, Clone, Copy)]
pub struct cl_long16(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64);


#[derive(Debug, Clone, Copy)]
pub struct cl_ulong1(u64);

#[derive(Debug, Clone, Copy)]
pub struct cl_ulong2(u64, u64);

#[derive(Debug, Clone, Copy)]
pub struct cl_ulong3(u64, u64, u64, u64);

#[derive(Debug, Clone, Copy)]
pub struct cl_ulong4(u64, u64, u64, u64);

#[derive(Debug, Clone, Copy)]
pub struct cl_ulong8(u64, u64, u64, u64, u64, u64, u64, u64);

#[derive(Debug, Clone, Copy)]
pub struct cl_ulong16(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);


#[derive(Debug, Clone, Copy)]
pub struct cl_float2(f32, f32);

#[derive(Debug, Clone, Copy)]
pub struct cl_float3(f32, f32, f32, f32);

#[derive(Debug, Clone, Copy)]
pub struct cl_float4(f32, f32, f32, f32);

#[derive(Debug, Clone, Copy)]
pub struct cl_float8(f32, f32, f32, f32, f32, f32, f32, f32);

#[derive(Debug, Clone, Copy)]
pub struct cl_float16(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32);


#[derive(Debug, Clone, Copy)]
pub struct cl_double2(f64, f64);

#[derive(Debug, Clone, Copy)]
pub struct cl_double3(f64, f64, f64, f64);

#[derive(Debug, Clone, Copy)]
pub struct cl_double4(f64, f64, f64, f64);

#[derive(Debug, Clone, Copy)]
pub struct cl_double8(f64, f64, f64, f64, f64, f64, f64, f64);

#[derive(Debug, Clone, Copy)]
pub struct cl_double16(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);


// VECTOR TYPE COMPLETE LIST: (taken from cl_platform.h v1.2):

// typedef vector unsigned char     __cl_uchar16;
// typedef vector signed char       __cl_char16;
// typedef vector unsigned short    __cl_ushort8;
// typedef vector signed short      __cl_short8;
// typedef vector unsigned int      __cl_uint4;
// typedef vector signed int        __cl_int4;
// typedef vector float             __cl_float4;

// typedef float __cl_float4   __attribute__((vector_size(16)));

// typedef cl_uchar    __cl_uchar16    __attribute__((vector_size(16)));
// typedef cl_char     __cl_char16     __attribute__((vector_size(16)));
// typedef cl_ushort   __cl_ushort8    __attribute__((vector_size(16)));
// typedef cl_short    __cl_short8     __attribute__((vector_size(16)));
// typedef cl_uint     __cl_uint4      __attribute__((vector_size(16)));
// typedef cl_int      __cl_int4       __attribute__((vector_size(16)));
// typedef cl_ulong    __cl_ulong2     __attribute__((vector_size(16)));
// typedef cl_long     __cl_long2      __attribute__((vector_size(16)));
// typedef cl_double   __cl_double2    __attribute__((vector_size(16)));

// typedef cl_uchar    __cl_uchar8     __attribute__((vector_size(8)));
// typedef cl_char     __cl_char8      __attribute__((vector_size(8)));
// typedef cl_ushort   __cl_ushort4    __attribute__((vector_size(8)));
// typedef cl_short    __cl_short4     __attribute__((vector_size(8)));
// typedef cl_uint     __cl_uint2      __attribute__((vector_size(8)));
// typedef cl_int      __cl_int2       __attribute__((vector_size(8)));
// typedef cl_ulong    __cl_ulong1     __attribute__((vector_size(8)));
// typedef cl_long     __cl_long1      __attribute__((vector_size(8)));
// typedef cl_float    __cl_float2     __attribute__((vector_size(8)));

// typedef cl_float    __cl_float8     __attribute__((vector_size(32)));
// typedef cl_double   __cl_double4    __attribute__((vector_size(32)));

// cl_char2
// cl_char4
// typedef  cl_char4  cl_char3;
// cl_char8
// cl_char16

// cl_uchar2
// cl_uchar4
// typedef  cl_uchar4  cl_uchar3;
// cl_uchar8
// cl_uchar16

// cl_short2
// cl_short4
// typedef  cl_short4  cl_short3;
// cl_short8
// cl_short16

// cl_ushort2
// cl_ushort4
// typedef  cl_ushort4  cl_ushort3;
// cl_ushort8
// cl_ushort16

// cl_int2
// cl_int4
// typedef  cl_int4  cl_int3;
// cl_int8
// cl_int16

// cl_uint2
// cl_uint4
// typedef  cl_uint4  cl_uint3;
// cl_uint8
// cl_uint16

// cl_long2
// cl_long4
// typedef  cl_long4  cl_long3;
// cl_long8
// cl_long16

// cl_ulong2
// cl_ulong4
// typedef  cl_ulong4  cl_ulong3;
// cl_ulong8
// cl_ulong16

// cl_float2
// cl_float4
// typedef  cl_float4  cl_float3;
// cl_float8
// cl_float16

// cl_double2
// cl_double4
// typedef  cl_double4  cl_double3;
// cl_double8
// cl_double16