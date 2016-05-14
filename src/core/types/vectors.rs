//! OpenCL Vector Types
//!
//! These types will have the same memory layout no matter which `repr` is
//! used. The default (`repr(rust)`) will be used until SIMD support is stable
//! and its use can be evaluated (shouldn't matter at all though). If using
//! any of these types *within* another struct, memory alignment must be
//! managed manually. If using within a `Vec` (typical usage, i.e.:
//! `Vec<cl_float4>`) you don't need to worry about it.
//!
//! [TODO]: Create a macro to implement `Index`, mathematical operations such
//! as `Add`, `Mul`, etc., and whatever else.
//!
//! Not sure about swizzling interfaces or if they'll ever be realistic.
//! Obviously for now just use .0, .1, .2, etc.
//!
//!

#![allow(non_camel_case_types)]

use core::{OclPrm, OclVec};
use std::ops::{Add};

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_char2(pub i8, pub i8);
unsafe impl OclPrm for cl_char2 {}
unsafe impl OclVec for cl_char2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_char3(pub i8, pub i8, pub i8, pub i8);
// unsafe impl OclPrm for cl_char3 {}
// unsafe impl OclVec for cl_char3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_char4(pub i8, pub i8, pub i8, pub i8);
unsafe impl OclPrm for cl_char4 {}
unsafe impl OclVec for cl_char4 {}

pub type cl_char3 = cl_char4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_char8(pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8);
unsafe impl OclPrm for cl_char8 {}
unsafe impl OclVec for cl_char8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_char16(pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, 
	pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8);
unsafe impl OclPrm for cl_char16 {}
unsafe impl OclVec for cl_char16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_uchar2(pub u8, pub u8);
unsafe impl OclPrm for cl_uchar2 {}
unsafe impl OclVec for cl_uchar2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_uchar3(pub u8, pub u8, pub u8, pub u8);
// unsafe impl OclPrm for cl_uchar3 {}
// unsafe impl OclVec for cl_uchar3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_uchar4(pub u8, pub u8, pub u8, pub u8);
unsafe impl OclPrm for cl_uchar4 {}
unsafe impl OclVec for cl_uchar4 {}

pub type cl_uchar3 = cl_uchar4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_uchar8(pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8);
unsafe impl OclPrm for cl_uchar8 {}
unsafe impl OclVec for cl_uchar8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_uchar16(pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, 
	pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8);
unsafe impl OclPrm for cl_uchar16 {}
unsafe impl OclVec for cl_uchar16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_short2(pub i16, pub i16);
unsafe impl OclPrm for cl_short2 {}
unsafe impl OclVec for cl_short2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_short3(pub i16, pub i16, pub i16, pub i16);
// unsafe impl OclPrm for cl_short3 {}
// unsafe impl OclVec for cl_short3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_short4(pub i16, pub i16, pub i16, pub i16);
unsafe impl OclPrm for cl_short4 {}
unsafe impl OclVec for cl_short4 {}

pub type cl_short3 = cl_short4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_short8(pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16);
unsafe impl OclPrm for cl_short8 {}
unsafe impl OclVec for cl_short8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_short16(pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, 
	pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16);
unsafe impl OclPrm for cl_short16 {}
unsafe impl OclVec for cl_short16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ushort2(pub u16, pub u16);
unsafe impl OclPrm for cl_ushort2 {}
unsafe impl OclVec for cl_ushort2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_ushort3(pub u16, pub u16, pub u16, pub u16);
// unsafe impl OclPrm for cl_ushort3 {}
// unsafe impl OclVec for cl_ushort3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ushort4(pub u16, pub u16, pub u16, pub u16);
unsafe impl OclPrm for cl_ushort4 {}
unsafe impl OclVec for cl_ushort4 {}

pub type cl_ushort3 = cl_ushort4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ushort8(pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16);
unsafe impl OclPrm for cl_ushort8 {}
unsafe impl OclVec for cl_ushort8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ushort16(pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, 
	pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16);
unsafe impl OclPrm for cl_ushort16 {}
unsafe impl OclVec for cl_ushort16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_int2(pub i32, pub i32);
unsafe impl OclPrm for cl_int2 {}
unsafe impl OclVec for cl_int2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_int3(pub i32, pub i32, pub i32, pub i32);
// unsafe impl OclPrm for cl_int3 {}
// unsafe impl OclVec for cl_int3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_int4(pub i32, pub i32, pub i32, pub i32);
unsafe impl OclPrm for cl_int4 {}
unsafe impl OclVec for cl_int4 {}

pub type cl_int3 = cl_int4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_int8(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);
unsafe impl OclPrm for cl_int8 {}
unsafe impl OclVec for cl_int8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_int16(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, 
	pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);
unsafe impl OclPrm for cl_int16 {}
unsafe impl OclVec for cl_int16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_uint2(pub u32, pub u32);
unsafe impl OclPrm for cl_uint2 {}
unsafe impl OclVec for cl_uint2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_uint3(pub u32, pub u32, pub u32, pub u32);
// unsafe impl OclPrm for cl_uint3 {}
// unsafe impl OclVec for cl_uint3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_uint4(pub u32, pub u32, pub u32, pub u32);
unsafe impl OclPrm for cl_uint4 {}
unsafe impl OclVec for cl_uint4 {}

pub type cl_uint3 = cl_uint4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_uint8(pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32);
unsafe impl OclPrm for cl_uint8 {}
unsafe impl OclVec for cl_uint8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_uint16(pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, 
	pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32);
unsafe impl OclPrm for cl_uint16 {}
unsafe impl OclVec for cl_uint16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_long1(pub i64);
unsafe impl OclPrm for cl_long1 {}
unsafe impl OclVec for cl_long1 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_long2(pub i64, pub i64);
unsafe impl OclPrm for cl_long2 {}
unsafe impl OclVec for cl_long2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_long3(pub i64, pub i64, pub i64, pub i64);
// unsafe impl OclPrm for cl_long3 {}
// unsafe impl OclVec for cl_long3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_long4(pub i64, pub i64, pub i64, pub i64);
unsafe impl OclPrm for cl_long4 {}
unsafe impl OclVec for cl_long4 {}

pub type cl_long3 = cl_long4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_long8(pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64);
unsafe impl OclPrm for cl_long8 {}
unsafe impl OclVec for cl_long8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_long16(pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, 
	pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64);
unsafe impl OclPrm for cl_long16 {}
unsafe impl OclVec for cl_long16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ulong1(pub u64);
unsafe impl OclPrm for cl_ulong1 {}
unsafe impl OclVec for cl_ulong1 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ulong2(pub u64, pub u64);
unsafe impl OclPrm for cl_ulong2 {}
unsafe impl OclVec for cl_ulong2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_ulong3(pub u64, pub u64, pub u64, pub u64);
// unsafe impl OclPrm for cl_ulong3 {}
// unsafe impl OclVec for cl_ulong3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ulong4(pub u64, pub u64, pub u64, pub u64);
unsafe impl OclPrm for cl_ulong4 {}
unsafe impl OclVec for cl_ulong4 {}

pub type cl_ulong3 = cl_ulong4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ulong8(pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64);
unsafe impl OclPrm for cl_ulong8 {}
unsafe impl OclVec for cl_ulong8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_ulong16(pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, 
	pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64);
unsafe impl OclPrm for cl_ulong16 {}
unsafe impl OclVec for cl_ulong16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_float2(pub f32, pub f32);
unsafe impl OclPrm for cl_float2 {}
unsafe impl OclVec for cl_float2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_float3(pub f32, pub f32, pub f32, pub f32);
// unsafe impl OclPrm for cl_float3 {}
// unsafe impl OclVec for cl_float3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_float4(pub f32, pub f32, pub f32, pub f32);
unsafe impl OclPrm for cl_float4 {}
unsafe impl OclVec for cl_float4 {}

impl Add<cl_float4> for cl_float4 {
	type Output = cl_float4;

	fn add(self, rhs: cl_float4) -> cl_float4 {
		cl_float4(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3)
	}
}

pub type cl_float3 = cl_float4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_float8(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);
unsafe impl OclPrm for cl_float8 {}
unsafe impl OclVec for cl_float8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_float16(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, 
	pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);
unsafe impl OclPrm for cl_float16 {}
unsafe impl OclVec for cl_float16 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_double2(pub f64, pub f64);
unsafe impl OclPrm for cl_double2 {}
unsafe impl OclVec for cl_double2 {}

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct cl_double3(pub f64, pub f64, pub f64, pub f64);
// unsafe impl OclPrm for cl_double3 {}
// unsafe impl OclVec for cl_double3 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_double4(pub f64, pub f64, pub f64, pub f64);
unsafe impl OclPrm for cl_double4 {}
unsafe impl OclVec for cl_double4 {}

pub type cl_double3 = cl_double4;

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_double8(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);
unsafe impl OclPrm for cl_double8 {}
unsafe impl OclVec for cl_double8 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct cl_double16(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, 
	pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);
unsafe impl OclPrm for cl_double16 {}
unsafe impl OclVec for cl_double16 {}



// VECTOR TYPE COMPLETE LIST (with redundancies): (taken from cl_platform.h v1.2):

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