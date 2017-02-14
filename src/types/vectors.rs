//! OpenCL vector types.
//!
//! These types *should* have the same memory layout no matter which `repr` is
//! used. The default (`repr(rust)`) will be used until SIMD support is stable
//! and its use can be evaluated. If using any of these types *within* another
//! struct, memory alignment must be managed manually using spacing, etc.
//! (OPEN QUESTION: Does using repr(simd) somehow help in this case?). If
//! using within a `Vec` (typical usage, i.e.: `Vec<ClFloat4>`) you don't need
//! to worry about it.
//!
//! [TODO]: Create a macro to implement `Index`, mathematical operations such
//! as `Add`, `Mul`, etc., and whatever else.
//!
//! Not sure about swizzling interfaces or if they'll ever be realistic.
//! Obviously for now just use .0, .1, .2, etc.
//!
//!

// #![allow(non_camel_case_types)]

use ::{OclPrm, OclVec};
use std::ops::{Add};


// ###### CL_CHAR ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClChar2(pub i8, pub i8);
unsafe impl OclPrm for ClChar2 {}
unsafe impl OclVec for ClChar2 {}
unsafe impl Send for ClChar2 {}
unsafe impl Sync for ClChar2 {}

#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClChar3(pub i8, pub i8, pub i8, i8);

impl ClChar3 {
    pub fn new(s0: i8, s1: i8, s2: i8) -> ClChar3 {
        ClChar3(s0, s1, s2, 0)
    }
}

unsafe impl OclPrm for ClChar3 {}
unsafe impl OclVec for ClChar3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClChar4(pub i8, pub i8, pub i8, pub i8);
unsafe impl OclPrm for ClChar4 {}
unsafe impl OclVec for ClChar4 {}


// pub type ClChar3 = ClChar4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClChar8(pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8);
unsafe impl OclPrm for ClChar8 {}
unsafe impl OclVec for ClChar8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClChar16(pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8,
    pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8);
unsafe impl OclPrm for ClChar16 {}
unsafe impl OclVec for ClChar16 {}


// ###### CL_UCHAR ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUchar2(pub u8, pub u8);
unsafe impl OclPrm for ClUchar2 {}
unsafe impl OclVec for ClUchar2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUchar3(pub u8, pub u8, pub u8, u8);

impl ClUchar3 {
    pub fn new(s0: u8, s1: u8, s2: u8) -> ClUchar3 {
        ClUchar3(s0, s1, s2, 0)
    }
}

unsafe impl OclPrm for ClUchar3 {}
unsafe impl OclVec for ClUchar3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUchar4(pub u8, pub u8, pub u8, pub u8);
unsafe impl OclPrm for ClUchar4 {}
unsafe impl OclVec for ClUchar4 {}


// pub type ClUchar3 = ClUchar4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUchar8(pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8);
unsafe impl OclPrm for ClUchar8 {}
unsafe impl OclVec for ClUchar8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUchar16(pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
    pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8);
unsafe impl OclPrm for ClUchar16 {}
unsafe impl OclVec for ClUchar16 {}


// ###### CL_SHORT ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClShort2(pub i16, pub i16);
unsafe impl OclPrm for ClShort2 {}
unsafe impl OclVec for ClShort2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClShort3(pub i16, pub i16, pub i16, i16);

impl ClShort3 {
    pub fn new(s0: i16, s1: i16, s2: i16) -> ClShort3 {
        ClShort3(s0, s1, s2, 0)
    }
}

unsafe impl OclPrm for ClShort3 {}
unsafe impl OclVec for ClShort3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClShort4(pub i16, pub i16, pub i16, pub i16);
unsafe impl OclPrm for ClShort4 {}
unsafe impl OclVec for ClShort4 {}


// pub type ClShort3 = ClShort4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClShort8(pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16);
unsafe impl OclPrm for ClShort8 {}
unsafe impl OclVec for ClShort8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClShort16(pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16,
    pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16);
unsafe impl OclPrm for ClShort16 {}
unsafe impl OclVec for ClShort16 {}


// ###### CL_USHORT ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUshort2(pub u16, pub u16);
unsafe impl OclPrm for ClUshort2 {}
unsafe impl OclVec for ClUshort2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUshort3(pub u16, pub u16, pub u16, u16);

impl ClUshort3 {
    pub fn new(s0: u16, s1: u16, s2: u16) -> ClUshort3 {
        ClUshort3(s0, s1, s2, 0)
    }
}

unsafe impl OclPrm for ClUshort3 {}
unsafe impl OclVec for ClUshort3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUshort4(pub u16, pub u16, pub u16, pub u16);
unsafe impl OclPrm for ClUshort4 {}
unsafe impl OclVec for ClUshort4 {}


// pub type ClUshort3 = ClUshort4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUshort8(pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16);
unsafe impl OclPrm for ClUshort8 {}
unsafe impl OclVec for ClUshort8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUshort16(pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16,
    pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16);
unsafe impl OclPrm for ClUshort16 {}
unsafe impl OclVec for ClUshort16 {}


// ###### CL_INT ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClInt2(pub i32, pub i32);
unsafe impl OclPrm for ClInt2 {}
unsafe impl OclVec for ClInt2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClInt3(pub i32, pub i32, pub i32, i32);

impl ClInt3 {
    pub fn new(s0: i32, s1: i32, s2: i32) -> ClInt3 {
        ClInt3(s0, s1, s2, 0)
    }
}

unsafe impl OclPrm for ClInt3 {}
unsafe impl OclVec for ClInt3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClInt4(pub i32, pub i32, pub i32, pub i32);
unsafe impl OclPrm for ClInt4 {}
unsafe impl OclVec for ClInt4 {}

impl ClInt4 {
    pub fn new(s0: i32, s1: i32, s2: i32, s3: i32) -> ClInt4 {
        ClInt4(s0, s1, s2, s3)
    }
}

impl Add<ClInt4> for ClInt4 {
    type Output = ClInt4;

    fn add(self, rhs: ClInt4) -> ClInt4 {
        ClInt4(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3)
    }
}


// pub type ClInt3 = ClInt4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClInt8(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);
unsafe impl OclPrm for ClInt8 {}
unsafe impl OclVec for ClInt8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClInt16(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32,
    pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);
unsafe impl OclPrm for ClInt16 {}
unsafe impl OclVec for ClInt16 {}


// ###### CL_UINT ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUint2(pub u32, pub u32);
unsafe impl OclPrm for ClUint2 {}
unsafe impl OclVec for ClUint2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUint3(pub u32, pub u32, pub u32, u32);

impl ClUint3 {
    pub fn new(s0: u32, s1: u32, s2: u32) -> ClUint3 {
        ClUint3(s0, s1, s2, 0)
    }
}

unsafe impl OclPrm for ClUint3 {}
unsafe impl OclVec for ClUint3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUint4(pub u32, pub u32, pub u32, pub u32);
unsafe impl OclPrm for ClUint4 {}
unsafe impl OclVec for ClUint4 {}


// pub type ClUint3 = ClUint4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUint8(pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32);
unsafe impl OclPrm for ClUint8 {}
unsafe impl OclVec for ClUint8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUint16(pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32,
    pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32);
unsafe impl OclPrm for ClUint16 {}
unsafe impl OclVec for ClUint16 {}


// ###### CL_LONG ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClLong1(pub i64);
unsafe impl OclPrm for ClLong1 {}
unsafe impl OclVec for ClLong1 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClLong2(pub i64, pub i64);
unsafe impl OclPrm for ClLong2 {}
unsafe impl OclVec for ClLong2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClLong3(pub i64, pub i64, pub i64, i64);

impl ClLong3 {
    pub fn new(s0: i64, s1: i64, s2: i64) -> ClLong3 {
        ClLong3(s0, s1, s2, 0)
    }
}

unsafe impl OclPrm for ClLong3 {}
unsafe impl OclVec for ClLong3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClLong4(pub i64, pub i64, pub i64, pub i64);
unsafe impl OclPrm for ClLong4 {}
unsafe impl OclVec for ClLong4 {}


// pub type ClLong3 = ClLong4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClLong8(pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64);
unsafe impl OclPrm for ClLong8 {}
unsafe impl OclVec for ClLong8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClLong16(pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64,
    pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64);
unsafe impl OclPrm for ClLong16 {}
unsafe impl OclVec for ClLong16 {}


// ###### CL_ULONG ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUlong1(pub u64);
unsafe impl OclPrm for ClUlong1 {}
unsafe impl OclVec for ClUlong1 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUlong2(pub u64, pub u64);
unsafe impl OclPrm for ClUlong2 {}
unsafe impl OclVec for ClUlong2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUlong3(pub u64, pub u64, pub u64, u64);

impl ClUlong3 {
    pub fn new(s0: u64, s1: u64, s2: u64) -> ClUlong3 {
        ClUlong3(s0, s1, s2, 0)
    }
}

unsafe impl OclPrm for ClUlong3 {}
unsafe impl OclVec for ClUlong3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUlong4(pub u64, pub u64, pub u64, pub u64);
unsafe impl OclPrm for ClUlong4 {}
unsafe impl OclVec for ClUlong4 {}


// pub type ClUlong3 = ClUlong4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUlong8(pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64);
unsafe impl OclPrm for ClUlong8 {}
unsafe impl OclVec for ClUlong8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClUlong16(pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64,
    pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64);
unsafe impl OclPrm for ClUlong16 {}
unsafe impl OclVec for ClUlong16 {}


// ###### CL_FLOAT ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClFloat2(pub f32, pub f32);
unsafe impl OclPrm for ClFloat2 {}
unsafe impl OclVec for ClFloat2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClFloat3(pub f32, pub f32, pub f32, f32);

impl ClFloat3 {
    pub fn new(s0: f32, s1: f32, s2: f32) -> ClFloat3 {
        ClFloat3(s0, s1, s2, 0.0)
    }
}

impl Add<ClFloat3> for ClFloat3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        ClFloat3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3)
    }
}

impl From<[f32; 3]> for ClFloat3 {
    fn from(f: [f32; 3]) -> Self {
        ClFloat3(f[0], f[1], f[2], 0.0)
    }
}

impl From<(f32, f32, f32)> for ClFloat3 {
    fn from(f: (f32, f32, f32)) -> Self {
        ClFloat3(f.0, f.1, f.2, 0.0)
    }
}

unsafe impl OclPrm for ClFloat3 {}
unsafe impl OclVec for ClFloat3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClFloat4(pub f32, pub f32, pub f32, pub f32);

impl ClFloat4 {
    pub fn new(s0: f32, s1: f32, s2: f32, s3: f32) -> ClFloat4 {
        ClFloat4(s0, s1, s2, s3)
    }
}

impl Add<ClFloat4> for ClFloat4 {
    type Output = ClFloat4;

    fn add(self, rhs: ClFloat4) -> ClFloat4 {
        ClFloat4(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3)
    }
}

unsafe impl OclPrm for ClFloat4 {}
unsafe impl OclVec for ClFloat4 {}
unsafe impl Send for ClFloat4 {}
unsafe impl Sync for ClFloat4 {}

// pub type ClFloat3 = ClFloat4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClFloat8(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);
unsafe impl OclPrm for ClFloat8 {}
unsafe impl OclVec for ClFloat8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClFloat16(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32,
    pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);
unsafe impl OclPrm for ClFloat16 {}
unsafe impl OclVec for ClFloat16 {}


// ###### CL_DOUBLE ######
#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClDouble2(pub f64, pub f64);
unsafe impl OclPrm for ClDouble2 {}
unsafe impl OclVec for ClDouble2 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClDouble3(pub f64, pub f64, pub f64, f64);

impl ClDouble3 {
    pub fn new(s0: f64, s1: f64, s2: f64) -> ClDouble3 {
        ClDouble3(s0, s1, s2, 0.0)
    }
}

unsafe impl OclPrm for ClDouble3 {}
unsafe impl OclVec for ClDouble3 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClDouble4(pub f64, pub f64, pub f64, pub f64);
unsafe impl OclPrm for ClDouble4 {}
unsafe impl OclVec for ClDouble4 {}


// pub type ClDouble3 = ClDouble4;


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClDouble8(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);
unsafe impl OclPrm for ClDouble8 {}
unsafe impl OclVec for ClDouble8 {}


#[derive(PartialEq, Debug, Clone, Copy, Default)]
pub struct ClDouble16(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64,
    pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);
unsafe impl OclPrm for ClDouble16 {}
unsafe impl OclVec for ClDouble16 {}
