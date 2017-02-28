//! OpenCL vector types.
//!
//!
//! [TODO]: Create a macro to implement mathematical operations such
//! as `Add`, `Mul`, etc., and whatever else.
//!
//!
//!

#![allow(unused_imports)]

use std::fmt::{self, Display, Formatter, Result as FmtResult, Error as FmtError};
pub use std::ops::{Add, Sub, Mul, Div, Rem, Index, Deref};
pub use num::{Zero, One, NumCast, Float, PrimInt};
pub use ::{OclPrm, OclVec, OclScl};

pub trait Splat {
    type Scalar: OclPrm;
    // type Output: OclVec;

    fn splat(Self::Scalar) -> Self;
}


#[macro_export]
macro_rules! expand_val {
    ($( $junk:expr, $val:expr ),+) => ( $($val),+ );
}


#[macro_export]
macro_rules! impl_common_cl_vec {
    ($name:ident, $cardinality:expr, $ty:ty, $( $field:ident ),+: $( $tr:ty ),+: $( $c:expr ),+) => {
        impl $name {
            #[inline]
            pub fn splat(val: $ty) -> $name {
                $name::new($( expand_val!($c, val)),+)
            }

            #[inline]
            pub fn zero() -> Self {
                $name::from([Zero::zero(); $cardinality])
            }

            #[inline]
            pub fn is_zero(&self) -> bool {
                *self == $name::from([Zero::zero(); $cardinality])
            }

            #[inline]
            pub fn one() -> Self {
                $name::from([One::one(); $cardinality])
            }
        }

        impl $crate::types::vectors::Deref for $name {
            type Target = [$ty];

            #[inline]
            fn deref(&self) -> &[$ty] {
                &self.0
            }
        }

        impl Zero for $name {
            #[inline]
            fn zero() -> Self {
                $name::zero()
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.is_zero()
            }
        }

        impl One for $name {
            #[inline]
            fn one() -> Self {
                $name::from([One::one(); $cardinality])
            }
        }

        impl Add for $name {
            type Output = Self;

            #[inline]
            fn add(self, rhs: $name) -> Self {
                $name::from([$( self[$c] + rhs[$c] ),+])
            }
        }

        impl Sub for $name {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: $name) -> Self {
                $name::from([$( self[$c] - rhs[$c] ),+])
            }
        }

        impl Mul for $name {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: $name) -> Self {
                $name::from([$( self[$c] * rhs[$c] ),+])
            }
        }

        impl Div for $name {
            type Output = Self;

            #[inline]
            fn div(self, rhs: $name) -> Self {
                $name::from([$( self[$c] / rhs[$c] ),+])
            }
        }

        impl Rem for $name {
            type Output = Self;

            #[inline]
            fn rem(self, rhs: $name) -> Self {
                $name::from([$( self[$c] / rhs[$c] ),+])
            }
        }

        impl Splat for $name {
            type Scalar = $ty;
            // type Output = Self;

            fn splat(val: $ty) -> Self {
                $name::splat(val)
            }
        }

        // impl $crate::types::vectors::Display for $name {
        //     fn fmt(&self, &mut $crate::types::vectors::Formatter) -> $crate::types::vectors::FmtResult<()> {
        //         write!(f, "{:?}", self)
        //     }
        // }
    }
}


// Vec3s need their own special treatment until some sort of repr(align)
// exists, if ever.
#[macro_export]
macro_rules! decl_impl_cl_vec {
    ($name:ident, 3, $ty:ty, $( $field:ident ),+: $( $tr:ty ),+: $( $c:expr ),+) => {
        #[derive(Debug, Clone, Copy, Default, PartialOrd)]
        pub struct $name([$ty; 4]);

        impl $name {
            pub fn new($( $field: $ty, )+) -> $name {
                $name([$( $field, )* Zero::zero()])
            }
        }

        impl From<[$ty; 3]> for $name {
            fn from(a: [$ty; 3]) -> $name {
                $name::new(a[0], a[1], a[2])
            }
        }

        impl From<$name> for [$ty; 3] {
            fn from(v: $name) -> [$ty; 3] {
                [v.0[0], v.0[1], v.0[2]]
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                (self.0[0] == other.0[0]) & (self.0[1] == other.0[1]) & (self.0[2] == other.0[2])
            }
        }

        impl_common_cl_vec!($name, 3, $ty, $( $field ),+: $( $tr ),+: $( $c ),+ );
    };
    ($name:ident, $cardinality:expr, $ty:ty, $( $field:ident ),+: $( $tr:ty ),+: $( $c:expr ),+) => {
        #[derive(Debug, Clone, Copy, Default, PartialOrd)]
        pub struct $name([$ty; $cardinality]);

        impl $name {
            pub fn new($( $field: $ty, )+) -> $name {
                $name([$( $field, )*])
            }
        }

        impl From<[$ty; $cardinality]> for $name {
            fn from(a: [$ty; $cardinality]) -> $name {
                $name(a)
            }
        }

        impl From<$name> for [$ty; $cardinality] {
            fn from(v: $name) -> [$ty; $cardinality] {
                v.0
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                $( (self[$c] == other[$c]) ) & +
            }
        }

        impl_common_cl_vec!($name, $cardinality, $ty, $( $field ),+: $( $tr ),+: $( $c ),+ );
    }
}


#[macro_export]
macro_rules! cl_vec {
    ($name:ident, 1, $ty:ty) => (
        decl_impl_cl_vec!($name, 1, $ty,
            s0:
            $ty:
            0
        );
    );
    ($name:ident, 2, $ty:ty) => (
        decl_impl_cl_vec!($name, 2, $ty,
            s0, s1:
            $ty, $ty:
            0, 1
        );
    );
    ($name:ident, 3, $ty:ty) => (
        decl_impl_cl_vec!($name, 3, $ty,
            s0, s1, s2:
            $ty, $ty, $ty:
            0, 1, 2
        );
    );
    ($name:ident, 4, $ty:ty) => (
        decl_impl_cl_vec!($name, 4, $ty,
            s0, s1, s2, s3:
            $ty, $ty, $ty, $ty:
            0, 1, 2, 3
        );
    );
    ($name:ident, 8, $ty:ty) => (
        decl_impl_cl_vec!($name, 8, $ty,
            s0, s1, s2, s3, s4, s5, s6, s7:
            $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty:
            0, 1, 2, 3, 4, 5, 6, 7
        );
    );
    ($name:ident, 16, $ty:ty) => (
        decl_impl_cl_vec!($name, 16, $ty,
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15:
            $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty:
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        );
    );
}




cl_vec!(ClInt2, 2, i32);
cl_vec!(ClInt4, 4, i32);

cl_vec!(ClFloat2, 2, f32);
cl_vec!(ClFloat4, 4, f32);





// // ###### CL_CHAR ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar2(pub i8, pub i8);
// unsafe impl OclPrm for ClChar2 {}
// unsafe impl OclVec for ClChar2 {}
cl_vec!(ClChar2, 2, i8);



// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar3(pub i8, pub i8, pub i8, i8);

// impl ClChar3 {
//     pub fn new(s0: i8, s1: i8, s2: i8) -> ClChar3 {
//         ClChar3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClChar3 {}
// unsafe impl OclVec for ClChar3 {}
cl_vec!(ClChar3, 3, i8);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar4(pub i8, pub i8, pub i8, pub i8);
// unsafe impl OclPrm for ClChar4 {}
// unsafe impl OclVec for ClChar4 {}

cl_vec!(ClChar4, 4, i8);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar8(pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8);
// unsafe impl OclPrm for ClChar8 {}
// unsafe impl OclVec for ClChar8 {}
cl_vec!(ClChar8, 8, i8);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar16(pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8,
//     pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8);
// unsafe impl OclPrm for ClChar16 {}
// unsafe impl OclVec for ClChar16 {}
cl_vec!(ClChar16, 16, i8);


// // ###### CL_UCHAR ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar2(pub u8, pub u8);
// unsafe impl OclPrm for ClUchar2 {}
// unsafe impl OclVec for ClUchar2 {}
cl_vec!(ClUchar2, 2, u8);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar3(pub u8, pub u8, pub u8, u8);

// impl ClUchar3 {
//     pub fn new(s0: u8, s1: u8, s2: u8) -> ClUchar3 {
//         ClUchar3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClUchar3 {}
// unsafe impl OclVec for ClUchar3 {}
cl_vec!(ClUchar3, 3, u8);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar4(pub u8, pub u8, pub u8, pub u8);
// unsafe impl OclPrm for ClUchar4 {}
// unsafe impl OclVec for ClUchar4 {}
cl_vec!(ClUchar4, 4, u8);


// pub type ClUchar3 = ClUchar4;


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar8(pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8);
// unsafe impl OclPrm for ClUchar8 {}
// unsafe impl OclVec for ClUchar8 {}
cl_vec!(ClUchar8, 8, u8);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar16(pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
//     pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8);
// unsafe impl OclPrm for ClUchar16 {}
// unsafe impl OclVec for ClUchar16 {}
cl_vec!(ClUchar16, 16, u8);


// // ###### CL_SHORT ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort2(pub i16, pub i16);
// unsafe impl OclPrm for ClShort2 {}
// unsafe impl OclVec for ClShort2 {}
cl_vec!(ClShort2, 2, i16);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort3(pub i16, pub i16, pub i16, i16);

// impl ClShort3 {
//     pub fn new(s0: i16, s1: i16, s2: i16) -> ClShort3 {
//         ClShort3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClShort3 {}
// unsafe impl OclVec for ClShort3 {}
cl_vec!(ClShort3, 3, i16);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort4(pub i16, pub i16, pub i16, pub i16);
// unsafe impl OclPrm for ClShort4 {}
// unsafe impl OclVec for ClShort4 {}
cl_vec!(ClShort4, 4, i16);


// pub type ClShort3 = ClShort4;


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort8(pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16);
// unsafe impl OclPrm for ClShort8 {}
// unsafe impl OclVec for ClShort8 {}
cl_vec!(ClShort8, 8, i16);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort16(pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16,
//     pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16);
// unsafe impl OclPrm for ClShort16 {}
// unsafe impl OclVec for ClShort16 {}
cl_vec!(ClShort16, 16, i16);

// // ###### CL_USHORT ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort2(pub u16, pub u16);
// unsafe impl OclPrm for ClUshort2 {}
// unsafe impl OclVec for ClUshort2 {}
cl_vec!(ClUshort2, 2, u16);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort3(pub u16, pub u16, pub u16, u16);

// impl ClUshort3 {
//     pub fn new(s0: u16, s1: u16, s2: u16) -> ClUshort3 {
//         ClUshort3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClUshort3 {}
// unsafe impl OclVec for ClUshort3 {}

cl_vec!(ClUshort3, 3, u16);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort4(pub u16, pub u16, pub u16, pub u16);
// unsafe impl OclPrm for ClUshort4 {}
// unsafe impl OclVec for ClUshort4 {}

cl_vec!(ClUshort4, 4, u16);



// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort8(pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16);
// unsafe impl OclPrm for ClUshort8 {}
// unsafe impl OclVec for ClUshort8 {}

cl_vec!(ClUshort8, 8, u16);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort16(pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16,
//     pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16);
// unsafe impl OclPrm for ClUshort16 {}
// unsafe impl OclVec for ClUshort16 {}

cl_vec!(ClUshort16, 16, u16);

// ###### CL_INT ######

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt2(pub i32, pub i32);
// unsafe impl OclPrm for ClInt2 {}
// unsafe impl OclVec for ClInt2 {}


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt3(pub i32, pub i32, pub i32, i32);

// impl ClInt3 {
//     pub fn new(s0: i32, s1: i32, s2: i32) -> ClInt3 {
//         ClInt3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClInt3 {}
// unsafe impl OclVec for ClInt3 {}

cl_vec!(ClInt3, 3, i32);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt4(pub i32, pub i32, pub i32, pub i32);
// unsafe impl OclPrm for ClInt4 {}
// unsafe impl OclVec for ClInt4 {}

// impl ClInt4 {
//     pub fn new(s0: i32, s1: i32, s2: i32, s3: i32) -> ClInt4 {
//         ClInt4(s0, s1, s2, s3)
//     }
// }

// impl Add<ClInt4> for ClInt4 {
//     type Output = ClInt4;

//     fn add(self, rhs: ClInt4) -> ClInt4 {
//         ClInt4(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3)
//     }
// }




// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt8(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);
// unsafe impl OclPrm for ClInt8 {}
// unsafe impl OclVec for ClInt8 {}

cl_vec!(ClInt8, 8, i32);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt16(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32,
//     pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);
// unsafe impl OclPrm for ClInt16 {}
// unsafe impl OclVec for ClInt16 {}


cl_vec!(ClInt16, 16, i32);

// // ###### CL_UINT ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint2(pub u32, pub u32);
// unsafe impl OclPrm for ClUint2 {}
// unsafe impl OclVec for ClUint2 {}

cl_vec!(ClUint2, 2, u32);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint3(pub u32, pub u32, pub u32, u32);

// impl ClUint3 {
//     pub fn new(s0: u32, s1: u32, s2: u32) -> ClUint3 {
//         ClUint3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClUint3 {}
// unsafe impl OclVec for ClUint3 {}

cl_vec!(ClUint3, 3, u32);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint4(pub u32, pub u32, pub u32, pub u32);
// unsafe impl OclPrm for ClUint4 {}
// unsafe impl OclVec for ClUint4 {}

cl_vec!(ClUint4, 4, u32);

// pub type ClUint3 = ClUint4;


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint8(pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32);
// unsafe impl OclPrm for ClUint8 {}
// unsafe impl OclVec for ClUint8 {}

cl_vec!(ClUint8, 8, i8);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint16(pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32,
//     pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32);
// unsafe impl OclPrm for ClUint16 {}
// unsafe impl OclVec for ClUint16 {}

cl_vec!(ClUint16, 16, u32);

// // ###### CL_LONG ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong1(pub i64);
// unsafe impl OclPrm for ClLong1 {}
// unsafe impl OclVec for ClLong1 {}

cl_vec!(ClLong, 1, i64);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong2(pub i64, pub i64);
// unsafe impl OclPrm for ClLong2 {}
// unsafe impl OclVec for ClLong2 {}

cl_vec!(ClLong2, 2, i64);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong3(pub i64, pub i64, pub i64, i64);

// impl ClLong3 {
//     pub fn new(s0: i64, s1: i64, s2: i64) -> ClLong3 {
//         ClLong3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClLong3 {}
// unsafe impl OclVec for ClLong3 {}

cl_vec!(ClLong3, 3, i64);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong4(pub i64, pub i64, pub i64, pub i64);
// unsafe impl OclPrm for ClLong4 {}
// unsafe impl OclVec for ClLong4 {}

cl_vec!(ClLong4, 4, i64);

// pub type ClLong3 = ClLong4;


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong8(pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64);
// unsafe impl OclPrm for ClLong8 {}
// unsafe impl OclVec for ClLong8 {}

cl_vec!(ClLong8, 8, i64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong16(pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64,
//     pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64);
// unsafe impl OclPrm for ClLong16 {}
// unsafe impl OclVec for ClLong16 {}

cl_vec!(ClLong16, 16, i64);


// // ###### CL_ULONG ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong1(pub u64);
// unsafe impl OclPrm for ClUlong1 {}
// unsafe impl OclVec for ClUlong1 {}

cl_vec!(ClUlong, 1, u64);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong2(pub u64, pub u64);
// unsafe impl OclPrm for ClUlong2 {}
// unsafe impl OclVec for ClUlong2 {}

cl_vec!(ClUlong2, 2, u64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong3(pub u64, pub u64, pub u64, u64);

// impl ClUlong3 {
//     pub fn new(s0: u64, s1: u64, s2: u64) -> ClUlong3 {
//         ClUlong3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClUlong3 {}
// unsafe impl OclVec for ClUlong3 {}

cl_vec!(ClUlong3, 3, u64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong4(pub u64, pub u64, pub u64, pub u64);
// unsafe impl OclPrm for ClUlong4 {}
// unsafe impl OclVec for ClUlong4 {}


cl_vec!(ClUlong4, 4, u64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong8(pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64);
// unsafe impl OclPrm for ClUlong8 {}
// unsafe impl OclVec for ClUlong8 {}

cl_vec!(ClUlong8, 8, u64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong16(pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64,
//     pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64);
// unsafe impl OclPrm for ClUlong16 {}
// unsafe impl OclVec for ClUlong16 {}

cl_vec!(ClUlong16, 16, u64);

// // ###### CL_FLOAT ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClFloat2(pub f32, pub f32);
// unsafe impl OclPrm for ClFloat2 {}
// unsafe impl OclVec for ClFloat2 {}


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClFloat3(pub f32, pub f32, pub f32, f32);

// impl ClFloat3 {
//     pub fn new(s0: f32, s1: f32, s2: f32) -> ClFloat3 {
//         ClFloat3(s0, s1, s2, 0.0)
//     }
// }

// impl Add<ClFloat3> for ClFloat3 {
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self {
//         ClFloat3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3)
//     }
// }

// impl From<[f32; 3]> for ClFloat3 {
//     fn from(f: [f32; 3]) -> Self {
//         ClFloat3(f[0], f[1], f[2], 0.0)
//     }
// }

// impl From<(f32, f32, f32)> for ClFloat3 {
//     fn from(f: (f32, f32, f32)) -> Self {
//         ClFloat3(f.0, f.1, f.2, 0.0)
//     }
// }

// unsafe impl OclPrm for ClFloat3 {}
// unsafe impl OclVec for ClFloat3 {}

cl_vec!(ClFloat3, 3, f32);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// // pub struct ClFloat4(pub f32, pub f32, pub f32, pub f32);
// pub struct ClFloat4([f32; 4]);

// impl ClFloat4 {
//     pub fn new(s0: f32, s1: f32, s2: f32, s3: f32) -> ClFloat4 {
//         ClFloat4([s0, s1, s2, s3])
//     }
// }

// impl Add<ClFloat4> for ClFloat4 {
//     type Output = ClFloat4;

//     #[inline]
//     fn add(self, rhs: ClFloat4) -> ClFloat4 {
//         ClFloat4([self[0] + rhs[0], self[1] + rhs[1], self[2] + rhs[2], self[3] + rhs[3]])
//     }
// }

// impl Deref for ClFloat4 {
//     type Target = [f32];

//     #[inline]
//     fn deref(&self) -> &[f32] {
//         &self.0
//     }
// }

// impl Index<usize> for ClFloat4 {
//     type Output = f32;

//     #[inline]
//     fn index(&self, idx: usize) -> &Self::Output {
//         &(**self)[idx]
//     }
// }

// unsafe impl OclPrm for ClFloat4 {}
// unsafe impl OclVec for ClFloat4 {}

// pub type ClFloat3 = ClFloat4;


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClFloat8(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);
// unsafe impl OclPrm for ClFloat8 {}
// unsafe impl OclVec for ClFloat8 {}

cl_vec!(ClFloat8, 8, f32);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClFloat16(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32,
//     pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);
// unsafe impl OclPrm for ClFloat16 {}
// unsafe impl OclVec for ClFloat16 {}

cl_vec!(ClFloat16, 16, f32);


// // ###### CL_DOUBLE ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble2(pub f64, pub f64);
// unsafe impl OclPrm for ClDouble2 {}
// unsafe impl OclVec for ClDouble2 {}

cl_vec!(ClDouble2, 2, f64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble3(pub f64, pub f64, pub f64, f64);

// impl ClDouble3 {
//     pub fn new(s0: f64, s1: f64, s2: f64) -> ClDouble3 {
//         ClDouble3(s0, s1, s2, 0.0)
//     }
// }

// unsafe impl OclPrm for ClDouble3 {}
// unsafe impl OclVec for ClDouble3 {}

cl_vec!(ClDouble3, 3, f64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble4(pub f64, pub f64, pub f64, pub f64);
// unsafe impl OclPrm for ClDouble4 {}
// unsafe impl OclVec for ClDouble4 {}

cl_vec!(ClDouble4, 4, f64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble8(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);
// unsafe impl OclPrm for ClDouble8 {}
// unsafe impl OclVec for ClDouble8 {}

cl_vec!(ClDouble8, 8, f64);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble16(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64,
//     pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);
// unsafe impl OclPrm for ClDouble16 {}
// unsafe impl OclVec for ClDouble16 {}

cl_vec!(ClDouble16, 16, f64);