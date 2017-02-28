//! OpenCL vector types.
//!
//! All operations use wrapped arithmetic and will not overflow.
//!
//! Some of these macros have been adapted (shamelessly copied) from those in
//! the standard library.
//!
//! [TODO (someday)]: Add scalar-widening operations (allowing vec * scl for example).

// #![allow(unused_imports)]

use std::fmt::{Display, Formatter, Result as FmtResult};
use std::ops::*;
use std::iter::{Sum, Product};
use num::{Zero, One};
use ::{OclPrm, OclVec};

pub trait Splat {
    type Scalar: OclPrm;

    fn splat(Self::Scalar) -> Self;
}


macro_rules! expand_val {
    ($( $junk:expr, $val:expr ),+) => ( $($val),+ );
}


// implements the unary operator "op &T"
// based on "op T" where T is expected to be `Copy`able
//
// Adapted from: `https://doc.rust-lang.org/src/core/internal_macros.rs.html`.
macro_rules! forward_ref_unop {
    (impl $imp:ident, $method:ident for $t:ty) => {
        impl<'a> $imp for &'a $t {
            type Output = <$t as $imp>::Output;

            #[inline]
            fn $method(self) -> <$t as $imp>::Output {
                $imp::$method(*self)
            }
        }
    }
}

// implements binary operators "&T op U", "T op &U", "&T op &U"
// based on "T op U" where T and U are expected to be `Copy`able
//
// Adapted from: `https://doc.rust-lang.org/src/core/internal_macros.rs.html`.
macro_rules! forward_ref_binop {
    (impl $imp:ident, $method:ident for $t:ty, $u:ty) => {
        impl<'a> $imp<$u> for &'a $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, rhs: $u) -> <$t as $imp<$u>>::Output {
                $imp::$method(*self, rhs)
            }
        }

        impl<'a> $imp<&'a $u> for $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, rhs: &'a $u) -> <$t as $imp<$u>>::Output {
                $imp::$method(self, *rhs)
            }
        }

        impl<'a, 'b> $imp<&'a $u> for &'b $t {
            type Output = <$t as $imp<$u>>::Output;

            #[inline]
            fn $method(self, rhs: &'a $u) -> <$t as $imp<$u>>::Output {
                $imp::$method(*self, *rhs)
            }
        }
    }
}


macro_rules! impl_sh_unsigned {
    ($name:ident, $f:ident: $( $idx:expr ),+) => (
        impl Shl<$f> for $name {
            type Output = $name;

            #[inline(always)]
            fn shl(self, rhs: $f) -> $name {
                // self.0.wrapping_shl((rhs & self::shift_max::$name as $f) as u32)
                $name::from([$( self[$idx] << rhs ),+])
            }
        }

        impl ShlAssign<$f> for $name {
            #[inline(always)]
            fn shl_assign(&mut self, rhs: $f) {
                *self = *self << rhs;
            }
        }

        impl Shr<$f> for $name {
            type Output = $name;

            #[inline(always)]
            fn shr(self, rhs: $f) -> $name {
                // self.0.wrapping_shr((rhs & self::shift_max::$name as $f) as u32)
                $name::from([$( self[$idx] >> rhs ),+])
            }
        }

        impl ShrAssign<$f> for $name {
            #[inline(always)]
            fn shr_assign(&mut self, rhs: $f) {
                *self = *self >> rhs;
            }
        }
    )
}


macro_rules! impl_sh_signed {
    ($name:ident, $f:ident: $( $idx:expr ),+) => (
        impl Shl<$f> for $name {
            type Output = $name;

            #[inline(always)]
            fn shl(self, rhs: $f) -> $name {
                if rhs < 0 {
                    // self.0.wrapping_shr((-rhs & self::shift_max::$name as $f) as u32)
                    $name::from([$( self[$idx] >> -rhs ),+])
                } else {
                    // self.0.wrapping_shl((rhs & self::shift_max::$name as $f) as u32)
                    $name::from([$( self[$idx] << rhs ),+])
                }
            }
        }

        impl ShlAssign<$f> for $name {
            #[inline(always)]
            fn shl_assign(&mut self, rhs: $f) {
                *self = *self << rhs;
            }
        }

        impl Shr<$f> for $name {
            type Output = $name;

            #[inline(always)]
            fn shr(self, rhs: $f) -> $name {
                if rhs < 0 {
                    // self.0.wrapping_shl((-rhs & self::shift_max::$name as $f) as u32)
                    $name::from([$( self[$idx] << -rhs ),+])
                } else {
                    // self.0.wrapping_shr((rhs & self::shift_max::$name as $f) as u32)
                    $name::from([$( self[$idx] >> rhs ),+])
                }
            }
        }

        impl ShrAssign<$f> for $name {
            #[inline(always)]
            fn shr_assign(&mut self, rhs: $f) {
                *self = *self >> rhs;
            }
        }
    )
}

// FIXME (#23545): uncomment the remaining impls
macro_rules! impl_sh_all {
    ($name:ident: $( $idx:expr ),+) => (
        //impl_sh_unsigned! { $name, u8: $( $idx ),+ }
        //impl_sh_unsigned! { $name, u16: $( $idx ),+ }
        //impl_sh_unsigned! { $name, u32: $( $idx ),+ }
        //impl_sh_unsigned! { $name, u64: $( $idx ),+ }
        impl_sh_unsigned! { $name, usize: $( $idx ),+ }

        //impl_sh_signed! { $name, i8: $( $idx ),+ }
        //impl_sh_signed! { $name, i16: $( $idx ),+ }
        //impl_sh_signed! { $name, i32: $( $idx ),+ }
        //impl_sh_signed! { $name, i64: $( $idx ),+ }
        //impl_sh_signed! { $name, isize: $( $idx ),+ }
    )
}


// Adapted from: `https://doc.rust-lang.org/src/core/iter/traits.rs.html`.
macro_rules! impl_sum_product {
    ($a:ident) => (
        impl Sum for $a {
            fn sum<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold($a::zero(), |a, b| a + b)
            }
        }

        impl Product for $a {
            fn product<I: Iterator<Item=$a>>(iter: I) -> $a {
                iter.fold($a::one(), |a, b| a * b)
            }
        }

        impl<'a> Sum<&'a $a> for $a {
            fn sum<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold($a::zero(), |a, b| a + *b)
            }
        }

        impl<'a> Product<&'a $a> for $a {
            fn product<I: Iterator<Item=&'a $a>>(iter: I) -> $a {
                iter.fold($a::one(), |a, b| a * *b)
            }
        }
    )
}

// Implements integer-specific operators.
//
// Adapted from: `https://doc.rust-lang.org/src/core/num/wrapping.rs.html`.
macro_rules! impl_int_ops {
    // ($($t:ty)*) => ($(
    ($name:ident, $cardinality:expr, $ty:ty, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        impl Add for $name {
            type Output = $name;

            #[inline(always)]
            fn add(self, rhs: $name) -> $name {
                $name::from([$( self[$idx].wrapping_add(rhs[$idx]) ),+])
            }
        }

        impl Sub for $name {
            type Output = $name;

            #[inline(always)]
            fn sub(self, rhs: $name) -> $name {
                $name::from([$( self[$idx].wrapping_sub(rhs[$idx]) ),+])
            }
        }

        impl Mul for $name {
            type Output = $name;

            #[inline(always)]
            fn mul(self, rhs: $name) -> $name {
                $name::from([$( self[$idx].wrapping_mul(rhs[$idx]) ),+])
            }
        }

        impl Div for $name {
            type Output = $name;

            #[inline(always)]
            fn div(self, rhs: $name) -> $name {
                $name::from([$( self[$idx].wrapping_div(rhs[$idx]) ),+])
            }
        }

        impl Rem for $name {
            type Output = $name;

            #[inline(always)]
            fn rem(self, rhs: $name) -> $name {
                $name::from([$( self[$idx].wrapping_rem(rhs[$idx]) ),+])
            }
        }

        impl Not for $name {
            type Output = $name;

            #[inline(always)]
            fn not(self) -> $name {
                $name::from([$( !self[$idx] ),+])
            }
        }
        forward_ref_unop! { impl Not, not for $name }

        impl BitXor for $name {
            type Output = $name;

            #[inline(always)]
            fn bitxor(self, rhs: $name) -> $name {
                $name::from([$( self[$idx] ^ rhs[$idx] ),+])
            }
        }
        forward_ref_binop! { impl BitXor, bitxor for $name, $name }

        impl BitXorAssign for $name {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: $name) {
                *self = *self ^ rhs;
            }
        }

        impl BitOr for $name {
            type Output = $name;

            #[inline(always)]
            fn bitor(self, rhs: $name) -> $name {
                $name::from([$( self[$idx] | rhs[$idx] ),+])
            }
        }
        forward_ref_binop! { impl BitOr, bitor for $name, $name }

        impl BitOrAssign for $name {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: $name) {
                *self = *self | rhs;
            }
        }

        impl BitAnd for $name {
            type Output = $name;

            #[inline(always)]
            fn bitand(self, rhs: $name) -> $name {
                $name::from([$( self[$idx] & rhs[$idx] ),+])
            }
        }
        forward_ref_binop! { impl BitAnd, bitand for $name, $name }

        impl BitAndAssign for $name {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: $name) {
                *self = *self & rhs;
            }
        }

        impl Neg for $name {
            type Output = $name;
            #[inline(always)]
            fn neg(self) -> $name {
                $name::from([$( 0 - self[$idx] ),+])
            }
        }
    }
}


// Implements floating-point-specific operators.
macro_rules! impl_float_ops {
    ($name:ident, $cardinality:expr, $ty:ty, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        impl Add for $name {
            type Output = $name;

            #[inline(always)]
            fn add(self, rhs: $name) -> $name {
                $name::from([$( self[$idx] + rhs[$idx] ),+])
            }
        }

        impl Sub for $name {
            type Output = $name;

            #[inline(always)]
            fn sub(self, rhs: $name) -> $name {
                $name::from([$( self[$idx] - rhs[$idx] ),+])
            }
        }

        impl Mul for $name {
            type Output = $name;

            #[inline(always)]
            fn mul(self, rhs: $name) -> $name {
                $name::from([$( self[$idx] * rhs[$idx] ),+])
            }
        }

        impl Div for $name {
            type Output = $name;

            #[inline(always)]
            fn div(self, rhs: $name) -> $name {
                $name::from([$( self[$idx] / rhs[$idx] ),+])
            }
        }

        impl Rem for $name {
            type Output = $name;

            #[inline(always)]
            fn rem(self, rhs: $name) -> $name {
                $name::from([$( self[$idx] % rhs[$idx] ),+])
            }
        }

        impl Neg for $name {
            type Output = $name;
            #[inline(always)]
            fn neg(self) -> $name {
                $name::from([$( 0. - self[$idx] ),+])
            }
        }

    }
}


// Implements operators common to both floating point and integer types.
macro_rules! impl_common {
    ($name:ident, $cardinality:expr, $ty:ty, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        impl $name {
            #[inline]
            pub fn splat(val: $ty) -> Self {
                $name::new($( expand_val!($idx, val)),+)
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

        impl Deref for $name {
            type Target = [$ty];

            #[inline]
            fn deref(&self) -> &[$ty] {
                &self.0
            }
        }

        impl DerefMut for $name {
            #[inline]
            fn deref_mut(&mut self) -> &mut [$ty] {
                &mut self.0
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

        impl Splat for $name {
            type Scalar = $ty;

            #[inline]
            fn splat(val: $ty) -> Self {
                $name::splat(val)
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut Formatter) -> FmtResult {
                write!(f, "{:?}", self)
            }
        }

        forward_ref_binop! { impl Add, add for $name, $name }

        impl AddAssign for $name {
            #[inline(always)]
            fn add_assign(&mut self, rhs: $name) {
                *self = *self + rhs;
            }
        }

        forward_ref_binop! { impl Sub, sub for $name, $name }

        impl SubAssign for $name {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: $name) {
                *self = *self - rhs;
            }
        }

        forward_ref_binop! { impl Mul, mul for $name, $name }

        impl MulAssign for $name {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: $name) {
                *self = *self * rhs;
            }
        }

        forward_ref_binop! { impl Div, div for $name, $name }

        impl DivAssign for $name {
            #[inline(always)]
            fn div_assign(&mut self, rhs: $name) {
                *self = *self / rhs;
            }
        }

        forward_ref_binop! { impl Rem, rem for $name, $name }

        impl RemAssign for $name {
            #[inline(always)]
            fn rem_assign(&mut self, rhs: $name) {
                *self = *self % rhs;
            }
        }

        forward_ref_unop! { impl Neg, neg for $name }

        impl_sum_product!($name);

        unsafe impl OclVec for $name {
            type Scalar = $ty;
        }
    }
}

// [NOTE]: Signed and unsigned paths are identical.
macro_rules! impl_cl_vec {
    ($name:ident, $cardinality:expr, $ty:ty, f, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        impl_common!($name, $cardinality, $ty, $( $field ),+: $( $tr ),+: $( $idx ),+ );
        impl_float_ops!($name, $cardinality, $ty, $( $field ),+: $( $tr ),+: $( $idx ),+ );
    };
    ($name:ident, $cardinality:expr, $ty:ty, i, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        impl_common!($name, $cardinality, $ty, $( $field ),+: $( $tr ),+: $( $idx ),+ );
        impl_int_ops!($name, $cardinality, $ty, $( $field ),+: $( $tr ),+: $( $idx ),+ );
        impl_sh_all!($name: $( $idx ),+);
    };
    ($name:ident, $cardinality:expr, $ty:ty, u, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        impl_common!($name, $cardinality, $ty, $( $field ),+: $( $tr ),+: $( $idx ),+ );
        impl_int_ops!($name, $cardinality, $ty, $( $field ),+: $( $tr ),+: $( $idx ),+ );
        impl_sh_all!($name: $( $idx ),+);
    };
}


// Vec3s need their own special treatment until some sort of repr(align)
// exists, if ever.
macro_rules! decl_impl_cl_vec {
    ($name:ident, 1, $ty:ty, $ty_fam:ident, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        #[derive(Debug, Clone, Copy, Default, PartialOrd)]
        pub struct $name([$ty; 1]);

        impl $name {
            #[inline]
            pub fn new($( $field: $ty, )+) -> $name {
                $name([$( $field, )*])
            }
        }

        impl From<[$ty; 1]> for $name {
            #[inline]
            fn from(a: [$ty; 1]) -> $name {
                $name::new(a[0])
            }
        }

        impl From<$name> for [$ty; 1] {
            #[inline]
            fn from(v: $name) -> [$ty; 1] {
                [v.0[0]]
            }
        }

        impl From<$ty> for $name {
            #[inline]
            fn from(s: $ty) -> $name {
                $name::new(s)
            }
        }

        impl From<$name> for $ty {
            #[inline]
            fn from(v: $name) -> $ty {
                v.0[0]
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, rhs: &Self) -> bool {
                self.0[0] == rhs.0[0]
            }
        }

        impl_cl_vec!($name, 1, $ty, $ty_fam, $( $field ),+: $( $tr ),+: $( $idx ),+ );
    };
    ($name:ident, 3, $ty:ty, $ty_fam:ident, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        #[derive(Debug, Clone, Copy, Default, PartialOrd)]
        pub struct $name([$ty; 4]);

        impl $name {
            #[inline]
            pub fn new($( $field: $ty, )+) -> $name {
                $name([$( $field, )* Zero::zero()])
            }
        }

        impl From<[$ty; 3]> for $name {
            #[inline]
            fn from(a: [$ty; 3]) -> $name {
                $name::new(a[0], a[1], a[2])
            }
        }

        impl From<$name> for [$ty; 3] {
            #[inline]
            fn from(v: $name) -> [$ty; 3] {
                [v.0[0], v.0[1], v.0[2]]
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, rhs: &Self) -> bool {
                (self.0[0] == rhs.0[0]) & (self.0[1] == rhs.0[1]) & (self.0[2] == rhs.0[2])
            }
        }

        impl_cl_vec!($name, 3, $ty, $ty_fam, $( $field ),+: $( $tr ),+: $( $idx ),+ );
    };
    ($name:ident, $cardinality:expr, $ty:ty, $ty_fam:ident, $( $field:ident ),+: $( $tr:ty ),+: $( $idx:expr ),+) => {
        #[derive(Debug, Clone, Copy, Default, PartialOrd)]
        pub struct $name([$ty; $cardinality]);

        impl $name {
            #[inline]
            pub fn new($( $field: $ty, )+) -> $name {
                $name([$( $field, )*])
            }
        }

        impl From<[$ty; $cardinality]> for $name {
            #[inline]
            fn from(a: [$ty; $cardinality]) -> $name {
                $name(a)
            }
        }

        impl From<$name> for [$ty; $cardinality] {
            #[inline]
            fn from(v: $name) -> [$ty; $cardinality] {
                v.0
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, rhs: &Self) -> bool {
                $( (self[$idx] == rhs[$idx]) ) & +
            }
        }

        impl_cl_vec!($name, $cardinality, $ty, $ty_fam, $( $field ),+: $( $tr ),+: $( $idx ),+ );
    }
}

macro_rules! cl_vec {
    ($name:ident, 1, $ty:ty, $ty_fam:ident) => (
        decl_impl_cl_vec!($name, 1, $ty, $ty_fam,
            s0:
            $ty:
            0
        );
    );
    ($name:ident, 2, $ty:ty, $ty_fam:ident) => (
        decl_impl_cl_vec!($name, 2, $ty, $ty_fam,
            s0, s1:
            $ty, $ty:
            0, 1
        );
    );
    ($name:ident, 3, $ty:ty, $ty_fam:ident) => (
        decl_impl_cl_vec!($name, 3, $ty, $ty_fam,
            s0, s1, s2:
            $ty, $ty, $ty:
            0, 1, 2
        );
    );
    ($name:ident, 4, $ty:ty, $ty_fam:ident) => (
        decl_impl_cl_vec!($name, 4, $ty, $ty_fam,
            s0, s1, s2, s3:
            $ty, $ty, $ty, $ty:
            0, 1, 2, 3
        );
    );
    ($name:ident, 8, $ty:ty, $ty_fam:ident) => (
        decl_impl_cl_vec!($name, 8, $ty, $ty_fam,
            s0, s1, s2, s3, s4, s5, s6, s7:
            $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty:
            0, 1, 2, 3, 4, 5, 6, 7
        );
    );
    ($name:ident, 16, $ty:ty, $ty_fam:ident) => (
        decl_impl_cl_vec!($name, 16, $ty, $ty_fam,
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15:
            $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty, $ty:
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        );
    );
}




// // ###### CL_CHAR ######
cl_vec!(Char, 1, i8, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar2(pub i8, pub i8);
// unsafe impl OclPrm for ClChar2 {}
// unsafe impl OclVec for ClChar2 {}
cl_vec!(Char2, 2, i8, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar3(pub i8, pub i8, pub i8, i8);

// impl ClChar3 {
//     pub fn new(s0: i8, s1: i8, s2: i8) -> ClChar3 {
//         ClChar3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClChar3 {}
// unsafe impl OclVec for ClChar3 {}
cl_vec!(Char3, 3, i8, i);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar4(pub i8, pub i8, pub i8, pub i8);
// unsafe impl OclPrm for ClChar4 {}
// unsafe impl OclVec for ClChar4 {}

cl_vec!(Char4, 4, i8, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar8(pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8);
// unsafe impl OclPrm for ClChar8 {}
// unsafe impl OclVec for ClChar8 {}
cl_vec!(Char8, 8, i8, i);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClChar16(pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8,
//     pub i8, pub i8, pub i8, pub i8, pub i8, pub i8, pub i8);
// unsafe impl OclPrm for ClChar16 {}
// unsafe impl OclVec for ClChar16 {}
cl_vec!(Char16, 16, i8, i);


// // ###### CL_UCHAR ######
cl_vec!(Uchar, 1, u8, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar2(pub u8, pub u8);
// unsafe impl OclPrm for ClUchar2 {}
// unsafe impl OclVec for ClUchar2 {}

cl_vec!(Uchar2, 2, u8, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar3(pub u8, pub u8, pub u8, u8);

// impl ClUchar3 {
//     pub fn new(s0: u8, s1: u8, s2: u8) -> ClUchar3 {
//         ClUchar3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClUchar3 {}
// unsafe impl OclVec for ClUchar3 {}
cl_vec!(Uchar3, 3, u8, u);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar4(pub u8, pub u8, pub u8, pub u8);
// unsafe impl OclPrm for ClUchar4 {}
// unsafe impl OclVec for ClUchar4 {}
cl_vec!(Uchar4, 4, u8, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar8(pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8);
// unsafe impl OclPrm for ClUchar8 {}
// unsafe impl OclVec for ClUchar8 {}
cl_vec!(Uchar8, 8, u8, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUchar16(pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8,
//     pub u8, pub u8, pub u8, pub u8, pub u8, pub u8, pub u8);
// unsafe impl OclPrm for ClUchar16 {}
// unsafe impl OclVec for ClUchar16 {}
cl_vec!(Uchar16, 16, u8, u);

// // ###### CL_SHORT ######
cl_vec!(Short, 1, i16, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort2(pub i16, pub i16);
// unsafe impl OclPrm for ClShort2 {}
// unsafe impl OclVec for ClShort2 {}

cl_vec!(Short2, 2, i16, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort3(pub i16, pub i16, pub i16, i16);

// impl ClShort3 {
//     pub fn new(s0: i16, s1: i16, s2: i16) -> ClShort3 {
//         ClShort3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClShort3 {}
// unsafe impl OclVec for ClShort3 {}
cl_vec!(Short3, 3, i16, i);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort4(pub i16, pub i16, pub i16, pub i16);
// unsafe impl OclPrm for ClShort4 {}
// unsafe impl OclVec for ClShort4 {}
cl_vec!(Short4, 4, i16, i);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort8(pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16);
// unsafe impl OclPrm for ClShort8 {}
// unsafe impl OclVec for ClShort8 {}
cl_vec!(Short8, 8, i16, i);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClShort16(pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16,
//     pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16, pub i16);
// unsafe impl OclPrm for ClShort16 {}
// unsafe impl OclVec for ClShort16 {}
cl_vec!(Short16, 16, i16, i);

// // ###### CL_USHORT ######
cl_vec!(Ushort, 1, u16, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort2(pub u16, pub u16);
// unsafe impl OclPrm for ClUshort2 {}
// unsafe impl OclVec for ClUshort2 {}

cl_vec!(Ushort2, 2, u16, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort3(pub u16, pub u16, pub u16, u16);

// impl ClUshort3 {
//     pub fn new(s0: u16, s1: u16, s2: u16) -> ClUshort3 {
//         ClUshort3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClUshort3 {}
// unsafe impl OclVec for ClUshort3 {}

cl_vec!(Ushort3, 3, u16, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort4(pub u16, pub u16, pub u16, pub u16);
// unsafe impl OclPrm for ClUshort4 {}
// unsafe impl OclVec for ClUshort4 {}

cl_vec!(Ushort4, 4, u16, u);


// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort8(pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16);
// unsafe impl OclPrm for ClUshort8 {}
// unsafe impl OclVec for ClUshort8 {}

cl_vec!(Ushort8, 8, u16, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUshort16(pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16,
//     pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16, pub u16);
// unsafe impl OclPrm for ClUshort16 {}
// unsafe impl OclVec for ClUshort16 {}

cl_vec!(Ushort16, 16, u16, u);

// ###### CL_INT ######
cl_vec!(Int, 1, i32, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt2(pub i32, pub i32);
// unsafe impl OclPrm for ClInt2 {}
// unsafe impl OclVec for ClInt2 {}

cl_vec!(Int2, 2, i32, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt3(pub i32, pub i32, pub i32, i32);

// impl ClInt3 {
//     pub fn new(s0: i32, s1: i32, s2: i32) -> ClInt3 {
//         ClInt3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClInt3 {}
// unsafe impl OclVec for ClInt3 {}

cl_vec!(Int3, 3, i32, i);

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

cl_vec!(Int4, 4, i32, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt8(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);
// unsafe impl OclPrm for ClInt8 {}
// unsafe impl OclVec for ClInt8 {}

cl_vec!(Int8, 8, i32, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClInt16(pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32,
//     pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32, pub i32);
// unsafe impl OclPrm for ClInt16 {}
// unsafe impl OclVec for ClInt16 {}

cl_vec!(Int16, 16, i32, i);

// // ###### CL_UINT ######
cl_vec!(Uint, 1, u32, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint2(pub u32, pub u32);
// unsafe impl OclPrm for ClUint2 {}
// unsafe impl OclVec for ClUint2 {}

cl_vec!(Uint2, 2, u32, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint3(pub u32, pub u32, pub u32, u32);

// impl ClUint3 {
//     pub fn new(s0: u32, s1: u32, s2: u32) -> ClUint3 {
//         ClUint3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClUint3 {}
// unsafe impl OclVec for ClUint3 {}

cl_vec!(Uint3, 3, u32, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint4(pub u32, pub u32, pub u32, pub u32);
// unsafe impl OclPrm for ClUint4 {}
// unsafe impl OclVec for ClUint4 {}

cl_vec!(Uint4, 4, u32, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint8(pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32);
// unsafe impl OclPrm for ClUint8 {}
// unsafe impl OclVec for ClUint8 {}

cl_vec!(Uint8, 8, i8, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUint16(pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32,
//     pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32, pub u32);
// unsafe impl OclPrm for ClUint16 {}
// unsafe impl OclVec for ClUint16 {}

cl_vec!(Uint16, 16, u32, u);

// // ###### CL_LONG ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong1(pub i64);
// unsafe impl OclPrm for ClLong1 {}
// unsafe impl OclVec for ClLong1 {}

cl_vec!(Long, 1, i64, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong2(pub i64, pub i64);
// unsafe impl OclPrm for ClLong2 {}
// unsafe impl OclVec for ClLong2 {}

cl_vec!(Long2, 2, i64, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong3(pub i64, pub i64, pub i64, i64);

// impl ClLong3 {
//     pub fn new(s0: i64, s1: i64, s2: i64) -> ClLong3 {
//         ClLong3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClLong3 {}
// unsafe impl OclVec for ClLong3 {}

cl_vec!(Long3, 3, i64, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong4(pub i64, pub i64, pub i64, pub i64);
// unsafe impl OclPrm for ClLong4 {}
// unsafe impl OclVec for ClLong4 {}

cl_vec!(Long4, 4, i64, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong8(pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64);
// unsafe impl OclPrm for ClLong8 {}
// unsafe impl OclVec for ClLong8 {}

cl_vec!(Long8, 8, i64, i);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClLong16(pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64,
//     pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64, pub i64);
// unsafe impl OclPrm for ClLong16 {}
// unsafe impl OclVec for ClLong16 {}

cl_vec!(Long16, 16, i64, i);

// // ###### CL_ULONG ######
// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong1(pub u64);
// unsafe impl OclPrm for ClUlong1 {}
// unsafe impl OclVec for ClUlong1 {}

cl_vec!(Ulong, 1, u64, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong2(pub u64, pub u64);
// unsafe impl OclPrm for ClUlong2 {}
// unsafe impl OclVec for ClUlong2 {}

cl_vec!(Ulong2, 2, u64, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong3(pub u64, pub u64, pub u64, u64);

// impl ClUlong3 {
//     pub fn new(s0: u64, s1: u64, s2: u64) -> ClUlong3 {
//         ClUlong3(s0, s1, s2, 0)
//     }
// }

// unsafe impl OclPrm for ClUlong3 {}
// unsafe impl OclVec for ClUlong3 {}

cl_vec!(Ulong3, 3, u64, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong4(pub u64, pub u64, pub u64, pub u64);
// unsafe impl OclPrm for ClUlong4 {}
// unsafe impl OclVec for ClUlong4 {}

cl_vec!(Ulong4, 4, u64, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong8(pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64);
// unsafe impl OclPrm for ClUlong8 {}
// unsafe impl OclVec for ClUlong8 {}

cl_vec!(Ulong8, 8, u64, u);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClUlong16(pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64,
//     pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64, pub u64);
// unsafe impl OclPrm for ClUlong16 {}
// unsafe impl OclVec for ClUlong16 {}

cl_vec!(Ulong16, 16, u64, u);

// // ###### CL_FLOAT ######
cl_vec!(Float, 1, f32, f);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClFloat2(pub f32, pub f32);
// unsafe impl OclPrm for ClFloat2 {}
// unsafe impl OclVec for ClFloat2 {}

cl_vec!(Float2, 2, f32, f);

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

cl_vec!(Float3, 3, f32, f);

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

cl_vec!(Float4, 4, f32, f);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClFloat8(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);
// unsafe impl OclPrm for ClFloat8 {}
// unsafe impl OclVec for ClFloat8 {}

cl_vec!(Float8, 8, f32, f);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClFloat16(pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32,
//     pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32, pub f32);
// unsafe impl OclPrm for ClFloat16 {}
// unsafe impl OclVec for ClFloat16 {}

cl_vec!(Float16, 16, f32, f);

// // ###### CL_DOUBLE ######
cl_vec!(Double, 1, f64, f);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble2(pub f64, pub f64);
// unsafe impl OclPrm for ClDouble2 {}
// unsafe impl OclVec for ClDouble2 {}

cl_vec!(Double2, 2, f64, f);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble3(pub f64, pub f64, pub f64, f64);

// impl ClDouble3 {
//     pub fn new(s0: f64, s1: f64, s2: f64) -> ClDouble3 {
//         ClDouble3(s0, s1, s2, 0.0)
//     }
// }

// unsafe impl OclPrm for ClDouble3 {}
// unsafe impl OclVec for ClDouble3 {}

cl_vec!(Double3, 3, f64, f);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble4(pub f64, pub f64, pub f64, pub f64);
// unsafe impl OclPrm for ClDouble4 {}
// unsafe impl OclVec for ClDouble4 {}

cl_vec!(Double4, 4, f64, f);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble8(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);
// unsafe impl OclPrm for ClDouble8 {}
// unsafe impl OclVec for ClDouble8 {}

cl_vec!(Double8, 8, f64, f);

// #[derive(PartialEq, Debug, Clone, Copy, Default)]
// pub struct ClDouble16(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64,
//     pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);
// unsafe impl OclPrm for ClDouble16 {}
// unsafe impl OclVec for ClDouble16 {}

cl_vec!(Double16, 16, f64, f);


// Copied from `https://doc.rust-lang.org/src/core/num/wrapping.rs.html`.
mod shift_max {
    #![allow(non_upper_case_globals, dead_code)]

    #[cfg(target_pointer_width = "16")]
    mod platform {
        pub const usize: u32 = super::u16;
        pub const isize: u32 = super::i16;
    }

    #[cfg(target_pointer_width = "32")]
    mod platform {
        pub const usize: u32 = super::u32;
        pub const isize: u32 = super::i32;
    }

    #[cfg(target_pointer_width = "64")]
    mod platform {
        pub const usize: u32 = super::u64;
        pub const isize: u32 = super::i64;
    }

    pub const i8: u32 = (1 << 3) - 1;
    pub const i16: u32 = (1 << 4) - 1;
    pub const i32: u32 = (1 << 5) - 1;
    pub const i64: u32 = (1 << 6) - 1;
    pub use self::platform::isize;

    pub const u8: u32 = i8;
    pub const u16: u32 = i16;
    pub const u32: u32 = i32;
    pub const u64: u32 = i64;
    pub use self::platform::usize;

    // Char, Char2, Char3, Char4, Char8, Char16,
    // Uchar, Uchar2, Uchar3, Uchar4, Uchar8, Uchar16,
    // Short, Short2, Short3, Short4, Short8, Short16,
    // Ushort, Ushort2, Ushort3, Ushort4, Ushort8, Ushort16,
    // Int, Int2, Int3, Int4, Int8, Int16,
    // Uint, Uint2, Uint3, Uint4, Uint8, Uint16,
    // Long, Long2, Long3, Long4, Long8, Long16,
    // Ulong, Ulong2, Ulong3, Ulong4, Ulong8, Ulong16,
    // Float, Float2, Float3, Float4, Float8, Float16,
    // Double, Double2, Double3, Double4, Double8, Double16,
}