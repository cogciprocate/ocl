//! A simple way to specify the sizes or offsets of up to three dimensions.
// use std::convert::Into;
use std::convert::From;
use std::fmt::Debug;
use std::ops::Index;
// use std::mem;
use num::{Num, ToPrimitive};
use error::{Result as OclResult, Error as OclError};
use standard::{MemDims, WorkDims};
use util;

/// Specifies a size or offset in up to three dimensions.
///
/// Custom types implementing `MemDims` and `WorkDims` should be created to
/// express more complex relationships between data shape and work size.
/// rather than using this one.
///
/// [FIXME] TODO: Much more explaination needed as soon as conventions solidify.
///
/// [UNSTABLE]: This type and its methods may be renamed or otherwise changed
/// at any time. This is still a work in progress.
///
#[derive(Clone, Debug, Copy)]
pub enum SpatialDims {
    Unspecified,
    One     (usize),
    Two     (usize, usize),
    Three   (usize, usize, usize),
}

impl SpatialDims {
    /// Returns a new `SpatialDims`.
    ///
    /// Dimensions must be specified in order from d0 -> d1 -> d2; i.e. `d1` 
    /// cannot be `Some(x)` if `d0` is `None`.
    pub fn new(d0: Option<usize>, d1: Option<usize>, d2: Option<usize>) -> OclResult<SpatialDims> {
        let std_err_msg = "Dimensions must be defined from left to right. If you define the 2nd \
            dimension, you must also define the 1st, etc.";

        if d2.is_some() { 
            if d1.is_some() && d0.is_some() {
                Ok(SpatialDims::Three(d0.unwrap(), d1.unwrap(), d2.unwrap()))
            } else {
                OclError::err(std_err_msg)
            }
        } else if d1.is_some() {
            if d0.is_some() {
                Ok(SpatialDims::Two(d1.unwrap(), d0.unwrap()))
            } else {
                OclError::err(std_err_msg)
            }
        } else if d0.is_some() {
            Ok(SpatialDims::One(d0.unwrap()))
        } else {
            Ok(SpatialDims::Unspecified)
        }
    }

    /// Returns the number of dimensions defined by this `SpatialDims`.
    pub fn dim_count(&self) -> u32 {
        match self {
            &SpatialDims::Unspecified => 0,
            &SpatialDims::Three(..) => 3,
            &SpatialDims::Two(..) => 2,
            &SpatialDims::One(..) => 1,
        }

    }

    /// Returns a 3D size or an error.
    pub fn to_size(&self) -> OclResult<[usize; 3]> {
        match self {
            &SpatialDims::Unspecified => Err(OclError::UnspecifiedDimensions),
            &SpatialDims::One(x) => Ok([x, 1, 1]),
            &SpatialDims::Two(x, y) => Ok([x, y, 1]),
            &SpatialDims::Three(x, y, z) => Ok([x, y, z]),
        }
    }

    /// Returns a 3D offset or an error.
    pub fn to_offset(&self) -> OclResult<[usize; 3]> {
        match self {
            &SpatialDims::Unspecified => Err(OclError::UnspecifiedDimensions),
            &SpatialDims::One(x) => Ok([x, 0, 0]),
            &SpatialDims::Two(x, y) => Ok([x, y, 0]),
            &SpatialDims::Three(x, y, z) => Ok([x, y, z]),
        }
    }

    /// Returns the product of all contained dimensional values (equivalent to
    /// a length, area, or volume depending on how many dimensions) or an
    /// error.
    pub fn to_len(&self) -> OclResult<usize> {
        match self {
            &SpatialDims::Unspecified => Err(OclError::UnspecifiedDimensions),
            &SpatialDims::Three(d0, d1, d2) => Ok(d0 * d1 * d2),
            &SpatialDims::Two(d0, d1) => Ok(d0 * d1),
            &SpatialDims::One(d0) => Ok(d0),
        }
    }


    // /// Returns a 3D size.
    // pub fn to_size(&self) -> [usize; 3] {
    //     // match self {
    //     //     &SpatialDims::Unspecified => 
    //     //     &SpatialDims::One(x) => [x, 1, 1],
    //     //     &SpatialDims::Two(x, y) => [x, y, 1],
    //     //     &SpatialDims::Three(x, y, z) => [x, y, z],
    //     // }
    //     self.to_size().expect("ocl::SpatialDims::to_size()")
    // }

    // /// Returns 3D offset.
    // pub fn to_offset(&self) -> [usize; 3] {
    //     // match self {
    //     //     &SpatialDims::Unspecified => [0, 0, 0],
    //     //     &SpatialDims::One(x) => [x, 0, 0],
    //     //     &SpatialDims::Two(x, y) => [x, y, 0],
    //     //     &SpatialDims::Three(x, y, z) => [x, y, z],
    //     // }
    //     self.try_to_offset().unwrap_or([0, 0, 0]);
    // }

    // /// Returns the product of all contained dimensional values (equivalent to
    // /// a length, area, or volume depending on how many dimensions).
    // pub fn to_len(&self) -> usize {
    //     self.try_to_len().unwrap_or(0)
    // }

    /// Takes the length and rounds it up to the nearest `incr` or an error.
    pub fn try_to_padded_len(&self, incr: usize) -> OclResult<usize> {
        Ok(util::padded_len(try!(self.to_len()), incr))
    }
}

impl MemDims for SpatialDims {
    fn padded_buffer_len(&self, incr: usize) -> OclResult<usize> {
        self.try_to_padded_len(incr)
    }
    fn to_size(&self) -> [usize; 3] { 
        self.to_size().expect("SpatialDims::<MemDims>::to_size()")
    }
}

impl WorkDims for SpatialDims {
    /// Returns the number of dimensions defined by this `SpatialDims`.
    fn dim_count(&self) -> u32 {
        self.dim_count()
    }

    fn to_work_size(&self) -> Option<[usize; 3]> {
        // match self {
        //     &SpatialDims::Unspecified => None,
        //     &SpatialDims::One(x) => Some([x, 1, 1]),
        //     &SpatialDims::Two(x, y) => Some([x, y, 1]),
        //     &SpatialDims::Three(x, y, z) => Some([x, y, z]),
        // }
        self.to_size().ok()
    }

    fn to_work_offset(&self) -> Option<[usize; 3]> {
        // match self {
        //     &SpatialDims::Unspecified => None,
        //     &SpatialDims::One(x) => Some([x, 0, 0]),
        //     &SpatialDims::Two(x, y) => Some([x, y, 0]),
        //     &SpatialDims::Three(x, y, z) => Some([x, y, z]),
        // }
        self.to_offset().ok()
    }
}


impl Index<usize> for SpatialDims {
    type Output = usize;

    fn index<'a>(&'a self, index: usize) -> &usize {
        match self {
            &SpatialDims::Unspecified => panic!("ocl::SpatialDims::index(): \
                Cannot index. No dimensions have been specified."),
            &SpatialDims::One(ref x) => {
                assert!(index == 0, "ocl::SpatialDims::index(): Index: [{}], out of range. \
                    Only one dimension avaliable.", index);
                x
            },
            &SpatialDims::Two(ref x, ref y) => {                
                match index {
                    0 => x,
                    1 => y,
                    _ => panic!("ocl::SpatialDims::index(): Index: [{}], out of range. \
                    Only two dimensions avaliable.", index),
                }
            },
            &SpatialDims::Three(ref x, ref y, ref z) => {
                match index {
                    0 => x,
                    1 => y,
                    2 => z,
                    _ => panic!("ocl::SpatialDims::index(): Index: [{}], out of range. \
                    Only three dimensions avaliable.", index),
                }
            },
        }
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<(T, )> for SpatialDims {
    fn from(val: (T, )) -> SpatialDims {
        SpatialDims::One(to_usize(val.0))
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a (T, )> for SpatialDims {
    fn from(val: &(T, )) -> SpatialDims {
        SpatialDims::One(to_usize(val.0))
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<[T; 1]> for SpatialDims {
    fn from(val: [T; 1]) -> SpatialDims {
        SpatialDims::One(to_usize(val[0]))
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a [T; 1]> for SpatialDims {
    fn from(val: &[T; 1]) -> SpatialDims {
        SpatialDims::One(to_usize(val[0]))
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<(T, T)> for SpatialDims {
    fn from(pair: (T, T)) -> SpatialDims {
        SpatialDims::Two(
            to_usize(pair.0), 
            to_usize(pair.1),
        )
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a (T, T)> for SpatialDims {
    fn from(pair: &(T, T)) -> SpatialDims {
        SpatialDims::Two(
            to_usize(pair.0), 
            to_usize(pair.1),
        )
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<[T; 2]> for SpatialDims {
    fn from(pair: [T; 2]) -> SpatialDims {
        SpatialDims::Two(
            to_usize(pair[0]), 
            to_usize(pair[1]),
        )
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a [T; 2]> for SpatialDims {
    fn from(pair: &[T; 2]) -> SpatialDims {
        SpatialDims::Two(
            to_usize(pair[0]), 
            to_usize(pair[1]),
        )
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<(T, T, T)> for SpatialDims {
    fn from(set: (T, T, T)) -> SpatialDims {
        SpatialDims::Three(
            to_usize(set.0), 
            to_usize(set.1), 
            to_usize(set.2),
        )
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a (T, T, T)> for SpatialDims {
    fn from(set: &(T, T, T)) -> SpatialDims {
        SpatialDims::Three(
            to_usize(set.0), 
            to_usize(set.1), 
            to_usize(set.2),
        )
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<[T; 3]> for SpatialDims {
    fn from(set: [T; 3]) -> SpatialDims {
        SpatialDims::Three(
            to_usize(set[0]), 
            to_usize(set[1]), 
            to_usize(set[2]),
        )
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a [T; 3]> for SpatialDims {
    fn from(set: &[T; 3]) -> SpatialDims {
        SpatialDims::Three(
            to_usize(set[0]), 
            to_usize(set[1]), 
            to_usize(set[2]),
        )
    }
}


#[inline]
pub fn to_usize<T: Num + ToPrimitive + Debug + Copy>(val: T) -> usize {
    val.to_usize().expect(&format!("Unable to convert the value '{:?}' into a 'SpatialDims'. \
        Dimensions must have positive values.", val))
}

