//! A simple way to specify the sizes of up to three dimensions.
// use std::convert::Into;
use std::convert::From;
use std::fmt::Debug;
use num::{Num, ToPrimitive};
use error::{Result as OclResult, Error as OclError};
use standard::{BufferDims, WorkDims};
use util;

/// A simple implementation of a type specifying the sizes of up to three
/// dimensions. 
///
/// Custom types implementing `BufferDims` can and should be created
/// to express more complex relationships between buffer and work size.
///
/// [FIXME] TODO: Much more explaination needed as soon as conventions solidify.
/// [UNSTABLE]: MAY BE CONSOLIDATED WITH `WorkDims`.
#[derive(Clone, Debug, Copy)]
pub enum SimpleDims {
    Unspecified,
    One     (usize),
    Two     (usize, usize),
    Three   (usize, usize, usize),
}

impl SimpleDims {
    /// Returns a new `SimpleDims`.
    ///
    /// Dimensions must be specified in order from d0 -> d1 -> d2; i.e. `d1` 
    /// cannot be `Some(x)` if `d0` is `None`.
    pub fn new(d0: Option<usize>, d1: Option<usize>, d2: Option<usize>) -> OclResult<SimpleDims> {
        let std_err_msg = "Dimensions must be defined from left to right. If you define the 2nd \
            dimension, you must also define the 1st, etc.";

        if d2.is_some() { 
            if d1.is_some() && d0.is_some() {
                Ok(SimpleDims::Three(d0.unwrap(), d1.unwrap(), d2.unwrap()))
            } else {
                OclError::err(std_err_msg)
            }
        } else if d1.is_some() {
            if d0.is_some() {
                Ok(SimpleDims::Two(d1.unwrap(), d0.unwrap()))
            } else {
                OclError::err(std_err_msg)
            }
        } else if d0.is_some() {
            Ok(SimpleDims::One(d0.unwrap()))
        } else {
            Ok(SimpleDims::Unspecified)
        }
    }

    /// Returns the number of dimensions defined by this `SimpleDims`.
    pub fn dim_count(&self) -> u32 {
        match self {
            &SimpleDims::Unspecified => 0,
            &SimpleDims::Three(..) => 3,
            &SimpleDims::Two(..) => 2,
            &SimpleDims::One(..) => 1,
        }

    }

    pub fn to_size(&self) -> [usize; 3] {
        match self {
            &SimpleDims::Unspecified => [1, 1, 1],
            &SimpleDims::One(x) => [x, 1, 1],
            &SimpleDims::Two(x, y) => [x, y, 1],
            &SimpleDims::Three(x, y, z) => [x, y, z],
        }
    }

    pub fn to_offset(&self) -> [usize; 3] {
        match self {
            &SimpleDims::Unspecified => [0, 0, 0],
            &SimpleDims::One(x) => [x, 0, 0],
            &SimpleDims::Two(x, y) => [x, y, 0],
            &SimpleDims::Three(x, y, z) => [x, y, z],
        }
    }    
}

impl BufferDims for SimpleDims {
    fn padded_buffer_len(&self, incr: usize) -> usize {
        let simple_len = match self {
            &SimpleDims::Unspecified => 0,
            &SimpleDims::Three(d0, d1, d2) => d0 * d1 * d2,
            &SimpleDims::Two(d0, d1) => d0 * d1,
            &SimpleDims::One(d0) => d0,
        };

        util::padded_len(simple_len, incr)
    }
}

impl WorkDims for SimpleDims {
    /// Returns the number of dimensions defined by this `SimpleDims`.
    fn dim_count(&self) -> u32 {
        self.dim_count()
    }

    fn to_work_size(&self) -> Option<[usize; 3]> {
        match self {
            &SimpleDims::Unspecified => None,
            &SimpleDims::One(x) => Some([x, 1, 1]),
            &SimpleDims::Two(x, y) => Some([x, y, 1]),
            &SimpleDims::Three(x, y, z) => Some([x, y, z]),
        }
    }

    fn to_work_offset(&self) -> Option<[usize; 3]> {
        match self {
            &SimpleDims::Unspecified => None,
            &SimpleDims::One(x) => Some([x, 0, 0]),
            &SimpleDims::Two(x, y) => Some([x, y, 0]),
            &SimpleDims::Three(x, y, z) => Some([x, y, z]),
        }
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<(T, )> for SimpleDims {
    fn from(val: (T, )) -> SimpleDims {
        SimpleDims::One(to_usize(val.0))
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a (T, )> for SimpleDims {
    fn from(val: &(T, )) -> SimpleDims {
        SimpleDims::One(to_usize(val.0))
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<[T; 1]> for SimpleDims {
    fn from(val: [T; 1]) -> SimpleDims {
        SimpleDims::One(to_usize(val[0]))
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a [T; 1]> for SimpleDims {
    fn from(val: &[T; 1]) -> SimpleDims {
        SimpleDims::One(to_usize(val[0]))
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<(T, T)> for SimpleDims {
    fn from(pair: (T, T)) -> SimpleDims {
        SimpleDims::Two(
            to_usize(pair.0), 
            to_usize(pair.1),
        )
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a (T, T)> for SimpleDims {
    fn from(pair: &(T, T)) -> SimpleDims {
        SimpleDims::Two(
            to_usize(pair.0), 
            to_usize(pair.1),
        )
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<[T; 2]> for SimpleDims {
    fn from(pair: [T; 2]) -> SimpleDims {
        SimpleDims::Two(
            to_usize(pair[0]), 
            to_usize(pair[1]),
        )
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a [T; 2]> for SimpleDims {
    fn from(pair: &[T; 2]) -> SimpleDims {
        SimpleDims::Two(
            to_usize(pair[0]), 
            to_usize(pair[1]),
        )
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<(T, T, T)> for SimpleDims {
    fn from(set: (T, T, T)) -> SimpleDims {
        SimpleDims::Three(
            to_usize(set.0), 
            to_usize(set.1), 
            to_usize(set.2),
        )
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a (T, T, T)> for SimpleDims {
    fn from(set: &(T, T, T)) -> SimpleDims {
        SimpleDims::Three(
            to_usize(set.0), 
            to_usize(set.1), 
            to_usize(set.2),
        )
    }
}

impl<T: Num + ToPrimitive + Debug + Copy> From<[T; 3]> for SimpleDims {
    fn from(set: [T; 3]) -> SimpleDims {
        SimpleDims::Three(
            to_usize(set[0]), 
            to_usize(set[1]), 
            to_usize(set[2]),
        )
    }
}

impl<'a, T: Num + ToPrimitive + Debug + Copy> From<&'a [T; 3]> for SimpleDims {
    fn from(set: &[T; 3]) -> SimpleDims {
        SimpleDims::Three(
            to_usize(set[0]), 
            to_usize(set[1]), 
            to_usize(set[2]),
        )
    }
}

#[inline]
pub fn to_usize<T: Num + ToPrimitive + Debug + Copy>(val: T) -> usize {
    val.to_usize().expect(&format!("Unable to convert the value '{:?}' into a SimpleDims. \
        Dimensions must have positive values.", val))
}

