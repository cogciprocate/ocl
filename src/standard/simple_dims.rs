//! A simple way to specify the sizes of up to three dimensions.
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
#[derive(Clone, Debug)]
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

    /// Returns a `WorkDims` corresponding to the dimensions of this `SimpleDims`.
    pub fn work_dims(&self) -> WorkDims {
        match self {
            &SimpleDims::Three(d0, d1, d2) => WorkDims::ThreeDims(d0, d1, d2),
            &SimpleDims::Two(d0, d1) => WorkDims::TwoDims(d0, d1),
            &SimpleDims::One(d0) => WorkDims::OneDim(d0),
            _ => WorkDims::Unspecified,
        }
    }
}

impl BufferDims for SimpleDims {
    fn padded_buffer_len(&self, incr: usize) -> usize {
        let simple_len = match self {
            &SimpleDims::Three(d0, d1, d2) => d0 * d1 * d2,
            &SimpleDims::Two(d0, d1) => d0 * d1,
            &SimpleDims::One(d0) => d0,
            _ => 0,
        };

        util::padded_len(simple_len, incr)
    }
}
