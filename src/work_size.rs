//! Defines the amount of work to be done by a kernel for each of up to three 
//! dimensions.
use std::ptr;
use libc::size_t;

/// Defines the amount of work to be done by a kernel for each of up to three 
/// dimensions.
///
/// [UNSTABLE]: MAY BE CONSOLIDATED WITH `SimpleDims`.
#[derive(PartialEq, Debug, Clone)]
pub enum WorkSize {
    Unspecified,
    OneDim      (usize),
    TwoDims     (usize, usize),
    ThreeDims   (usize, usize, usize),
}

impl WorkSize {
    /// Returns the number of dimensions defined by this `WorkSize`.
    pub fn dim_count(&self) -> u32 {
        match self {
            &WorkSize::ThreeDims(..)        => 3,
            &WorkSize::TwoDims(..)      => 2,
            &WorkSize::OneDim(..)       => 1,
            &WorkSize::Unspecified      => 0,
        }

    }

    /// Returns the amount work to be done in three dimensional terms.
    pub fn complete_worksize(&self) -> (usize, usize, usize) {
        match self {
            &WorkSize::OneDim(x) => (x, 1, 1),
            &WorkSize::TwoDims(x, y) => (x, y, 1),
            &WorkSize::ThreeDims(x, y, z) => (x, y, z),
            _ => (1, 1, 1)
        }
    }

    pub fn as_work_offset(&self) -> Option<[usize; 3]> {
        match self {
            &WorkSize::OneDim(x) => Some([x, 0, 0]),
            &WorkSize::TwoDims(x, y) => Some([x, y, 0]),
            &WorkSize::ThreeDims(x, y, z) => Some([x, y, z]),
            _ => None
        }
    }

    pub fn as_work_size(&self) -> Option<[usize; 3]> {
        match self {
            &WorkSize::OneDim(x) => Some([x, 1, 1]),
            &WorkSize::TwoDims(x, y) => Some([x, y, 1]),
            &WorkSize::ThreeDims(x, y, z) => Some([x, y, z]),
            _ => None
        }
    }

    /// Returns a raw pointer to the enum.
    pub fn as_ptr(&self) -> *const size_t {
        match self {
            &WorkSize::OneDim(x) => {
                &x as *const usize as *const size_t
            },
            &WorkSize::TwoDims(x, _) => {
                &x as *const usize as *const size_t
            },
            &WorkSize::ThreeDims(x, _, _) => {
                &x as *const usize as *const size_t
            },
            _ => ptr::null(),
        }
    }
}
