//! Defines the amount of work to be done by a kernel for each of up to three 
//! dimensions.
// use std::ptr;
// use libc::size_t;

/// Defines the amount of work to be done by a kernel for each of up to three 
/// dimensions.
///
/// [UNSTABLE]: MAY BE CONSOLIDATED WITH `SimpleDims`.
#[derive(PartialEq, Debug, Clone)]
pub enum WorkDims {
    Unspecified,
    OneDim      (usize),
    TwoDims     (usize, usize),
    ThreeDims   (usize, usize, usize),
}

impl WorkDims {
    /// Returns the number of dimensions defined by this `WorkDims`.
    pub fn dim_count(&self) -> u32 {
        match self {
            &WorkDims::ThreeDims(..)        => 3,
            &WorkDims::TwoDims(..)      => 2,
            &WorkDims::OneDim(..)       => 1,
            &WorkDims::Unspecified      => 0,
        }

    }

    // /// Returns the amount work to be done in three dimensional terms.
    // pub fn complete_worksize(&self) -> (usize, usize, usize) {
    //     match self {
    //         &WorkDims::OneDim(x) => (x, 1, 1),
    //         &WorkDims::TwoDims(x, y) => (x, y, 1),
    //         &WorkDims::ThreeDims(x, y, z) => (x, y, z),
    //         _ => (1, 1, 1)
    //     }
    // }

    pub fn as_core(&self) -> Option<[usize; 3]> {
        match self {
            &WorkDims::OneDim(x) => Some([x, 0, 0]),
            &WorkDims::TwoDims(x, y) => Some([x, y, 0]),
            &WorkDims::ThreeDims(x, y, z) => Some([x, y, z]),
            _ => None
        }
    }

    // pub fn as_work_dims(&self) -> Option<[usize; 3]> {
    //     match self {
    //         &WorkDims::OneDim(x) => Some([x, 1, 1]),
    //         &WorkDims::TwoDims(x, y) => Some([x, y, 1]),
    //         &WorkDims::ThreeDims(x, y, z) => Some([x, y, z]),
    //         _ => None
    //     }
    // }

    // /// Returns a core pointer to the enum.
    // pub fn as_ptr(&self) -> *const size_t {
    //     match self {
    //         &WorkDims::OneDim(x) => {
    //             &x as *const usize as *const size_t
    //         },
    //         &WorkDims::TwoDims(x, _) => {
    //             &x as *const usize as *const size_t
    //         },
    //         &WorkDims::ThreeDims(x, _, _) => {
    //             &x as *const usize as *const size_t
    //         },
    //         _ => ptr::null(),
    //     }
    // }
}
