//! Utility and debugging functions.
//!
//! ## Stability
//!
//! Printing functions may be moved/renamed/removed at any time.
use crate::{OclPrm, OclScl};
use num_traits::PrimInt;
use std::iter;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::string::FromUtf8Error;

//=============================================================================
//================================= MACROS ====================================
//=============================================================================

//=============================================================================
//================================ STATICS ====================================
//=============================================================================

pub mod colors {
    //! ASCII Color Palette
    //!
    //! Used for printing functions.
    //
    // TODO: Remove or feature gate printing related code.

    pub static TAB: &'static str = "    ";

    pub static C_DEFAULT: &'static str = "\x1b[0m";
    pub static C_UNDER: &'static str = "\x1b[1m";

    // 30–37
    pub static C_RED: &'static str = "\x1b[31m";
    pub static C_BRED: &'static str = "\x1b[1;31m";
    pub static C_GRN: &'static str = "\x1b[32m";
    pub static C_BGRN: &'static str = "\x1b[1;32m";
    pub static C_ORA: &'static str = "\x1b[33m";
    pub static C_DBL: &'static str = "\x1b[34m";
    pub static C_PUR: &'static str = "\x1b[35m";
    pub static C_CYA: &'static str = "\x1b[36m";
    pub static C_LGR: &'static str = "\x1b[37m";
    // [ADDME] 38: Extended Colors
    // pub static C_EXT38: &'static str = "\x1b[38m";
    pub static C_DFLT: &'static str = "\x1b[39m";

    // 90-97
    pub static C_DGR: &'static str = "\x1b[90m";
    pub static C_LRD: &'static str = "\x1b[91m";
    pub static C_YEL: &'static str = "\x1b[93m";
    pub static C_BLU: &'static str = "\x1b[94m";
    pub static C_LBL: &'static str = "\x1b[94m";
    pub static C_MAG: &'static str = "\x1b[95m";
    // [ADDME] 38: Extended Colors
    // pub static C_EXT38: &'static str = "\x1b[38m";

    pub static BGC_DEFAULT: &'static str = "\x1b[49m";
    pub static BGC_GRN: &'static str = "\x1b[42m";
    pub static BGC_PUR: &'static str = "\x1b[45m";
    pub static BGC_LGR: &'static str = "\x1b[47m";
    pub static BGC_DGR: &'static str = "\x1b[100m";
}

//=============================================================================
//=========================== UTILITY FUNCTIONS ===============================
//=============================================================================

/// An error caused by a utility function.
#[derive(Debug, thiserror::Error)]
pub enum UtilError {
    #[error(
        "The size of the source byte slice ({src} bytes) does not match \
        the size of the destination type ({dst} bytes)."
    )]
    BytesTo { src: usize, dst: usize },
    #[error(
        "The size of the source byte vector ({src} bytes) does not match \
        the size of the destination type ({dst} bytes)."
    )]
    BytesInto { src: usize, dst: usize },
    #[error(
        "The size of the source byte vector ({src} bytes) is not evenly \
        divisible by the size of the destination type ({dst} bytes)."
    )]
    BytesIntoVec { src: usize, dst: usize },
    #[error(
        "The size of the source byte slice ({src} bytes) is not evenly \
        divisible by the size of the destination type ({dst} bytes)."
    )]
    BytesToVec { src: usize, dst: usize },
    #[error("Unable to convert bytes into string: {0}")]
    BytesIntoString(#[from] FromUtf8Error),
}

/// Copies a byte slice to a new `u32`.
///
/// ### Stability
///
/// May depricate in favor of `bytes_to`
///
pub fn bytes_to_u32(bytes: &[u8]) -> u32 {
    debug_assert!(bytes.len() == 4);

    u32::from(bytes[0])
        | (u32::from(bytes[1]) << 8)
        | (u32::from(bytes[2]) << 16)
        | (u32::from(bytes[3]) << 24)
}

/// Copies a slice of bytes to a new value of arbitrary type.
///
/// ### Safety
///
/// You may want to wear a helmet.
///
pub unsafe fn bytes_to<T>(bytes: &[u8]) -> Result<T, UtilError> {
    if mem::size_of::<T>() == bytes.len() {
        let mut new_val = mem::MaybeUninit::<T>::uninit();
        ptr::copy(bytes.as_ptr(), new_val.as_mut_ptr() as *mut u8, bytes.len());
        Ok(new_val.assume_init())
    } else {
        Err(UtilError::BytesTo {
            src: bytes.len(),
            dst: mem::size_of::<T>(),
        })
    }
}

/// Converts a vector of bytes into a value of arbitrary type.
///
/// ### Safety
///
/// Roughly equivalent to a weekend in Tijuana.
///
// [NOTE]: Not sure this is the best or simplest way to do this but whatever.
// Would be nice to not even have to copy anything and just basically
// transmute the vector into the result type. [TODO]: Fiddle with this
// at some point.
//
pub unsafe fn bytes_into<T>(vec: Vec<u8>) -> Result<T, UtilError> {
    if mem::size_of::<T>() == vec.len() {
        let mut new_val = mem::MaybeUninit::<T>::uninit();
        ptr::copy(vec.as_ptr(), new_val.as_mut_ptr() as *mut u8, vec.len());
        Ok(new_val.assume_init())
    } else {
        Err(UtilError::BytesInto {
            src: vec.len(),
            dst: mem::size_of::<T>(),
        })
    }
}

/// Converts a vector of bytes into a vector of arbitrary type.
///
/// ### Safety
///
/// Ummm... Say what?
///
/// TODO: Consider using `alloc::heap::reallocate_inplace` equivalent.
///
pub unsafe fn bytes_into_vec<T>(mut vec: Vec<u8>) -> Result<Vec<T>, UtilError> {
    // debug_assert!(vec.len() % mem::size_of::<T>() == 0);
    if vec.len() % mem::size_of::<T>() == 0 {
        let new_len = vec.len() / mem::size_of::<T>();
        let new_cap = vec.capacity() / mem::size_of::<T>();
        let ptr = vec.as_mut_ptr();
        mem::forget(vec);
        let mut new_vec: Vec<T> = Vec::from_raw_parts(ptr as *mut T, new_len, new_cap);
        new_vec.shrink_to_fit();
        Ok(new_vec)
    } else {
        Err(UtilError::BytesIntoVec {
            src: vec.len(),
            dst: mem::size_of::<T>(),
        })
    }
}

/// Copies a slice of bytes into a vector of arbitrary type.
///
/// ### Safety
///
/// Negative.
///
pub unsafe fn bytes_to_vec<T>(bytes: &[u8]) -> Result<Vec<T>, UtilError> {
    // debug_assert!(bytes.len() % mem::size_of::<T>() == 0);
    if bytes.len() % mem::size_of::<T>() == 0 {
        let new_len = bytes.len() / mem::size_of::<T>();
        let mut new_vec: Vec<T> = Vec::with_capacity(new_len);
        ptr::copy(
            bytes.as_ptr(),
            new_vec.as_mut_ptr() as *mut _ as *mut u8,
            bytes.len(),
        );
        new_vec.set_len(new_len);
        Ok(new_vec)
    } else {
        Err(UtilError::BytesToVec {
            src: bytes.len(),
            dst: mem::size_of::<T>(),
        })
    }
}

/// Converts a byte Vec into a string, removing the trailing null byte if it
/// exists.
pub fn bytes_into_string(mut bytes: Vec<u8>) -> Result<String, UtilError> {
    if bytes.last() == Some(&0u8) {
        bytes.pop();
    }

    String::from_utf8(bytes)
        .map(|str| String::from(str.trim()))
        .map_err(UtilError::BytesIntoString)
}

/// [UNTESTED] Copies an arbitrary primitive or struct into core bytes.
///
/// ### Depth
///
/// This is not a deep copy, will only copy the surface of primitives, structs,
/// etc. Not 100% sure about what happens with other types but should copy
/// everything zero levels deep.
///
/// ### Endianness
///
/// 98% sure (speculative) this will always be correct due to the driver
/// automatically taking it into account.
///
/// ### Safety
///
/// Don't ask.
///
/// [FIXME]: Evaluate the ins and outs of this and lock this down a bit.
pub unsafe fn into_bytes<T>(val: T) -> Vec<u8> {
    // let big_endian = false;
    let size = mem::size_of::<T>();
    let mut new_vec: Vec<u8> = iter::repeat(0).take(size).collect();

    ptr::copy(&val as *const _ as *const u8, new_vec.as_mut_ptr(), size);

    // if big_endian {
    //     new_vec = new_vec.into_iter().rev().collect();
    // }

    new_vec
}

/// Pads `len` to make it evenly divisible by `incr`.
pub fn padded_len(len: usize, incr: usize) -> usize {
    let len_mod = len % incr;

    if len_mod == 0 {
        len
    } else {
        let pad = incr - len_mod;
        let padded_len = len + pad;
        debug_assert_eq!(padded_len % incr, 0);
        padded_len
    }
}

/// An error caused by `util::vec_remove_rebuild`.
#[derive(thiserror::Error, Debug)]
pub enum VecRemoveRebuildError {
    #[error("Remove list is longer than source vector.")]
    TooLong,
    #[error(
        "'remove_list' contains at least one out of range index: [{idx}] \
        ('orig_vec' length: {orig_len})."
    )]
    OutOfRange { idx: usize, orig_len: usize },
}

/// Batch removes elements from a vector using a list of indices to remove.
///
/// Will create a new vector and do a streamlined rebuild if
/// `remove_list.len()` > `rebuild_threshold`. Threshold should typically be
/// set very low (less than probably 5 or 10) as it's expensive to remove one
/// by one.
///
pub fn vec_remove_rebuild<T: Clone + Copy>(
    orig_vec: &mut Vec<T>,
    remove_list: &[usize],
    rebuild_threshold: usize,
) -> Result<(), VecRemoveRebuildError> {
    if remove_list.len() > orig_vec.len() {
        return Err(VecRemoveRebuildError::TooLong);
    }
    let orig_len = orig_vec.len();

    // If the list is below threshold
    if remove_list.len() <= rebuild_threshold {
        for &idx in remove_list.iter().rev() {
            if idx < orig_len {
                orig_vec.remove(idx);
            } else {
                return Err(VecRemoveRebuildError::OutOfRange { idx, orig_len });
            }
        }
    } else {
        unsafe {
            let mut remove_markers: Vec<bool> = iter::repeat(true).take(orig_len).collect();

            // Build a sparse list of which elements to remove:
            for &idx in remove_list.iter() {
                if idx < orig_len {
                    *remove_markers.get_unchecked_mut(idx) = false;
                } else {
                    return Err(VecRemoveRebuildError::OutOfRange { idx, orig_len });
                }
            }

            let mut new_len = 0usize;

            // Iterate through remove_markers and orig_vec, pushing when the marker is false:
            for idx in 0..orig_len {
                if *remove_markers.get_unchecked(idx) {
                    *orig_vec.get_unchecked_mut(new_len) = *orig_vec.get_unchecked(idx);
                    new_len += 1;
                }
            }

            debug_assert_eq!(new_len, orig_len - remove_list.len());
            orig_vec.set_len(new_len);
        }
    }

    Ok(())
}

/// Wraps (`%`) each value in the list `vals` if it equals or exceeds `val_n`.
pub fn wrap_vals<T: OclPrm + PrimInt>(vals: &[T], val_n: T) -> Vec<T> {
    vals.iter().map(|&v| v % val_n).collect()
}

// /// Converts a length in `T` to a size in bytes.
// #[inline]
// pub fn len_to_size<T>(len: usize) -> usize {
//     len * mem::size_of::<T>()
// }

// /// Converts lengths in `T` to sizes in bytes for a `[usize; 3]`.
// #[inline]
// pub fn len3_to_size3<T>(lens: [usize; 3]) -> [usize; 3] {
//     [len_to_size::<T>(lens[0]), len_to_size::<T>(lens[1]), len_to_size::<T>(lens[2])]
// }

// /// Converts lengths in `T` to sizes in bytes for a `&[usize]`.
// pub fn lens_to_sizes<T>(lens: &[usize]) -> Vec<usize> {
//     lens.iter().map(|len| len * mem::size_of::<T>()).collect()
// }

//=============================================================================
//=========================== PRINTING FUNCTIONS ==============================
//=============================================================================

/// Prints bytes as hex.
pub fn print_bytes_as_hex(bytes: &[u8]) {
    print!("0x");

    for &byte in bytes.iter() {
        print!("{:x}", byte);
    }
}

#[allow(unused_assignments, unused_variables)]
/// [UNSTABLE]: MAY BE REMOVED AT ANY TIME
/// Prints a vector to stdout. Used for debugging.
//
// TODO: Remove or feature gate printing related code.
//
pub fn print_slice<T: OclScl>(
    vec: &[T],
    every: usize,
    val_range: Option<(T, T)>,
    idx_range: Option<Range<usize>>,
    show_zeros: bool,
) {
    print!(
        "{cdgr}[{cg}{}{cdgr}/{}",
        vec.len(),
        every,
        cg = colors::C_GRN,
        cdgr = colors::C_DGR
    );

    let (vr_start, vr_end) = match val_range {
        Some(vr) => {
            print!(";({}-{})", vr.0, vr.1);
            vr
        }
        None => (Default::default(), Default::default()),
    };

    let (ir_start, ir_end) = match idx_range {
        Some(ref ir) => {
            print!(";[{}..{}]", ir.start, ir.end);
            (ir.start, ir.end)
        }
        None => (0usize, 0usize),
    };

    print!("]:{cd} ", cd = colors::C_DEFAULT,);

    let mut ttl_nz = 0usize;
    let mut ttl_ir = 0usize;
    let mut within_idx_range = true;
    let mut within_val_range = true;
    let mut hi: T = vr_start;
    let mut lo: T = vr_end;
    let mut sum: i64 = 0;
    let mut ttl_prntd: usize = 0;
    let len = vec.len();

    let mut color: &'static str = colors::C_DEFAULT;
    let mut prnt: bool = false;

    // Yes, this clusterfuck needs rewriting someday
    for (i, item) in vec.iter().enumerate() {
        prnt = false;

        if every != 0 {
            if i % every == 0 {
                prnt = true;
            } else {
                prnt = false;
            }
        }

        if idx_range.is_some() {
            let ir = idx_range.as_ref().expect("ocl::buffer::print_vec()");

            if i < ir_start || i >= ir_end {
                prnt = false;
                within_idx_range = false;
            } else {
                within_idx_range = true;
            }
        } else {
            within_idx_range = true;
        }

        if val_range.is_some() {
            if *item < vr_start || *item > vr_end {
                prnt = false;
                within_val_range = false;
            } else {
                if within_idx_range {
                    // if *item == Default::default() {
                    //     ttl_ir += 1;
                    // } else {
                    //     ttl_ir += 1;
                    // }
                    ttl_ir += 1;
                }

                within_val_range = true;
            }
        }

        if within_idx_range && within_val_range {
            sum += item.to_i64().expect("ocl::buffer::print_vec(): vec[i]");

            if *item > hi {
                hi = *item
            };

            if *item < lo {
                lo = *item
            };

            if vec[i] != Default::default() {
                ttl_nz += 1usize;
                color = colors::C_ORA;
            } else if show_zeros {
                color = colors::C_DEFAULT;
            } else {
                prnt = false;
            }
        }

        if prnt {
            print!(
                "{cg}[{cd}{}{cg}:{cc}{}{cg}]{cd}",
                i,
                vec[i],
                cc = color,
                cd = colors::C_DEFAULT,
                cg = colors::C_DGR
            );
            ttl_prntd += 1;
        }
    }

    let mut anz: f32 = 0f32;
    let mut nz_pct: f32 = 0f32;

    let mut ir_pct: f32 = 0f32;
    let mut avg_ir: f32 = 0f32;

    if ttl_nz > 0 {
        anz = sum as f32 / ttl_nz as f32;
        nz_pct = (ttl_nz as f32 / len as f32) * 100f32;
        //print!( "[ttl_nz: {}, nz_pct: {:.0}%, len: {}]", ttl_nz, nz_pct, len);
    }

    if ttl_ir > 0 {
        avg_ir = sum as f32 / ttl_ir as f32;
        ir_pct = (ttl_ir as f32 / len as f32) * 100f32;
        //print!( "[ttl_nz: {}, nz_pct: {:.0}%, len: {}]", ttl_nz, nz_pct, len);
    }

    println!(
        "{cdgr}; (nz:{clbl}{}{cdgr}({clbl}{:.2}%{cdgr}),\
        ir:{clbl}{}{cdgr}({clbl}{:.2}%{cdgr}),hi:{},lo:{},anz:{:.2},prntd:{}){cd} ",
        ttl_nz,
        nz_pct,
        ttl_ir,
        ir_pct,
        hi,
        lo,
        anz,
        ttl_prntd,
        cd = colors::C_DEFAULT,
        clbl = colors::C_LBL,
        cdgr = colors::C_DGR
    );
}

pub fn print_simple<T: OclScl>(slice: &[T]) {
    print_slice(slice, 1, None, None, true);
}

pub fn print_val_range<T: OclScl>(slice: &[T], every: usize, val_range: Option<(T, T)>) {
    print_slice(slice, every, val_range, None, true);
}

#[cfg(test)]
mod tests {
    // use std::iter;

    #[test]
    fn remove_rebuild() {
        let mut primary_vals: Vec<u32> = (0..(1 << 18)).map(|v| v).collect();
        let orig_len = primary_vals.len();

        let mut bad_indices: Vec<usize> = Vec::<usize>::with_capacity(1 << 16);
        let mut idx = 0;

        // Mark every whateverth value 'bad':
        for &val in primary_vals.iter() {
            if (val % 19 == 0) || (val % 31 == 0) || (val % 107 == 0) {
                bad_indices.push(idx);
            }

            idx += 1;
        }

        println!(
            "util::tests::remove_rebuild(): bad_indices: {}",
            bad_indices.len()
        );

        // Remove the bad values:
        super::vec_remove_rebuild(&mut primary_vals, &bad_indices[..], 3)
            .expect("util::tests::remove_rebuild()");

        // Check:
        for &val in primary_vals.iter() {
            if (val % 19 == 0) || (val % 31 == 0) || (val % 107 == 0) {
                panic!(
                    "util::tests::remove_rebuild(): Value: '{}' found in list!",
                    val
                );
            }
        }

        assert_eq!(orig_len, primary_vals.len() + bad_indices.len());
    }
}
