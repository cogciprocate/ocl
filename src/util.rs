//! Utility and debugging functions.
//!
//! Largely untested.
//! 
//! ## Stability
//!
//! Printing functions may be moved/renamed/removed at any time.
use std::ops::Range;
use std::mem;
use std::ptr;
use std::iter;
use num::{Integer, FromPrimitive, ToPrimitive};
use rand;
use rand::distributions::{IndependentSample, Range as RandRange};
use error::{Result as OclResult, Error as OclError};

use core::OclPrm;

//=============================================================================
//================================= MACROS ====================================
//=============================================================================

// /// `print!` with all the glorious colors of the the ANSI rainbow.
// ///
// /// Watch out for the leprechaun at the end of that rainbow. Seriously.
// ///
// /// #### Usage
// /// 
// /// `printc!(yellow: "Number of banana peels on head: {}", hat_height);`
// ///
// /// See [`colorify!` docs](/ocl/ocl/macro.colorify!.html)
// /// for a current list of colors.
// #[macro_export]
// macro_rules! printc {
//     ($c:ident: $fmt:expr) => ( print!(colorify!($c: $fmt)) );
//     ($c:ident: $fmt:expr, $($arg:tt)*) => ( print!(colorify!($c: $fmt), $($arg)*) );
// }

// /// `println!` with color.
// ///
// /// #### Usage
// /// 
// /// `printlnc!(orange: "Number of baggies filled while walking dogs: {}", bag_count);`
// ///
// /// See [`colorify!` docs](/ocl/ocl/macro.colorify!.html)
// /// for a current list of colors.
// #[macro_export]
// macro_rules! printlnc {
//     ($c:ident: $fmt:expr) => ( print!(concat!(colorify!($c: $fmt), "\n")) );
//     ($c:ident: $fmt:expr, $($arg:tt)*) => ( print!(concat!(colorify!($c: $fmt), "\n"), $($arg)*) );
// }

// /// Adds color to a formatting literal.
// ///
// /// #### Usage
// /// 
// /// `writeln!(fmtr, colorify!(red: "Number of zombies killed: {}"), zombie_kills);`
// ///
// #[macro_export]
// macro_rules! colorify {
//     (default: $s:expr) => ( concat!("\x1b[0m", $s, "\x1b[0m") );
//     (red: $s:expr) => ( concat!("\x1b[31m", $s, "\x1b[0m") );
//     (red_bold: $s:expr) => ( concat!("\x1b[1;31m", $s, "\x1b[0m") );
//     (green: $s:expr) => ( concat!("\x1b[32m", $s, "\x1b[0m") );
//     (green_bold: $s:expr) => ( concat!("\x1b[1;32m", $s, "\x1b[0m") );
//     (orange: $s:expr) => ( concat!("\x1b[33m", $s, "\x1b[0m") );
//     (yellow_bold: $s:expr) => ( concat!("\x1b[1;33m", $s, "\x1b[0m") );
//     (blue: $s:expr) => ( concat!("\x1b[34m", $s, "\x1b[0m") );
//     (blue_bold: $s:expr) => ( concat!("\x1b[1;34m", $s, "\x1b[0m") );
//     (purple: $s:expr) => ( concat!("\x1b[35m", $s, "\x1b[0m") );
//     (purple_bold: $s:expr) => ( concat!("\x1b[1;35m", $s, "\x1b[0m") );
//     (cyan: $s:expr) => ( concat!("\x1b[36m", $s, "\x1b[0m") );
//     (cyan_bold: $s:expr) => ( concat!("\x1b[1;36m", $s, "\x1b[0m") );
//     (light_grey: $s:expr) => ( concat!("\x1b[37m", $s, "\x1b[0m") );
//     (white_bold: $s:expr) => ( concat!("\x1b[1;37m", $s, "\x1b[0m") );
//     (dark_grey: $s:expr) => ( concat!("\x1b[90m", $s, "\x1b[0m") );
//     (dark_grey_bold: $s:expr) => ( concat!("\x1b[1;90m", $s, "\x1b[0m") );
//     (peach: $s:expr) => ( concat!("\x1b[91m", $s, "\x1b[0m") );
//     (peach_bold: $s:expr) => ( concat!("\x1b[1;91m", $s, "\x1b[0m") );
//     (lime: $s:expr) => ( concat!("\x1b[92m", $s, "\x1b[0m") );
//     (lime_bold: $s:expr) => ( concat!("\x1b[1;92m", $s, "\x1b[0m") );
//     (yellow: $s:expr) => ( concat!("\x1b[93m", $s, "\x1b[0m") );
//     (yellow_bold2: $s:expr) => ( concat!("\x1b[1;93m", $s, "\x1b[0m") );
//     (royal_blue: $s:expr) => ( concat!("\x1b[94m", $s, "\x1b[0m") );
//     (royal_blue_bold: $s:expr) => ( concat!("\x1b[1;94m", $s, "\x1b[0m") );
//     (magenta: $s:expr) => ( concat!("\x1b[95m", $s, "\x1b[0m") );
//     (magenta_bold: $s:expr) => ( concat!("\x1b[1;95m", $s, "\x1b[0m") );
//     (teal: $s:expr) => ( concat!("\x1b[96m", $s, "\x1b[0m") );
//     (teal_bold: $s:expr) => ( concat!("\x1b[1;96m", $s, "\x1b[0m") );
//     (white: $s:expr) => ( concat!("\x1b[97m", $s, "\x1b[0m") );
//     (white_bold2: $s:expr) => ( concat!("\x1b[1;97m", $s, "\x1b[0m") );
// }

// #[macro_export]
// macro_rules! printy {
//     ($fmt:expr) => ( print!(yellowify!($fmt)) );
//     ($fmt:expr, $($arg:tt)*) => ( print!(yellowify!($fmt), $($arg)*) );
// }

// #[macro_export]
// macro_rules! yellowify {
//     ($s:expr) => (concat!("\x1b[93m", $s, "\x1b[0m"));
// }

//=============================================================================
//================================ STATICS ====================================
//=============================================================================

pub mod colors {
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

/// Copies a byte slice to a new `u32`.
///
/// ### Stability
///
/// May depricate in favor of `bytes_to`
///
pub fn bytes_to_u32(bytes: &[u8]) -> u32 {
    debug_assert!(bytes.len() == 4);
    
    bytes[0] as u32 | 
    ((bytes[1] as u32) << 8) |
    ((bytes[2] as u32) << 16) |
    ((bytes[3] as u32) << 24)
}

/// Converts a vector of bytes into a value of arbitrary type.
///
/// ### Safety
///
/// Roughly equivalent to a weekend in Tijuana.
///
// [NOTE]: Not sure this is the best or simplest way to do this but whatever.
// Would be nice to not even have to copy anything and just basically 
// transmute the vector into the result type. TODO: Fiddle with this 
// at some point. 
//
pub unsafe fn bytes_into<T>(vec: Vec<u8>) -> T {
    let byte_count = mem::size_of::<u8>() * vec.len();
    assert_eq!(mem::size_of::<T>(), byte_count);

    let mut new_val: T = mem::uninitialized();

    ptr::copy(vec.as_ptr(), &mut new_val as *mut _ as *mut u8, byte_count);

    new_val
}

/// Copies a slice of bytes to a new value of arbitrary type.
///
/// ### Safety
///
/// You may want to wear a helmet.
///
pub unsafe fn bytes_to<T>(bytes: &[u8]) -> T {
    let byte_count = mem::size_of::<u8>() * bytes.len();
    assert_eq!(mem::size_of::<T>(), byte_count);

    let mut new_val: T = mem::uninitialized();

    ptr::copy(bytes.as_ptr(), &mut new_val as *mut _ as *mut u8, byte_count);

    new_val
}

/// Converts a vector of bytes into a vector of arbitrary type.
///
/// ### Safety
///
/// Ummm... Say what?
///
/// TODO: Consider using alloc::heap::reallocate_inplace` equivalent.
///
pub unsafe fn bytes_into_vec<T>(mut vec: Vec<u8>) -> Vec<T> {
    debug_assert!(vec.len() % mem::size_of::<T>() == 0);
    let new_len = vec.len() / mem::size_of::<T>();
    let new_cap = vec.capacity() / mem::size_of::<T>();
    let ptr = vec.as_mut_ptr();
    mem::forget(vec);
    let mut new_vec: Vec<T> = Vec::from_raw_parts(ptr as *mut T, new_len, new_cap);
    new_vec.shrink_to_fit();
    new_vec
}

/// Copies a slice of bytes into a vector of arbitrary type.
///
/// ### Safety
///
/// Negative.
///
pub unsafe fn bytes_to_vec<T>(bytes: &[u8]) -> Vec<T> {
    debug_assert!(bytes.len() % mem::size_of::<T>() == 0);
    let new_len = bytes.len() / mem::size_of::<T>();
    let mut new_vec: Vec<T> = Vec::with_capacity(new_len);
    ptr::copy(bytes.as_ptr(), new_vec.as_mut_ptr() as *mut _ as *mut u8, bytes.len());
    new_vec.set_len(new_len);
    new_vec
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

/// Batch removes elements from a vector using a list of indices to remove.
///
/// Will create a new vector and do a streamlined rebuild if 
/// `remove_list.len()` > `rebuild_threshold`. Threshold should typically be
/// set very low (less than probably 5 or 10) as it's expensive to remove one 
/// by one.
///
/// ### Safety
///
/// Should be perfectly safe, just need to test it a bit.
///
pub fn vec_remove_rebuild<T: Clone + Copy>(orig_vec: &mut Vec<T>, remove_list: &[usize], 
                rebuild_threshold: usize) -> OclResult<()> {
    if remove_list.len() > orig_vec.len() { 
        return OclError::err("ocl::util::vec_remove_rebuild: Remove list is longer than source
            vector.");
    }
    let orig_len = orig_vec.len();

    // If the list is below threshold
    if remove_list.len() <= rebuild_threshold {
        for &idx in remove_list.iter().rev() {
            if idx < orig_len { 
                 orig_vec.remove(idx);
            } else {
                return OclError::err(format!("ocl::util::remove_rebuild_vec: 'remove_list' contains
                at least one out of range index: [{}] ('orig_vec.len()': {}).", idx, orig_len));
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
                    return OclError::err(format!("ocl::util::remove_rebuild_vec: 'remove_list' contains
                    at least one out of range index: [{}] ('orig_vec.len()': {}).", idx, orig_len));
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
pub fn wrap_vals<T: OclPrm + Integer>(vals: &[T], val_n: T) -> Vec<T> {
    vals.iter().map(|&v| v % val_n).collect()
}



/// Returns a vector with length `size` containing random values in the (half-open)
/// range `[vals.0, vals.1)`.
pub fn scrambled_vec<T: OclPrm>(vals: (T, T), size: usize) -> Vec<T> {
    assert!(size > 0, "\nbuffer::shuffled_vec(): Vector size must be greater than zero.");
    assert!(vals.0 < vals.1, "\nbuffer::shuffled_vec(): Minimum value must be less than maximum.");
    let mut rng = rand::weak_rng();
    let range = RandRange::new(vals.0, vals.1);

    (0..size).map(|_| range.ind_sample(&mut rng)).take(size as usize).collect()
}

/// Returns a vector with length `size` which is first filled with each integer value
/// in the (inclusive) range `[vals.0, vals.1]`. If `size` is greater than the 
/// number of integers in the aforementioned range, the integers will repeat. After
/// being filled with `size` values, the vector is shuffled and the order of its
/// values is randomized.
pub fn shuffled_vec<T: OclPrm>(vals: (T, T), size: usize) -> Vec<T> {
    let mut vec: Vec<T> = Vec::with_capacity(size);
    assert!(size > 0, "\nbuffer::shuffled_vec(): Vector size must be greater than zero.");
    assert!(vals.0 < vals.1, "\nbuffer::shuffled_vec(): Minimum value must be less than maximum.");
    let min = vals.0.to_i64().expect("\nbuffer::shuffled_vec(), min");
    let max = vals.1.to_i64().expect("\nbuffer::shuffled_vec(), max") + 1;
    let mut range = (min..max).cycle();

    for _ in 0..size {
        vec.push(FromPrimitive::from_i64(range.next().expect("\nbuffer::shuffled_vec(), range")).expect("\nbuffer::shuffled_vec(), from_usize"));
    }

    shuffle(&mut vec);
    vec
}


/// Shuffles the values in a vector using a single pass of Fisher-Yates with a
/// weak (not cryptographically secure) random number generator.
pub fn shuffle<T: OclPrm>(vec: &mut [T]) {
    let len = vec.len();
    let mut rng = rand::weak_rng();
    let mut ridx: usize;
    let mut tmp: T;

    for i in 0..len {
        ridx = RandRange::new(i, len).ind_sample(&mut rng);
        tmp = vec[i];
        vec[i] = vec[ridx];
        vec[ridx] = tmp;
    }
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

/// Does what is says it's gonna.
pub fn print_bytes_as_hex(bytes: &Vec<u8>) {
    print!("0x");

    for &byte in bytes.iter() {
        print!("{:x}", byte);
    }
}


#[allow(unused_assignments, unused_variables)]
/// [UNSTABLE]: MAY BE REMOVED AT ANY TIME
/// Prints a vector to stdout. Used for debugging.
pub fn print_slice<T: OclPrm>(
            vec: &[T], 
            every: usize, 
            val_range: Option<(T, T)>, 
            idx_range: Option<Range<usize>>,
            show_zeros: bool, 
            ) {
    print!( "{cdgr}[{cg}{}{cdgr}/{}", vec.len(), every, cg = colors::C_GRN, cdgr = colors::C_DGR);

    let (vr_start, vr_end) = match val_range {
        Some(vr) => {
            print!( ";({}-{})", vr.0, vr.1);
            vr
        },

        None => (Default::default(), Default::default()),
    };

    let (ir_start, ir_end) = match idx_range {
        Some(ref ir) => {
            print!( ";[{}..{}]", ir.start, ir.end);
            (ir.start, ir.end)
        },

        None => (0usize, 0usize),
    };

    print!( "]:{cd} ", cd = colors::C_DEFAULT,);

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
    for i in 0..vec.len() {

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
            if vec[i] < vr_start || vec[i] > vr_end {
                prnt = false;
                within_val_range = false;
            } else {
                if within_idx_range {
                    if vec[i] == Default::default() {
                        ttl_ir += 1;
                    } else {
                        ttl_ir += 1;
                    }
                }

                within_val_range = true;
            }
        } 

        if within_idx_range && within_val_range {
            sum += vec[i].to_i64().expect("ocl::buffer::print_vec(): vec[i]");

            if vec[i] > hi { hi = vec[i] };

            if vec[i] < lo { lo = vec[i] };

            if vec[i] != Default::default() {
                ttl_nz += 1usize;
                color = colors::C_ORA;
            } else {
                if show_zeros {
                    color = colors::C_DEFAULT;
                } else {
                    prnt = false;
                }
            }
        }

        if prnt {
            print!( "{cg}[{cd}{}{cg}:{cc}{}{cg}]{cd}", i, vec[i], cc = color, cd = colors::C_DEFAULT, cg = colors::C_DGR);
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


    println!("{cdgr} ;(nz:{clbl}{}{cdgr}({clbl}{:.2}%{cdgr}),\
        ir:{clbl}{}{cdgr}({clbl}{:.2}%{cdgr}),hi:{},lo:{},anz:{:.2},prntd:{}){cd} ", 
        ttl_nz, nz_pct, ttl_ir, ir_pct, hi, lo, anz, ttl_prntd, cd = colors::C_DEFAULT, clbl = colors::C_LBL, cdgr = colors::C_DGR);
}


pub fn print_simple<T: OclPrm>(slice: &[T]) {
    print_slice(slice, 1, None, None, true);
}



pub fn print_val_range<T: OclPrm>(slice: &[T], every: usize, val_range: Option<(T, T)>) {
    print_slice(slice, every, val_range, None, true);
}


#[cfg(test)]
mod tests {
    // use std::iter;

    #[test]
    fn remove_rebuild() {
        let mut primary_vals: Vec<u32> = (0..(2 << 18)).map(|v| v).collect();
        let orig_len = primary_vals.len();

        let mut bad_indices: Vec<usize> = Vec::<usize>::with_capacity(2 << 16);
        let mut idx = 0;

        // Mark every whateverth value 'bad':
        for &val in primary_vals.iter() {
            if (val % 19 == 0) || (val % 31 == 0) || (val % 107 == 0) {
                bad_indices.push(idx);
            }

            idx += 1;
        }

        println!("util::tests::remove_rebuild(): bad_indices: {}", bad_indices.len());
     
        // Remove the bad values:
        super::vec_remove_rebuild(&mut primary_vals, &bad_indices[..], 3)
            .expect("util::tests::remove_rebuild()");

        // Check:
        for &val in primary_vals.iter() {
            if (val % 19 == 0) || (val % 31 == 0) || (val % 107 == 0) {
                panic!("util::tests::remove_rebuild(): Value: '{}' found in list!", val);
            }
        }

        assert_eq!(orig_len, primary_vals.len() + bad_indices.len());
    }
}