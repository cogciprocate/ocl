//! Tools for rendering objects to the screen or as text.
use std::ops::Range;

use super::OclNum;
// MT: Mini-tab: 4 spaces ('mini' compared to the huge tab on certain terminals)
pub static MT: &'static str = "    "; 

pub static C_DEFAULT: &'static str = "\x1b[0m";
pub static C_UNDER: &'static str = "\x1b[1m";

pub static C_RED: &'static str = "\x1b[31m";
pub static C_BRED: &'static str = "\x1b[1;31m";
pub static C_GRN: &'static str = "\x1b[32m";
pub static C_BGRN: &'static str = "\x1b[1;32m";
pub static C_ORA: &'static str = "\x1b[33m";
pub static C_DBL: &'static str = "\x1b[34m";
pub static C_PUR: &'static str = "\x1b[35m";
pub static C_CYA: &'static str = "\x1b[36m";
pub static C_LGR: &'static str = "\x1b[37m";
pub static C_DGR: &'static str = "\x1b[90m";
pub static C_LRD: &'static str = "\x1b[91m";
pub static C_YEL: &'static str = "\x1b[93m";
pub static C_BLU: &'static str = "\x1b[94m";
pub static C_MAG: &'static str = "\x1b[95m";
pub static C_LBL: &'static str = "\x1b[94m";

pub static BGC_DEFAULT: &'static str = "\x1b[49m";
pub static BGC_GRN: &'static str = "\x1b[42m";
pub static BGC_PUR: &'static str = "\x1b[45m";
pub static BGC_LGR: &'static str = "\x1b[47m";
pub static BGC_DGR: &'static str = "\x1b[100m";


#[allow(unused_assignments, unused_variables)]
/// [UNSTABLE]: MAY BE REMOVED AT ANY TIME
/// Prints a vector to stdout. Used for debugging.
pub fn print_vec<T: OclNum>(
            vec: &[T], 
            every: usize, 
            val_range: Option<(T, T)>, 
            idx_range: Option<Range<usize>>,
            show_zeros: bool, 
    )
{
    print!( "{cdgr}[{cg}{}{cdgr}/{}", vec.len(), every, cg = C_GRN, cdgr = C_DGR);

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

    print!( "]:{cd} ", cd = C_DEFAULT,);

    let mut ttl_nz = 0usize;
    let mut ttl_ir = 0usize;
    let mut within_idx_range = true;
    let mut within_val_range = true;
    let mut hi: T = vr_start;
    let mut lo: T = vr_end;
    let mut sum: i64 = 0;
    let mut ttl_prntd: usize = 0;
    let len = vec.len();


    let mut color: &'static str = C_DEFAULT;
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
                color = C_ORA;
            } else {
                if show_zeros {
                    color = C_DEFAULT;
                } else {
                    prnt = false;
                }
            }
        }

        if prnt {
            print!( "{cg}[{cd}{}{cg}:{cc}{}{cg}]{cd}", i, vec[i], cc = color, cd = C_DEFAULT, cg = C_DGR);
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
        ttl_nz, nz_pct, ttl_ir, ir_pct, hi, lo, anz, ttl_prntd, cd = C_DEFAULT, clbl = C_LBL, cdgr = C_DGR);
}

