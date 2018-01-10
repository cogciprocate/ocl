//! These tests are fairly trivial. Most actual regression testing is done
//! by running tests with [bismit](https://github.com/cogciprocate/bismit).
//!
//! Lots more tests needed (what's new?).
//!
//! TODO: port some of bismit's tests over.
//!
//! [NOTE]: These tests are being migrated from the original `ocl` crate.
//!
//! TODO: Finish porting tests from ocl.
//!

#![allow(dead_code)]

pub mod build_error;
pub mod buffer_copy;
// pub mod buffer_ops_rect;
// pub mod image_ops;
pub mod buffer_fill;
// pub mod clear_completed;
// pub mod concurrent;
// pub mod kernel_arg_ptr;
pub mod vector_types;
// pub mod context_props;

use rand::{self, Rng};
use error::{Result as OclCoreResult};
use ::{OclScl, PlatformId, DeviceId, Context};

const PRINT_ITERS_MAX: i32 = 3;
const PRINT_SLICES_MAX: usize = 16;
const PRINT: bool = false;


/// Returns one context for each device on each platform available.
pub fn get_available_contexts() -> Vec<(PlatformId, DeviceId, Context)> {
    use ::{DeviceInfo, DeviceInfoResult};
    let mut contexts = Vec::with_capacity(16);

    for platform in ::get_platform_ids().unwrap() {
        for device in ::get_device_ids(&platform, None, None).unwrap() {
            match ::get_device_info(device, DeviceInfo::Available).unwrap() {
                DeviceInfoResult::Available(r) => if !r { continue; },
                _ => unreachable!(),
            }

            let context_properties = ::ContextProperties::new().platform(platform);
            let context = ::create_context(Some(&context_properties),
                &[device], None, None).unwrap();

            contexts.push((platform, device, context));
        }
    }

    contexts
}

fn gen_region_origin(dims: &[usize; 3]) -> ([usize; 3], [usize; 3]) {
    let mut rng = rand::weak_rng();

    let region = [
        rng.gen_range(1, dims[0] + 1),
        rng.gen_range(1, dims[1] + 1),
        rng.gen_range(1, dims[2] + 1),
    ];

    let origin = [
        rng.gen_range(0, (dims[0] - region[0]) + 1),
        rng.gen_range(0, (dims[1] - region[1]) + 1),
        rng.gen_range(0, (dims[2] - region[2]) + 1),
    ];

    (origin, region)
}

fn within_region(coords: [usize; 3], region_ofs: [usize; 3], region_size: [usize; 3]) -> bool {
    let mut within: bool = true;
    for i in 0..3 {
        within &= coords[i] >= region_ofs[i] && coords[i] < (region_ofs[i] + region_size[i]);
    }
    within
}

fn verify_vec_rect<T: OclScl>(origin: [usize; 3], region: [usize; 3], in_region_val: T,
            out_region_val: T, vec_dims: [usize; 3], ele_per_coord: usize, vec: &[T],
            ttl_runs: i32, print: bool) -> OclCoreResult<()>
{
    let mut print = PRINT && print && ttl_runs <= PRINT_ITERS_MAX;
    let slices_to_print = PRINT_SLICES_MAX;
    let mut result = Ok(());

    if print {
        println!("Verifying run: '{}', origin: {:?}, region: {:?}, vec_dims: {:?}", ttl_runs,
            origin, region, vec_dims);
    }

    for z in 0..vec_dims[2] {
        for y in 0..vec_dims[1] {
            for x in 0..vec_dims[0] {
                let pixel = (z * vec_dims[1] * vec_dims[0]) +
                    (y * vec_dims[0]) + x;
                let idz = pixel * ele_per_coord;

                for id in 0..ele_per_coord {
                    let idx = idz + id;

                    // Print:
                    if print {
                        if within_region([x, y, z], origin, region) {
                            if vec[idx] == in_region_val {
                                // printc!(lime: "[{:02}]", vec[idx]);
                                print!("[{:02}]", vec[idx]);
                            } else {
                                // printc!(red_bold: "[{:02}]", vec[idx]);
                                print!("[{:02}]", vec[idx]);
                            }
                        } else {
                            if vec[idx] == out_region_val {
                                // printc!(dark_grey: "[{:02}]", vec[idx]);
                                print!("[{:02}]", vec[idx]);
                            } else {
                                // printc!(yellow: "[{:02}]", vec[idx]);
                                print!("[{:02}]", vec[idx]);
                            }
                        }
                    }

                    // Verify:
                    if result.is_ok() {
                        if within_region([x, y, z], origin, region) {
                            if vec[idx] != in_region_val {
                                result = Err(format!("vec[{}] should be '{}' but is '{}'",
                                    idx, in_region_val, vec[idx]).into());
                            }
                        } else {
                            if vec[idx] != out_region_val {
                                result = Err(format!("vec[{}] should be '{}' but is '{}'",
                                    idx, out_region_val, vec[idx]).into());
                            }
                        }
                    }
                }
            }
            if print { print!("\n"); }
        }
        if print { print!("\n"); }
        if print && z >= slices_to_print { print = false };
    }
    if print { print!("\n"); }

    result
}
