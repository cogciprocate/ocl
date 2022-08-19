extern crate ocl;
extern crate rand;
extern crate num_traits;
extern crate futures;
#[macro_use] extern crate failure;

pub mod sub_buffer_pool;
pub mod command_graph;
pub mod work_pool;
pub mod full_device_info;

pub use self::sub_buffer_pool::SubBufferPool;
pub use self::command_graph::{CommandGraph, Command, CommandDetails, KernelArgBuffer, RwCmdIdxs};
pub use self::work_pool::WorkPool;

use rand::distributions::uniform::{SampleRange, SampleUniform};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand::distributions::Uniform;
use num_traits::FromPrimitive;
use ocl::OclScl;


/// Returns a vector with length `size` containing random values in the (half-open)
/// range `[vals.0, vals.1)`.
pub fn scrambled_vec<T: OclScl + SampleUniform>(vals: (T, T), size: usize) -> Vec<T> {
    assert!(size > 0, "\nbuffer::shuffled_vec(): Vector size must be greater than zero.");
    assert!(vals.0 < vals.1, "\nbuffer::shuffled_vec(): Minimum value must be less than maximum.");
    let mut rng = SmallRng::from_entropy();
    let range = Uniform::new(vals.0, vals.1);

    (0..size).map(|_| rng.sample(&range)).take(size as usize).collect()
}

/// Returns a vector with length `size` which is first filled with each integer value
/// in the (inclusive) range `[vals.0, vals.1]`. If `size` is greater than the
/// number of integers in the aforementioned range, the integers will repeat. After
/// being filled with `size` values, the vector is shuffled and the order of its
/// values is randomized.
pub fn shuffled_vec<T: OclScl>(vals: (T, T), size: usize) -> Vec<T> {
    let mut vec: Vec<T> = Vec::with_capacity(size);
    assert!(size > 0, "\nbuffer::shuffled_vec(): Vector size must be greater than zero.");
    assert!(vals.0 < vals.1, "\nbuffer::shuffled_vec(): Minimum value must be less than maximum.");
    let min = vals.0.to_i64().expect("\nbuffer::shuffled_vec(), min");
    let max = vals.1.to_i64().expect("\nbuffer::shuffled_vec(), max") + 1;
    let mut range = (min..max).cycle();

    for _ in 0..size {
        vec.push(FromPrimitive::from_i64(range.next().expect("\nbuffer::shuffled_vec(), range"))
            .expect("\nbuffer::shuffled_vec(), from_usize"));
    }

    shuffle(&mut vec);
    vec
}


/// Shuffles the values in a vector using a single pass of Fisher-Yates with a
/// weak (not cryptographically secure) random number generator.
pub fn shuffle<T: OclScl>(vec: &mut [T]) {
    let len = vec.len();
    let mut rng = SmallRng::from_entropy();
    let mut ridx: usize;
    let mut tmp: T;

    for i in 0..len {
        ridx = (i..len).sample_single(&mut rng);
        tmp = vec[i];
        vec[i] = vec[ridx];
        vec[ridx] = tmp;
    }
}