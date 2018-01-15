extern crate ocl;
extern crate rand;
extern crate num_traits;

pub mod sub_buffer_pool;
pub mod command_graph;

pub use self::sub_buffer_pool::SubBufferPool;
pub use self::command_graph::{CommandGraph, Command, CommandDetails, KernelArgBuffer, RwCmdIdxs};

// use rand;
use rand::distributions::{IndependentSample, Range as RandRange};
use rand::distributions::range::SampleRange;
use num_traits::FromPrimitive;
use ocl::OclScl;


/// Returns a vector with length `size` containing random values in the (half-open)
/// range `[vals.0, vals.1)`.
pub fn scrambled_vec<T: OclScl + SampleRange>(vals: (T, T), size: usize) -> Vec<T> {
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