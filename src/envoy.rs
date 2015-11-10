use std::iter::{ self };
use std::slice::{ Iter };
use rand::{ self };
use rand::distributions::{ IndependentSample, Range as RandRange };
use num::{ FromPrimitive, ToPrimitive };
use std::ops::{ Range, Index, IndexMut };

use cl_h;
use super::{ fmt, OclNum, ProQueue, EnvoyDims };


impl<'a, T> EnvoyDims for &'a T where T: EnvoyDims {
    fn padded_envoy_len(&self, pq: &ProQueue) -> usize { (*self).padded_envoy_len(pq) }
}

pub type AxonState = Envoy<u8>;
pub type DendriteState = Envoy<u8>;
pub type SynapseState = Envoy<u8>;

pub struct Envoy<T> {
	vec: Vec<T>,
	buf: cl_h::cl_mem,
	pq: ProQueue,
}

impl<T: OclNum> Envoy<T> {
	pub fn new<E: EnvoyDims>(dims: E, init_val: T, pq: &ProQueue) -> Envoy<T> {
		let len = dims.padded_envoy_len(pq) as usize;
		let vec: Vec<T> = iter::repeat(init_val).take(len).collect();

		Envoy::_new(vec, pq)
	}

	// SHUFFLED(): max_val is inclusive!
	pub fn shuffled<E: EnvoyDims>(dims: E, min_val: T, max_val: T, pq: &ProQueue) -> Envoy<T> {
		let len = dims.padded_envoy_len(pq) as usize;
		let vec: Vec<T> = shuffled_vec(len, min_val, max_val);

		Envoy::_new(vec, pq)
	}

	fn _new(mut vec: Vec<T>, pq: &ProQueue) -> Envoy<T> {
		let buf: cl_h::cl_mem = super::create_buffer(&mut vec, pq.context());

		let mut envoy = Envoy {
			vec: vec,
			buf: buf,
			pq: pq.clone(),
		};

		envoy.write();

		envoy
	}

	pub fn write(&mut self) {
		// self.pq.enqueue_write_buffer(self);
		super::enqueue_write_buffer(&self.vec, self.buf, self.pq.cmd_queue(), 0);
	}

	pub fn write_direct(&self, sdr: &[T], offset: usize) {
		super::enqueue_write_buffer(sdr, self.buf, self.pq.cmd_queue(), offset);
	}

	pub fn read(&mut self) {
		super::enqueue_read_buffer(&mut self.vec, self.buf, self.pq.cmd_queue(), 0);
	}

	pub fn read_direct(&self, sdr: &mut [T], offset: usize) {
		super::enqueue_read_buffer(sdr, self.buf, self.pq.cmd_queue(), offset);
	}

	pub fn set_all_to(&mut self, val: T) {
		for ele in self.vec.iter_mut() {
			*ele = val;
		}
		self.write();
	}

	pub fn set_range_to(&mut self, val: T, range: Range<usize>) {
		for idx in range {
			self.vec[idx] = val;
		}
		self.write();
	}

	pub fn len(&self) -> usize {
		self.vec.len()
	}

	pub fn print_simple(&mut self) {
		self.print(1, None, None, true);
    }

    pub fn print_val_range(&mut self, every: usize, val_range: Option<(T, T)>,) {
		self.print(every, val_range, None, true);
    }

    // PRINT(): <<<<< TODO: only pq.read() the idx_range (just slice self.vec to match and bypass .read()) >>>>>
    pub fn print(&mut self, every: usize, val_range: Option<(T, T)>, 
    			idx_range: Option<Range<usize>>, zeros: bool)
	{
    	self.read();

		fmt::print_vec(&self.vec[..], every, val_range, idx_range, zeros);
	}

	/// Resize Envoy. Dangles any references kernels may have had to the buffer. [REWORDME]
	// RELEASES OLD BUFFER -- IF ANY KERNELS HAD REFERENCES TO IT THEY BREAK
	pub unsafe fn resize(&mut self, new_dims: &EnvoyDims, val: T) {		
		self.release();
		self.vec.resize(new_dims.padded_envoy_len(&self.pq) as usize, val);
		self.buf = super::create_buffer(&mut self.vec, self.pq.context());
		// JUST TO VERIFY
		self.write();
	}

    pub fn release(&mut self) {
		super::release_mem_object(self.buf);
	}

	pub fn vec(&self) -> &Vec<T> {
		&self.vec
	}

	pub fn vec_mut(&mut self) -> &mut Vec<T> {
		&mut self.vec
	}

	pub fn buf(&self) -> cl_h::cl_mem {
		self.buf
	}

	pub fn iter<'a>(&'a self) -> Iter<'a, T> {
		self.vec.iter()
	}
}

impl<'b, T> Index<&'b usize> for Envoy<T> {
    type Output = T;

    fn index<'a>(&'a self, index: &'b usize) -> &'a T {
        &self.vec[..][*index]
    }
}

impl<'b, T> IndexMut<&'b usize> for Envoy<T> {
    fn index_mut<'a>(&'a mut self, index: &'b usize) -> &'a mut T {
        &mut self.vec[..][*index]
    }
}

impl<T> Index<usize> for Envoy<T> {
    type Output = T;

    fn index<'a>(&'a self, index: usize) -> &'a T {
        &self.vec[..][index]
    }
}

impl<T> IndexMut<usize> for Envoy<T> {
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut T {
        &mut self.vec[..][index]
    }
}

// impl<'a, T> Iterator<


// impl<T: OclNum> Display for Envoy<T> {
//     fn fmt(&self, fmtr: &mut Formatter) -> FmtResult {
//     	// self.print(1, None, None, true)
//     	let mut tmp_vec = Vec::with_capacity(self.vec.len());
//     	self.read_direct(&mut tmp_vec[..], 0);
//     	fmt::fmt_vec(fmtr.buf, &tmp_vec[..], 1, None, None, true)
// 	}
// }


pub fn shuffled_vec<T: OclNum>(size: usize, min_val: T, max_val: T) -> Vec<T> {
	let mut vec: Vec<T> = Vec::with_capacity(size);

	assert!(size > 0, "\ncl_h::envoy::shuffled_vec(): Vector size must be greater than zero.");
	assert!(min_val < max_val, "\ncl_h::envoy::shuffled_vec(): Minimum value must be less than maximum.");

	let min = min_val.to_i64().expect("\ncl_h::envoy::shuffled_vec(), min");
	let max = max_val.to_i64().expect("\ncl_h::envoy::shuffled_vec(), max") + 1;

	let mut range = (min..max).cycle();

	for _ in 0..size {
		vec.push(FromPrimitive::from_i64(range.next().expect("\ncl_h::envoy::shuffled_vec(), range")).expect("\ncl_h::envoy::shuffled_vec(), from_usize"));
	}

	shuffle_vec(&mut vec);

	vec

}


// Fisher-Yates
pub fn shuffle_vec<T: OclNum>(vec: &mut Vec<T>) {
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



// #[cfg(test)]
pub mod tests {
	use super::{ Envoy };
	use super::super::{ OclNum };
	use std::num::Zero;

	pub trait EnvoyTest<T> {
		fn read_idx_direct(&self, idx: usize) -> T;
	}

	impl<T: OclNum> EnvoyTest<T> for Envoy<T> {
		fn read_idx_direct(&self, idx: usize) -> T {
			let mut buffer = vec![Zero::zero()];
			self.read_direct(&mut buffer[0..1], idx);
			buffer[0]
		}
	}
}
