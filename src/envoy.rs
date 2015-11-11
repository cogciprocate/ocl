use std::iter::{ self };
use std::slice::{ Iter };
use rand::{ self };
use rand::distributions::{ IndependentSample, Range as RandRange };
use num::{ FromPrimitive, ToPrimitive };
use std::ops::{ Range, Index, IndexMut };

use cl_h;
use super::{ fmt, OclNum, ProQueue, EnvoyDims, EventList };


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
		let buf: cl_h::cl_mem = super::create_buf(&mut vec, pq.context());

		let mut envoy = Envoy {
			vec: vec,
			buf: buf,
			pq: pq.clone(),
		};

		envoy.write_wait();

		envoy
	}

	/// Write contents of `self.vec` to remote device, waiting on events in 
	/// `wait_list` and adding a new event to `dest_list`.
	pub fn write(&mut self, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) {
		super::enqueue_write_buffer(self.pq.cmd_queue(), self.buf, false, &self.vec, 0, 
			wait_list, dest_list);
	}

	/// Write contents of `data` to remote device with remote offset of `offset`, 
	/// waiting on events in `wait_list` and adding a new event to `dest_list`.
	pub fn write_direct(&mut self, data: &[T], offset: usize, wait_list: Option<&EventList>, 
				dest_list: Option<&mut EventList>) 
	{
		super::enqueue_write_buffer(self.pq.cmd_queue(), self.buf, false, data, offset, 
			wait_list, dest_list);
	}

	/// Write contents of `self.vec` to remote device and block until complete.
	pub fn write_wait(&mut self) {
		super::enqueue_write_buffer(self.pq.cmd_queue(), self.buf, true, &self.vec, 0, None, None);
	}

	/// Read remote device contents into `self.vec`, waiting on events in `wait_list`
	/// and adding a new event to `dest_list`.
	pub fn read(&mut self, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) {
		super::enqueue_read_buffer(self.pq.cmd_queue(), self.buf, false, &mut self.vec, 0, 
			wait_list, dest_list);
	}

	/// Read remote device contents with remote offset of `offset` into `data`,
	/// waiting on events in `wait_list` and adding a new event to `dest_list`.
	pub fn read_direct(&self, data: &mut [T], offset: usize, wait_list: Option<&EventList>, 
				dest_list: Option<&mut EventList>) 
	{
		super::enqueue_read_buffer(self.pq.cmd_queue(), self.buf, false, data, offset, 
			wait_list, dest_list);
	}

	/// Read remote device contents into `self.vec` and block until complete.
	pub fn read_wait(&mut self) {
		super::enqueue_read_buffer(self.pq.cmd_queue(), self.buf, true, &mut self.vec, 0, None, None);
	}	

	pub fn set_all_to(&mut self, val: T) {
		for ele in self.vec.iter_mut() {
			*ele = val;
		}

		self.write_wait();
	}

	pub fn set_range_to(&mut self, val: T, range: Range<usize>) {
		for idx in range {
			self.vec[idx] = val;
		}
		self.write_wait();
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

    // PRINT(): <<<<< TODO: only pq.read_wait() the idx_range (just slice self.vec to match and bypass .read_wait()) >>>>>
    pub fn print(&mut self, every: usize, val_range: Option<(T, T)>, 
    			idx_range: Option<Range<usize>>, zeros: bool)
	{
    	self.read_wait();

		fmt::print_vec(&self.vec[..], every, val_range, idx_range, zeros);
	}

	/// Resize Envoy. Dangles any references kernels may have had to the buffer. [REWORDME]
	// RELEASES OLD BUFFER -- IF ANY KERNELS HAD REFERENCES TO IT THEY BREAK
	pub unsafe fn resize(&mut self, new_dims: &EnvoyDims, val: T) {		
		self.release();
		self.vec.resize(new_dims.padded_envoy_len(&self.pq) as usize, val);
		self.buf = super::create_buf(&mut self.vec, self.pq.context());
		// JUST TO VERIFY
		self.write_wait();
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
			self.read_direct(&mut buffer[0..1], idx, None, None);
			buffer[0]
		}
	}
}
