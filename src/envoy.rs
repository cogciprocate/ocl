use std::iter::{ self };
use std::slice::{ Iter };
use rand::{ self };
use rand::distributions::{ IndependentSample, Range as RandRange };
use num::{ FromPrimitive, ToPrimitive };
use std::ops::{ Range, Index, IndexMut };

use cl_h::{ self, cl_mem };
use super::{ fmt, OclNum, Queue, EnvoyDims, EventList };

impl<'a, T> EnvoyDims for &'a T where T: EnvoyDims {
    fn padded_envoy_len(&self, incr: usize) -> usize { (*self).padded_envoy_len(incr) }
}

pub type AxonState = Envoy<u8>;
pub type DendriteState = Envoy<u8>;
pub type SynapseState = Envoy<u8>;

// [FIXME] TODO: Check that type size is <= the maximum supported by device.
pub struct Envoy<T> {
	vec: Vec<T>,
	buffer_obj: cl_mem,
	queue: Queue,
}

impl<T: OclNum> Envoy<T> {
	/// Creates a new Envoy with dimensions: `dims` and inital value: `init_val` 
	/// within the command queue associated with `pq`.
	pub fn new<E: EnvoyDims>(dims: E, init_val: T, queue: &Queue) -> Envoy<T> {
		let len = dims.padded_envoy_len(super::get_max_work_group_size(queue.device_id()));
		let vec: Vec<T> = iter::repeat(init_val).take(len).collect();

		Envoy::_new(vec, queue)
	}

	/// Creates a new Envoy initialized with a series of integers ranging from `min_val`
	/// to `max_val` which are shuffled randomly.
	// Note: max_val is inclusive.
	pub fn shuffled<E: EnvoyDims>(dims: E, min_val: T, max_val: T, queue: &Queue) -> Envoy<T> {
		let len = dims.padded_envoy_len(super::get_max_work_group_size(queue.device_id()));
		let vec: Vec<T> = shuffled_vec(len, min_val, max_val);

		Envoy::_new(vec, queue)
	}

	/// Creates a new Envoy initialized with random values within the range `min_val..max_val`
	/// (exclusive).
	// Note: max_val is exclusive.
	pub fn scrambled<E: EnvoyDims>(dims: E, min_val: T, max_val: T, queue: &Queue) -> Envoy<T> {
		let len = dims.padded_envoy_len(super::get_max_work_group_size(queue.device_id()));
		let vec: Vec<T> = scrambled_vec(len, min_val, max_val);

		Envoy::_new(vec, queue)
	}

	fn _new(mut vec: Vec<T>, queue: &Queue) -> Envoy<T> {
		let buffer_obj: cl_mem = super::create_buffer(&mut vec, queue.context_obj(), 
			cl_h::CL_MEM_READ_WRITE);

		Envoy {
			vec: vec,
			buffer_obj: buffer_obj,
			queue: queue.clone(),
		}
	}

	/// After waiting on events in `wait_list` to finish, writes the contents of
	/// 'self.vec' to the remote device data buffer and adds a new event to `dest_list`
	pub fn write(&mut self, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) {
		super::enqueue_write_buffer(self.queue.obj(), self.buffer_obj, dest_list.is_none(), 
			&self.vec, 0, wait_list, dest_list);
	}

	/// After waiting on events in `wait_list` to finish, writes the contents of
	/// 'data' to the remote device data buffer with a remote offset of `offset`
	/// and adds a new event to `dest_list`
	pub fn write_direct(&mut self, data: &[T], offset: usize, wait_list: Option<&EventList>, 
				dest_list: Option<&mut EventList>) 
	{
		super::enqueue_write_buffer(self.queue.obj(), self.buffer_obj, dest_list.is_none(), 
			data, offset, wait_list, dest_list);
	}

	/// Writes the contents of `self.vec` to the remote device data buffer and 
	/// blocks until completed.
	pub fn write_wait(&mut self) {
		super::enqueue_write_buffer(self.queue.obj(), self.buffer_obj, true, &self.vec, 0, None, None);
	}

	/// After waiting on events in `wait_list` to finish, reads the remote device 
	/// data buffer into 'self.vec' and adds a new event to `dest_list`.
	pub fn read(&mut self, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) {
		super::enqueue_read_buffer(self.queue.obj(), self.buffer_obj, dest_list.is_none(), 
			&mut self.vec, 0, wait_list, dest_list);
	}

	/// After waiting on events in `wait_list` to finish, reads the remote device 
	/// data buffer with a remote offset of `offset` into `data` and adds a new 
	/// event to `dest_list`.
	pub fn read_direct(&self, data: &mut [T], offset: usize, wait_list: Option<&EventList>, 
				dest_list: Option<&mut EventList>) 
	{
		super::enqueue_read_buffer(self.queue.obj(), self.buffer_obj, dest_list.is_none(), 
			data, offset, wait_list, dest_list);
	}

	/// Reads the remote device data buffer into `self.vec` and blocks until completed.
	pub fn read_wait(&mut self) {
		super::enqueue_read_buffer(self.queue.obj(), self.buffer_obj, true, &mut self.vec, 0, None, None);
	}	

	/// Blocks until the underlying command queue has completed all commands.
	pub fn wait(&self) {
		self.queue.finish();
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

	/// Resizes Envoy. Dangles any references kernels may have had to the buffer. [REWORDME]
	// RELEASES OLD BUFFER -- IF ANY KERNELS HAD REFERENCES TO IT THEY BREAK
	pub unsafe fn resize(&mut self, new_dims: &EnvoyDims, val: T) {		
		self.release();
		let new_len = new_dims.padded_envoy_len(super::get_max_work_group_size(self.queue.device_id()));
		self.vec.resize(new_len, val);
		self.buffer_obj = super::create_buffer(&mut self.vec, self.queue.context_obj(), cl_h::CL_MEM_READ_WRITE);
		// JUST TO VERIFY
		// self.write_wait();
	}

    pub fn release(&mut self) {
		super::release_mem_object(self.buffer_obj);
	}

	/// Returns a reference to the local vector associated with this envoy.
	pub fn vec(&self) -> &Vec<T> {
		&self.vec
	}

	/// Returns a mutable reference to the local vector associated with this envoy.
	/// 
	/// # Safety
	/// - Could cause data collisions, etc.
	pub unsafe fn vec_mut(&mut self) -> &mut Vec<T> {
		&mut self.vec
	}

	pub fn buffer_obj(&self) -> cl_mem {
		self.buffer_obj
	}

	/// Returns a reference to the program/command queue associated with this envoy.
	pub fn queue(&self) -> &Queue {
		&self.queue
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


pub fn scrambled_vec<T: OclNum>(size: usize, min_val: T, max_val: T) -> Vec<T> {
	assert!(size > 0, "\nenvoy::shuffled_vec(): Vector size must be greater than zero.");
	assert!(min_val < max_val, "\nenvoy::shuffled_vec(): Minimum value must be less than maximum.");
	// let mut vec: Vec<T> = Vec::with_capacity(size);
	let mut rng = rand::weak_rng();
	let range = RandRange::new(min_val, max_val);

	(0..size).map(|_| range.ind_sample(&mut rng)).take(size as usize).collect()

	// for _ in 0..size {
		
	// }

	// vec
}


pub fn shuffled_vec<T: OclNum>(size: usize, min_val: T, max_val: T) -> Vec<T> {
	let mut vec: Vec<T> = Vec::with_capacity(size);

	assert!(size > 0, "\nenvoy::shuffled_vec(): Vector size must be greater than zero.");
	assert!(min_val < max_val, "\nenvoy::shuffled_vec(): Minimum value must be less than maximum.");

	let min = min_val.to_i64().expect("\nenvoy::shuffled_vec(), min");
	let max = max_val.to_i64().expect("\nenvoy::shuffled_vec(), max") + 1;

	let mut range = (min..max).cycle();

	for _ in 0..size {
		vec.push(FromPrimitive::from_i64(range.next().expect("\nenvoy::shuffled_vec(), range")).expect("\nenvoy::shuffled_vec(), from_usize"));
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
