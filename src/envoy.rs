
use std::iter;
use std::slice::{Iter, IterMut};
use rand;
use rand::distributions::{IndependentSample, Range as RandRange};
use num::{FromPrimitive, ToPrimitive};
use std::ops::{Range, RangeFull, Index, IndexMut};
use std::default::Default;

use cl_h::{self, cl_mem, cl_bitfield};
use super::{fmt, OclNum, Queue, EnvoyDims, EventList, OclError, OclResult};

static VEC_OPT_ERR_MSG: &'static str = "No host side vector defined for this Envoy. \
	You must create this Envoy using 'Envoy::with_vec()' (et al.) in order to call this function.";

enum VecOption<T> {
	None,
	Some(Vec<T>),
}

impl<T> VecOption<T> {
	fn as_ref(&self) -> OclResult<&Vec<T>> {
		match self {
			&VecOption::Some(ref vec) => {
				Ok(vec)
			},
			&VecOption::None => Err(OclError::new(VEC_OPT_ERR_MSG)),
		}
	}

	fn as_ref_mut(&mut self) -> OclResult<&mut Vec<T>> {
		match self {
			&mut VecOption::Some(ref mut vec) => {
				Ok(vec)
			},
			&mut VecOption::None => Err(OclError::new(VEC_OPT_ERR_MSG)),
		}
	}

	fn is_some(&self) -> bool {
		if let &VecOption::None = self { false } else { true }
	}
}

/// An array with an optional local working copy of data. One copy of data is 
/// stored in a buffer on the device associated with `queue`. A second is stored 
/// as an optional vector (`vec`) in host memory. Basically just a buffer with an
/// optional built-in vector for use as a workspace etc.
///
/// The host side vector must be manually synchronized with the device side buffer 
/// using the `.fill_vec()`, `.flush_vec()`, etc. Data within the contained vector 
/// should generally be considered stale except immediately after a fill/flush 
/// (exception: pinned memory).
///
/// Fill/flush functions are for convenience and are equivalent to the psuedocode: 
/// read_into(self.vec) and write_from(self.vec).
///
// TODO: Check that type size (sizeof(T)) is <= the maximum supported by device.
pub struct Envoy<T> {
	// vec: Vec<T>,
	buffer_obj: cl_mem,
	queue: Queue,
	len: usize,
	vec: VecOption<T>,
}

///
/// # Panics
/// All functions will panic upon any OpenCL error.
impl<T: OclNum> Envoy<T> {
	/// Creates a new read/write Envoy with dimensions: `dims` which will use the 
	/// command queue: `queue` (and its associated device and context) for all operations.
	///
	/// The device side buffer will be allocated a size based on the maximum workgroup 
	/// size of the device. This helps ensure that kernels do not attempt to read 
	/// from or write to memory beyond the length of the buffer (see crate level 
	/// documentation for more details about how dimensions are used). The buffer
	/// will be initialized with a sensible default value (probably `0`).
	///
	/// # Panics 
	/// The returned Envoy contains no host side vector. Functions associated with
	/// one such as `.flush_vec()`, `fill_vec()`, etc. will panic.
	pub fn new<E: EnvoyDims>(dims: E, queue: &Queue) -> Envoy<T> {
		let len = dims.padded_envoy_len(super::get_max_work_group_size(queue.device_id()));
		let init_val = T::default();
		Envoy::_new((init_val, len), queue)
	}

	/// Creates a new read/write Envoy with a host side working copy of data.
	/// Host vector and device buffer are initialized with a sensible default value.
	pub fn with_vec<E: EnvoyDims>(dims: E, queue: &Queue) -> Envoy<T> {
		let len = dims.padded_envoy_len(super::get_max_work_group_size(queue.device_id()));
		let vec: Vec<T> = iter::repeat(T::default()).take(len).collect();

		Envoy::_with_vec(vec, queue)
	}

	/// [MARKED FOR POSSIBLE REMOVAL]: Convenience function.
	/// Creates a new read/write Envoy with a host side working copy of data.
	/// Host vector and device buffer are initialized with the value, `init_val`.
	pub fn with_vec_initialized_to<E: EnvoyDims>(init_val: T, dims: E, queue: &Queue) -> Envoy<T> {
		let len = dims.padded_envoy_len(super::get_max_work_group_size(queue.device_id()));
		let vec: Vec<T> = iter::repeat(init_val).take(len).collect();

		Envoy::_with_vec(vec, queue)
	}

	/// [MARKED FOR POSSIBLE REMOVAL]: Convenience function.
	/// Creates a new read/write Envoy with a vector initialized with a series of 
	/// integers ranging from `min_val` to `max_val` which are shuffled randomly.
	// Note: max_val is inclusive.
	pub fn with_vec_shuffled<E: EnvoyDims>(min_val: T, max_val: T, dims: E, queue: &Queue) 
			-> Envoy<T> 
	{
		let len = dims.padded_envoy_len(super::get_max_work_group_size(queue.device_id()));
		let vec: Vec<T> = shuffled_vec(len, min_val, max_val);

		Envoy::_with_vec(vec, queue)
	}

	/// [MARKED FOR POSSIBLE REMOVAL]: Convenience function.
	/// Creates a new read/write Envoy with a vector initialized with random values 
	/// within the range `min_val..max_val` (exclusive).
	// Note: max_val is exclusive.
	pub fn with_vec_scrambled<E: EnvoyDims>(min_val: T, max_val: T, dims: E, queue: &Queue) 
			-> Envoy<T> 
	{
		let len = dims.padded_envoy_len(super::get_max_work_group_size(queue.device_id()));
		let vec: Vec<T> = scrambled_vec(len, min_val, max_val);

		Envoy::_with_vec(vec, queue)
	}	

	/// Creates a new Envoy with caller-managed buffer length, type, flags, and 
	/// initialization.
	///
	/// # Examples
	/// See `examples/envoy_unchecked.rs`.
	///
	/// # Parameter Reference Documentation
	/// Refer to the following page for information about how to configure flags and
	/// other options for the buffer.
	/// [https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html)
	///
	/// # Safety
	/// No creation time checks are made to help prevent device buffer overruns, 
	/// etc. Caller should be sure to initialize the device side memory with a 
	/// write. Any host side memory pointed to by `host_ptr` is at even greater
	/// risk if used with `CL_MEM_USE_HOST_PTR` (see below).
	///
	/// [IMPORTANT] Practically every read and write to an Envoy created in this way is
	/// potentially unsafe. Because `.read()` and `.write()` do not require an 
	/// unsafe block, their implied promises about safety may be broken at any time.
	///
	/// *You need to know what you're doing and be extra careful using an Envoy created 
	/// with this function because it badly breaks Rust's usual safety promises even
	/// outside of an unsafe block.*
	///
	/// **This is horribly un-idiomatic Rust. You have been warned.**
	///
	/// NOTE: The above important warnings probably only apply to buffers created with 
	/// the `CL_MEM_USE_HOST_PTR` flag because the memory is considered 'pinned' but 
	/// there may also be implementation specific issues which haven't been considered 
	/// or are unknown.
	///
	/// # Stability
	/// The functionality of an Envoy used in this way may eventually be moved into a
	/// separate type, potentially called just `Buffer` which can manage its inherent
	/// unsafety without breaking promises. [update: probably not]
	/// 
	pub unsafe fn new_raw_unchecked(flags: cl_bitfield, len: usize, host_ptr: Option<&[T]>, 
				queue: &Queue) -> Envoy<T> 
	{
		let buffer_obj: cl_mem = super::create_buffer(host_ptr, Some((T::default(), len)), 
			queue.context_obj(), flags);

		Envoy {
			buffer_obj: buffer_obj,
			queue: queue.clone(),
			len: len,
			vec: VecOption::None,
		}
	}

	fn _new(iv_len: (T, usize), queue: &Queue) -> Envoy<T> {
		let buffer_obj: cl_mem = super::create_buffer(None, Some(iv_len), 
			queue.context_obj(), cl_h::CL_MEM_READ_WRITE | cl_h::CL_MEM_COPY_HOST_PTR);

		Envoy {			
			buffer_obj: buffer_obj,
			queue: queue.clone(),
			len: iv_len.1,
			vec: VecOption::None,
		}
	}

	fn _with_vec(mut vec: Vec<T>, queue: &Queue) -> Envoy<T> {
		let buffer_obj: cl_mem = super::create_buffer(Some(&mut vec), None, queue.context_obj(), 
			cl_h::CL_MEM_READ_WRITE | cl_h::CL_MEM_COPY_HOST_PTR);

		Envoy {		
			buffer_obj: buffer_obj,
			queue: queue.clone(),
			len: vec.len(),	
			vec: VecOption::Some(vec),
		}
	}

	/// After waiting on events in `wait_list` to finish, writes the contents of
	/// 'data' to the remote device data buffer with a remote offset of `offset`
	/// and adds a new event to `dest_list`
	///
	/// If the `dest_list` event list is `None`, the write will be blocking.
	pub fn write_direct(&mut self, data: &[T], offset: usize, wait_list: Option<&EventList>, 
				dest_list: Option<&mut EventList>) 
	{
		let block_after_write = dest_list.is_none();
		super::enqueue_write_buffer(self.queue.obj(), self.buffer_obj, block_after_write, 
			data, offset, wait_list, dest_list);
	}

	/// After waiting on events in `wait_list` to finish, writes the contents of
	/// 'self.vec' to the remote device data buffer and adds a new event to `dest_list`
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	pub fn flush_vec(&mut self, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) {
		let vec = self.vec.as_ref_mut().expect("Envoy::flush_vec()");
		super::enqueue_write_buffer(self.queue.obj(), self.buffer_obj, dest_list.is_none(), 
			vec, 0, wait_list, dest_list);
	}

	/// Writes the contents of `self.vec` to the remote device data buffer and 
	/// blocks until completed. 
	///
	/// Equivalent to `.flush_vec(None, None)`.
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	pub fn flush_vec_wait(&mut self) {
		let vec = self.vec.as_ref_mut().expect("Envoy::flush_vec_wait()");
		super::enqueue_write_buffer(self.queue.obj(), self.buffer_obj, true, vec, 0, None, None);
	}

	/// After waiting on events in `wait_list` to finish, reads the remote device 
	/// data buffer with a remote offset of `offset` into `data` and adds a new 
	/// event to `dest_list`.
	///
	/// If the `dest_list` event list is `None`, the read will be blocking.
	pub fn read_direct(&self, data: &mut [T], offset: usize, wait_list: Option<&EventList>, 
				dest_list: Option<&mut EventList>) 
	{
		let block_after_read = dest_list.is_none();
		super::enqueue_read_buffer(self.queue.obj(), self.buffer_obj, block_after_read, 
			data, offset, wait_list, dest_list);
	}

	/// After waiting on events in `wait_list` to finish, reads the remote device 
	/// data buffer into 'self.vec' and adds a new event to `dest_list`.
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	pub fn fill_vec(&mut self, wait_list: Option<&EventList>, dest_list: Option<&mut EventList>) {
		let vec = self.vec.as_ref_mut().expect("Envoy::fill_vec()");
		super::enqueue_read_buffer(self.queue.obj(), self.buffer_obj, dest_list.is_none(), 
			vec, 0, wait_list, dest_list);
	}	

	/// Reads the remote device data buffer into `self.vec` and blocks until completed.
	///
	/// Equivalent to `.fill_vec(None, None)`.
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	pub fn fill_vec_wait(&mut self) {
		let vec = self.vec.as_ref_mut().expect("Envoy::read_wait()");
		super::enqueue_read_buffer(self.queue.obj(), self.buffer_obj, 
			true, vec, 0, None, None);
	}	

	/// Blocks until the underlying command queue has completed all commands.
	pub fn wait(&self) {
		self.queue.finish();
	}

	/// [MARKED FOR POSSIBLE REMOVAL]: Convenience function.
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	/// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
	pub fn set_all_to(&mut self, val: T) {
		{
			let vec = self.vec.as_ref_mut().expect("Envoy::set_all_to()");
			for ele in vec.iter_mut() {
				*ele = val;
			}
		}

		self.flush_vec_wait();
	}

	/// [MARKED FOR POSSIBLE REMOVAL]: Convenience function.
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	/// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
	pub fn set_range_to(&mut self, val: T, range: Range<usize>) {		
		{
			let vec = self.vec.as_ref_mut().expect("Envoy::set_range_to()");
			// for idx in range {
				// self.vec[idx] = val;
			for ele in vec[range].iter_mut() {
				*ele = val;
			}
		}

		self.flush_vec_wait();
	}

	/// Returns the length of the Envoy.
	///
	/// This is the length of both the device side buffer and the host side vector,
	/// if any. This may not agree with desired dataset size because it will have been
	/// rounded up to the nearest maximum workgroup size of the device on which it was
	/// created.
	pub fn len(&self) -> usize {
		debug_assert!((if let VecOption::Some(ref vec) = self.vec { vec.len() } 
			else { self.len }) == self.len);
		self.len
	}

	/// [MARKED FOR POSSIBLE REMOVAL]: Convenience function. (move to tests module?)
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	/// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
	pub fn print_simple(&mut self) {
		self.print(1, None, None, true);
    }

    /// [MARKED FOR POSSIBLE REMOVAL]: Convenience function. (move to tests module?)
    ///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	/// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
    pub fn print_val_range(&mut self, every: usize, val_range: Option<(T, T)>,) {
		self.print(every, val_range, None, true);
    }

    /// [MARKED FOR POSSIBLE REMOVAL]: Convenience function. (move to tests module?)
    /// [FIXME]: CREATE A VECTOR FOR PRINTING IF NONE EXISTS
    ///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	/// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
    pub fn print(&mut self, every: usize, val_range: Option<(T, T)>, 
    			idx_range_opt: Option<Range<usize>>, zeros: bool)
	{
		let idx_range = match idx_range_opt.clone() {
			Some(r) => r,
			None => 0..self.len(),
		};

		let vec = self.vec.as_ref_mut().expect("Envoy::print()");

		super::enqueue_read_buffer(self.queue.obj(), self.buffer_obj, true, 
			&mut vec[idx_range.clone()], idx_range.start, None, None);
		fmt::print_vec(&vec[..], every, val_range, idx_range_opt, zeros);

	}

	/// Resizes Envoy. Recreates device side buffer and dangles any references 
	/// kernels may have had to the old buffer.
	///
	/// # Safety
	/// [IMPORTANT]: You must manually reassign any kernel arguments which may have 
	/// had a reference to the (device side) buffer associated with this Envoy.
	pub unsafe fn resize(&mut self, new_dims: &EnvoyDims/*, val: T*/) {
		self.release();
		let new_len = new_dims.padded_envoy_len(super::get_max_work_group_size(
			self.queue.device_id()));

		match self.vec {
			VecOption::Some(ref mut vec) => {
				vec.resize(new_len, T::default());
				self.buffer_obj = super::create_buffer(Some(vec), None, self.queue.context_obj(), 
					cl_h::CL_MEM_READ_WRITE | cl_h::CL_MEM_COPY_HOST_PTR);
			},
			VecOption::None => {
				self.len = new_len;
				// let vec: Vec<T> = iter::repeat(T::default()).take(new_len).collect();
				self.buffer_obj = super::create_buffer(None, Some((T::default(), new_len)), 
					self.queue.context_obj(), cl_h::CL_MEM_READ_WRITE);
			},
		};
	}

	/// Decrements the reference count associated with the previous buffer object, 
	/// `self.buffer_obj`.
    pub fn release(&mut self) {
		super::release_mem_object(self.buffer_obj);
	}

	/// Returns a reference to the local vector associated with this envoy.
	///
	/// # Failures
	/// Returns an error if this envoy contains no vector.
	pub fn vec(&self) -> OclResult<&Vec<T>> {
		self.vec.as_ref()
	}

	/// Returns a mutable reference to the local vector associated with this envoy.
	/// 
	/// # Failures
	/// Returns an error if this envoy contains no vector.
	///
	/// # Safety
	/// Could cause data collisions, etc. Probably not unsafe strictly speaking 
	/// (is it?) but marked as such to alert the caller to any potential 
	/// synchronization issues.
	pub unsafe fn vec_mut(&mut self) -> OclResult<&mut Vec<T>> {
		self.vec.as_ref_mut()
	}

	/// Returns an immutable reference to the value located at index `idx`, bypassing 
	/// bounds and enum variant checks.
	///
	/// # Safety
	/// Assumes `self.vec` is a `VecOption::Vec(_)` and that the index `idx` is within
	/// the vector bounds.
	pub unsafe fn get_unchecked(&self, idx: usize) -> &T {
		debug_assert!(self.vec.is_some() && idx < self.len);
		let vec_ptr: *const Vec<T> = &self.vec as *const VecOption<T> as *const Vec<T>;
		(*vec_ptr).get_unchecked(idx) 
	}

	/// Returns a mutable reference to the value located at index `idx`, bypassing 
	/// bounds and enum variant checks.
	///
	/// # Safety
	/// Assumes `self.vec` is a `VecOption::Vec(_)` and that the index `idx` is within
	/// bounds. Might eat all the laundry.
	pub unsafe fn get_unchecked_mut(&mut self, idx: usize) -> &mut T {		
		debug_assert!(self.vec.is_some() && idx < self.len);
		let vec_ptr_mut: *mut Vec<T> = &mut self.vec as *mut VecOption<T> as *mut Vec<T>;
		(*vec_ptr_mut).get_unchecked_mut(idx) 
	}

	/// Returns a copy of the device buffer object reference.
	pub fn buffer_obj(&self) -> cl_mem {
		self.buffer_obj
	}

	/// Returns a reference to the program/command queue associated with this envoy.
	pub fn queue(&self) -> &Queue {
		&self.queue
	}

	/// Changes the queue used by this Envoy for reads and writes, etc.
	///
	/// # Safety
	/// Not all implications of changing the queue, particularly if the new queue
	/// is associated with a new device which has different workgroup size dimensions,
	/// have been considered or are dealt with. For now, considering these cases is
	/// left to the caller. It's probably a good idea to at least call `.resize()`
	/// after calling this function.
	pub unsafe fn set_queue(&mut self, queue: Queue) {
		self.queue = queue;
	}

	/// Returns an iterator to a contained vector.
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	pub fn iter<'a>(&'a self) -> Iter<'a, T> {
		self.vec.as_ref().expect("Envoy::iter()").iter()
	}

	/// Returns a mutable iterator to a contained vector.
	///
	/// # Panics
	/// Panics if this Envoy contains no vector.
	pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, T> {
		self.vec.as_ref_mut().expect("Envoy::iter()").iter_mut()
	}
}

impl<T> Index<usize> for Envoy<T> {
    type Output = T;
    /// # Panics
	/// Panics if this Envoy contains no vector.
    fn index<'a>(&'a self, index: usize) -> &'a T {
        &self.vec.as_ref().expect("Envoy::index()")[..][index]
    }
}

impl<T> IndexMut<usize> for Envoy<T> {
	/// # Panics
	/// Panics if this Envoy contains no vector.
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut T {
    	&mut self.vec.as_ref_mut().expect("Envoy::index_mut()")[..][index]
    }
}

impl<'b, T> Index<&'b usize> for Envoy<T> {
    type Output = T;
    /// # Panics
	/// Panics if this Envoy contains no vector.
    fn index<'a>(&'a self, index: &'b usize) -> &'a T {
        &self.vec.as_ref().expect("Envoy::index()")[..][*index]
    }
}

impl<'b, T> IndexMut<&'b usize> for Envoy<T> {
	/// # Panics
	/// Panics if this Envoy contains no vector.
    fn index_mut<'a>(&'a mut self, index: &'b usize) -> &'a mut T {
        &mut self.vec.as_ref_mut().expect("Envoy::index_mut()")[..][*index]
    }
}

impl<T> Index<Range<usize>> for Envoy<T> {
    type Output = [T];
    /// # Panics
	/// Panics if this Envoy contains no vector.
    fn index<'a>(&'a self, range: Range<usize>) -> &'a [T] {
        &self.vec.as_ref().expect("Envoy::index()")[range]
    }
}

impl<T> IndexMut<Range<usize>> for Envoy<T> {
	/// # Panics
	/// Panics if this Envoy contains no vector.
    fn index_mut<'a>(&'a mut self, range: Range<usize>) -> &'a mut [T] {
    	&mut self.vec.as_ref_mut().expect("Envoy::index_mut()")[range]
    }
}

impl<T> Index<RangeFull> for Envoy<T> {
    type Output = [T];
    /// # Panics
	/// Panics if this Envoy contains no vector.
    fn index<'a>(&'a self, range: RangeFull) -> &'a [T] {
        &self.vec.as_ref().expect("Envoy::index()")[range]
    }
}

impl<T> IndexMut<RangeFull> for Envoy<T> {
	/// # Panics
	/// Panics if this Envoy contains no vector.
    fn index_mut<'a>(&'a mut self, range: RangeFull) -> &'a mut [T] {
    	&mut self.vec.as_ref_mut().expect("Envoy::index_mut()")[range]
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
	let mut rng = rand::weak_rng();
	let range = RandRange::new(min_val, max_val);

	(0..size).map(|_| range.ind_sample(&mut rng)).take(size as usize).collect()
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
