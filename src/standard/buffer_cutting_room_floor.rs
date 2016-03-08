use std::ops::{Range, RangeFull, Index, IndexMut};
use std::default::Default;
use std::slice::{Iter, IterMut};
use util;

static VEC_OPT_ERR_MSG: &'static str = "No host side vector defined for this Buffer. \
    You must create this Buffer using 'Buffer::with_vec()' (et al.) in order to call this method.";


pub trait BufferExtras<T: OclPrm> {
    fn with_vec_initialized_to<D: MemLen>(init_val: T, dims: D, queue: &Queue) -> Buffer<T>;
    fn with_vec_shuffled<D: MemLen>(vals: (T, T), dims: D, queue: &Queue) -> Buffer<T>;
    fn with_vec_scrambled<D: MemLen>(vals: (T, T), dims: D, queue: &Queue) -> Buffer<T>;
    fn.cmd().fill([&mut self, val: T]).enq() -> OclResult<()>;
    fn set_range_to(&mut self, val: T, range: Range<usize>) -> OclResult<()>;
    unsafe fn resize<B: MemLen>(&mut self, new_dims: &B, queue: &Queue);
    fn print_simple(&mut self);
    fn print_val_range(&mut self, every: usize, val_range: Option<(T, T)>,);
    fn print(&mut self, every: usize, val_range: Option<(T, T)>, 
                    idx_range_opt: Option<Range<usize>>, zeros: bool);
}


// An option type mainly just for convenient error handling.
#[derive(Debug, Clone)]
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

    fn as_mut(&mut self) -> OclResult<&mut Vec<T>> {
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




    /// Creates a new read/write Buffer with dimensions: `dims` which will use the 
    /// command queue: `queue` (and its associated device and context) for all operations.
    ///
    /// The device side buffer will be allocated a size based on the maximum workgroup 
    /// size of the device. This helps ensure that kernels do not attempt to read 
    /// from or write to memory beyond the length of the buffer (see crate level 
    /// documentation for more details about how dimensions are used). The buffer
    /// will be initialized with a sensible default value (probably `0`).
    ///
    /// ## Other Method Panics
    /// The returned Buffer contains no host side vector. Functions associated with
    /// one such as `.enqueue_flush_vec()`, `enqueue_fill_vec()`, etc. will panic.
    /// [FIXME]: Return result.
    pub fn new<D: MemLen>(dims: D, queue: &Queue) -> Buffer<T> {
        let dims: SpatialDims = dims.to_lens().into(); 
        // let len = dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");
        let len = dims.to_len();
        Buffer::_new(dims, len, queue)
    }

    /// Creates a new read/write Buffer with a host side working copy of data.
    /// Host vector and device buffer are initialized with a sensible default value.
    /// [FIXME]: Return result.
    pub fn with_vec<D: MemLen>(dims: D, queue: &Queue) -> Buffer<T> {
        let len = dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");
        let vec: Vec<T> = std::iter::repeat(T::default()).take(len).collect();

        Buffer::_with_vec(vec, queue)
    }

    /// Creates a new Buffer with caller-managed buffer length, type, flags, and 
    /// initialization.
    ///
    /// - Does not optimize size. 
    ///
    /// [DOCUMENTATION OUT OF DATE]
    ///
    /// ## Examples
    /// See `examples/buffer_unchecked.rs`.
    ///
    /// ## Parameter Reference Documentation
    /// Refer to the following page for information about how to configure flags and
    /// other options for the buffer.
    /// [https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateBuffer.html)
    ///
    /// ## Safety
    /// No creation time checks are made to help prevent device buffer overruns, 
    /// etc. Caller should be sure to initialize the device side memory with a 
    /// write. Any host side memory pointed to by `host_ptr` is at even greater
    /// risk if used with `CL_MEM_USE_HOST_PTR` (see below).
    ///
    /// [IMPORTANT] Practically every read and write to an Buffer created in this way is
    /// potentially unsafe. Because `.enqueue_read()` and `.enqueue_write()` do not require an 
    /// unsafe block, their implied promises about safety may be broken at any time.
    ///
    /// *You need to know what you're doing and be extra careful using an Buffer created 
    /// with this method because it badly breaks Rust's usual safety promises even
    /// outside of an unsafe block.*
    ///
    /// **This is horribly un-idiomatic Rust. You have been warned.**
    ///
    /// NOTE: The above important warnings probably only apply to buffers created with 
    /// the `CL_MEM_USE_HOST_PTR` flag because the memory is considered 'pinned' but 
    /// there may also be implementation specific issues which haven't been considered 
    /// or are unknown.
    ///
    /// [FIXME]: Return result.
    /// [FIXME]: Update docs.
    pub unsafe fn new_unchecked<S>(flags: MemFlags, dims: S, host_ptr: Option<&[T]>, 
                queue: &Queue) -> Buffer<T> where S: Into<SpatialDims>
    {
        let dims = dims.into();
        let len = dims.to_len();
        let obj_core = core::create_buffer(queue.context_core_as_ref(), flags, len,
            host_ptr).expect("[FIXME: TEMPORARY]: Buffer::_new():");

        Buffer {
            obj_core: obj_core,
            // command_queue_obj_core: queue.core_as_ref().clone(),
            queue: queue.clone(),
            dims: dims,
            len: len,
            _data: PhantomData,
            // vec: VecOption::None,
        }
    }

    /// Consolidated constructor for Buffers without vectors.
    /// [FIXME]: Return result.
    fn _new(dims: SpatialDims, len: usize, queue: &Queue) -> Buffer<T> {
        let obj_core = unsafe { core::create_buffer::<T>(queue.context_core_as_ref(),
            core::MEM_READ_WRITE, len, None).expect("Buffer::_new()") };

        Buffer {            
            obj_core: obj_core,
            command_queue_obj_core: queue.core_as_ref().clone(),
            dims: dims,
            len: len,
            _data: PhantomData,
            // vec: VecOption::None,
        }
    }


    // Consolidated constructor for Buffers with vectors.
    /// [FIXME]: Return result.
    fn _with_vec(mut vec: Vec<T>, queue: &Queue) -> Buffer<T> {
        let obj_core = unsafe { core::create_buffer(queue.context_core_as_ref(), 
            core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR, vec.len(), Some(&mut vec))
            .expect("Buffer::_with_vec()") };

        Buffer {        
            obj_core: obj_core,
            command_queue_obj_core: queue.core_as_ref().clone(),
            len: vec.len(), 
            vec: VecOption::Some(vec),
        }
    }

    /// Enqueues reading `data.len() * mem::size_of::<T>()` bytes from the device 
    /// buffer into `data` with a remote offset of `offset`.
    ///
    /// Setting `queue` to `None` will use the default queue set during creation.
    /// Otherwise, the queue passed will be used for this call only.
    ///
    /// Will optionally wait for events in `ewait` to finish 
    /// before reading. Will also optionally add a new event associated with
    /// the read to `enew`.
    ///
    /// [UPDATE] If the `enew` event list is `None`, the read will be blocking, otherwise
    /// returns immediately.
    ///
    /// ## Safety
    ///
    /// Bad things will happen if the memory referred to by `data` is freed and
    /// reallocated before the read completes. It's up to the caller to make 
    /// sure that the new event added to `enew` completes. Use 
    /// 'enew.last()' right after the calling `::read_async` to get a.
    /// reference to the event associated with the read. [NOTE: Improved ease
    /// of use is coming to the event api eventually]
    ///
    /// ## Errors
    ///
    /// `offset` must be less than the length of the buffer.
    ///
    /// The length of `data` must be less than the length of the buffer minus `offset`.
    ///
    /// Errors upon any OpenCL error.
    ///
    /// [UNSTABLE: Likely to be depricated in favor of `::cmd`.
    unsafe fn enqueue_read(&self, queue: Option<&Queue>, block: bool, offset: usize, data: &mut [T],
                ewait: Option<&EventList>, enew: Option<&mut ClEventPtrNew>) -> OclResult<()>
    {
        // assert!(offset < self.len(), "Buffer::read{{_async}}(): Offset out of range.");
        // assert!(data.len() <= self.len() - offset, 
        //     "Buffer::read{{_async}}(): Data length out of range.");
        if offset >= self.len() { 
            return OclError::err("Buffer::read{{_async}}(): Offset out of range."); }
        if data.len() > self.len() - offset {
            return OclError::err("Buffer::read{{_async}}(): Data length out of range."); }

        let command_queue = match queue {
            Some(q) => q,
            None => &self.queue,
        };

        // let blocking_read = enew.is_none();
        core::enqueue_read_buffer(command_queue, &self.obj_core, block, 
            // offset, data, ewait.map(|el| el.core_as_ref()), enew.map(|el| el.core_as_mut()))
            offset, data, ewait, enew)
    }

    /// Enqueues writing `data.len() * mem::size_of::<T>()` bytes from `data` to the 
    /// device buffer with a remote offset of `offset`.
    ///
    /// Setting `queue` to `None` will use the default queue set during creation.
    /// Otherwise, the queue passed will be used for this call only.
    ///
    /// Will optionally wait for events in `ewait` to finish before writing. 
    /// Will also optionally add a new event associated with the write to `enew`.
    ///
    /// [UPDATE] If the `enew` event list is `None`, the write will be blocking, otherwise
    /// returns immediately.
    ///
    /// ## Data Integrity
    ///
    /// Ensure that the memory referred to by `data` is unmolested until the 
    /// write completes if passing a `enew`.
    ///
    /// ## Errors
    ///
    /// `offset` must be less than the length of the buffer.
    ///
    /// The length of `data` must be less than the length of the buffer minus `offset`.
    ///
    /// Errors upon any OpenCL error.
    /// 
    /// [UNSTABLE: Likely to be depricated in favor of `::cmd`.
    fn enqueue_write(&self, queue: Option<&Queue>, block: bool, offset: usize, data: &[T], 
                ewait: Option<&EventList>, enew: Option<&mut ClEventPtrNew>) -> OclResult<()>
    {        
        // assert!(offset < self.len(), "Buffer::write{{_async}}(): Offset out of range.");
        // assert!(data.len() <= self.len() - offset, 
        //     "Buffer::write{{_async}}(): Data length out of range.");
        if offset >= self.len() { 
            return OclError::err("Buffer::write{{_async}}(): Offset out of range."); }
        if data.len() > self.len() - offset {
            return OclError::err("Buffer::write{{_async}}(): Data length out of range."); }

        let command_queue = match queue {
            Some(q) => q,
            None => &self.queue,
        };

        // let blocking_write = enew.is_none();
        core::enqueue_write_buffer(command_queue, &self.obj_core, block, 
            // offset, data, ewait.map(|el| el.core_as_ref()), enew.map(|el| el.core_as_mut()))
            offset, data, ewait, enew)
    }

    /// Reads `data.len() * mem::size_of::<T>()` bytes from the (remote) device buffer 
    /// into `data` with a remote offset of `offset` and blocks until complete.
    ///
    /// ## Errors
    ///
    /// `offset` must be less than the length of the buffer.
    ///
    /// The length of `data` must be less than the length of the buffer minus `offset`.
    ///
    /// Errors upon any OpenCL error.
    pub fn read_old(&self, data: &mut [T])
    {
        // Safe due to being a blocking read (right?).
        unsafe { self.enqueue_read(None, true, 0, data, None, None).expect("ocl::Buffer::read()") }
    }

    /// Writes `data.len() * mem::size_of::<T>()` bytes from `data` to the (remote) 
    /// device buffer with a remote offset of `offset` and blocks until complete.
    ///
    /// ## Panics
    ///
    /// `offset` must be less than the length of the buffer.
    ///
    /// The length of `data` must be less than the length of the buffer minus `offset`.
    ///
    /// Errors upon any OpenCL error.
    pub fn write_old(&self, data: &[T])
    {
        self.enqueue_write(None, true, 0, data, None, None).expect("ocl::Buffer::read()")
    }

    /// After waiting on events in `ewait` to finish, reads the remote device 
    /// data buffer into 'self.vec' and adds a new event to `enew`.
    ///
    /// [UPDATE] Will block until the read is complete and the internal vector is filled if 
    /// `enew` is `None`.
    ///
    /// ## Safety 
    ///
    /// Currently up to the caller to ensure this `Buffer` lives long enough
    /// for the read to complete.
    ///
    /// TODO: Keep an internal eventlist to track pending reads and cancel them
    /// if this `Buffer` is destroyed beforehand.
    ///
    /// ## Errors
    ///
    /// Errors if this Buffer contains no vector or upon any OpenCL error.
    unsafe fn enqueue_fill_vec(&mut self, block: bool, ewait: Option<&EventList>, 
                enew: Option<&mut ClEventPtrNew>) -> OclResult<()>
    {
        debug_assert!(self.vec.as_ref().unwrap().len() == self.len());
        let vec = try!(self.vec.as_mut());
        core::enqueue_read_buffer(&self.command_queue_obj_core, &self.obj_core, block, 
            // 0, vec, ewait.map(|el| el.core_as_ref()), enew.map(|el| el.core_as_mut()))
            0, vec, ewait, enew)
    }

    /// Reads the remote device data buffer into `self.vec` and blocks until completed.
    ///
    /// Equivalent to `.enqueue_fill_vec(true, None, None)`.
    ///
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector or upon any OpenCL error.
    pub fn fill_vec(&mut self) {
        // Safe due to being a blocking read (right?).
        unsafe { self.enqueue_fill_vec(true, None, None).expect("Buffer::fill_vec()"); }
    } 

    /// After waiting on events in `ewait` to finish, writes the contents of
    /// 'self.vec' to the remote device data buffer and adds a new event to `enew`.
    ///
    /// ## Data Integrity
    ///
    /// Ensure that this `Buffer` lives until until the write completes if 
    /// passing a `enew`.
    ///
    /// [UPDATE] Will block until the write is complete if `enew` is None.
    ///
    /// ## Errors
    ///
    /// Errors if this Buffer contains no vector or upon any OpenCL error.
    fn enqueue_flush_vec(&mut self, block: bool, ewait: Option<&EventList>, 
                enew: Option<&mut ClEventPtrNew>) -> OclResult<()>
    {
        debug_assert!(self.vec.as_ref().unwrap().len() == self.len());
        let vec = try!(self.vec.as_mut());
        core::enqueue_write_buffer(&self.command_queue_obj_core, &self.obj_core, block, 
            // 0, vec, ewait.map(|el| el.core_as_ref()), enew.map(|el| el.core_as_mut()))
            0, vec, ewait, enew)
    }

    /// Writes the contents of `self.vec` to the remote device data buffer and 
    /// blocks until completed. 
    ///
    /// Equivalent to `.enqueue_flush_vec(true, None, None)`.
    ///
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector or upon any OpenCL error.
    pub fn flush_vec(&mut self) {
        self.enqueue_flush_vec(true, None, None).expect("Buffer::flush_vec()");
    }  

    /// Returns a reference to the local vector associated with this buffer.
    ///
    /// Contents of this vector may change during use due to previously enqueued
    /// reads. ([FIXME]: Is this a safety issue?)
    ///
    /// ## Failures
    ///
    /// [FIXME: UPDATE DOC] Returns an error if this buffer contains no vector.
    #[inline]
    pub fn vec(&self) -> &Vec<T> {
        self.vec.as_ref().expect("Buffer::vec()")
    }

    /// Returns an iterator to a contained vector.
    ///
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    pub fn iter<'a>(&'a self) -> Iter<'a, T> {
        self.vec.as_ref().expect("Buffer::iter()").iter()
    }

    /// Returns a mutable iterator to a contained vector.
    ///
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, T> {
        self.vec.as_mut().expect("Buffer::iter()").iter_mut()
    }


    /// Returns a mutable reference to the local vector associated with this buffer.
    ///
    /// Contents of this vector may change during use due to previously enqueued
    /// read.
    /// 
    /// ## Failures
    ///
    /// [FIXME: UPDATE DOC] Returns an error if this buffer contains no vector.
    ///
    /// ## Safety
    ///
    /// Could cause data collisions, etc. May not be unsafe strictly speaking
    /// (is it?) but marked as such to alert the caller to any potential 
    /// synchronization issues from previously enqueued reads.
    #[inline]
    pub unsafe fn vec_mut(&mut self) -> &mut Vec<T> {
        self.vec.as_mut().expect("Buffer::vec_mut()")
    }

    /// Returns an immutable reference to the value located at index `idx`, bypassing 
    /// bounds and enum variant checks.
    ///
    /// ## Safety
    ///
    /// Assumes `self.vec` is a `VecOption::Vec` and that the index `idx` is within
    /// the vector bounds.
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> &T {
        debug_assert!(self.vec.is_some() && idx < self.len);
        let vec_ptr: *const Vec<T> = &self.vec as *const VecOption<T> as *const Vec<T>;
        (*vec_ptr).get_unchecked(idx) 
    }

    /// Returns a mutable reference to the value located at index `idx`, bypassing 
    /// bounds and enum variant checks.
    ///
    /// ## Safety
    ///
    /// Assumes `self.vec` is a `VecOption::Vec` and that the index `idx` is within
    /// bounds. Can eat all the laundry.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, idx: usize) -> &mut T {      
        debug_assert!(self.vec.is_some() && idx < self.len);
        let vec_ptr_mut: *mut Vec<T> = &mut self.vec as *mut VecOption<T> as *mut Vec<T>;
        (*vec_ptr_mut).get_unchecked_mut(idx) 
    }








impl<T: OclPrm> BufferExtras<T> for Buffer<T> {
    /// [UNSTABLE]: Convenience method.
    /// Creates a new read/write Buffer with a host side working copy of data.
    /// Host vector and device buffer are initialized with the value, `init_val`.
    /// [FIXME]: Return result.
    fn with_vec_initialized_to<D: MemLen>(init_val: T, dims: D, queue: &Queue) -> Buffer<T> {
        let len = dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");
        let vec: Vec<T> = std::iter::repeat(init_val).take(len).collect();

        Buffer::_with_vec(vec, queue)
    }

    /// [UNSTABLE]: Convenience method.
    /// Creates a new read/write Buffer with a vector initialized with a series of 
    /// integers ranging from `vals.0` to `vals.1` (closed) which are shuffled 
    /// randomly.
    ///
    /// Note: Even if the Buffer type is a floating point type, the values returned
    /// will still be integral values (e.g.: 1.0, 2.0, 3.0, etc.).
    ///
    /// ## Security
    ///
    /// Resulting values are not cryptographically secure.
    /// [FIXME]: Return result.
    // Note: vals.1 is inclusive.
    fn with_vec_shuffled<D: MemLen>(vals: (T, T), dims: D, queue: &Queue) 
            -> Buffer<T> 
    {
        let len = dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");
        let vec: Vec<T> = util::shuffled_vec(len, vals);

        Buffer::_with_vec(vec, queue)
    }

    /// [UNSTABLE]: Convenience method.
    /// Creates a new read/write Buffer with a vector initialized with random values 
    /// within the (half-open) range `vals.0..vals.1`.
    ///
    /// ## Security
    ///
    /// Resulting values are not cryptographically secure.
    /// [FIXME]: Return result.
    // Note: vals.1 is exclusive.
    fn with_vec_scrambled<D: MemLen>(vals: (T, T), dims: D, queue: &Queue) 
            -> Buffer<T> 
    {
        let len = dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");
        let vec: Vec<T> = util::scrambled_vec(len, vals);

        Buffer::_with_vec(vec, queue)
    }   

    /// [UNSTABLE]: Convenience method.
    ///
    /// ## Panics [UPDATE ME]
    /// Panics if this Buffer contains no vector.
    /// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
    /// TODO: Consider adding to `BufferCmd`.
    fn.cmd().fill([&mut self, val: T]).enq() -> OclResult<()> {
        {
            let vec = try!(self.vec.as_mut());
            for ele in vec.iter_mut() {
                *ele = val;
            }
        }

        self.enqueue_flush_vec(true, None, None)
    }

    /// [UNSTABLE]: Convenience method.
    ///
    /// ## Panics [UPDATE ME]
    ///
    /// Panics if this Buffer contains no vector.
    ///
    /// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
    /// TODO: Consider adding to `BufferCmd`.
    fn set_range_to(&mut self, val: T, range: Range<usize>) -> OclResult<()> {       
        {
            let vec = try!(self.vec.as_mut());
            // for idx in range {
                // self.vec[idx] = val;
            for ele in vec[range].iter_mut() {
                *ele = val;
            }
        }

        self.enqueue_flush_vec(true, None, None)
    }

    /// [UNSTABLE]: Resizes Buffer. Recreates device side buffer and dangles any references 
    /// kernels may have had to the old buffer.
    ///
    /// ## Safety
    ///
    /// [IMPORTANT]: You must manually reassign any kernel arguments which may have 
    /// had a reference to the (device side) buffer associated with this Buffer.
    /// [FIXME]: Return result.
    unsafe fn resize<B: MemLen>(&mut self, new_dims: &B, queue: &Queue) {
        // self.release();
        let new_len = new_dims.to_len_padded(queue.device().max_wg_size()).expect("[FIXME]: Buffer::new: TEMP");

        match self.vec {
            VecOption::Some(ref mut vec) => {
                vec.resize(new_len, T::default());
                self.obj_core = core::create_buffer(queue.context_core_as_ref(), 
                    core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR, self.len, Some(vec))
                    .expect("[FIXME: TEMPORARY]: Buffer::_resize()");
            },
            VecOption::None => {
                self.len = new_len;
                // let vec: Vec<T> = std::iter::repeat(T::default()).take(new_len).collect();
                self.obj_core = core::create_buffer::<T>(queue.context_core_as_ref(), 
                    core::MEM_READ_WRITE, self.len, None)
                    .expect("[FIXME: TEMPORARY]: Buffer::_resize()");
            },
        };
    }

    
    /// Blocks the current thread until the underlying command queue has
    /// completed all commands.
    pub fn wait(&self) {
        core::finish(&self.queue).unwrap();
    }

    /// [UNSTABLE]: Convenience method.
    ///
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    /// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
    fn print_simple(&mut self) {
        self.print(1, None, None, true);
    }

    /// [UNSTABLE]: Convenience method. 
    ///
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    /// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
    fn print_val_range(&mut self, every: usize, val_range: Option<(T, T)>,) {
        self.print(every, val_range, None, true);
    }


    /// [UNSTABLE]: Convenience/debugging method. May be moved/renamed/deleted.
    /// [FIXME]: CREATE AN EMPTY VECTOR FOR PRINTING IF NONE EXISTS INSTEAD
    /// OF PANICING.
    ///
    ///
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    /// [FIXME]: GET WORKING EVEN WITH NO CONTAINED VECTOR
    fn print(&mut self, every: usize, val_range: Option<(T, T)>, 
                idx_range_opt: Option<Range<usize>>, zeros: bool)
    {
        let idx_range = match idx_range_opt.clone() {
            Some(r) => r,
            None => 0..self.len(),
        };

        let vec = self.vec.as_mut().expect("Buffer::print()");

        unsafe { core::enqueue_read_buffer::<T, EventList>(
            &self.command_queue_obj_core, &self.obj_core, true, idx_range.start, 
            &mut vec[idx_range.clone()], None, None).unwrap() };
        util::print_slice(&vec[..], every, val_range, idx_range_opt, zeros);

    }
}



impl<T: OclPrm> Index<usize> for Buffer<T> {
    type Output = T;
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    ///
    #[inline]
    fn index<'a>(&'a self, index: usize) -> &'a T {
        &self.vec.as_ref().expect("Buffer::index()")[..][index]
    }
}

impl<T: OclPrm> IndexMut<usize> for Buffer<T> {
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    ///
    #[inline]
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut T {
        &mut self.vec.as_mut().expect("Buffer::index_mut()")[..][index]
    }
}

impl<'b, T: OclPrm> Index<&'b usize> for Buffer<T> {
    type Output = T;
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    ///
    #[inline]
    fn index<'a>(&'a self, index: &'b usize) -> &'a T {
        &self.vec.as_ref().expect("Buffer::index()")[..][*index]
    }
}

impl<'b, T: OclPrm> IndexMut<&'b usize> for Buffer<T> {
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    ///
    #[inline]
    fn index_mut<'a>(&'a mut self, index: &'b usize) -> &'a mut T {
        &mut self.vec.as_mut().expect("Buffer::index_mut()")[..][*index]
    }
}

impl<T: OclPrm> Index<Range<usize>> for Buffer<T> {
    type Output = [T];
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    ///
    #[inline]
    fn index<'a>(&'a self, range: Range<usize>) -> &'a [T] {
        &self.vec.as_ref().expect("Buffer::index()")[range]
    }
}

impl<T: OclPrm> IndexMut<Range<usize>> for Buffer<T> {
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    ///
    #[inline]
    fn index_mut<'a>(&'a mut self, range: Range<usize>) -> &'a mut [T] {
        &mut self.vec.as_mut().expect("Buffer::index_mut()")[range]
    }
}

impl<T: OclPrm> Index<RangeFull> for Buffer<T> {
    type Output = [T];
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    ///
    #[inline]
    fn index<'a>(&'a self, range: RangeFull) -> &'a [T] {
        &self.vec.as_ref().expect("Buffer::index()")[range]
    }
}

impl<T: OclPrm> IndexMut<RangeFull> for Buffer<T> {
    /// ## Panics
    ///
    /// Panics if this Buffer contains no vector.
    ///
    #[inline]
    fn index_mut<'a>(&'a mut self, range: RangeFull) -> &'a mut [T] {
        &mut self.vec.as_mut().expect("Buffer::index_mut()")[range]
    }
}

// #[cfg(test)]
#[cfg(not(release))]
pub mod tests {
    use super::Buffer;
    use core::OclPrm;
    use std::num::Zero;

    /// Test functions available to external crates.
    pub trait BufferTest<T> {
        fn read_idx_direct(&self, idx: usize) -> T;
    }

    impl<T: OclPrm> BufferTest<T> for Buffer<T> {
        // Throw caution to the wind (this is potentially unsafe).
        fn read_idx_direct(&self, idx: usize) -> T {
            let mut buffer = vec![Zero::zero()];
            self.cmd().read(&mut buffer[0..1]).offset(idx).enq().unwrap();
            buffer[0]
        }
    }
}

