//! A sub-buffer allocator.

use ocl::flags::MemFlags;
use ocl::traits::OclPrm;
use ocl::{Buffer, Queue};
use std::collections::{HashMap, LinkedList};

pub struct PoolRegion {
    buffer_id: usize,
    origin: u32,
    len: u32,
}

/// A simple (linear search) sub-buffer allocator.
pub struct SubBufferPool<T: OclPrm> {
    buffer: Buffer<T>,
    regions: LinkedList<PoolRegion>,
    sub_buffers: HashMap<usize, Buffer<T>>,
    align: u32,
    _next_uid: usize,
}

impl<T: OclPrm> SubBufferPool<T> {
    /// Returns a new buffer pool.
    pub fn new(len: u32, default_queue: Queue) -> SubBufferPool<T> {
        let align = default_queue.device().mem_base_addr_align().unwrap();
        let flags = MemFlags::new().alloc_host_ptr().read_write();

        let buffer = Buffer::<T>::builder()
            .queue(default_queue)
            .flags(flags)
            .len(len as usize)
            .build()
            .unwrap();

        SubBufferPool {
            buffer,
            regions: LinkedList::new(),
            sub_buffers: HashMap::new(),
            align,
            _next_uid: 0,
        }
    }

    fn next_valid_align(&self, unaligned_origin: u32) -> u32 {
        ((unaligned_origin / self.align) + 1) * self.align
    }

    fn next_uid(&mut self) -> usize {
        self._next_uid += 1;
        self._next_uid - 1
    }

    fn insert_region(&mut self, region: PoolRegion, region_idx: usize) {
        let mut tail = self.regions.split_off(region_idx);
        self.regions.push_back(region);
        self.regions.append(&mut tail);
    }

    fn create_sub_buffer(
        &mut self,
        region_idx: usize,
        flags: Option<MemFlags>,
        origin: u32,
        len: u32,
    ) -> usize {
        let buffer_id = self.next_uid();
        let region = PoolRegion {
            buffer_id,
            origin,
            len,
        };
        let sbuf = self
            .buffer
            .create_sub_buffer(flags, region.origin as usize, region.len as usize)
            .unwrap();
        if let Some(idx) = self.sub_buffers.insert(region.buffer_id, sbuf) {
            panic!("Duplicate indexes: {}", idx);
        }
        self.insert_region(region, region_idx);
        buffer_id
    }

    /// Allocates space for and creates a new sub-buffer then returns the
    /// buffer id which can be used to `::get` or `::free` it.
    pub fn alloc(&mut self, len: u32, flags: Option<MemFlags>) -> Result<usize, ()> {
        assert!(self.regions.len() == self.sub_buffers.len());

        match self.regions.front() {
            Some(_) => {
                let mut end_prev = 0;
                let mut create_at = None;

                for (region_idx, region) in self.regions.iter().enumerate() {
                    if region.origin - end_prev >= len {
                        create_at = Some(region_idx);
                        break;
                    } else {
                        end_prev = self.next_valid_align(region.origin + region.len);
                    }
                }

                if let Some(region_idx) = create_at {
                    Ok(self.create_sub_buffer(region_idx, flags, end_prev, len))
                } else if self.buffer.len() as u32 - end_prev >= len {
                    let region_idx = self.regions.len();
                    Ok(self.create_sub_buffer(region_idx, flags, end_prev, len))
                } else {
                    Err(())
                }
            }
            None => Ok(self.create_sub_buffer(0, flags, 0, len)),
        }
    }

    /// Deallocates the buffer identified by `buffer_id` or returns it back in
    /// the event of a failure.
    pub fn free(&mut self, buffer_id: usize) -> Result<(), usize> {
        let mut region_idx = None;

        // The `SubBuffer` drops here when it goes out of scope:
        if let Some(_) = self.sub_buffers.remove(&buffer_id) {
            region_idx = self.regions.iter().position(|r| r.buffer_id == buffer_id);
        }

        if let Some(r_idx) = region_idx {
            let mut tail = self.regions.split_off(r_idx);
            tail.pop_front().ok_or(buffer_id)?;
            self.regions.append(&mut tail);
            Ok(())
        } else {
            Err(buffer_id)
        }
    }

    /// Returns a reference to the sub-buffer identified by `buffer_id`.
    pub fn get(&self, buffer_id: usize) -> Option<&Buffer<T>> {
        self.sub_buffers.get(&buffer_id)
    }

    /// Returns a mutable reference to the sub-buffer identified by `buffer_id`.
    #[allow(dead_code)]
    pub fn get_mut(&mut self, buffer_id: usize) -> Option<&mut Buffer<T>> {
        self.sub_buffers.get_mut(&buffer_id)
    }

    /// Defragments the buffer. Be sure to `::finish()` any and all command
    /// queues you may be using before doing this.
    ///
    /// All kernels with a buffer argument set to any of the sub-buffers in
    /// this pool will need to be created anew or arguments refreshed (use
    /// `Kernel::arg_..._named` when initializing arguments in order to change
    /// them later).
    #[allow(dead_code)]
    pub fn defrag(&mut self) {
        // - Iterate through sub-buffers, dropping and recreating at the
        //   leftmost (lowest) available address within the buffer, optionally
        //   copying old data to the new position in device memory.
        // - Rebuild regions list (or just update offsets).
        unimplemented!();
    }

    /// Shrinks or grows and defragments the main buffer. Invalidates all
    /// kernels referencing sub-buffers in this pool. See `::defrag` for more
    /// information.
    #[allow(dead_code, unused_variables)]
    pub fn resize(&mut self, len: u32) {
        // Allocate a new buffer then copy old buffer contents one sub-buffer
        // at a time, defragmenting in the process.
        unimplemented!();
    }
}
