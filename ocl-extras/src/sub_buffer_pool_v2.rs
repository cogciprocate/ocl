//! A sub-buffer allocator.

use std::collections::{LinkedList, HashMap};
use ocl::{Queue, Buffer, SubBuffer, OclPrm};
use ocl::flags::MemFlags;


pub struct FreeRegion {
    base_chunk_idx: usize,
    len: u32,
}

pub struct PoolRegion {
    buffer_id: usize,
    origin: u32,
    len: u32,
}


pub struct SubBufferPool {
    buffer: Buffer<T>,
    sub_buffers: HashMap<usize, SubBuffer<T>>,
    regions: BTreeMap<PoolRegion>,
    chunk_size: u32,
    chunk_count: u32,
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
            .dims(len)
            .build().unwrap();

        SubBufferPool {
            buffer: buffer,
            regions: LinkedList::new(),
            sub_buffers: HashMap::new(),
            align: align,
            _next_uid: 0,
        }
    }

    fn next_uid(&mut self) -> usize {
        self._next_uid += 1;
        self._next_uid - 1
    }

    fn next_chunk_idx(&self, addr: u32) -> usize {
        (addr / self.chunk_size) + 1)
    }

    fn create_sub_buffer(&mut self, region_idx: usize, flags: Option<MemFlags>,
            origin: u32, len: u32) -> usize
    {
        let buffer_id = self.next_uid();
        let region = PoolRegion { buffer_id: buffer_id, origin: origin, len: len };
        let sbuf = self.buffer.create_sub_buffer(flags, region.origin, region.len).unwrap();
        if let Some(idx) = self.sub_buffers.insert(region.buffer_id, sbuf) {
            panic!("Duplicate indexes: {}", idx); }
        self.insert_region(region, region_idx);
        buffer_id
    }
}