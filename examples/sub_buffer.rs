//! A basic example of dynamically generated sub-buffers and programs.
//!
//!

#![allow(unused_variables)]

extern crate ocl;
extern crate rand;

use rand::distributions::Range as RandRange;
use std::collections::{LinkedList, HashMap};
use ocl::{flags, Platform, Device, Context, Queue, Program,
    Buffer, Kernel, SubBuffer, OclPrm};
use ocl::aliases::ClFloat4;

const INITIAL_PROGRAM_COUNT: usize = 25;
const INITIAL_BUFFER_LEN: usize = 2 << 23; // 256MiB of ClFloat4
const SUB_BUF_MIN_LEN: usize = 2 << 11; // 64KiB of ClFloat4
const SUB_BUF_MAX_LEN: usize = 2 << 15; // 1MiB of ClFloat4


struct PoolRegion {
    buffer_id: usize,
    origin: usize,
    len: usize,
}


/// A simple but slow sub-buffer allocator based on a linked list.
pub struct BufferPool<T: OclPrm> {
    buffer: Buffer<T>,
    regions: LinkedList<PoolRegion>,
    sub_buffers: HashMap<usize, SubBuffer<T>>,
    _next_uid: usize,
}

impl<T: OclPrm> BufferPool<T> {
    /// Returns a new buffer pool.
    pub fn new(len: usize, default_queue: Queue) -> BufferPool<T> {
        let buffer = Buffer::<T>::new(default_queue, Some(flags::MEM_READ_WRITE),
            len, None).unwrap();

        BufferPool {
            buffer: buffer,
            regions: LinkedList::new(),
            sub_buffers: HashMap::new(),
            _next_uid: 0,
        }
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

    fn create_sub_buffer(&mut self, region_idx: usize, origin: usize, len: usize) -> usize {
        let buffer_id = self.next_uid();
        let region = PoolRegion { buffer_id: buffer_id, origin: origin, len: len };
        let sbuf = self.buffer.create_sub_buffer(Some(flags::MEM_ALLOC_HOST_PTR),
            region.origin, region.len).unwrap();
        self.sub_buffers.insert(region.buffer_id, sbuf).unwrap();
        self.insert_region(region, region_idx);
        buffer_id
    }

    /// Allocates space for and creates a new sub-buffer then returns the
    /// buffer id which can be used to `::get` or `::free` it.
    pub fn alloc(&mut self, len: usize) -> Result<usize, ()> {
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
                        end_prev = region.origin + region.len;
                    }
                }

                if let Some(region_idx) = create_at {
                    Ok(self.create_sub_buffer(region_idx, end_prev, len))
                } else if self.buffer.len() - end_prev >= len {
                    let region_idx = self.regions.len();
                    Ok(self.create_sub_buffer(region_idx, end_prev, len))
                } else {
                    Err(())
                }
            },
            None => {
                Ok(self.create_sub_buffer(0, 0, len))
            },
        }
    }

    /// Deallocates the buffer identified by `buffer_id`. Returns the same
    /// `buffer_id` if successful.
    pub fn free(&mut self, buffer_id: usize) -> Result<(), ()> {
        let mut region_idx = None;

        // SubBuffer drops here when it goes out of scope:
        if let Some(_) = self.sub_buffers.remove(&buffer_id) {
            region_idx = self.regions.iter().position(|r| r.buffer_id == buffer_id);
        }

        if let Some(r_idx) = region_idx {
            let mut tail = self.regions.split_off(r_idx);
            tail.pop_front().ok_or(())?;
            self.regions.append(&mut tail);
            Ok(())
        } else {
            Err(())
        }
    }

    /// Returns a reference to the sub-buffer identified by `buffer_id`.
    pub fn get(&self, buffer_id: usize) -> Option<&SubBuffer<T>> {
        self.sub_buffers.get(&buffer_id)
    }

    /// Returns a mutable reference to the sub-buffer identified by `buffer_id`.
    pub fn get_mut(&mut self, buffer_id: usize) -> Option<&mut SubBuffer<T>> {
        self.sub_buffers.get_mut(&buffer_id)
    }

    /// Defragments the buffer. Be sure to `::finish()` any and all command
    /// queues you may be using before doing this.
    ///
    /// All kernels with a buffer argument set to any of the sub-buffers in
    /// this pool will need to be created anew or arguments refreshed (use
    /// `Kernel::arg_..._named` when initializing arguments in order to change
    /// them later).
    pub fn defrag(&mut self) {
        unimplemented!();
    }

    /// Shrinks or grows and defragments the main buffer. Invalidates all
    /// kernels referencing sub-buffers in this pool. See `::defrag` for more
    /// information.
    pub fn resize(&mut self, len: usize) {
        // Allocate a new buffer then copy old buffer contents, defragmenting
        // in the process.
        unimplemented!();
    }
}




/// A mock-up function which could dynamically generate kernel source but
/// instead just randomly returns an addition or subtraction kernel.
///
/// [NOTE]: Using OpenCL 2.1+ one would be able to return a SPIR-V IL binary
/// instead of an uncompiled string which would be more straightforward to
/// create from a structured graph, not to mention considerably faster both to
/// generate and for the device to compile and use.
///
/// Indeed, thinking about it, one could already achieve the same effect by
/// targeting LLVM instead and using tools found at:
/// `https://github.com/KhronosGroup/SPIRV-LLVM` to convert as necessary. I
/// believe all/most OpenCL vendors have offline LLVM -> Binary compilers for
/// older hardware. [TODO]: Investigate this.
///
fn gen_kernel(kernel_name: &str) -> String {
    let op = if rand::random() { "+" } else { "-" };

    format!(
        r#"
            __kernel void {}(__global float4* buffer, float4 values) {{
                buffer[get_global_id(0)] {}= values;
            }}
        "#,
        kernel_name, op
    )
}


fn main() {
    // Set up using defaults:
    let platform = Platform::default();
    let device = Device::first(platform);
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();
    let queue = Queue::new(&context, device, None).unwrap();

    let pool: BufferPool<ClFloat4> = BufferPool::new(INITIAL_BUFFER_LEN, queue.clone());

    let buffer_size_range = RandRange::new(SUB_BUF_MIN_LEN, SUB_BUF_MAX_LEN);

    struct TaskChain {
        queue: Queue,
        buffer_ids: Vec<usize>,
        programs: Vec<Program>,
        kernels: Vec<Kernel>,
    }

    let tasks: Vec<TaskChain> = Vec::new();


    {

    }



    // let kernel_src = gen_kernel("add", ());

    // let program = Program::builder()
    //     .devices(device)
    //     .src(src)
    //     .build(&context).unwrap();

    // let kernel = Kernel::new("add", &program, &queue).unwrap()
    //     .gws(&dims)
    //     .arg_buf(&buffer)
    //     .arg_scl(ClFloat4(10., 10., 10., 10.));

    // kernel.cmd()
    //     .queue(&queue)
    //     .gwo(kernel.get_gwo())
    //     .gws(&dims)
    //     .lws(kernel.get_lws())
    //     .ewait_opt(None)
    //     .enew_opt(None)
    //     .enq().unwrap();

    // buffer.cmd()
    //     .queue(&queue)
    //     .block(true)
    //     .offset(0)
    //     .read(&mut vec)
    //     .ewait_opt(None)
    //     .enew_opt(None)
    //     .enq().unwrap();

    // println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);}
}


