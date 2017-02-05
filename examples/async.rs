//! An example of dynamically generated sub-buffers and programs plus a few
//! other basic implementations of memory and command flow structures.
//!
//! ### Notes / TODO
//!
//! - Error handling incomplete. Ideally each major structure has its own
//!   error type and in a real program one master error type to rule them all.
//! - Some methods unimplemented.
//!

#![allow(unused_variables, dead_code)]

extern crate ocl;
extern crate rand;

/////// [FIXME]: TEMPORARY:
extern crate libc;

use std::slice;
use rand::distributions::{IndependentSample, Range as RandRange};
use std::collections::{LinkedList, HashMap, BTreeSet};
use ocl::{flags, Platform, Device, Context, Queue, Program, Buffer, Kernel, SubBuffer, OclPrm,
    Event, EventList};
use ocl::flags::{CommandQueueProperties, MemFlags};
use ocl::aliases::ClFloat4;

const INITIAL_PROGRAM_COUNT: usize = 25;
const INITIAL_BUFFER_LEN: u32 = 2 << 23; // 256MiB of ClFloat4
const SUB_BUF_MIN_LEN: u32 = 2 << 11; // 64KiB of ClFloat4
const SUB_BUF_MAX_LEN: u32 = 2 << 15; // 1MiB of ClFloat4

/////// [FIXME]: TEMPORARY:
const PRINT_DEBUG: bool = true;


struct PoolRegion {
    buffer_id: usize,
    origin: u32,
    len: u32,
}


/// A simple but slow sub-buffer allocator.
struct BufferPool<T: OclPrm> {
    buffer: Buffer<T>,
    regions: LinkedList<PoolRegion>,
    sub_buffers: HashMap<usize, SubBuffer<T>>,
    align: u32,
    _next_uid: usize,
}

impl<T: OclPrm> BufferPool<T> {
    /// Returns a new buffer pool.
    pub fn new(len: u32, default_queue: Queue) -> BufferPool<T> {
        let align = default_queue.device().mem_base_addr_align().unwrap();

        let buffer = Buffer::<T>::new(default_queue, Some(flags::MEM_READ_WRITE),
            len, None).unwrap();

        BufferPool {
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

    fn insert_region(&mut self, region: PoolRegion, region_idx: usize) {
        let mut tail = self.regions.split_off(region_idx);
        self.regions.push_back(region);
        self.regions.append(&mut tail);
    }

    fn create_sub_buffer(&mut self, region_idx: usize, flags: Option<flags::MemFlags>,
            unaligned_origin: u32, len: u32) -> usize
    {
        let buffer_id = self.next_uid();
        let origin = ((unaligned_origin / self.align) + 1) * self.align;
        let region = PoolRegion { buffer_id: buffer_id, origin: origin, len: len };
        let sbuf = self.buffer.create_sub_buffer(flags, region.origin, region.len).unwrap();
        if let Some(idx) = self.sub_buffers.insert(region.buffer_id, sbuf) {
            panic!("Duplicate indexes"); }
        self.insert_region(region, region_idx);
        buffer_id
    }

    /// Allocates space for and creates a new sub-buffer then returns the
    /// buffer id which can be used to `::get` or `::free` it.
    pub fn alloc(&mut self, len: u32, flags: Option<flags::MemFlags>) -> Result<usize, ()> {
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
                    Ok(self.create_sub_buffer(region_idx, flags, end_prev, len))
                } else if self.buffer.len() as u32 - end_prev >= len {
                    let region_idx = self.regions.len();
                    Ok(self.create_sub_buffer(region_idx, flags, end_prev, len))
                } else {
                    Err(())
                }
            },
            None => {
                Ok(self.create_sub_buffer(0, flags, 0, len))
            },
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
            tail.pop_front().ok_or(buffer_id)   ?;
            self.regions.append(&mut tail);
            Ok(())
        } else {
            Err(buffer_id)
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
        // - Iterate through sub-buffers, dropping and recreating at the
        //   leftmost (lowest) available address within the buffer, optionally
        //   copying old data to the new position in device memory.
        // - Rebuild regions list (or just update offsets).
        unimplemented!();
    }

    /// Shrinks or grows and defragments the main buffer. Invalidates all
    /// kernels referencing sub-buffers in this pool. See `::defrag` for more
    /// information.
    pub fn resize(&mut self, len: u32) {
        // Allocate a new buffer then copy old buffer contents one sub-buffer
        // at a time, defragmenting in the process.
        unimplemented!();
    }
}


struct RwCmdIdxs {
    writers: Vec<usize>,
    readers: Vec<usize>,
}

impl RwCmdIdxs {
    fn new() -> RwCmdIdxs {
        RwCmdIdxs { writers: Vec::new(), readers: Vec::new() }
    }
}


struct KernelArgBuffer {
    arg_idx: usize,
    buffer_id: usize,
}

impl KernelArgBuffer {
    pub fn new(arg_idx: usize, buffer_id: usize) -> KernelArgBuffer {
        KernelArgBuffer { arg_idx: arg_idx, buffer_id: buffer_id }
    }
}


enum CommandDetails {
    Write { target: usize },
    Read { source: usize },
    Kernel { id: usize, sources: Vec<KernelArgBuffer>, targets: Vec<KernelArgBuffer> },
}

impl CommandDetails {
    pub fn sources(&self) -> Vec<usize> {
        match *self {
            CommandDetails::Read { source } => vec![source],
            CommandDetails::Write { .. } => vec![],
            CommandDetails::Kernel { ref sources, .. } => {
                sources.iter().map(|arg| arg.buffer_id).collect()
            },
        }
    }

    pub fn targets(&self) -> Vec<usize> {
        match *self {
            CommandDetails::Read { .. } => vec![],
            CommandDetails::Write { target } => vec![target],
            CommandDetails::Kernel { ref targets, .. } => {
                targets.iter().map(|arg| arg.buffer_id).collect()
            },
        }
    }
}


struct Command {
    details: CommandDetails,
    event: Option<Event>,
    requisite_commands: Vec<usize>,
    requisite_events: EventList,
}

impl Command {
    pub fn new(details: CommandDetails) -> Command {
        Command {
            details: details,
            event: None,
            requisite_commands: Vec::new(),
            requisite_events: EventList::new(),
        }
    }

    /// Returns a list of commands which both precede a command and which
    /// write to a block of memory which is read from by that command.
    pub fn preceding_writers(&self, cmds: &HashMap<usize, RwCmdIdxs>) -> BTreeSet<usize> {
        let pre_writers = self.details.sources().iter().flat_map(|cmd_src_block|
                cmds.get(cmd_src_block).unwrap().writers.iter().cloned()).collect();

        // if PRINT_DEBUG { println!("##### [{}]: Preceding Writers: {:?}", cmd_idx, pre_writers); }
        pre_writers
    }

    /// Returns a list of commands which both follow a command and which read
    /// from a block of memory which is written to by that command.
    pub fn following_readers(&self, cmds: &HashMap<usize, RwCmdIdxs>) -> BTreeSet<usize> {
        let fol_readers = self.details.targets().iter().flat_map(|cmd_tar_block|
                cmds.get(cmd_tar_block).unwrap().readers.iter().cloned()).collect();

        // if PRINT_DEBUG { println!("##### [{}]: Following Readers: {:?}", cmd_idx, fol_readers); }
        fol_readers
    }
}


/// A sequence dependency graph representing the temporal requirements of each
/// asynchronous read, write, and kernel (commands) for a particular task.
///
/// Obviously this is an overkill for this example but this graph is flexible
/// enough to schedule execution correctly and optimally with arbitrarily many
/// parallel tasks with arbitrary duration reads, writes and kernels.
///
/// Note that in this example we are using `buffer_id` a `usize` to represent
/// memory regions (because that's what the allocator above is using) but we
/// could easily use multiple part, complex identifiers/keys. For example, we
/// may have a program with a large number of buffers which are organized into
/// a complex hierarchy or some other arbitrary structure. We could swap
/// `buffer_id` for some value which represented that as long as the
/// identifier we used could uniquely identify each subsection of memory. We
/// could also use ranges of values and do an overlap check and have
/// byte-level precision.
///
struct CommandGraph {
    commands: Vec<Command>,
    locked: bool,
    next_cmd_idx: usize,
}

impl CommandGraph {
    /// Returns a new, empty graph.
    pub fn new() -> CommandGraph {
        CommandGraph {
            commands: Vec::new(),
            locked: false,
            next_cmd_idx: 0,
        }
    }

    /// Adds a new command and returns the command index if successful.
    pub fn add(&mut self, command: Command) -> Result<usize, ()> {
        if self.locked { return Err(()); }
        self.commands.push(command);
        Ok(self.commands.len() - 1)
    }

    /// Returns a sub-buffer map which contains every command that reads from
    /// or writes to each sub-buffer.
    fn readers_and_writers_by_sub_buffer(&self) -> HashMap<usize, RwCmdIdxs> {
        let mut cmds = HashMap::new();
        if PRINT_DEBUG { println!("\n##### Readers and Writers by Memory Block:"); }
        if PRINT_DEBUG { println!("#####"); }

        for (cmd_idx, cmd) in self.commands.iter().enumerate() {
            if PRINT_DEBUG { println!("##### Command [{}]:", cmd_idx); }
            if PRINT_DEBUG {  println!("#####     [Sources:]"); }

            for cmd_src_block in cmd.details.sources().into_iter() {
                let rw_cmd_idxs = cmds.entry(cmd_src_block.clone())
                    .or_insert(RwCmdIdxs::new());

                rw_cmd_idxs.readers.push(cmd_idx);
                if PRINT_DEBUG { println!("#####     [{}]: {:?}", rw_cmd_idxs.readers.len() - 1, cmd_src_block); }
            }

            if PRINT_DEBUG { println!("#####     [Targets:]"); }

            for cmd_tar_block in cmd.details.targets().into_iter() {
                let rw_cmd_idxs = cmds.entry(cmd_tar_block.clone())
                    .or_insert(RwCmdIdxs::new());

                rw_cmd_idxs.writers.push(cmd_idx);
                if PRINT_DEBUG { println!("#####     [{}]: {:?}", rw_cmd_idxs.writers.len() - 1, cmd_tar_block); }
            }

            if PRINT_DEBUG { println!("#####"); }
        }

        cmds
    }

    /// Populates the list of requisite commands necessary for building the
    /// correct event wait list for each command.
    pub fn populate_requisites(&mut self) {
        let cmds = self.readers_and_writers_by_sub_buffer();
        if PRINT_DEBUG { println!("\n##### Preceding Writers and Following Readers:"); }
        if PRINT_DEBUG { println!("#####"); }

        for (cmd_idx, cmd) in self.commands.iter_mut().enumerate() {
            if PRINT_DEBUG { println!("##### Command [{}]:", cmd_idx); }
            assert!(cmd.requisite_commands.is_empty());

            for &req_cmd_idx in cmd.preceding_writers(&cmds).iter()
                    .chain(cmd.following_readers(&cmds).iter())
            {
                cmd.requisite_commands.push(req_cmd_idx);
            }

            cmd.requisite_commands.shrink_to_fit();
            if PRINT_DEBUG { println!("#####"); }
        }

        self.commands.shrink_to_fit();
        self.locked = true;
    }
}


struct Task {
    cmd_graph: CommandGraph,
    queue: Queue,
    // programs: Vec<Program>,
    kernels: Vec<Kernel>,
    buffers: Vec<usize>,
}

impl Task {
    /// Returns a new, empty task.
    pub fn new(queue: Queue) -> Task {
        Task {
            cmd_graph: CommandGraph::new(),
            queue: queue,
            // programs: Vec::new(),
            kernels: Vec::new(),
            buffers: Vec::new(),
        }
    }

    /// Adds a new write command.
    pub fn add_write_command(&mut self, target_buffer_id: usize) -> Result<usize, ()> {
        self.cmd_graph.add(Command::new(CommandDetails::Write { target: target_buffer_id }))
    }

    /// Adds a new read command.
    pub fn add_read_command(&mut self, source_buffer_id: usize) -> Result<usize, ()> {
        self.cmd_graph.add(Command::new(CommandDetails::Read { source: source_buffer_id }))
    }

    /// Adds a new kernel.
    pub fn add_kernel(&mut self, kernel: Kernel, source_buffer_ids: Vec<KernelArgBuffer>, target_buffer_ids: Vec<KernelArgBuffer>)
        -> Result<usize, ()>
    {
        self.kernels.push(kernel);

        self.cmd_graph.add(Command::new(CommandDetails::Kernel {
            id: self.kernels.len() - 1,
            sources: source_buffer_ids,
            targets: target_buffer_ids,
        }))
    }

    pub fn write<T: OclPrm>(&self, cmd_idx: usize, buf_pool: &BufferPool<T>) -> Result<&mut [T], ()> {
        let tar_buf_id = match self.cmd_graph.commands[cmd_idx].details {
            CommandDetails::Write { target } => target,
            _ => return Err(()),
        };

        let buf = buf_pool.get(tar_buf_id).unwrap();

        unsafe {
            Ok(
                slice::from_raw_parts_mut(
                    ocl::core::enqueue_map_buffer::<T>(
                        &self.queue,
                        buf,
                        true,
                        flags::MAP_WRITE_INVALIDATE_REGION,
                        0,
                        buf.len(),
                        None,
                        None,
                        ).unwrap() as *mut T
                    ,
                    buf.len()
                )
            )
        }
    }

    pub fn read<T: OclPrm>(&self, cmd_idx: usize, buf_pool: &BufferPool<T>) -> Result<&[T], ()> {
        let src_buf_id = match self.cmd_graph.commands[cmd_idx].details {
            CommandDetails::Read { source } => source,
            _ => return Err(()),
        };

        let buf = buf_pool.get(src_buf_id).unwrap();

        unsafe {
            Ok(
                slice::from_raw_parts(
                    ocl::core::enqueue_map_buffer::<T>(
                        &self.queue,
                        buf,
                        true,
                        flags::MAP_READ,
                        0,
                        buf.len(),
                        None,
                        None,
                        ).unwrap() as *const T
                    ,
                    buf.len()
                )
            )
        }
    }

    pub fn kernel(&self, cmd_idx: usize) -> Result<(), ()> {
        let kernel_id = match self.cmd_graph.commands[cmd_idx].details {
            CommandDetails::Kernel { id, .. } => id,
            _ => return Err(()),
        };

        self.kernels[kernel_id].cmd().enq().unwrap();
        Ok(())
    }
}


/// A mock-up function which could dynamically generate kernel source but
/// instead just returns either an addition or subtraction kernel.
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
fn gen_kern_src(kernel_name: &str, add: bool) -> String {
    let op = if add { "+" } else { "-" };

    format!(
        r#"
            __kernel void {}(
                __global float4* in,
                float4 values,
                __global float4* out)
            {{
                uint idx = get_global_id(0);
                out[idx] = in[idx] {} values;
            }}
        "#,
        kernel_name, op
    )
}


/// Creates a large number of complex asynchronous tasks and verifies that
/// they each executed correctly without blocking or explicitly using any fences
fn main() {
    // Flags and buffer size range:
    let ooo_queue_flag = Some(flags::QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    let write_buf_flags = Some(flags::MEM_ALLOC_HOST_PTR | flags::MEM_READ_ONLY |
        flags::MEM_HOST_WRITE_ONLY);
    let read_buf_flags = Some(flags::MEM_ALLOC_HOST_PTR | flags::MEM_WRITE_ONLY |
        flags::MEM_HOST_READ_ONLY);
    let buffer_size_range = RandRange::new(SUB_BUF_MIN_LEN, SUB_BUF_MAX_LEN);
    let mut rng = rand::weak_rng();

    // Set up context using defaults:
    let platform = Platform::default();
    let device = Device::first(platform);
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    // All queues are out of order and coordinated by the command graph. This
    // example doesn't really need two because they're already out of order
    // (I'm fairly sure about this) but it's good practice to have a separate
    // I/O queue so we'll stick with it for now.
    let kernel_queue = Queue::new(&context, device, ooo_queue_flag).unwrap();
    let io_queue = Queue::new(&context, device, ooo_queue_flag).unwrap();

    let mut buf_pool: BufferPool<ClFloat4> = BufferPool::new(INITIAL_BUFFER_LEN, io_queue);
    let mut tasks: HashMap<usize, Task> = HashMap::new();

    let mut pool_full = false;

    // Create some arbitrary tasks until our buffer pool is full:
    while !pool_full {
        for j in 0..300 {
            // This task will:
            //
            // 1. Write data
            // 2. Run one kernel
            // 3. Read data

            // Vary the work size:
            let work_size = buffer_size_range.ind_sample(&mut rng);

            // The container for this task:
            let mut task = Task::new(kernel_queue.clone());

            // Allocate our input and output buffers:
            let write_buf_id_res = buf_pool.alloc(work_size, write_buf_flags);
            let read_buf_id_res = buf_pool.alloc(work_size, read_buf_flags);

            // Make sure we didn't exceed the buffer pool:
            if write_buf_id_res.is_err() || read_buf_id_res.is_err() {
                pool_full = true;
                println!("Buffer pool is full.");
                break;
            }

            let write_buf_id = write_buf_id_res.unwrap();
            let read_buf_id = write_buf_id_res.unwrap();

            let program = Program::builder()
                .devices(device)
                // .src(gen_kern_src("kern", rand::random()))
                .src(gen_kern_src("kern", true))
                .build(&context).unwrap();

            let kern = Kernel::new("kern", &program, kernel_queue.clone()).unwrap()
                .gws(work_size)
                .arg_buf(buf_pool.get(write_buf_id).unwrap())
                .arg_scl(ClFloat4(100., 100., 100., 100.))
                .arg_buf(buf_pool.get(read_buf_id).unwrap());

            // Initial write to device:
            assert!(task.add_write_command(write_buf_id).unwrap() == 0);

            // Kernel:
            assert!(task.add_kernel(kern,
                vec![KernelArgBuffer::new(0, write_buf_id)],
                vec![KernelArgBuffer::new(2, read_buf_id)]).unwrap() == 1);

            // Final read from device:
            assert!(task.add_read_command(read_buf_id).unwrap() == 2);

            // Insert task. Use a buffer_id as the key since it'll be unique:
            tasks.insert(write_buf_id, task);
        }
    }

    let mut val_count = 0usize;

    for task in tasks.values() {
        match task.write(0, &buf_pool) {
            Ok(data) => {
                let write_buf_id = match task.cmd_graph.commands[0].details {
                    CommandDetails::Write { target } => target,
                    _ => panic!(),
                };

                let buf = buf_pool.get(write_buf_id).unwrap();

                assert!(data.len() == buf.len());

                for val in data.iter_mut() {
                    *val = ClFloat4(0., 0., 0., 0.);
                }

                ocl::core::enqueue_unmap_mem_object(
                    &task.queue,
                    buf,
                    data.as_ptr() as *mut libc::c_void,
                    None,
                    None,
                ).unwrap();
            },
            Err(_) => panic!("Error attempting to write."),
        }

        task.kernel(1).unwrap();

        match task.read(2, &buf_pool) {
            Ok(data) => {
                let read_buf_id = match task.cmd_graph.commands[2].details {
                    CommandDetails::Read { source } => source,
                    _ => panic!(),
                };

                let buf = buf_pool.get(read_buf_id).unwrap();
                assert!(data.len() == buf.len());

                for val in data.iter() {
                    if *val != ClFloat4(100., 100., 100., 100.) {
                        panic!("Invalid value: {:?}", val);
                    }
                    val_count += 1;
                }

                // println!("All {} values correctly equal '{:?}'!", data.len(), data.first().unwrap());

                ocl::core::enqueue_unmap_mem_object(
                    &task.queue,
                    buf,
                    data.as_ptr() as *mut libc::c_void,
                    None,
                    None,
                ).unwrap();
            },
            Err(_) => panic!("Error attempting to write."),
        }
    }

    println!("All {} values are correct!", val_count);
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


    // struct TaskChain {
    //     queue: Queue,
    //     buffer_ids: Vec<usize>,
    //     programs: Vec<Program>,
    //     kernels: Vec<Kernel>,
    // }