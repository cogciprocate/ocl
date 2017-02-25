//! An example of dynamically generated sub-buffers and programs plus a few
//! other basic implementations of useful memory and command flow structures.
//!
//! ### Futures
//!
//! The one thing this example is missing is cyclical repeating tasks. To
//! implement these currently requires either the use of event callbacks,
//! restricting oneself to using only a single thread, or manually managing
//! coordination between threads using channels, etc. Since the current best
//! of these three solutions, event callbacks, are buggy on certain platforms
//! due to perpetually flawed drivers (ahem, Nvidia), cyclical tasks have been
//! omitted from this example for now.
//!
//! Fear not. In the very near future, futures-rs integration is coming. This
//! will bring a very nice way to inject host code into the execution stream
//! of a complex task. When futures arrive, in the future, this example will
//! be updated to be as cool as Marty McFly.
//!
//!
//! ### Notes / TODO
//!
//! - Error handling incomplete. Ideally each major structure has its own
//!   error type and in a real program one master error type to rule them all.
//! - Some methods unimplemented.
//! - Futures integration.
//!

#![allow(non_snake_case)]

extern crate rand;
extern crate chrono;
extern crate ocl;

use rand::{Rng, XorShiftRng};
use rand::distributions::{IndependentSample, Range as RandRange};
use std::collections::{LinkedList, HashMap, BTreeSet};
use ocl::{Platform, Device, Context, Queue, Program, Buffer, Kernel, SubBuffer, OclPrm,
    Event, EventList, MemMap};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::aliases::ClFloat4;

const INITIAL_BUFFER_LEN: u32 = 1 << 23; // 256MiB of ClFloat4
const SUB_BUF_MIN_LEN: u32 = 1 << 11; // 64KiB of ClFloat4
const SUB_BUF_MAX_LEN: u32 = 1 << 15; // 1MiB of ClFloat4


struct PoolRegion {
    buffer_id: usize,
    origin: u32,
    len: u32,
}

/// A simple (linear search) sub-buffer allocator.
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
        let flags = MemFlags::new().alloc_host_ptr().read_write();
        let buffer = Buffer::<T>::builder()
            .queue(default_queue)
            .flags(flags)
            .dims(len)
            .fill_val(Default::default(), None::<()>)
            .build().unwrap();

        BufferPool {
            buffer: buffer,
            regions: LinkedList::new(),
            sub_buffers: HashMap::new(),
            align: align,
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
    #[allow(dead_code)]
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


struct RwCmdIdxs {
    writers: Vec<usize>,
    readers: Vec<usize>,
}

impl RwCmdIdxs {
    fn new() -> RwCmdIdxs {
        RwCmdIdxs { writers: Vec::new(), readers: Vec::new() }
    }
}

#[allow(dead_code)]
struct KernelArgBuffer {
    arg_idx: usize, // Will be used when refreshing kernels after defragging or resizing.
    buffer_id: usize,
}

impl KernelArgBuffer {
    pub fn new(arg_idx: usize, buffer_id: usize) -> KernelArgBuffer {
        KernelArgBuffer { arg_idx: arg_idx, buffer_id: buffer_id }
    }
}


/// Details of a queuable command.
enum CommandDetails {
    Fill { target: usize },
    Write { target: usize },
    Read { source: usize },
    Copy { source: usize, target: usize },
    Kernel { id: usize, sources: Vec<KernelArgBuffer>, targets: Vec<KernelArgBuffer> },
}

impl CommandDetails {
    pub fn sources(&self) -> Vec<usize> {
        match *self {
            CommandDetails::Fill { .. } => vec![],
            CommandDetails::Read { source } => vec![source],
            CommandDetails::Write { .. } => vec![],
            CommandDetails::Copy { source, .. } => vec![source],
            CommandDetails::Kernel { ref sources, .. } => {
                sources.iter().map(|arg| arg.buffer_id).collect()
            },
        }
    }

    pub fn targets(&self) -> Vec<usize> {
        match *self {
            CommandDetails::Fill { target } => vec![target],
            CommandDetails::Read { .. } => vec![],
            CommandDetails::Write { target } => vec![target],
            CommandDetails::Copy { target, .. } => vec![target],
            CommandDetails::Kernel { ref targets, .. } => {
                targets.iter().map(|arg| arg.buffer_id).collect()
            },
        }
    }
}


struct Command {
    details: CommandDetails,
    event: Option<Event>,
    requisite_events: EventList,
}

impl Command {
    pub fn new(details: CommandDetails) -> Command {
        Command {
            details: details,
            event: None,
            requisite_events: EventList::new(),
        }
    }

    /// Returns a list of commands which both precede a command and which
    /// write to a block of memory which is read from by that command.
    pub fn preceding_writers(&self, cmds: &HashMap<usize, RwCmdIdxs>) -> BTreeSet<usize> {
        let pre_writers = self.details.sources().iter().flat_map(|cmd_src_block|
                cmds.get(cmd_src_block).unwrap().writers.iter().cloned()).collect();

        pre_writers
    }

    /// Returns a list of commands which both follow a command and which read
    /// from a block of memory which is written to by that command.
    pub fn following_readers(&self, cmds: &HashMap<usize, RwCmdIdxs>) -> BTreeSet<usize> {
        let fol_readers = self.details.targets().iter().flat_map(|cmd_tar_block|
                cmds.get(cmd_tar_block).unwrap().readers.iter().cloned()).collect();

        fol_readers
    }
}


/// A sequence dependency graph representing the temporal requirements of each
/// asynchronous read, write, copy, and kernel (commands) for a particular
/// task.
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
    command_requisites: Vec<Vec<usize>>,
    locked: bool,
    next_cmd_idx: usize,
}

impl CommandGraph {
    /// Returns a new, empty graph.
    pub fn new() -> CommandGraph {
        CommandGraph {
            commands: Vec::new(),
            command_requisites: Vec::new(),
            locked: false,
            next_cmd_idx: 0,
        }
    }

    /// Adds a new command and returns the command index if successful.
    pub fn add(&mut self, command: Command) -> Result<usize, ()> {
        if self.locked { return Err(()); }
        self.commands.push(command);
        self.command_requisites.push(Vec::new());
        Ok(self.commands.len() - 1)
    }

    /// Returns a sub-buffer map which contains every command that reads from
    /// or writes to each sub-buffer.
    fn readers_and_writers_by_sub_buffer(&self) -> HashMap<usize, RwCmdIdxs> {
        let mut cmds = HashMap::new();

        for (cmd_idx, cmd) in self.commands.iter().enumerate() {
            for cmd_src_block in cmd.details.sources().into_iter() {
                let rw_cmd_idxs = cmds.entry(cmd_src_block.clone())
                    .or_insert(RwCmdIdxs::new());

                rw_cmd_idxs.readers.push(cmd_idx);
            }

            for cmd_tar_block in cmd.details.targets().into_iter() {
                let rw_cmd_idxs = cmds.entry(cmd_tar_block.clone())
                    .or_insert(RwCmdIdxs::new());

                rw_cmd_idxs.writers.push(cmd_idx);
            }
        }

        cmds
    }

    /// Populates the list of requisite commands necessary for building the
    /// correct event wait list for each command.
    pub fn populate_requisites(&mut self) {
        let cmds = self.readers_and_writers_by_sub_buffer();

        for (cmd_idx, cmd) in self.commands.iter_mut().enumerate() {
            assert!(self.command_requisites[cmd_idx].is_empty());

            for &req_cmd_idx in cmd.preceding_writers(&cmds).iter()
                    .chain(cmd.following_readers(&cmds).iter())
            {
                self.command_requisites[cmd_idx].push(req_cmd_idx);
            }

            self.command_requisites[cmd_idx].shrink_to_fit();
        }

        self.commands.shrink_to_fit();
        self.command_requisites.shrink_to_fit();
        self.locked = true;
    }

    /// Returns the list of requisite events for a command.
    pub fn get_req_events(&mut self, cmd_idx: usize) -> Result<&EventList, &'static str> {
        if !self.locked { return Err("Call '::populate_requisites' first."); }
        if self.next_cmd_idx != cmd_idx { return Err("Command events requested out of order."); }

        self.commands[cmd_idx].requisite_events.clear();

        for &req_idx in self.command_requisites[cmd_idx].iter() {
            if let Some(event) = self.commands[req_idx].event.clone() {
                self.commands[cmd_idx].requisite_events.push(event);
            }
        }

        Ok(&self.commands[cmd_idx].requisite_events)
    }

    /// Sets the event associated with the completion of a command.
    pub fn set_cmd_event(&mut self, cmd_idx: usize, event: Event) -> Result<(), &'static str> {
        if !self.locked { return Err("Call '::populate_requisites' first."); }

        self.commands[cmd_idx].event = Some(event);

        if (self.next_cmd_idx + 1) == self.commands.len() {
            self.next_cmd_idx = 0;
        } else {
            self.next_cmd_idx += 1;
        }

        Ok(())
    }
}


/// The specific details and pieces needed to execute the commands in the
/// command graph.
struct Task {
    cmd_graph: CommandGraph,
    queue: Queue,
    kernels: Vec<Kernel>,
    expected_result: Option<ClFloat4>,
}

impl Task{
    /// Returns a new, empty task.
    pub fn new(queue: Queue) -> Task {
        Task {
            cmd_graph: CommandGraph::new(),
            queue: queue,
            kernels: Vec::new(),
            expected_result: None,
        }
    }

    /// Adds a new write command.
    pub fn add_fill_command(&mut self, target_buffer_id: usize) -> Result<usize, ()> {
        self.cmd_graph.add(Command::new(CommandDetails::Fill { target: target_buffer_id }))
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
    pub fn add_kernel(&mut self, kernel: Kernel, source_buffer_ids: Vec<KernelArgBuffer>,
            target_buffer_ids: Vec<KernelArgBuffer>) -> Result<usize, ()>
    {
        self.kernels.push(kernel);

        self.cmd_graph.add(Command::new(CommandDetails::Kernel {
            id: self.kernels.len() - 1,
            sources: source_buffer_ids,
            targets: target_buffer_ids,
        }))
    }

    /// Adds a new copy command.
    pub fn add_copy_command(&mut self, source_buffer_id: usize, target_buffer_id: usize)
            -> Result<usize, ()>
    {
        self.cmd_graph.add(Command::new(CommandDetails::Copy {
            source: source_buffer_id,
            target: target_buffer_id,
        }))
    }

    pub fn set_expected_result(&mut self, expected_result: ClFloat4) {
        self.expected_result = Some(expected_result)
    }

    /// Fill a buffer with a pattern of data:
    pub fn fill<T: OclPrm>(&mut self, pattern: T, cmd_idx: usize, buf_pool: &BufferPool<T>) {
        let buffer_id = match self.cmd_graph.commands[cmd_idx].details {
            CommandDetails::Fill { target } => target,
            _ => panic!("Task::fill: Not a fill command."),
        };

        let mut ev = Event::empty();
        let buf = buf_pool.get(buffer_id).unwrap();

        buf.cmd().fill(pattern, None)
            .ewait(self.cmd_graph.get_req_events(cmd_idx).unwrap())
            .enew(&mut ev)
            .enq().unwrap();

        self.cmd_graph.set_cmd_event(cmd_idx, ev).unwrap();
    }

    /// Map some memory for reading or writing.
    pub fn map<T: OclPrm>(&mut self, cmd_idx: usize, buf_pool: &BufferPool<T>) -> MemMap<T> {
        let (buffer_id, flags) = match self.cmd_graph.commands[cmd_idx].details {
            CommandDetails::Write { target } => (target, MapFlags::new().write_invalidate_region()),
            CommandDetails::Read { source } => (source, MapFlags::new().read()),
            _ => panic!("Task::map: Not a write or read command."),
        };

        let buf = buf_pool.get(buffer_id).unwrap();

        buf.cmd().map().flags(flags)
            .ewait(self.cmd_graph.get_req_events(cmd_idx).unwrap())
            .enq().unwrap()
    }

    /// Unmap mapped memory.
    pub fn unmap<T: OclPrm>(&mut self, mut data: MemMap<T>, cmd_idx: usize, _: &BufferPool<T>) {
        // let buffer_id = match self.cmd_graph.commands[cmd_idx].details {
        //     CommandDetails::Write { target } => target,
        //     CommandDetails::Read { source } => source,
        //     _ => panic!("Task::unmap: Not a write or read command."),
        // };

        let mut ev = Event::empty();

        data.enqueue_unmap(Some(&self.queue), None::<EventList>, Some(&mut ev)).unwrap();

        self.cmd_graph.set_cmd_event(cmd_idx, ev).unwrap();
    }

    /// Copy contents of one buffer to another.
    pub fn copy<T: OclPrm>(&mut self, cmd_idx: usize, buf_pool: &BufferPool<T>) {
        let (src_buf_id, tar_buf_id) = match self.cmd_graph.commands[cmd_idx].details {
            CommandDetails::Copy { source, target } => (source, target),
            _ => panic!("Task::copy: Not a copy command."),
        };

        let mut ev = Event::empty();
        let src_buf = buf_pool.get(src_buf_id).unwrap();
        let tar_buf = buf_pool.get(tar_buf_id).unwrap();

        src_buf.cmd().copy(tar_buf, None, None)
            .ewait(self.cmd_graph.get_req_events(cmd_idx).unwrap())
            .enew(&mut ev)
            .enq().unwrap();

        self.cmd_graph.set_cmd_event(cmd_idx, ev).unwrap();
    }

    /// Enqueue a kernel.
    pub fn kernel(&mut self, cmd_idx: usize) {
        let kernel_id = match self.cmd_graph.commands[cmd_idx].details {
            CommandDetails::Kernel { id, .. } => id,
            _ => panic!("Task::kernel: Not a kernel command."),
        };

        let mut ev = Event::empty();

        self.kernels[kernel_id].cmd().enew(&mut ev)
            .ewait(self.cmd_graph.get_req_events(cmd_idx).unwrap())
            .enq().unwrap();

        self.cmd_graph.set_cmd_event(cmd_idx, ev).unwrap();
    }
}

fn coeff(add: bool) -> f32 {
    if add { 1. } else { -1. }
}

/// A very simple kernel source generator. Imagine something cooler here.
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
fn gen_kern_src(kernel_name: &str, simple: bool, add: bool) -> String {
    let op = if add { "+" } else { "-" };

    if simple {
        format!(r#"__kernel void {}(
                __global float4* in,
                float4 values,
                __global float4* out)
            {{
                uint idx = get_global_id(0);
                out[idx] = in[idx] {} values;
            }}"#
            ,
            kernel_name, op
        )
    } else {
        format!(r#"__kernel void {}(
                __global float4* in_0,
                __global float4* in_1,
                __global float4* in_2,
                float4 values,
                __global float4* out)
            {{
                uint idx = get_global_id(0);
                out[idx] = in_0[idx] {} in_1[idx] {} in_2[idx] {} values;
            }}"#
            ,
            kernel_name, op, op, op
        )
    }
}


/// Returns a simple task.
///
/// This task will:
///
/// 0.) Write data
/// 1.) Run one kernel
/// 2.) Read data
///
fn create_simple_task(device: Device, context: &Context, buf_pool: &mut BufferPool<ClFloat4>,
    work_size: u32, queue: Queue) -> Result<Task, ()>
{
    let write_buf_flags = Some(MemFlags::new().read_only().host_write_only());
    let read_buf_flags = Some(MemFlags::new().write_only().host_read_only());

    // The container for this task:
    let mut task = Task::new(queue.clone());

    // Allocate our input buffer:
    let write_buf_id = match buf_pool.alloc(work_size, write_buf_flags) {
        Ok(buf_id) => buf_id,
        Err(_) => return Err(()),
    };

    // Allocate our output buffer, freeing the unused input buffer upon error.
    let read_buf_id = match buf_pool.alloc(work_size, read_buf_flags) {
        Ok(buf_id) => buf_id,
        Err(_) => {
            buf_pool.free(write_buf_id).ok();
            return Err(());
        },
    };

    let program = Program::builder()
        .devices(device)
        .src(gen_kern_src("kern", true, true))
        .build(context).unwrap();

    let kern = Kernel::new("kern", &program, queue).unwrap()
        .gws(work_size)
        .arg_buf(buf_pool.get(write_buf_id).unwrap())
        .arg_vec(ClFloat4(100., 100., 100., 100.))
        .arg_buf(buf_pool.get(read_buf_id).unwrap());

    // 0.) Initial write to device:
    assert!(task.add_write_command(write_buf_id).unwrap() == 0);

    // 1.) Kernel:
    assert!(task.add_kernel(kern,
        vec![KernelArgBuffer::new(0, write_buf_id)],
        vec![KernelArgBuffer::new(2, read_buf_id)]).unwrap() == 1);

    // 2.) Final read from device:
    assert!(task.add_read_command(read_buf_id).unwrap() == 2);

    // Populate the command graph:
    task.cmd_graph.populate_requisites();
    Ok(task)
}

fn run_simple_task(task: &mut Task, buf_pool: &BufferPool<ClFloat4>) {
    // 0.) Write a bunch of 50's:
    let mut data = task.map(0, buf_pool);

    for val in data.iter_mut() {
        *val = ClFloat4(50., 50., 50., 50.);
    }

    task.unmap(data, 0, buf_pool);

    // 1.) Run kernel (adds 100 to everything):
    task.kernel(1);
}


fn verify_simple_task(task: &mut Task, buf_pool: &BufferPool<ClFloat4>, correct_val_count: &mut usize) {
    // 2.) Read results and verifies them:
    let data = task.map(2, buf_pool);

    for val in data.iter() {
        assert_eq!(*val, ClFloat4(150., 150., 150., 150.));
        *correct_val_count += 1;
    }

    task.unmap(data, 2, buf_pool);
}


/// Returns a complex task.
///
/// This task will:
///
/// 0.) Write:   host_mem  -> buffer[0]
/// 1.) Kernel:  buffer[0] -> kernel_A -> buffer[1]
/// 2.) Copy:    buffer[1] -> buffer[2]
/// 3.) Copy:    buffer[1] -> buffer[3]
/// 4.) Fill:              -> buffer[4]
/// 5.) Kernel:  buffer[2] ->
///              buffer[3] ->
///              buffer[4] -> kernel_B -> buffer[5]
/// 6.) Kernel:  buffer[5] -> kernel_C -> buffer[6]
/// 7.) Read:    buffer[6] -> host_mem
///
fn create_complex_task(device: Device, context: &Context, buf_pool: &mut BufferPool<ClFloat4>,
    work_size: u32, queue: Queue, rng: &mut XorShiftRng) -> Result<Task, ()>
{
    // The container for this task:
    let mut task = Task::new(queue.clone());

    let buffer_count = 7;

    // Allocate our buffers:
    let buffer_id_res: Vec<_> = (0..buffer_count).map(|i| {
        let flags = match i {
            0 => Some(MemFlags::new().read_only().host_write_only()),
            1...5 => Some(MemFlags::new().read_write().host_no_access()),
            6 => Some(MemFlags::new().write_only().host_read_only()),
            _ => panic!("Only 7 buffers are configured."),
        };

        buf_pool.alloc(work_size, flags)
    }).collect();

    let mut buffer_ids = Vec::with_capacity(buffer_count);

    // Add valid buffer_ids to the final list, being sure to deallocate all
    // previously allocated buffers from this task in the event of a failure:
    for idx in 0..buffer_count {
        match buffer_id_res[idx] {
            Ok(buf_id) => buffer_ids.push(buf_id),
            Err(_) => {
                for prev_idx in 0..idx {
                    buf_pool.free(buffer_id_res[prev_idx].unwrap()).ok();
                }
                return Err(());
            }
        }
    }

    let kern_a_sign = rng.gen();
    let kern_b_sign = rng.gen();
    let kern_c_sign = rng.gen();
    let kern_a_val = RandRange::new(-1000., 1000.).ind_sample(rng);
    let kern_b_val = RandRange::new(-500., 500.).ind_sample(rng);
    let kern_c_val = RandRange::new(-2000., 2000.).ind_sample(rng);

    let program = Program::builder()
        .devices(device)
        .src(gen_kern_src("kernel_A", true, kern_a_sign))
        .src(gen_kern_src("kernel_B", false, kern_b_sign))
        .src(gen_kern_src("kernel_C", true, kern_c_sign))
        .build(context).unwrap();

    let kernel_A = Kernel::new("kernel_A", &program, queue.clone()).unwrap()
        .gws(work_size)
        .arg_buf(buf_pool.get(buffer_ids[0]).unwrap())
        .arg_vec(ClFloat4(kern_a_val, kern_a_val, kern_a_val, kern_a_val))
        .arg_buf(buf_pool.get(buffer_ids[1]).unwrap());

    let kernel_B = Kernel::new("kernel_B", &program, queue.clone()).unwrap()
        .gws(work_size)
        .arg_buf(buf_pool.get(buffer_ids[2]).unwrap())
        .arg_buf(buf_pool.get(buffer_ids[3]).unwrap())
        .arg_buf(buf_pool.get(buffer_ids[4]).unwrap())
        .arg_vec(ClFloat4(kern_b_val, kern_b_val, kern_b_val, kern_b_val))
        .arg_buf(buf_pool.get(buffer_ids[5]).unwrap());

    let kernel_C = Kernel::new("kernel_C", &program, queue).unwrap()
        .gws(work_size)
        .arg_buf(buf_pool.get(buffer_ids[5]).unwrap())
        .arg_vec(ClFloat4(kern_c_val, kern_c_val, kern_c_val, kern_c_val))
        .arg_buf(buf_pool.get(buffer_ids[6]).unwrap());

    // 0.) Initially write 500s:
    assert!(task.add_write_command(buffer_ids[0]).unwrap() == 0);

    // 1.) Kernel A -- Add values:
    assert!(task.add_kernel(kernel_A,
        vec![KernelArgBuffer::new(0, buffer_ids[0])],
        vec![KernelArgBuffer::new(2, buffer_ids[1])]).unwrap() == 1);

    // 2.) Copy from buffer[1] to buffer[2]:
    assert!(task.add_copy_command(buffer_ids[1], buffer_ids[2]).unwrap() == 2);

    // 3.) Copy from buffer[1] to buffer[3]:
    assert!(task.add_copy_command(buffer_ids[1], buffer_ids[3]).unwrap() == 3);

    // 4.) Fill buffer[4] with 50s:
    assert!(task.add_fill_command(buffer_ids[4]).unwrap() == 4);

    // 5.) Kernel B -- Sum buffers and add values:
    assert!(task.add_kernel(kernel_B,
        vec![KernelArgBuffer::new(0, buffer_ids[2]),
            KernelArgBuffer::new(1, buffer_ids[3]),
            KernelArgBuffer::new(2, buffer_ids[4])],
        vec![KernelArgBuffer::new(4, buffer_ids[5])]).unwrap() == 5);

    // 6.) Kernel C -- Subtract values:
    assert!(task.add_kernel(kernel_C,
        vec![KernelArgBuffer::new(0, buffer_ids[5])],
        vec![KernelArgBuffer::new(2, buffer_ids[6])]).unwrap() == 6);

    // 7.) Final read from device:
    assert!(task.add_read_command(buffer_ids[6]).unwrap() == 7);

    // Calculate expected result value:
    let kern_a_out_val = 500. + (coeff(kern_a_sign) * kern_a_val);
    let kern_b_out_val = kern_a_out_val +
        (coeff(kern_b_sign) * kern_a_out_val) +
        (coeff(kern_b_sign) * 50.) +
        (coeff(kern_b_sign) * kern_b_val);
    let kern_c_out_val = kern_b_out_val + (coeff(kern_c_sign) * kern_c_val);
    task.set_expected_result(ClFloat4(kern_c_out_val, kern_c_out_val, kern_c_out_val, kern_c_out_val));

    // Populate the command graph:
    task.cmd_graph.populate_requisites();
    Ok(task)
}

fn run_complex_task(task: &mut Task, buf_pool: &BufferPool<ClFloat4>) {
    // 0.) Initially write 500s:
    let write_cmd_idx = 0;
    let mut data = task.map(write_cmd_idx, buf_pool);

    for val in data.iter_mut() {
        *val = ClFloat4(500., 500., 500., 500.);
    }

    task.unmap(data, write_cmd_idx, buf_pool);

    // 1.) Kernel A -- Add values:
    task.kernel(1);

    // 2.) Copy from buffer[1] to buffer[2]:
    task.copy(2, buf_pool);

    // 3.) Copy from buffer[1] to buffer[3]:
    task.copy(3, buf_pool);

    // 4.) Fill buffer[4] with 50s:
    task.fill(ClFloat4(50., 50., 50., 50.), 4, buf_pool);

    // 5.) Kernel B -- Sum buffers and add values:
    task.kernel(5);

    // 6.) Kernel C -- Subtract values:
    task.kernel(6);

}

fn verify_complex_task(task: &mut Task, buf_pool: &BufferPool<ClFloat4>, correct_val_count: &mut usize) {
    // 7.) Final read from device:
    let read_cmd_idx = 7;
    let data = task.map(read_cmd_idx, buf_pool);

    let expected_result = task.expected_result.unwrap();

    for val in data.iter() {
        assert_eq!(*val, expected_result);
        *correct_val_count += 1;
    }

    task.unmap(data, read_cmd_idx, buf_pool);
}


/// Creates a large number of both simple and complex asynchronous tasks and
/// verifies that they all execute correctly.
fn main() {
    // Buffer/work size range:
    let buffer_size_range = RandRange::new(SUB_BUF_MIN_LEN, SUB_BUF_MAX_LEN);
    let mut rng = rand::weak_rng();

    // Set up context using defaults:
    let platform = Platform::default();
    let device = Device::first(platform);
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    // Out of order queues (coordinated by the command graph):
    let io_queue = Queue::new(&context, device, Some(CommandQueueProperties::new().out_of_order())).unwrap();
    let kern_queue = Queue::new(&context, device, Some(CommandQueueProperties::new().out_of_order())).unwrap();

    let mut buf_pool: BufferPool<ClFloat4> = BufferPool::new(INITIAL_BUFFER_LEN, io_queue);
    let mut simple_tasks: Vec<Task> = Vec::new();
    let mut complex_tasks: Vec<Task> = Vec::new();
    let mut pool_full = false;

    let start_time = chrono::Local::now();
    println!("Creating tasks...");

    // Create some arbitrary tasks until our buffer pool is full:
    while !pool_full {
        // Add 3 simple tasks:
        for _ in 0..3 {
            // Random work size:
            let work_size = buffer_size_range.ind_sample(&mut rng);

            // Create task if there is room in the buffer pool:
            let task = match create_simple_task(device, &context, &mut buf_pool,
                    work_size, kern_queue.clone())
            {
                Ok(task) => task,
                Err(_) => {
                    pool_full = true;
                    println!("Buffer pool is full.");
                    break;
                },
            };

            // Add task to the list:
            simple_tasks.push(task);
        }

        // Add 5 complex tasks:
        for _ in 0..5 {
            // Random work size:
            let work_size = buffer_size_range.ind_sample(&mut rng);

            // Create task if there is room in the buffer pool:
            let task = match create_complex_task(device, &context, &mut buf_pool,
                    work_size, kern_queue.clone(), &mut rng)
            {
                Ok(task) => task,
                Err(_) => {
                    pool_full = true;
                    println!("Buffer pool is full.");
                    break;
                },
            };

            // Add task to the list:
            complex_tasks.push(task);
        }
    }


    let create_duration = chrono::Local::now() - start_time;
    let mut correct_val_count = 0usize;
    println!("Running tasks...");

    // Run tasks:
    for task in simple_tasks.iter_mut() {
        run_simple_task(task, &buf_pool);
    }

    for task in complex_tasks.iter_mut() {
        run_complex_task(task, &buf_pool);
    }


    let run_duration = chrono::Local::now() - start_time - create_duration;
    println!("Reading and verifying task results...");

    // Read and verify task results:
    for task in simple_tasks.iter_mut() {
        verify_simple_task(task, &buf_pool, &mut correct_val_count);
    }

    for task in complex_tasks.iter_mut() {
        verify_complex_task(task, &buf_pool, &mut correct_val_count);
    }


    let verify_duration = chrono::Local::now() - start_time - create_duration - run_duration;

    println!("All {} (float4) result values from {} simple and {} complex tasks are correct! \n\
        Durations => | Create: {} | Run: {} | Verify: {} | ",
        correct_val_count, simple_tasks.len(), complex_tasks.len(), fmt_duration(create_duration),
        fmt_duration(run_duration), fmt_duration(verify_duration));
}


fn fmt_duration(duration: chrono::Duration) -> String {
    let el_sec = duration.num_seconds();
    let el_ms = duration.num_milliseconds() - (el_sec * 1000);
    format!("{}.{} seconds", el_sec, el_ms)
}