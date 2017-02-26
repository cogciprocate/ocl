//! A little bit of everything.
//!
//! Repeatedly creates and enqueues one of two types of task, simple or
//! complex. Both types rely on the command graph and futures to synchronize
//! their completion with each other and with thread pool offloaded host-side
//! I/O and processing.
//!
//!

extern crate libc;
extern crate futures;
extern crate futures_cpupool;
extern crate rand;
extern crate chrono;
extern crate ocl;
extern crate ocl_extras as extras;
#[macro_use] extern crate colorify;

use rand::{Rng, XorShiftRng};
use rand::distributions::{IndependentSample, Range as RandRange};
use futures::{stream, Future, Sink, Stream, Join};
use futures::sync::mpsc::{self, Sender};
use futures_cpupool::{CpuPool, CpuFuture};
use ocl::{Platform, Device, Context, Queue, Program, Kernel, OclPrm,
    Event, EventList, FutureMemMap};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::aliases::ClFloat4;
use ocl::async::{Error as AsyncError};
use extras::{SubBufferPool, CommandGraph, Command, CommandDetails, KernelArgBuffer};

const INITIAL_BUFFER_LEN: u32 = 1 << 24; // 512MiB of ClFloat4
const SUB_BUF_MIN_LEN: u32 = 1 << 15; // 1MiB of ClFloat4
const SUB_BUF_MAX_LEN: u32 = 1 << 19; // 16MiB of ClFloat4


enum TaskKind {
    Simple,
    Complex,
}

/// The specific details and pieces needed to execute the commands in the
/// command graph.
#[allow(dead_code)]
struct Task {
    task_id: usize,
    cmd_graph: CommandGraph,
    kernels: Vec<Kernel>,
    expected_result: Option<ClFloat4>,
    kind: TaskKind,
    work_size: u32,
    finish_events: EventList,
}

impl Task {
    /// Returns a new, empty task.
    pub fn new(task_id: usize, kind: TaskKind, work_size: u32) -> Task {
        Task {
            task_id: task_id,
            cmd_graph: CommandGraph::new(),
            kernels: Vec::new(),
            expected_result: None,
            kind: kind,
            work_size: work_size,
            finish_events: EventList::new(),
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

    /// Set the expected final value.
    pub fn set_expected_result(&mut self, expected_result: ClFloat4) {
        self.expected_result = Some(expected_result)
    }

    /// Return a list of 'open' events, those that aren't already being waited on.
    #[allow(dead_code)]
    pub fn get_finish_events(&mut self) -> &mut EventList {
        self.finish_events.clear();
        self.cmd_graph.get_finish_events(&mut self.finish_events);
        &mut self.finish_events
    }

    /// Fill a buffer with a pattern of data:
    pub fn fill<T: OclPrm>(&self, pattern: T, cmd_idx: usize, buf_pool: &SubBufferPool<T>) {
        let buffer_id = match *self.cmd_graph.commands()[cmd_idx].details() {
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
    pub fn map<T: OclPrm>(&self, cmd_idx: usize, buf_pool: &SubBufferPool<T>) -> FutureMemMap<T>
    {
        let (buffer_id, flags) = match *self.cmd_graph.commands()[cmd_idx].details(){
            CommandDetails::Write { target } => (target, MapFlags::new().write_invalidate_region()),
            CommandDetails::Read { source } => (source, MapFlags::new().read()),
            _ => panic!("Task::map: Not a write or read command."),
        };

        let buf = buf_pool.get(buffer_id).unwrap();

        let mut future_data = buf.cmd().map().flags(flags)
            .ewait(self.cmd_graph.get_req_events(cmd_idx).unwrap())
            .enq_async().unwrap();

        let unmap_event_target = future_data.create_unmap_event().unwrap().clone();
        self.cmd_graph.set_cmd_event(cmd_idx, unmap_event_target.into()).unwrap();

        future_data
    }

    /// Copy contents of one buffer to another.
    pub fn copy<T: OclPrm>(&self, cmd_idx: usize, buf_pool: &SubBufferPool<T>) {
        let (src_buf_id, tar_buf_id) = match *self.cmd_graph.commands()[cmd_idx].details(){
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
    pub fn kernel(&self, cmd_idx: usize) {
        let kernel_id = match *self.cmd_graph.commands()[cmd_idx].details(){
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
fn gen_kern_src(kernel_name: &str, type_str: &str, simple: bool, add: bool) -> String {
    let op = if add { "+" } else { "-" };

    if simple {
        format!(r#"__kernel void {}(
                __global {ts}* in,
                {ts} values,
                __global {ts}* out)
            {{
                uint idx = get_global_id(0);
                out[idx] = in[idx] {} values;
            }}"#
            ,
            kn=kernel_name, op=op, ts=type_str
        )
    } else {
        format!(r#"__kernel void {kn}(
                __global {ts}* in_0,
                __global {ts}* in_1,
                __global {ts}* in_2,
                {ts} values,
                __global {ts}* out)
            {{
                uint idx = get_global_id(0);
                out[idx] = in_0[idx] {op} in_1[idx] {op} in_2[idx] {op} values;
            }}"#
            ,
            kn=kernel_name, op=op, ts=type_str
        )
    }
}



//#############################################################################
//#############################################################################
//############################## SIMPLE TASK ##################################
//#############################################################################
//#############################################################################
/// Returns a simple task.
///
/// This task will:
///
/// (0) Write data
/// (1) Run one kernel
/// (2) Read data
///
fn create_simple_task(task_id: usize, device: Device, context: &Context,
        buf_pool: &mut SubBufferPool<ClFloat4>, work_size: u32, queues: &[Queue]) -> Result<Task, ()>
{
    let write_buf_flags = Some(MemFlags::new().read_only() | MemFlags::new().host_write_only());
    let read_buf_flags = Some(MemFlags::new().write_only() | MemFlags::new().host_read_only());

    // The container for this task:
    let mut task = Task::new(task_id, TaskKind::Simple, work_size);

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

    buf_pool.get_mut(write_buf_id).unwrap().set_default_queue(queues[0].clone());
    buf_pool.get_mut(read_buf_id).unwrap().set_default_queue(queues[1].clone());

    let program = Program::builder()
        .devices(device)
        .src(gen_kern_src("kern", "float4", true, true))
        .build(context).unwrap();

    let kern = Kernel::new("kern", &program).unwrap()
        .queue(queues[2].clone())
        .gws(work_size)
        .arg_buf(buf_pool.get(write_buf_id).unwrap())
        .arg_vec(ClFloat4(100., 100., 100., 100.))
        .arg_buf(buf_pool.get(read_buf_id).unwrap());

    // (0) Initial write to device:
    assert!(task.add_write_command(write_buf_id).unwrap() == 0);

    // (1) Kernel:
    assert!(task.add_kernel(kern,
        vec![KernelArgBuffer::new(0, write_buf_id)],
        vec![KernelArgBuffer::new(2, read_buf_id)]).unwrap() == 1);

    // (2) Final read from device:
    assert!(task.add_read_command(read_buf_id).unwrap() == 2);

    // Populate the command graph:
    task.cmd_graph.populate_requisites();
    Ok(task)
}

/// Enqueues a unique simple task as defined above.
fn enqueue_simple_task(task: &Task, buf_pool: &SubBufferPool<ClFloat4>, thread_pool: &CpuPool,
        tx: Sender<usize>) -> Join<CpuFuture<usize, AsyncError>, CpuFuture<Sender<usize>, AsyncError>>
{
    // Do some extra work:
    let task_id = task.task_id;

    // (0) Write a bunch of 50's:
    let write = task.map(0, &buf_pool).and_then(move |mut data| {
        for val in data.iter_mut() {
            *val = ClFloat4(50., 50., 50., 50.);
        }

        printlnc!(green: "Task [{}] (simple): Buffer initialized.", task_id);

        Ok(task_id)
    });

    // let write_spawned = thread_pool.spawn(write).wait().unwrap();
    let write_spawned = thread_pool.spawn(write);

    // (1) Run kernel (adds 100 to everything):
    task.kernel(1);

    // (2) Read results and verify them:
    let verify = task.map(2, &buf_pool)
        .and_then(move |data| {
            let mut val_count = 0usize;

            for val in data.iter() {
                let correct_val = ClFloat4(150., 150., 150., 150.);
                if *val != correct_val {
                    return Err(format!("Result value mismatch: {:?} != {:?}", val, correct_val).into())
                }
                val_count += 1;
            }

            printlnc!(yellow: "Task [{}] (simple): Verify successful: \
                {} values correct.", task_id, val_count);

            Ok(tx.send(val_count))
        })
        .and_then(|send| send.map_err(|e| AsyncError::from(e)));

    let verify_spawned = thread_pool.spawn(verify);

    write_spawned.join(verify_spawned)
    // write_spawned.join(verify_spawned).wait().unwrap();
}


//#############################################################################
//#############################################################################
//############################# COMPLEX TASK ##################################
//#############################################################################
//#############################################################################
/// Returns a complex task.
///
/// This task will:
///
/// (0) Write:   host_mem  -> buffer[0]
/// (1) Kernel:  buffer[0] -> kernel_a -> buffer[1]
/// (2) Copy:    buffer[1] -> buffer[2]
/// (3) Copy:    buffer[1] -> buffer[3]
/// (4) Fill:              -> buffer[4]
/// (5) Kernel:  buffer[2] ->
///              buffer[3] ->
///              buffer[4] -> kernel_b -> buffer[5]
/// (6) Kernel:  buffer[5] -> kernel_c -> buffer[6]
/// (7) Read:    buffer[6] -> host_mem
///
fn create_complex_task(task_id: usize, device: Device, context: &Context,
        buf_pool: &mut SubBufferPool<ClFloat4>, work_size: u32, queues: &[Queue],
        rng: &mut XorShiftRng) -> Result<Task, ()>
{
    // The container for this task:
    let mut task = Task::new(task_id, TaskKind::Complex, work_size);

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
            Ok(buf_id) => {
                // Set a unique queue for each buffer to avoid deadlocks:
                buf_pool.get_mut(buf_id).unwrap().set_default_queue(queues[idx].clone());
                buffer_ids.push(buf_id)
            },
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
        .src(gen_kern_src("kernel_a", "float4", true, kern_a_sign))
        .src(gen_kern_src("kernel_b", "float4", false, kern_b_sign))
        .src(gen_kern_src("kernel_c", "float4", true, kern_c_sign))
        .build(context).unwrap();

    let kernel_a = Kernel::new("kernel_a", &program).unwrap()
        .queue(queues[7].clone())
        .gws(work_size)
        .arg_buf(buf_pool.get(buffer_ids[0]).unwrap())
        .arg_vec(ClFloat4(kern_a_val, kern_a_val, kern_a_val, kern_a_val))
        .arg_buf(buf_pool.get(buffer_ids[1]).unwrap());

    let kernel_b = Kernel::new("kernel_b", &program).unwrap()
        .queue(queues[7].clone())
        .gws(work_size)
        .arg_buf(buf_pool.get(buffer_ids[2]).unwrap())
        .arg_buf(buf_pool.get(buffer_ids[3]).unwrap())
        .arg_buf(buf_pool.get(buffer_ids[4]).unwrap())
        .arg_vec(ClFloat4(kern_b_val, kern_b_val, kern_b_val, kern_b_val))
        .arg_buf(buf_pool.get(buffer_ids[5]).unwrap());

    let kernel_c = Kernel::new("kernel_c", &program).unwrap()
        .queue(queues[7].clone())
        .gws(work_size)
        .arg_buf(buf_pool.get(buffer_ids[5]).unwrap())
        .arg_vec(ClFloat4(kern_c_val, kern_c_val, kern_c_val, kern_c_val))
        .arg_buf(buf_pool.get(buffer_ids[6]).unwrap());

    // (0) Initially write 500s:
    assert!(task.add_write_command(buffer_ids[0]).unwrap() == 0);

    // (1) Kernel A -- Add values:
    assert!(task.add_kernel(kernel_a,
        vec![KernelArgBuffer::new(0, buffer_ids[0])],
        vec![KernelArgBuffer::new(2, buffer_ids[1])]).unwrap() == 1);

    // (2) Copy from buffer[1] to buffer[2]:
    assert!(task.add_copy_command(buffer_ids[1], buffer_ids[2]).unwrap() == 2);

    // (3) Copy from buffer[1] to buffer[3]:
    assert!(task.add_copy_command(buffer_ids[1], buffer_ids[3]).unwrap() == 3);

    // (4) Fill buffer[4] with 50s:
    assert!(task.add_fill_command(buffer_ids[4]).unwrap() == 4);

    // (5) Kernel B -- Sum buffers and add values:
    assert!(task.add_kernel(kernel_b,
        vec![KernelArgBuffer::new(0, buffer_ids[2]),
            KernelArgBuffer::new(1, buffer_ids[3]),
            KernelArgBuffer::new(2, buffer_ids[4])],
        vec![KernelArgBuffer::new(4, buffer_ids[5])]).unwrap() == 5);

    // (6) Kernel C -- Subtract values:
    assert!(task.add_kernel(kernel_c,
        vec![KernelArgBuffer::new(0, buffer_ids[5])],
        vec![KernelArgBuffer::new(2, buffer_ids[6])]).unwrap() == 6);

    // (7) Final read from device:
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

/// Enqueues a unique complex task as defined above.
fn enqueue_complex_task(task: &Task, buf_pool: &SubBufferPool<ClFloat4>, thread_pool: &CpuPool,
        tx: Sender<usize>) -> Join<CpuFuture<usize, AsyncError>, CpuFuture<Sender<usize>, AsyncError>>
{
    let task_id = task.task_id;

    // (0) Initially write 500s:
    let write = task.map(0, &buf_pool).and_then(move |mut data| {
        for val in data.iter_mut() {
            *val = ClFloat4(500., 500., 500., 500.);
        }

        printlnc!(green_bold: "Task [{}] (complex): Buffer initialized.", task_id);

        Ok(task_id)
    });

    // (1) Kernel A -- Add values:
    task.kernel(1);

    // (2) Copy from buffer[1] to buffer[2]:
    task.copy(2, buf_pool);

    // (3) Copy from buffer[1] to buffer[3]:
    task.copy(3, buf_pool);

    // (4) Fill buffer[4] with 50s:
    task.fill(ClFloat4(50., 50., 50., 50.), 4, buf_pool);

    // (5) Kernel B -- Sum buffers and add values:
    task.kernel(5);

    // (6) Kernel C -- Subtract values:
    task.kernel(6);

    // (7) Finally read and verify:
    let expected_result = task.expected_result.unwrap();

    let verify = task.map(7, &buf_pool)
        .and_then(move |data| {
            let mut val_count = 0usize;

            for val in data.iter() {
                let correct_val = expected_result;
                if *val != correct_val {
                    return Err(format!("Result value mismatch: {:?} != {:?}", val, correct_val).into())
                }
                val_count += 1;
            }

            printlnc!(yellow_bold: "Task [{}] (complex): Verify successful: \
                {} values correct.", task_id, val_count);

            Ok(tx.send(val_count))
        })
        .and_then(|send| send.map_err(|e| AsyncError::from(e)));

    let write_spawned = thread_pool.spawn(write);
    let verify_spawned = thread_pool.spawn(verify);

    write_spawned.join(verify_spawned)
}


/// Returns a nicely formatted duration in seconds.
fn fmt_duration(duration: chrono::Duration) -> String {
    let el_sec = duration.num_seconds();
    let el_ms = duration.num_milliseconds() - (el_sec * 1000);
    format!("{}.{} seconds", el_sec, el_ms)
}


/// Creates a large number of both simple and complex asynchronous tasks and
/// verifies that they all execute correctly.
pub fn main() {
    // Buffer/work size range:
    let buffer_size_range = RandRange::new(SUB_BUF_MIN_LEN, SUB_BUF_MAX_LEN);
    let mut rng = rand::weak_rng();

    // Set up context using defaults:
    let platform = Platform::default();
    printlnc!(blue: "Platform: {}", platform.name());

    // let device = Device::first(platform);
    let device_idx = RandRange::new(0, 15).ind_sample(&mut rng);

    let device = Device::specifier()
        .wrapping_indices(vec![device_idx])
        .to_device_list(Some(platform)).unwrap()[0];

    printlnc!(teal: "Device: {} {}", device.vendor(), device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    // Queues (events coordinated by command graph):
    let queue_flags = Some(CommandQueueProperties::new().out_of_order());
    let queues_simple: Vec<_> = (0..3).map(|_| Queue::new(&context, device, queue_flags).unwrap())
        .collect();
    let queues_complex: Vec<_> = (0..8).map(|_| Queue::new(&context, device, queue_flags).unwrap())
        .collect();

    // A pool of available device side memory (one big buffer with an attached allocator).
    let mut buf_pool: SubBufferPool<ClFloat4> = SubBufferPool::new(INITIAL_BUFFER_LEN,
        Queue::new(&context, device, queue_flags).unwrap());
    let mut tasks = Vec::with_capacity(256);

    // Our thread pool for offloading reading, writing, and other host-side processing.
    let thread_pool = CpuPool::new_num_cpus();
    let mut correct_val_count = 0usize;

    // Channels are used to communicate result counts (this isn't really
    // necessary here but shown for demonstration):
    let (tx, mut rx) = mpsc::channel(1);

    let start_time = chrono::Local::now();
    printlnc!(white_bold: "Creating and enqueuing tasks...");

    // Create some arbitrary tasks until our buffer pool is full:
    loop {
        // Random work size:
        let work_size = buffer_size_range.ind_sample(&mut rng);

        // Create task if there is room in the buffer pool:
        let task_id = tasks.len();

        let task_res = if rng.gen() {
        // let task_res = if false {
            create_simple_task(task_id, device, &context, &mut buf_pool, work_size,
                &queues_simple)
        } else {
            create_complex_task(task_id, device, &context, &mut buf_pool, work_size,
                &queues_complex, &mut rng)
        };

        let task = match task_res {
            Ok(task) => task,
            Err(_) => {
                println!("Buffer pool is now full.");
                break;
            },
        };

        match task.kind {
            TaskKind::Simple => tasks.push(enqueue_simple_task(&task, &buf_pool, &thread_pool, tx.clone())),
            TaskKind::Complex => tasks.push(enqueue_complex_task(&task, &buf_pool, &thread_pool, tx.clone())),
        }
    }

    let create_enqueue_duration = chrono::Local::now() - start_time;
    let task_count = tasks.len();
    printlnc!(white_bold: "Waiting on {} tasks to complete...", task_count);

    stream::futures_unordered(tasks).for_each(|(task_id, _)| {
        printlnc!(orange: "Task [{}]: Complete.", task_id);
        Ok(())
    }).wait().unwrap();

    rx.close();

    for count in rx.wait() {
        correct_val_count += count.unwrap();
    }

    let run_duration = chrono::Local::now() - start_time - create_enqueue_duration;
    let total_duration = chrono::Local::now() - start_time;

    printlnc!(white_bold: "\nAll {} (float4) result values from {} tasks are correct! \n\
        Durations => | Create/Enqueue: {} | Run: {} | Total: {}",
        correct_val_count, task_count, fmt_duration(create_enqueue_duration),
        fmt_duration(run_duration), fmt_duration(total_duration));
}
