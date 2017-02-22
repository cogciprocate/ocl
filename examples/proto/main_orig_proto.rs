#![allow(dead_code, unused_variables, unused_imports, unused_mut, unreachable_code)]


// use std::io;
// use std::thread;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use libc::c_void;
use chrono;
use futures::{future, stream, Async, Sink, Stream};
use futures::future::*;
use futures::sync::mpsc::{self, Receiver, Sender, UnboundedSender};
use futures_cpupool::{CpuPool, CpuFuture};
use tokio_core::reactor::{Core, Handle};
use tokio_timer::{Timer, Sleep, TimerError};
// use self::dinglehopper::Dinglehopper;

use rand::{self, Rng, XorShiftRng};
use rand::distributions::{IndependentSample, Range as RandRange};
use std::collections::{LinkedList, HashMap, BTreeSet};
use ocl::{core, async, Platform, Device, Context, Queue, Program, Buffer, Kernel, SubBuffer, OclPrm,
    Event, EventList, FutureMemMap, MemMap, Error as OclError};
// use ocl::core::{FutureMemMap, MemMap};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::aliases::ClFloat4;
use ocl::async::{FutureResult as FutureAsyncResult, Error as AsyncError};

use extras_proto::{SubBufferPool, CommandGraph, Command, CommandDetails, KernelArgBuffer, RwCmdIdxs};
use switches::{Switches, SWITCHES};

// // ORIGINAL SIZES:
const INITIAL_BUFFER_LEN: u32 = 2 << 23; // 256MiB of ClFloat4
const SUB_BUF_MIN_LEN: u32 = 2 << 11; // 64KiB of ClFloat4
const SUB_BUF_MAX_LEN: u32 = 2 << 15; // 1MiB of ClFloat4

// // LARGER BUFFER:
// const INITIAL_BUFFER_LEN: u32 = 2 << 25; // 1GiB of ClFloat4

// // LARGER SUB-BUFFERS:
// const INITIAL_BUFFER_LEN: u32 = 2 << 23; // 256MiB of ClFloat4
// const SUB_BUF_MIN_LEN: u32 = 2 << 19; // 16MiB of ClFloat4
// const SUB_BUF_MAX_LEN: u32 = 2 << 21; // 64MiB of ClFloat4

// // STILL LARGER SUB-BUFFERS:
// const INITIAL_BUFFER_LEN: u32 = 2 << 25; // 512MiB of ClFloat4
// const SUB_BUF_MIN_LEN: u32 = 2 << 23;
// const SUB_BUF_MAX_LEN: u32 = 2 << 24;



enum TaskKind {
    Simple,
    Complex,
}


/// The specific details and pieces needed to execute the commands in the
/// command graph.
struct Task {
    task_id: usize,
    cmd_graph: CommandGraph,
    // queue: Queue,
    kernels: Vec<Kernel>,
    expected_result: Option<ClFloat4>,
    kind: TaskKind,
    work_size: u32,
    finish_events: EventList,
}

impl Task{
    /// Returns a new, empty task.
    pub fn new(task_id: usize, /*queue: Queue,*/ kind: TaskKind, work_size: u32) -> Task {
        Task {
            task_id: task_id,
            cmd_graph: CommandGraph::new(),
            // queue: queue,
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

    pub fn set_expected_result(&mut self, expected_result: ClFloat4) {
        self.expected_result = Some(expected_result)
    }

    pub fn get_finish_events(&mut self) -> &mut EventList {
        self.finish_events.clear();
        self.cmd_graph.get_finish_events(&mut self.finish_events);
        &mut self.finish_events
    }

    /// Fill a buffer with a pattern of data:
    pub fn fill<T: OclPrm>(&self, pattern: T, cmd_idx: usize, buf_pool: &SubBufferPool<T>)
            -> FutureAsyncResult<()>
    {
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

        future::ok(())
    }

    /// Map some memory for reading or writing.
    pub fn map<T: OclPrm>(&self, cmd_idx: usize, buf_pool: &SubBufferPool<T>,
            /*thread_pool: &CpuPool*/) -> FutureMemMap<T>
    {
        let (buffer_id, flags) = match *self.cmd_graph.commands()[cmd_idx].details(){
            CommandDetails::Write { target } => (target, MapFlags::write_invalidate_region()),
            CommandDetails::Read { source } => (source, MapFlags::read()),
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

    // /// Unmap mapped memory.
    // pub fn unmap<T: OclPrm>(&mut self, data: &mut MemMap<T>, cmd_idx: usize,
    //         buf_pool: &SubBufferPool<T>)
    // {
    //     let buffer_id = match self.cmd_graph.commands()[cmd_idx].details(){
    //         CommandDetails::Write { target } => target,
    //         CommandDetails::Read { source } => source,
    //         _ => panic!("Task::unmap: Not a write or read command."),
    //     };

    //     let mut ev = Event::empty();

    //     data.unmap(None, None, Some(&mut ev)).unwrap();

    //     self.cmd_graph.set_cmd_event(cmd_idx, ev).unwrap();
    // }

    /// Copy contents of one buffer to another.
    pub fn copy<T: OclPrm>(&self, cmd_idx: usize, buf_pool: &SubBufferPool<T>)
             -> FutureAsyncResult<()>
    {
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

        future::ok(())
    }


    //#############################################################################
    //#############################################################################
    //########################### ENQUEUE A KERNEL ################################
    //#############################################################################
    //#############################################################################

    /// Enqueue a kernel.
    pub fn kernel(&self, cmd_idx: usize) -> FutureAsyncResult<()> {
        let kernel_id = match *self.cmd_graph.commands()[cmd_idx].details(){
            CommandDetails::Kernel { id, .. } => id,
            _ => panic!("Task::kernel: Not a kernel command."),
        };

        let mut ev = Event::empty();

        self.kernels[kernel_id].cmd().enew(&mut ev)
            .ewait(self.cmd_graph.get_req_events(cmd_idx).unwrap())
            .enq().unwrap();

        // println!("Setting command completion event for kernel [task: {}, kernel_id: {}, cmd_idx: {}]. \
        //     Event: {:?}.", self.task_id, kernel_id, cmd_idx, ev);

        self.cmd_graph.set_cmd_event(cmd_idx, ev).unwrap();

        future::ok(())
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




// /// Returns a complex task.
// ///
// /// This task will:
// ///
// /// (0) Write:   host_mem  -> buffer[0]
// /// (1) Kernel:  buffer[0] -> kernel_a -> buffer[1]
// /// (2) Copy:    buffer[1] -> buffer[2]
// /// (3) Copy:    buffer[1] -> buffer[3]
// /// (4) Fill:              -> buffer[4]
// /// (5) Kernel:  buffer[2] ->
// ///              buffer[3] ->
// ///              buffer[4] -> kernel_b -> buffer[5]
// /// (6) Kernel:  buffer[5] -> kernel_c -> buffer[6]
// /// (7) Read:    buffer[6] -> host_mem
// ///
// fn create_complex_task(task_id: usize, device: Device, context: &Context,
//         buf_pool: &mut verify<ClFloat4>, work_size: u32, queue: Queue,
//         rng: &mut XorShiftRng) -> Result<Task, ()>
// {
//     // The container for this task:
//     let mut task = Task::new(task_id, queue.clone(), TaskKind::Complex, work_size);

//     let buffer_count = 7;

//     // Allocate our buffers:
//     let buffer_id_res: Vec<_> = (0..buffer_count).map(|i| {
//         let flags = match i {
//             0 => Some(MemFlags::read_only() | MemFlags::host_write_only()),
//             1...5 => Some(MemFlags::read_write() | MemFlags::host_no_access()),
//             6 => Some(MemFlags::write_only() | MemFlags::host_read_only()),
//             _ => panic!("Only 7 buffers are configured."),
//         };

//         buf_pool.alloc(work_size, flags)
//     }).collect();

//     let mut buffer_ids = Vec::with_capacity(buffer_count);

//     // Add valid buffer_ids to the final list, being sure to deallocate all
//     // previously allocated buffers from this task in the event of a failure:
//     for idx in 0..buffer_count {
//         match buffer_id_res[idx] {
//             Ok(buf_id) => buffer_ids.push(buf_id),
//             Err(_) => {
//                 for prev_idx in 0..idx {
//                     buf_pool.free(buffer_id_res[prev_idx].unwrap()).ok();
//                 }
//                 return Err(());
//             }
//         }
//     }

//     let kern_a_sign = rng.gen();
//     let kern_b_sign = rng.gen();
//     let kern_c_sign = rng.gen();
//     let kern_a_val = RandRange::new(-1000., 1000.).ind_sample(rng);
//     let kern_b_val = RandRange::new(-500., 500.).ind_sample(rng);
//     let kern_c_val = RandRange::new(-2000., 2000.).ind_sample(rng);

//     let program = Program::builder()
//         .devices(device)
//         .src(gen_kern_src("kernel_a", "float4", true, kern_a_sign))
//         .src(gen_kern_src("kernel_b", "float4", false, kern_b_sign))
//         .src(gen_kern_src("kernel_c", "float4", true, kern_c_sign))
//         .build(context).unwrap();

//     let kernel_a = Kernel::new("kernel_a", &program, queue.clone()).unwrap()
//         .gws(work_size)
//         .arg_buf(buf_pool.get(buffer_ids[0]).unwrap())
//         .arg_vec(ClFloat4(kern_a_val, kern_a_val, kern_a_val, kern_a_val))
//         .arg_buf(buf_pool.get(buffer_ids[1]).unwrap());

//     let kernel_b = Kernel::new("kernel_b", &program, queue.clone()).unwrap()
//         .gws(work_size)
//         .arg_buf(buf_pool.get(buffer_ids[2]).unwrap())
//         .arg_buf(buf_pool.get(buffer_ids[3]).unwrap())
//         .arg_buf(buf_pool.get(buffer_ids[4]).unwrap())
//         .arg_vec(ClFloat4(kern_b_val, kern_b_val, kern_b_val, kern_b_val))
//         .arg_buf(buf_pool.get(buffer_ids[5]).unwrap());

//     let kernel_c = Kernel::new("kernel_c", &program, queue).unwrap()
//         .gws(work_size)
//         .arg_buf(buf_pool.get(buffer_ids[5]).unwrap())
//         .arg_vec(ClFloat4(kern_c_val, kern_c_val, kern_c_val, kern_c_val))
//         .arg_buf(buf_pool.get(buffer_ids[6]).unwrap());

//     // (0) Initially write 500s:
//     assert!(task.add_write_command(buffer_ids[0]).unwrap() == 0);

//     // (1) Kernel A -- Add values:
//     assert!(task.add_kernel(kernel_a,
//         vec![KernelArgBuffer::new(0, buffer_ids[0])],
//         vec![KernelArgBuffer::new(2, buffer_ids[1])]).unwrap() == 1);

//     // (2) Copy from buffer[1] to buffer[2]:
//     assert!(task.add_copy_command(buffer_ids[1], buffer_ids[2]).unwrap() == 2);

//     // (3) Copy from buffer[1] to buffer[3]:
//     assert!(task.add_copy_command(buffer_ids[1], buffer_ids[3]).unwrap() == 3);

//     // (4) Fill buffer[4] with 50s:
//     assert!(task.add_fill_command(buffer_ids[4]).unwrap() == 4);

//     // (5) Kernel B -- Sum buffers and add values:
//     assert!(task.add_kernel(kernel_b,
//         vec![KernelArgBuffer::new(0, buffer_ids[2]),
//             KernelArgBuffer::new(1, buffer_ids[3]),
//             KernelArgBuffer::new(2, buffer_ids[4])],
//         vec![KernelArgBuffer::new(4, buffer_ids[5])]).unwrap() == 5);

//     // (6) Kernel C -- Subtract values:
//     assert!(task.add_kernel(kernel_c,
//         vec![KernelArgBuffer::new(0, buffer_ids[5])],
//         vec![KernelArgBuffer::new(2, buffer_ids[6])]).unwrap() == 6);

//     // (7) Final read from device:
//     assert!(task.add_read_command(buffer_ids[6]).unwrap() == 7);

//     // Calculate expected result value:
//     let kern_a_out_val = 500. + (coeff(kern_a_sign) * kern_a_val);
//     let kern_b_out_val = kern_a_out_val +
//         (coeff(kern_b_sign) * kern_a_out_val) +
//         (coeff(kern_b_sign) * 50.) +
//         (coeff(kern_b_sign) * kern_b_val);
//     let kern_c_out_val = kern_b_out_val + (coeff(kern_c_sign) * kern_c_val);
//     task.set_expected_result(ClFloat4(kern_c_out_val, kern_c_out_val, kern_c_out_val, kern_c_out_val));

//     // Populate the command graph:
//     task.cmd_graph.populate_requisites();
//     Ok(task)
// }


// fn enqueue_complex_task(task: &mut Task, buf_pool: &SubBufferPool<ClFloat4>, thread_pool: &CpuPool) {
//     // (0) Initially write 500s:
//     let write_cmd_idx = 0;
//     let future_data = task.map(write_cmd_idx, buf_pool, thread_pool);
//     println!("Awaiting Data...");
//     let pooled_data = thread_pool.spawn(future_data);
//     let mut data = pooled_data.wait().unwrap();
//     println!("Data Ready.");

//     for val in data.iter_mut() {
//         *val = ClFloat4(500., 500., 500., 500.);
//     }

//     task.unmap(&mut data, write_cmd_idx, buf_pool, thread_pool);

//     // (1) Kernel A -- Add values:
//     task.kernel(1);

//     // (2) Copy from buffer[1] to buffer[2]:
//     task.copy(2, buf_pool);

//     // (3) Copy from buffer[1] to buffer[3]:
//     task.copy(3, buf_pool);

//     // (4) Fill buffer[4] with 50s:
//     task.fill(ClFloat4(50., 50., 50., 50.), 4, buf_pool);

//     // (5) Kernel B -- Sum buffers and add values:
//     task.kernel(5);

//     // (6) Kernel C -- Subtract values:
//     task.kernel(6);

// }


// fn verify_complex_task(task: &mut Task, buf_pool: &SubBufferPool<ClFloat4>, thread_pool: &CpuPool,
//         correct_val_count: &mut usize)
// {
//     // (7) Final read from device:
//     let read_cmd_idx = 7;
//     let future_data = task.map(read_cmd_idx, buf_pool, thread_pool);
//     let pooled_data = thread_pool.spawn(future_data);
//     let mut data = pooled_data.wait().unwrap();

//     let expected_result = task.expected_result.unwrap();

//     for val in data.iter() {
//         assert_eq!(*val, expected_result);
//         *correct_val_count += 1;
//     }

//     task.unmap(&mut data, read_cmd_idx, buf_pool, thread_pool);
// }




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
        buf_pool: &mut SubBufferPool<ClFloat4>, work_size: u32, write_queue: Queue,
        read_queue: Queue, kern_queue: Queue) -> Result<Task, ()>
{
    let write_buf_flags = Some(MemFlags::read_only() | MemFlags::host_write_only());
    let read_buf_flags = Some(MemFlags::write_only() | MemFlags::host_read_only());
    // let write_buf_flags = Some(MemFlags::read_only() | MemFlags::host_write_only() | MemFlags::alloc_host_ptr());
    // let read_buf_flags = Some(MemFlags::write_only() | MemFlags::host_read_only() | MemFlags::alloc_host_ptr());

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

    buf_pool.get_mut(write_buf_id).unwrap().set_default_queue(write_queue);
    buf_pool.get_mut(read_buf_id).unwrap().set_default_queue(read_queue);

    let program = Program::builder()
        .devices(device)
        .src(gen_kern_src("kern", "float4", true, true))
        .build(context).unwrap();

    let kern = Kernel::new("kern", &program, kern_queue).unwrap()
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



// fn enqueue_simple_task(task: &Task, buf_pool: &SubBufferPool<ClFloat4>, /*handle: &Handle,*/
//         mut tx: Sender<usize>) /*-> BoxFuture<usize, AsyncError>*/
// {
//     // (0) Write a bunch of 50's:
//     // thread_pool.spawn(task.map(0, buf_pool)
//     //     .and_then(move |mut data| {
//     //         for val in data.iter_mut() {
//     //             *val = ClFloat4(50., 50., 50., 50.);
//     //         }

//     //         println!("Data has been written.");

//     //         Ok(())
//     //     })
//     // ).forget();
//     // // ).wait().unwrap();

//     // // (1) Run kernel (adds 100 to everything):
//     // task.kernel(1)

//     // // (2) Read results and verify them:
//     // thread_pool.spawn(task.map(2, buf_pool)
//     //     .and_then(move |mut data| {
//     //         let mut val_count = 0usize;

//     //         for val in data.iter() {
//     //             let correct_val = ClFloat4(150., 150., 150., 150.);
//     //             if *val != correct_val {
//     //                 return Err(format!("Result value mismatch: {:?} != {:?}", val, correct_val).into())
//     //             }
//     //             val_count += 1;
//     //         }

//     //         println!("Verify done.");

//     //         Ok(val_count)
//     //     })
//     //     .and_then(move |val_count| {
//     //         Ok(tx.send(val_count))
//     //     })
//     // // ).forget();
//     // ).wait().unwrap().wait().unwrap();


//     // // (1) Run kernel (adds 100 to everything):
//     // handle.spawn(task.kernel(1)
//     //     .then(|_| Ok(()))
//     //     // .wait().unwrap();
//     // );

//     // (2) Read results and verify them:
//     // handle.spawn(task.map(2, buf_pool)
//     //     .and_then(move |mut data| {
//     //         let mut val_count = 0usize;

//     //         for val in data.iter() {
//     //             let correct_val = ClFloat4(150., 150., 150., 150.);
//     //             if *val != correct_val {
//     //                 return Err(format!("Result value mismatch: {:?} != {:?}", val, correct_val).into())
//     //             }
//     //             val_count += 1;
//     //         }

//     //         print!("Verify done. ");

//     //         // Ok(val_count)
//     //         // Ok(tx.send(val_count).and_then(|_| Ok(())))
//     //         Ok(tx.send(val_count).wait())

//     //     })
//     //     // .and_then(move |val_count| {
//     //     //     Ok(tx.send(val_count))
//     //     // })
//     //     .then(|res| { /*res.unwrap();*/ Ok(()) })
//     //     // .wait().unwrap()
//     //     // .wait().unwrap();
//     // );


//     println!("(0)");

//     // (0) Write a bunch of 50's:
//     let write = task.map(0, buf_pool).and_then(move |mut data| {
//         for val in data.iter_mut() {
//             *val = ClFloat4(50., 50., 50., 50.);
//         }

//         println!("Data has been written. ");

//         Ok(())
//     });

//     println!("(1)");

//     // (1) Run kernel (adds 100 to everything):
//     let kernel = task.kernel(1);

//     println!("(2)");

//     // (2) Read results and verify them:
//     let verify = task.map(2, buf_pool).and_then(move |mut data| {
//         let mut val_count = 0usize;

//         for val in data.iter() {
//             let correct_val = ClFloat4(150., 150., 150., 150.);
//             if *val != correct_val {
//                 return Err(format!("Result value mismatch: {:?} != {:?}", val, correct_val).into())
//             }
//             val_count += 1;
//         }

//         println!("Verify done. ");

//         Ok(val_count)
//         // Ok(tx.send(val_count).and_then(|_| Ok(())))
//         // Ok(tx.send(val_count).wait())
//     });

//     println!(" Returning... ");

//     write
//         .and_then(|_| kernel)
//         .and_then(|_| verify)
//         .boxed()
//         .wait().unwrap();


//     // println!("Task[{}], Command[2] (map) has been spawned.", task.task_id);


//     // let (mut data, vals_checked) = pooled_data.wait().unwrap();
//     // *correct_val_count += vals_checked;

//     //#########################################################################
//     //############################## UNMAP ####################################
//     //#########################################################################

//     // data.enqueue_unmap(None, None::<Event>, None::<&mut Event>).unwrap();

//     // task.get_finish_events();
//     // assert!(task.finish_events.len() == 1);
// }





/// Creates a large number of both simple and complex asynchronous tasks and
/// verifies that they all execute correctly.
pub fn main() {
    use std::mem;
    use ocl::core::{Event as EventCore};

    // println!("Event: {}", mem::size_of::<[Event; 10]>());
    // println!("EventCore: {}", mem::size_of::<[EventCore; 10]>());
    // println!("UserEventCore: {}", mem::size_of::<[UserEventCore; 10]>());
    // return;


    // Buffer/work size range:
    let buffer_size_range = RandRange::new(SUB_BUF_MIN_LEN, SUB_BUF_MAX_LEN);
    let mut rng = rand::weak_rng();

    // Set up context using defaults:
    let platform = Platform::default();
    println!("Platform: {}", platform.name());

    // let device = Device::first(platform);
    let device_idx = 1;

    let device = Device::specifier()
        .wrapping_indices(vec![device_idx])
        .to_device_list(Some(&platform)).unwrap()[0];

    println!("Device: {} {}", device.vendor(), device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    // Queues (events coordinated by command graph):
    let queue_flags = Some(SWITCHES.queue_flags);
    let write_queue = Queue::new(&context, device, queue_flags).unwrap();
    let read_queue = Queue::new(&context, device, queue_flags).unwrap();
    let kern_queue = Queue::new(&context, device, queue_flags).unwrap();

    let mut buf_pool: SubBufferPool<ClFloat4> = SubBufferPool::new(INITIAL_BUFFER_LEN, write_queue.clone());
    // let mut simple_tasks: Vec<Task> = Vec::new();
    // let mut complex_tasks: Vec<Task> = Vec::new();
    let mut tasks: HashMap<usize, Task> = HashMap::new();
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
            let task_id = tasks.len();
            let task = match create_simple_task(task_id, device, &context, &mut buf_pool,
                    work_size, write_queue.clone(), read_queue.clone(), kern_queue.clone())
            {
                Ok(task) => task,
                Err(_) => {
                    pool_full = true;
                    println!("Buffer pool is full.");
                    break;
                },
            };

            // // Add task to the list:
            // simple_tasks.push(task);

            // Add task to the list:
            // Rc::new(tasks.insert(task_id, task));
            tasks.insert(task_id, task);
        }

        // // Add 5 complex tasks:
        // for _ in 0..5 {
        //     // Random work size:
        //     let work_size = buffer_size_range.ind_sample(&mut rng);

        //     // Create task if there is room in the buffer pool:
        //     let task_id = tasks.len();
        //     let task = match create_complex_task(task_id, device, &context, &mut buf_pool,
        //             work_size, kern_queue.clone(), &mut rng)
        //     {
        //         Ok(task) => task,
        //         Err(_) => {
        //             pool_full = true;
        //             println!("Buffer pool is full.");
        //             break;
        //         },
        //     };

        //     // // Add task to the list:
        //     // complex_tasks.push(task);

        //     // Add task to the list:
        //     tasks.insert(task_id, task);
        // }
    }

    let thread_pool = CpuPool::new_num_cpus();
    let mut core = Core::new().unwrap();
    let handle = core.handle();
    // let remote  = core.remote();

    let mut correct_val_count = 0usize;
    let (tx, mut rx) = mpsc::channel(1);
    // let (tx, mut rx) = mpsc::unbounded();

    let create_duration = chrono::Local::now() - start_time;

    println!("Enqueuing tasks...");
    // println!("Reading and verifying task results...");

    // let future_tasks = ;

    let mut futures = Vec::with_capacity(tasks.len());

    // let enqueue_all = stream::iter(tasks.values().map(|t| Ok::<_, ()>(t))).for_each(|task| {
    for task in tasks.values() {
        let tx = tx.clone();

        // println!("(0)");

        let write_map = task.map(0, &buf_pool);

        // (0) Write a bunch of 50's:
        let write_write = write_map
            .and_then(move |mut data| {
                for val in data.iter_mut() {
                    *val = ClFloat4(50., 50., 50., 50.);
                }

                // println!("Data has been written. ");

                Ok(())
            })
            // .then(|_| Ok::<_, ()>(()))
        // .wait().unwrap();
        ;

        // write_write.wait().unwrap();
        // let write_spawn = thread_pool.spawn(write_write);
        // write_futures.push(write_write);
        let write_spawned = thread_pool.spawn(write_write);

        // println!("(1)");

        // (1) Run kernel (adds 100 to everything):
        let kernel_future = task.kernel(1);

        // println!("(2)");

        // (2) Read results and verify them:
        let verify =
            // .and_then(|_| {
                task.map(2, &buf_pool).and_then(move |mut data| {
                    let mut val_count = 0usize;

                    for val in data.iter() {
                        let correct_val = ClFloat4(150., 150., 150., 150.);
                        if *val != correct_val {
                            return Err(format!("Result value mismatch: {:?} != {:?}", val, correct_val).into())
                        }
                        val_count += 1;
                    }

                    // println!("Verify done. ");

                    // Ok(val_count)
                    // Ok(tx.send(val_count).and_then(|_| Ok(()))) // <---- DOESN'T WORK (not sure why)
                    Ok(tx.send(val_count).wait())
                })
            // })
            // .and_then(|_| Ok(()))
        ;

        let verify_spawned = thread_pool.spawn(verify);

        futures.push(write_spawned.join(verify_spawned));
    }
    // );


    stream::futures_unordered(futures).for_each(|_| {
        // correct_val_count.set(correct_val_count.get() + val_count);
        // println!("Task: {} has completed.", task_id);
        Ok(())
    }).wait().unwrap();

    // thread_pool.spawn_fn(|| {
    //     for future in futures {
    //         // spawn.forget();
    //         future.wait().unwrap();
    //     }

    //     Ok::<(), ()>(())
    // }).wait().unwrap();




    // for task in tasks.values() {
    //     println!("WTF");

    //     let val = match task.kind {
    //         // TaskKind::Simple => enqueue_simple_task(task, &buf_pool, &thread_pool, tx.clone()),
    //         TaskKind::Simple => enqueue_simple_task(task, &buf_pool, tx.clone()),
    //         // TaskKind::Complex => enqueue_complex_task(task, &buf_pool, &thread_pool),
    //         TaskKind::Complex => unreachable!(),
    //     };
    //     // .wait().unwrap();

    //     // println!("[{:?}]", val);
    // }

    // let enqueue_all = stream::iter(tasks.values().map(|t| Ok::<_, ()>(t))).for_each(|task| {
    //     handle.spawn(
    //         match task.kind {
    //             // TaskKind::Simple => enqueue_simple_task(task, &buf_pool, &thread_pool, tx.clone()),
    //             TaskKind::Simple => enqueue_simple_task(task, &buf_pool, tx.clone()),
    //             // TaskKind::Complex => enqueue_complex_task(task, &buf_pool, &thread_pool),
    //             TaskKind::Complex => unreachable!(),
    //         }
    //         .then(|_| Ok(()))
    //     );
    //     // print!("{} ", count);

    //     Ok(())
    // });

    print!("\n");

    let _ = tx;

    // println!("Core running...");

    // let all = stream::iter(write_futures.iter().map(|r| Ok::<_, ()>(r)).into_future();

    // core.run(all).unwrap();

    // println!("\nCore done.");

    rx.close();

    for count in rx.wait() {
        // println!("Count: {}", count.unwrap());
        correct_val_count += count.unwrap();
    }

    let enqueue_duration = chrono::Local::now() - start_time - create_duration;
    let run_duration = chrono::Local::now() - start_time - create_duration - enqueue_duration;

    // for task in tasks.values() {

    //     // let (mut data, vals_checked) = pooled_data.wait().unwrap();
    //     // correct_val_count += vals_checked;
    // }

    let final_duration = chrono::Local::now() - start_time - create_duration - enqueue_duration - run_duration;

    // println!("All {} (float4) result values from {} simple and {} complex tasks are correct! \n\
    //     Durations => | Create: {} | Run: {} | Verify: {} | ",
    //     correct_val_count, simple_tasks.len(), complex_tasks.len(), fmt_duration(create_duration),
    //     fmt_duration(run_duration), fmt_duration(verify_duration));
    printlnc!(yellow_bold: "All {} (float4) result values from {} tasks are correct! \n\
        Durations => | Create: {} | Enqueue: {} | Run: {} | Final: {} | ",
        correct_val_count, tasks.len(), fmt_duration(create_duration),
        fmt_duration(enqueue_duration), fmt_duration(run_duration),
        fmt_duration(final_duration));
}


fn fmt_duration(duration: chrono::Duration) -> String {
    let el_sec = duration.num_seconds();
    let el_ms = duration.num_milliseconds() - (el_sec * 1000);
    format!("{}.{} seconds", el_sec, el_ms)
}





// /// Returns a simple task.
// ///
// /// This task will:
// ///
// /// (0) Write data
// /// (1) Run one kernel
// /// (2) Read data
// ///
// fn create_simple_task(task_id: usize, device: Device, context: &Context,
//         buf_pool: &mut SubBufferPool<ClFloat4>, work_size: u32, queue: Queue) -> Result<Task, ()>
// {
//     let write_buf_flags = Some(MemFlags::read_only() | MemFlags::host_write_only());
//     let read_buf_flags = Some(MemFlags::write_only() | MemFlags::host_read_only());
//     // let write_buf_flags = Some(MemFlags::read_only() | MemFlags::host_write_only() | MemFlags::alloc_host_ptr());
//     // let read_buf_flags = Some(MemFlags::write_only() | MemFlags::host_read_only() | MemFlags::alloc_host_ptr());

//     // The container for this task:
//     let mut task = Task::new(task_id, queue.clone(), TaskKind::Simple, work_size);

//     // Allocate our input buffer:
//     let write_buf_id = match buf_pool.alloc(work_size, write_buf_flags) {
//         Ok(buf_id) => buf_id,
//         Err(_) => return Err(()),
//     };

//     // Allocate our output buffer, freeing the unused input buffer upon error.
//     let read_buf_id = match buf_pool.alloc(work_size, read_buf_flags) {
//         Ok(buf_id) => buf_id,
//         Err(_) => {
//             buf_pool.free(write_buf_id).ok();
//             return Err(());
//         },
//     };

//     let write_buf = buf_pool.get(write_buf_id).unwrap();
//     let read_buf = buf_pool.get(read_buf_id).unwrap();

//     let program = Program::builder()
//         .devices(device)
//         .src(gen_kern_src("kern", "float4", true, true))
//         .build(context).unwrap();

//     let kern = Kernel::new("kern", &program, queue).unwrap()
//         .gws(work_size)
//         .arg_buf(write_buf)
//         .arg_vec(ClFloat4(100., 100., 100., 100.))
//         .arg_buf(read_buf);

//     // (0) Initial write to device:
//     assert!(task.add_write_command(write_buf_id).unwrap() == 0);

//     // (1) Kernel:
//     assert!(task.add_kernel(kern,
//         vec![KernelArgBuffer::new(0, write_buf_id)],
//         vec![KernelArgBuffer::new(2, read_buf_id)]).unwrap() == 1);

//     // (2) Final read from device:
//     assert!(task.add_read_command(read_buf_id).unwrap() == 2);

//     // Populate the command graph:
//     task.cmd_graph.populate_requisites();
//     Ok(task)
// }


// //#############################################################################
// //#############################################################################
// //######################### INITIATE SIMPLE TASK ##############################
// //#############################################################################
// //#############################################################################


// fn initiate_simple_task(task: &mut Task, buf_pool: &SubBufferPool<ClFloat4>, thread_pool: &CpuPool) {
//     // (0) Write a bunch of 50's:
//     let cmd_idx = 0;
//     let task_id = task.task_id;

//     //#########################################################################
//     //############################### MAP #####################################
//     //#########################################################################

//     let (buffer_id, flags) = match *task.cmd_graph.commands()[cmd_idx].details(){
//         CommandDetails::Write { target } => (target, MapFlags::write_invalidate_region()),
//         CommandDetails::Read { source } => (source, MapFlags::read()),
//         _ => panic!("Task::map: Not a write or read command."),
//     };

//     let buf = buf_pool.get(buffer_id).unwrap();

//     let mut map_event = Event::empty();

//     let mut future_data: FutureMemMap<ClFloat4> = buf.cmd().map().flags(flags)
//         .ewait(task.cmd_graph.get_req_events(cmd_idx).unwrap())
//         .enew(&mut map_event)
//         .enq_async().unwrap();

//     let unmap_event_target = future_data.create_unmap_event().unwrap().clone();
//     task.cmd_graph.set_cmd_event(cmd_idx, unmap_event_target.into()).unwrap();

//     //#########################################################################
//     //############################## WRITE ####################################
//     //#########################################################################

//     let pooled_data = thread_pool.spawn(future_data.and_then(move |mut data| {
//         for val in data.iter_mut() {
//             *val = ClFloat4(50., 50., 50., 50.);
//         }

//         Ok(data)
//     }));

//     let data = pooled_data.wait().unwrap();

//     //#########################################################################
//     //############################# KERNEL ####################################
//     //#########################################################################

//     // (1) Run kernel (adds 100 to everything):
//     task.kernel(1);
// }


// //#############################################################################
// //#############################################################################
// //########################### VERIFY SIMPLE TASK ##############################
// //#############################################################################
// //#############################################################################

// fn verify_simple_task(task: &mut Task, buf_pool: &SubBufferPool<ClFloat4>, thread_pool: &CpuPool,
//         correct_val_count: &mut usize)
// {
//     // (2) Read results and verify them:
//     let cmd_idx = 2;
//     let task_id = task.task_id;

//     //#########################################################################
//     //############################### MAP #####################################
//     //#########################################################################

//     // (2) Read results and verify them:
//     let (buffer_id, flags) = match *task.cmd_graph.commands()[cmd_idx].details(){
//         CommandDetails::Write { target } => (target, MapFlags::write_invalidate_region()),
//         CommandDetails::Read { source } => (source, MapFlags::read()),
//         _ => panic!("Task::map: Not a write or read command."),
//     };

//     let buf = buf_pool.get(buffer_id).unwrap();

//     let mut future_data = buf.cmd().map().flags(flags)
//         .ewait(task.cmd_graph.get_req_events(2).unwrap())
//         .enq_async().unwrap();

//     let unmap_event = future_data.create_unmap_event().unwrap().clone();
//     task.cmd_graph.set_cmd_event(cmd_idx, unmap_event.into()).unwrap();

//     //#########################################################################
//     //############################## READ #####################################
//     //#########################################################################

//     let pooled_data = thread_pool.spawn(future_data.and_then(move |mut data| {
//     // let pooled_data = future_data.and_then(move |mut data| {
//         let mut val_count = 0usize;

//         for val in data.iter() {
//             let correct_val = ClFloat4(150., 150., 150., 150.);
//             if *val != correct_val {
//                 return Err(format!("Result value mismatch: {:?} != {:?}",
//                     val, correct_val).into())
//             }

//             val_count += 1;
//         }

//         Ok((data, val_count))
//     }));
//     // });

//     let (mut data, vals_checked) = pooled_data.wait().unwrap();
//     *correct_val_count += vals_checked;

//     //#########################################################################
//     //############################## UNMAP ####################################
//     //#########################################################################

//     data.enqueue_unmap(None, None::<Event>, None::<&mut Event>).unwrap();

//     task.get_finish_events();
//     assert!(task.finish_events.len() == 1);
// }




