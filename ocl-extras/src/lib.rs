#![allow(dead_code, unused_variables, unused_imports, unused_mut, unreachable_code)]

extern crate libc;
extern crate futures;
extern crate futures_cpupool;
extern crate tokio_core;
extern crate tokio_timer;
extern crate rand;
extern crate chrono;
extern crate ocl;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate colorify;


pub mod sub_buffer_pool;
pub mod command_graph;

pub use self::sub_buffer_pool::{SubBufferPool};
pub use self::command_graph::{CommandGraph, Command, CommandDetails, KernelArgBuffer, RwCmdIdxs};