#![allow(dead_code, unused_variables, unused_imports, unused_mut, unreachable_code)]

extern crate libc;
extern crate futures;
extern crate futures_cpupool;
extern crate tokio_core;
extern crate tokio_timer;

extern crate chrono;
extern crate ocl;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate colorify;
extern crate rand;


pub mod sub_buffer_pool;
pub mod command_graph;

pub use libc::c_void;
pub use self::sub_buffer_pool::{SubBufferPool};
pub use self::command_graph::{CommandGraph, Command, CommandDetails, KernelArgBuffer, RwCmdIdxs};
pub use self::util::{Duration, DateTime, Local};
pub use self::util::{ADD_KERN_SRC};
pub use self::util::{fmt_duration, now};



/// Because what good is a library without a module named `util`.
mod util {
    pub use chrono::{Duration, DateTime, Local};

    pub static ADD_KERN_SRC: &'static str = r#"
        __kernel void add(
            __global float4* in,
            float4 values,
            __global float4* out)
        {
            uint idx = get_global_id(0);
            out[idx] = in[idx] + values;
        }
    "#;




    pub fn fmt_duration(duration: Duration) -> String {
        let el_sec = duration.num_seconds();
        let el_ms = duration.num_milliseconds() - (el_sec * 1000);
        format!("{}.{} seconds", el_sec, el_ms)
    }

    pub fn now() -> DateTime<Local> {
        Local::now()
    }
}