//! Types related to futures and asynchrony.

extern crate qutex;

mod buffer_sink;
mod buffer_stream;
mod future_mem_map;
mod mem_map;
mod order_lock;
mod rw_vec;

pub use self::buffer_sink::{BufferSink, FutureFlush, Inner as BufferSinkInner};
pub use self::buffer_stream::{BufferStream, FutureFlood, Inner as BufferStreamInner};
pub use self::future_mem_map::FutureMemMap;
pub use self::mem_map::MemMap;
pub use self::order_lock::{
    FutureGuard, FutureReadGuard, FutureWriteGuard, OrderGuard, OrderLock, ReadGuard, WriteGuard,
};
pub use self::rw_vec::RwVec;

// * TODO: Implement this:
//
// pub struct EventListTrigger {
//     wait_events: EventList,
//     completion_event: UserEvent,
//     callback_is_set: bool,
// }

// pub struct EventTrigger {
//     wait_event: Event,
//     completion_event: UserEvent,
//     callback_is_set: bool,
// }

// impl EventTrigger {
//     pub fn new(wait_event: Event, completion_event: UserEvent) -> EventTrigger {
//         EventTrigger {
//             wait_event: wait_event,
//             completion_event: completion_event ,
//             callback_is_set: false,
//         }
//     }
// }
