//! An OpenCL command queue.
// use std::mem;
// use std::ptr;

use raw::{self, CommandQueue as CommandQueueRaw, DeviceId as DeviceIdRaw, Context as ContextRaw};
use super::Context;

/// A command queue.
///
/// # Destruction
///
/// Underlying queue object is destroyed automatically.
///
// TODO: Implement a constructor which accepts a DeviceIdRaw.
pub struct Queue {
    obj_raw: CommandQueueRaw,
    context_obj_raw: ContextRaw,
    device_id_raw: DeviceIdRaw,
}

impl Queue {
    /// Returns a new Queue on the device specified by `device_idx`. 
    ///
    /// 'device_idx` refers to a index in the list of devices generated when creating
    /// `context`. For a list of these devices, call `context.device_ids()`. If 
    /// `device_idx` is out of range, it will automatically 'wrap around' via a 
    /// modulo operation and therefore is valid up to the limit of `usize`. See
    /// the documentation for `Context` for more information.
    /// 
    /// [FIXME]: Return result.
    pub fn new(context: &Context, device_idx: Option<usize>) -> Queue {
        let device_idxs = match device_idx {
            Some(idx) => vec![idx],
            None => Vec::with_capacity(0),
        };

        let device_ids_raw = context.resolve_device_idxs(&device_idxs);
        assert!(device_ids_raw.len() == 1, "Queue::new: Error resolving device ids.");
        let device_id_raw = device_ids_raw[0].clone();

        let obj_raw = raw::create_command_queue(context.obj_raw(), &device_id_raw)
            .expect("[FIXME: TEMPORARY]: Queue::new():"); 

        Queue {
            obj_raw: obj_raw,
            context_obj_raw: context.obj_raw().clone(),
            device_id_raw: device_id_raw, 
        }
    }   

    /// Blocks until all commands in this queue have completed.
    pub fn finish(&self) {
        raw::finish(&self.obj_raw).unwrap();
    }

    /// Returns the OpenCL command queue object associated with this queue.
    pub fn obj_raw(&self) -> &CommandQueueRaw {
        &self.obj_raw
    }

    /// Returns the OpenCL context object associated with this queue.
    pub fn context_obj_raw(&self) -> &ContextRaw {
        &self.context_obj_raw
    }

    /// Returns the OpenCL device id associated with this queue.
    ///
    /// Not to be confused with the zero-indexed `device_idx` passed to `::new()`
    /// when creating this queue.
    pub fn device_id_raw(&self) -> &DeviceIdRaw {
        &self.device_id_raw
    }

    // /// Decrements the reference counter of the associated OpenCL command queue object.
    // pub fn release(&mut self) {
    //     raw::release_command_queue(self.obj_raw).unwrap();
    // }
}

// impl Drop for Queue {
//     fn drop(&mut self) {
//         // println!("DROPPING COMMAND QUEUE");
//         raw::release_command_queue(&self.obj_raw).unwrap();
//     }
// }
