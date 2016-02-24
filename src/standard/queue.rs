//! An OpenCL command queue.

use core::{self, CommandQueue as CommandQueueCore, DeviceId as DeviceIdCore, Context as ContextCore};
use standard::{Context, Device};

/// A command queue.
///
/// # Destruction
///
/// Underlying queue object is destroyed automatically.
///
// TODO: Implement a constructor which accepts a DeviceIdCore.
#[derive(Clone, Debug)]
pub struct Queue {
    obj_core: CommandQueueCore,
    context_obj_core: ContextCore,
    device_id_core: DeviceIdCore,
}

impl Queue {
    /// Returns a new Queue on the device specified by `device`. 
    ///
    /// Not specifying `device` will default to the first available device
    /// associated with `context`.
    ///
    /// [FIXME]: Return result.
    pub fn new(context: &Context, device: Option<Device>) -> Queue {
        // let device_idxs = match device_idx {
        //     Some(idx) => vec![idx],
        //     None => Vec::with_capacity(0),
        // };

        // let device_ids_core = context.resolve_device_idxs(&device_idxs);
        // assert!(device_ids_core.len() == 1, "Queue::new_by_device_index: Error resolving device ids.");
        // let device_id_core = device_ids_core[0].clone();

        let device_id_core = match device {
            Some(d) => d.as_core().clone(),
            None => context.get_device_by_index(0).as_core().clone(),
        };

        let obj_core = core::create_command_queue(context.core_as_ref(), &device_id_core)
            .expect("[FIXME: TEMPORARY]: Queue::new_by_device_index():"); 

        Queue {
            obj_core: obj_core,
            context_obj_core: context.core_as_ref().clone(),
            device_id_core: device_id_core, 
        }
    }


    /// Returns a new Queue on the device specified by `device_idx`. 
    ///
    /// 'device_idx` refers to a index in the list of devices generated when creating
    /// `context`. For a list of these devices, call `context.device_ids()`. If 
    /// `device_idx` is out of range, it will automatically 'wrap around' via a 
    /// modulo operation and therefore is valid up to the limit of `usize`. See
    /// the documentation for `Context` for more information.
    /// 
    /// [FIXME]: Return result.
    pub fn new_by_device_index(context: &Context, device_idx: Option<usize>) -> Queue {
        let device_idxs = match device_idx {
            Some(idx) => vec![idx],
            None => Vec::with_capacity(0),
        };

        let device_ids_core = context.resolve_device_idxs(&device_idxs);
        assert!(device_ids_core.len() == 1, "Queue::new_by_device_index: Error resolving device ids.");
        let device_id_core = device_ids_core[0].clone();

        let obj_core = core::create_command_queue(context.core_as_ref(), &device_id_core)
            .expect("[FIXME: TEMPORARY]: Queue::new_by_device_index():"); 

        Queue {
            obj_core: obj_core,
            context_obj_core: context.core_as_ref().clone(),
            device_id_core: device_id_core, 
        }
    }  


    /// Blocks until all commands in this queue have completed.
    pub fn finish(&self) {
        core::finish(&self.obj_core).unwrap();
    }

    /// Returns the OpenCL command queue object associated with this queue.
    pub fn core_as_ref(&self) -> &CommandQueueCore {
        &self.obj_core
    }

    /// Returns the OpenCL context object associated with this queue.
    pub fn context_core_as_ref(&self) -> &ContextCore {
        &self.context_obj_core
    }

    /// Returns the OpenCL device id associated with this queue.
    ///
    /// Not to be confused with the zero-indexed `device_idx` passed to `::new()`
    /// when creating this queue.
    pub fn device_core_as_ref(&self) -> &DeviceIdCore {
        &self.device_id_core
    }
}

