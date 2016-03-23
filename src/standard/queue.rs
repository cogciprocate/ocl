//! An OpenCL command queue.

use std;
use std::ops::{Deref, DerefMut};
use error::{Result as OclResult};
use core::{self, CommandQueue as CommandQueueCore, Context as ContextCore,
    CommandQueueInfo, CommandQueueInfoResult};
use standard::{Context, Device};

/// A command queue which manages all actions taken on kernels, buffers, and
/// images.
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
    device: Device,
}

impl Queue {
    /// Returns a new Queue on the device specified by `device`. 
    ///
    /// Not specifying `device` will default to the first available device
    /// associated with `context`.
    ///
    /// [FIXME]: Return result.
    pub fn new(context: &Context, device: Device) -> OclResult<Queue> {
        // let device_idxs = match device_idx {
        //     Some(idx) => vec![idx],
        //     None => Vec::with_capacity(0),
        // };

        // let device_ids_core = context.resolve_device_idxs(&device_idxs);
        // assert!(device_ids_core.len() == 1, "Queue::new_by_device_index: Error resolving device ids.");
        // let device_id_core = device_ids_core[0].clone();

        // let device = match device {
        //     Some(d) => d,
        //     None => context.devices()[0],
        // };

        let obj_core = try!(core::create_command_queue(context, &device));

        Ok(Queue {
            obj_core: obj_core,
            context_obj_core: context.core_as_ref().clone(),
            device: device, 
        })
    }


    // /// Returns a new Queue on the device specified by `device_idx`. 
    // ///
    // /// 'device_idx` refers to a index in the list of devices generated when creating
    // /// `context`. For a list of these devices, call `context.device_ids()`. If 
    // /// `device_idx` is out of range, it will automatically 'wrap around' via a 
    // /// modulo operation and therefore is valid up to the limit of `usize`. See
    // /// the documentation for `Context` for more information.
    // /// 
    // /// [FIXME]: Return result.
    // pub fn new_by_device_index(context: &Context, device_idx: Option<usize>) -> Queue {
    //     let device_idxs = match device_idx {
    //         Some(idx) => vec![idx],
    //         None => Vec::with_capacity(0),
    //     };


    //     let devices = context.resolve_device_idxs(&device_idxs);
    //     println!("QUEUE DEVICES: {:?}", devices);
    //     assert!(devices.len() == 1, "Queue::new_by_device_index: Error resolving device ids.");
    //     let device = devices[0].clone();

    //     let obj_core = core::create_command_queue(context, &device)
    //         .expect("[FIXME: TEMPORARY]: Queue::new_by_device_index():"); 

    //     Queue {
    //         obj_core: obj_core,
    //         context_obj_core: context.core_as_ref().clone(),
    //         device: device, 
    //     }
    // }  


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
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns info about this queue.
    pub fn info(&self, info_kind: CommandQueueInfo) -> CommandQueueInfoResult {
        // match core::get_command_queue_info(&self.obj_core, info_kind) {
        //     Ok(res) => res,
        //     Err(err) => CommandQueueInfoResult::Error(Box::new(err)),
        // }        
        core::get_command_queue_info(&self.obj_core, info_kind)
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Queue")
            .field("Context", &self.info(CommandQueueInfo::Context))
            .field("Device", &self.info(CommandQueueInfo::Device))
            .field("ReferenceCount", &self.info(CommandQueueInfo::ReferenceCount))
            .field("Properties", &self.info(CommandQueueInfo::Properties))
            .finish()
    }
}

impl std::fmt::Display for Queue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl AsRef<CommandQueueCore> for Queue {
    fn as_ref(&self) -> &CommandQueueCore {
        &self.obj_core
    }
}

impl Deref for Queue {
    type Target = CommandQueueCore;

    fn deref(&self) -> &CommandQueueCore {
        &self.obj_core
    }
}

impl DerefMut for Queue {
    fn deref_mut(&mut self) -> &mut CommandQueueCore {
        &mut self.obj_core
    }
}
