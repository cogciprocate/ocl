//! An OpenCL command queue.

use std;
use core::{self, CommandQueue as CommandQueueCore, Context as ContextCore,
    CommandQueueInfo, CommandQueueInfoResult};
use standard::{self, Context, Device};

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
    device: Device,
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

        let device = match device {
            Some(d) => d.clone(),
            None => context.get_device_by_index(0).clone(),
        };

        let obj_core = core::create_command_queue(context.core_as_ref(), &device)
            .expect("[FIXME: TEMPORARY]: Queue::new_by_device_index():"); 

        Queue {
            obj_core: obj_core,
            context_obj_core: context.core_as_ref().clone(),
            device: device, 
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

        let devices = context.resolve_device_idxs(&device_idxs);
        assert!(devices.len() == 1, "Queue::new_by_device_index: Error resolving device ids.");
        let device = devices[0].clone();

        let obj_core = core::create_command_queue(context.core_as_ref(), &device)
            .expect("[FIXME: TEMPORARY]: Queue::new_by_device_index():"); 

        Queue {
            obj_core: obj_core,
            context_obj_core: context.core_as_ref().clone(),
            device: device, 
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
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns info about this queue.
    pub fn info(&self, info_kind: CommandQueueInfo) -> CommandQueueInfoResult {
        match core::get_command_queue_info(&self.obj_core, info_kind) {
            Ok(res) => res,
            Err(err) => CommandQueueInfoResult::Error(Box::new(err)),
        }        
    }
}




impl std::fmt::Display for Queue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // write!(f, "{}", &self.to_string())
        let (begin, delim, end) = if standard::INFO_FORMAT_MULTILINE {
            ("\n", "\n", "\n")
        } else {
            ("{ ", ", ", " }")
        };

        // TemporaryPlaceholderVariant(Vec<u8>),
        // Context(Context),
        // Device(DeviceId),
        // ReferenceCount(u32),
        // Properties(CommandQueueProperties),
        // Error(Box<OclError>),

        write!(f, "[Queue]: {b}\
                Context: {}{d}\
                Device: {}{d}\
                ReferenceCount: {}{d}\
                Properties: {}{e}\
            ",
            self.info(CommandQueueInfo::Context),
            self.info(CommandQueueInfo::Device),
            self.info(CommandQueueInfo::ReferenceCount),
            self.info(CommandQueueInfo::Properties),
            b = begin,
            d = delim,
            e = end,
        )
    }
}
