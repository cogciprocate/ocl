//! An `OpenCL` command queue.

use std;
use std::ops::{Deref, DerefMut};
use error::{Result as OclResult};
use core::{self, CommandQueue as CommandQueueCore, Context as ContextCore,
    CommandQueueInfo, CommandQueueInfoResult};
use standard::{Context, Device};

/// A command queue which manages all actions taken on kernels, buffers, and
/// images.
///
/// ## Destruction
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
    pub fn new(context: &Context, device: Device) -> OclResult<Queue> {
        let obj_core = try!(core::create_command_queue(context, &device));

        Ok(Queue {
            obj_core: obj_core,
            context_obj_core: context.core_as_ref().clone(),
            device: device,
        })
    }

    /// Blocks until all commands in this queue have completed before returning.
    pub fn finish(&self) {
        core::finish(&self.obj_core).unwrap();
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    pub fn core_as_ref(&self) -> &CommandQueueCore {
        &self.obj_core
    }

    /// Returns a reference to the core pointer wrapper of the context
    /// associated with this queue, usable by functions in the `core` module.
    pub fn context_core_as_ref(&self) -> &ContextCore {
        &self.context_obj_core
    }

    /// Returns the `OpenCL` device associated with this queue.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns info about this queue.
    pub fn info(&self, info_kind: CommandQueueInfo) -> CommandQueueInfoResult {
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
