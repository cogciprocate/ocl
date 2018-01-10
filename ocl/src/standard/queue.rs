//! An `OpenCL` command queue.

use std;
use std::ops::{Deref, DerefMut};
use core::{self, Result as OclCoreResult, CommandQueue as CommandQueueCore, CommandQueueInfo,
    CommandQueueInfoResult, OpenclVersion, CommandQueueProperties, ClWaitListPtr, ClContextPtr};
use error::{Error as OclError, Result as OclResult};
use standard::{Context, Device, Event};

/// A command queue which manages all actions taken on kernels, buffers, and
/// images.
///
///
//
// * TODO: Consider implementing a constructor which accepts a DeviceIdCore and
// creates a context and queue from it.
//
//
#[derive(Clone, Debug)]
pub struct Queue {
    obj_core: CommandQueueCore,
    device_version: OpenclVersion,
}

impl Queue {
    /// Returns a new Queue on the device specified by `device`.
    pub fn new(context: &Context, device: Device, properties: Option<CommandQueueProperties>)
            -> OclResult<Queue> {
        let obj_core = core::create_command_queue(context, &device, properties)?;
        let device_version = device.version()?;

        Ok(Queue {
            obj_core: obj_core,
            device_version: device_version,
        })
    }

    /// Issues all previously queued OpenCL commands to the device.
    pub fn flush(&self) -> OclResult<()> {
        core::flush(&self.obj_core).map_err(OclError::from)
    }

    /// Blocks until all commands in this queue have completed before returning.
    pub fn finish(&self) -> OclResult<()> {
        core::finish(&self.obj_core).map_err(OclError::from)
    }

    /// Enqueues a marker command which waits for either a list of events to
    /// complete, or all previously enqueued commands to complete.
    pub fn enqueue_marker<Ewl>(&self, ewait: Option<Ewl>) -> OclResult<Event>
            where Ewl: ClWaitListPtr
    {
        let mut marker_event = Event::empty();
        core::enqueue_marker_with_wait_list(&self.obj_core, ewait, Some(&mut marker_event),
                Some(&self.device_version)).map(|_| marker_event)
            .map_err(OclError::from)
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn as_core(&self) -> &CommandQueueCore {
        &self.obj_core
    }

    /// Returns a copy of the Context associated with this queue.
    pub fn context(&self) -> Context {
        self.obj_core.context().map(Context::from).unwrap()
    }

    /// Returns the `OpenCL` device associated with this queue.
    pub fn device(&self) -> Device {
        self.obj_core.device().map(Device::from).unwrap()
    }

    /// Returns the cached device version.
    pub fn device_version(&self) -> OpenclVersion {
        self.device_version
    }

    /// Returns info about this queue.
    pub fn info(&self, info_kind: CommandQueueInfo) -> OclCoreResult<CommandQueueInfoResult> {
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

impl AsRef<Queue> for Queue {
    fn as_ref(&self) -> &Queue {
        self
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

unsafe impl<'a> ClContextPtr for &'a Queue {
    fn as_ptr(&self) -> ::ffi::cl_context {
        self.context_ptr().expect("<&Queue as ClContextPtr>::as_ptr: \
            Unable to obtain a context pointer.")
    }
}