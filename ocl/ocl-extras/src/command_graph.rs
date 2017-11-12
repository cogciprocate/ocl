//! A command requisite-dependency graph.
//!
//! ### Queues and Deadlocks
//!
//! When offloading host-side processing to a thread pool or foreign thread, a
//! few important guidelines are important to keep in mind in order to avoid
//! deadlocks. This usually applies to unmap or write commands but could
//! equally apply to any other command which triggers a delayed-user-callback
//! chain.
//!
//! Let's say we have a host-mapped buffer (`MemMap`) which needs to write
//! data to that mapped area and then enqueue an unmap moving the data to a
//! device... [TODO: FINISH THIS]
//!
//! When an offloaded task completes, it will inevitably enqueue another
//! command or trigger some other action which will affect commands which are
//! **already** in a command queue.
//!
//!

use std::cell::{Cell, RefCell, Ref};
use std::collections::{HashMap, BTreeSet};
use ocl::{Event, EventList};

pub struct RwCmdIdxs {
    writers: Vec<usize>,
    readers: Vec<usize>,
}

impl RwCmdIdxs {
    fn new() -> RwCmdIdxs {
        RwCmdIdxs { writers: Vec::new(), readers: Vec::new() }
    }
}

#[allow(dead_code)]
pub struct KernelArgBuffer {
    arg_idx: usize, // Will be used when refreshing kernels after defragging or resizing.
    buffer_id: usize,
}

impl KernelArgBuffer {
    pub fn new(arg_idx: usize, buffer_id: usize) -> KernelArgBuffer {
        KernelArgBuffer { arg_idx: arg_idx, buffer_id: buffer_id }
    }
}


/// Details of a queuable command.
pub enum CommandDetails {
    Fill { target: usize },
    Write { target: usize },
    Read { source: usize },
    Copy { source: usize, target: usize },
    Kernel { id: usize, sources: Vec<KernelArgBuffer>, targets: Vec<KernelArgBuffer> },
}

impl CommandDetails {
    pub fn sources(&self) -> Vec<usize> {
        match *self {
            CommandDetails::Fill { .. } => vec![],
            CommandDetails::Read { source } => vec![source],
            CommandDetails::Write { .. } => vec![],
            CommandDetails::Copy { source, .. } => vec![source],
            CommandDetails::Kernel { ref sources, .. } => {
                sources.iter().map(|arg| arg.buffer_id).collect()
            },
        }
    }

    pub fn targets(&self) -> Vec<usize> {
        match *self {
            CommandDetails::Fill { target } => vec![target],
            CommandDetails::Read { .. } => vec![],
            CommandDetails::Write { target } => vec![target],
            CommandDetails::Copy { target, .. } => vec![target],
            CommandDetails::Kernel { ref targets, .. } => {
                targets.iter().map(|arg| arg.buffer_id).collect()
            },
        }
    }
}


pub struct Command {
    details: CommandDetails,
    event: RefCell<Option<Event>>,
    requisite_events: RefCell<EventList>,
}

impl Command {
    pub fn new(details: CommandDetails) -> Command {
        Command {
            details: details,
            event: RefCell::new(None),
            requisite_events: RefCell::new(EventList::new()),
        }
    }

    /// Returns a list of commands which both precede a command and which
    /// write to a block of memory which is read from by that command.
    pub fn preceding_writers(&self, cmds: &HashMap<usize, RwCmdIdxs>) -> BTreeSet<usize> {
        self.details.sources().iter().flat_map(|cmd_src_block|
            cmds.get(cmd_src_block).unwrap().writers.iter().cloned()).collect()
    }

    /// Returns a list of commands which both follow a command and which read
    /// from a block of memory which is written to by that command.
    pub fn following_readers(&self, cmds: &HashMap<usize, RwCmdIdxs>) -> BTreeSet<usize> {
        self.details.targets().iter().flat_map(|cmd_tar_block|
            cmds.get(cmd_tar_block).unwrap().readers.iter().cloned()).collect()
    }

    pub fn details(&self) -> &CommandDetails { &self.details }
}


/// A directional sequence dependency graph representing the temporal
/// requirements of each asynchronous read, write, copy, and kernel (commands)
/// for a particular task.
///
/// Obviously this is an overkill for this example but this graph is flexible
/// enough to schedule execution correctly and optimally with arbitrarily many
/// parallel tasks with arbitrary duration reads, writes and kernels.
///
/// Note that in this example we are using `buffer_id` a `usize` to represent
/// memory regions (because that's what the allocator above is using) but we
/// could easily use multiple part, complex identifiers/keys. For example, we
/// may have a program with a large number of buffers which are organized into
/// a complex hierarchy or some other arbitrary structure. We could swap
/// `buffer_id` for some value which represented that as long as the
/// identifier we used could uniquely identify each subsection of memory. We
/// could also use ranges of values and do an overlap check and have
/// byte-level precision.
///
pub struct CommandGraph {
    commands: Vec<Command>,
    command_requisites: Vec<Vec<usize>>,
    ends: (Vec<usize>, Vec<usize>),
    locked: bool,
    next_cmd_idx: Cell<usize>,
}

impl CommandGraph {
    /// Returns a new, empty graph.
    pub fn new() -> CommandGraph {
        CommandGraph {
            commands: Vec::new(),
            command_requisites: Vec::new(),
            ends: (Vec::new(), Vec::new()),
            locked: false,
            next_cmd_idx: Cell::new(0),
        }
    }

    /// Adds a new command and returns the command index if successful.
    pub fn add(&mut self, command: Command) -> Result<usize, ()> {
        if self.locked { return Err(()); }
        self.commands.push(command);
        self.command_requisites.push(Vec::new());
        Ok(self.commands.len() - 1)
    }

    /// Returns a sub-buffer map which contains every command that reads from
    /// or writes to each sub-buffer.
    fn readers_and_writers_by_buffer(&self) -> HashMap<usize, RwCmdIdxs> {
        let mut cmds = HashMap::new();

        for (cmd_idx, cmd) in self.commands.iter().enumerate() {
            for cmd_src_block in cmd.details.sources().into_iter() {
                let rw_cmd_idxs = cmds.entry(cmd_src_block.clone())
                    .or_insert(RwCmdIdxs::new());

                rw_cmd_idxs.readers.push(cmd_idx);
            }

            for cmd_tar_block in cmd.details.targets().into_iter() {
                let rw_cmd_idxs = cmds.entry(cmd_tar_block.clone())
                    .or_insert(RwCmdIdxs::new());

                rw_cmd_idxs.writers.push(cmd_idx);
            }
        }

        cmds
    }

    /// Populates the list of requisite commands necessary for each command.
    ///
    /// Requisite commands (preceding writers and following readers) for a
    /// command are those which are causally linked and must come either
    /// directly before or after. By determining whether or not a command
    /// comes directly before or after another we can determine the
    /// causal/temporal relationship between any two nodes on the graph.
    ///
    /// Nodes without any preceding writers or following readers are start or
    /// finish endpoints respectively. It's possible for a graph to have no
    /// endpoints, in which case the graph is closed and at least partially
    /// cyclical.
    ///
    pub fn populate_requisites(&mut self) {
        let cmds = self.readers_and_writers_by_buffer();

        for (cmd_idx, cmd) in self.commands.iter_mut().enumerate() {
            assert!(self.command_requisites[cmd_idx].is_empty());

            // Get all commands which must precede the current `cmd`.
            let preceding_writers = cmd.preceding_writers(&cmds);

            // If there are none, `cmd` is a start endpoint.
            if preceding_writers.len() == 0 { self.ends.0.push(cmd_idx); }

            // Otherwise add them to the list of requisites.
            for &req_cmd_idx in preceding_writers.iter() {
                self.command_requisites[cmd_idx].push(req_cmd_idx);
            }

            // Get all commands which must follow the current `cmd`.
            let following_readers = cmd.following_readers(&cmds);

            // If there are none, `cmd` is a finish endpoint.
            if following_readers.len() == 0 { self.ends.1.push(cmd_idx); }

            // Otherwise add them to the list of requisites.
            for &req_cmd_idx in following_readers.iter() {
                self.command_requisites[cmd_idx].push(req_cmd_idx);
            }

            self.command_requisites[cmd_idx].shrink_to_fit();
        }

        self.commands.shrink_to_fit();
        self.command_requisites.shrink_to_fit();
        self.locked = true;
    }

    /// Returns the list of requisite events for a command.
    pub fn get_req_events(&self, cmd_idx: usize) -> Result<Ref<EventList>, &'static str> {
        if !self.locked { return Err("Call '::populate_requisites' first."); }
        if self.next_cmd_idx.get() != cmd_idx { return Err("Command events requested out of order."); }

        self.commands.get(cmd_idx).unwrap().requisite_events.borrow_mut().clear();

        for &req_idx in self.command_requisites[cmd_idx].iter() {
            let event_opt = self.commands[req_idx].event.borrow().clone();

            if let Some(event) = event_opt {
                self.commands[cmd_idx].requisite_events.borrow_mut().push(event);
            }
        }

        Ok(self.commands[cmd_idx].requisite_events.borrow())
    }

    /// Sets the event associated with the completion of a command.
    pub fn set_cmd_event(&self, cmd_idx: usize, event: Event) -> Result<(), &'static str> {
        if !self.locked { return Err("Call '::populate_requisites' first."); }

        // let event_opt = self.commands[req_idx].event.borrow();

        *self.commands.get(cmd_idx).unwrap().event.borrow_mut() = Some(event);

        if (self.next_cmd_idx.get() + 1) == self.commands.len() {
            self.next_cmd_idx.set(0);
        } else {
            // self.next_cmd_idx += 1;
            self.next_cmd_idx.set(self.next_cmd_idx.get() + 1);
        }

        Ok(())
    }

    pub fn commands<'a>(&'a self) -> &'a [Command] {
        self.commands.as_slice()
    }

    pub fn get_finish_events (&self, event_list: &mut EventList) {
        assert!(self.next_cmd_idx.get() == 0, "Finish events can only be determined \
            for each cycle just after the graph has set its last cmd event.");

        for &cmd_idx in self.ends.1.iter() {
            let event_opt = self.commands[cmd_idx].event.borrow().clone();

            if let Some(event) = event_opt {
                event_list.push(event);
            }
        }
    }
}
