//! A command requisite-dependency graph.

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
    event: Option<Event>,
    requisite_events: EventList,
}

impl Command {
    pub fn new(details: CommandDetails) -> Command {
        Command {
            details: details,
            event: None,
            requisite_events: EventList::new(),
        }
    }

    /// Returns a list of commands which both precede a command and which
    /// write to a block of memory which is read from by that command.
    pub fn preceding_writers(&self, cmds: &HashMap<usize, RwCmdIdxs>) -> BTreeSet<usize> {
        let pre_writers = self.details.sources().iter().flat_map(|cmd_src_block|
                cmds.get(cmd_src_block).unwrap().writers.iter().cloned()).collect();

        pre_writers
    }

    /// Returns a list of commands which both follow a command and which read
    /// from a block of memory which is written to by that command.
    pub fn following_readers(&self, cmds: &HashMap<usize, RwCmdIdxs>) -> BTreeSet<usize> {
        let fol_readers = self.details.targets().iter().flat_map(|cmd_tar_block|
                cmds.get(cmd_tar_block).unwrap().readers.iter().cloned()).collect();

        fol_readers
    }

    pub fn details(&self) -> &CommandDetails { &self.details }
}


/// A sequence dependency graph representing the temporal requirements of each
/// asynchronous read, write, copy, and kernel (commands) for a particular
/// task.
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
    locked: bool,
    next_cmd_idx: usize,
}

impl CommandGraph {
    /// Returns a new, empty graph.
    pub fn new() -> CommandGraph {
        CommandGraph {
            commands: Vec::new(),
            command_requisites: Vec::new(),
            locked: false,
            next_cmd_idx: 0,
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
    fn readers_and_writers_by_sub_buffer(&self) -> HashMap<usize, RwCmdIdxs> {
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

    /// Populates the list of requisite commands necessary for building the
    /// correct event wait list for each command.
    pub fn populate_requisites(&mut self) {
        let cmds = self.readers_and_writers_by_sub_buffer();

        for (cmd_idx, cmd) in self.commands.iter_mut().enumerate() {
            assert!(self.command_requisites[cmd_idx].is_empty());

            for &req_cmd_idx in cmd.preceding_writers(&cmds).iter()
                    .chain(cmd.following_readers(&cmds).iter())
            {
                self.command_requisites[cmd_idx].push(req_cmd_idx);
            }

            self.command_requisites[cmd_idx].shrink_to_fit();
        }

        self.commands.shrink_to_fit();
        self.command_requisites.shrink_to_fit();
        self.locked = true;
    }

    /// Returns the list of requisite events for a command.
    pub fn get_req_events(&mut self, cmd_idx: usize) -> Result<&EventList, &'static str> {
        if !self.locked { return Err("Call '::populate_requisites' first."); }
        if self.next_cmd_idx != cmd_idx { return Err("Command events requested out of order."); }

        self.commands[cmd_idx].requisite_events.clear().unwrap();

        for &req_idx in self.command_requisites[cmd_idx].iter() {
            if let Some(event) = self.commands[req_idx].event.clone() {
                self.commands[cmd_idx].requisite_events.push(event);
            }
        }

        Ok(&self.commands[cmd_idx].requisite_events)
    }

    /// Sets the event associated with the completion of a command.
    pub fn set_cmd_event(&mut self, cmd_idx: usize, event: Event) -> Result<(), &'static str> {
        if !self.locked { return Err("Call '::populate_requisites' first."); }

        self.commands[cmd_idx].event = Some(event);

        if (self.next_cmd_idx + 1) == self.commands.len() {
            self.next_cmd_idx = 0;
        } else {
            self.next_cmd_idx += 1;
        }

        Ok(())
    }

    pub fn commands<'a>(&'a self) -> &'a [Command] {
        self.commands.as_slice()
    }
}
