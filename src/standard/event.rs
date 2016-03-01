//! An OpenCL event.

use std;
use std::ops::{Deref, DerefMut};
use std::convert::Into;
use core::{self, Event as EventCore, EventInfo, EventInfoResult, ProfilingInfo, ProfilingInfoResult};


/// An event representing a command or user created event.
#[derive(Clone, Debug)]
pub struct Event(EventCore);

impl Event {
    /// Creates a new `Event` from a `EventCore`.
    ///
    /// ## Safety 
    ///
    /// Not meant to be called directly.
    pub fn new(event_core: EventCore) -> Event {
        Event(event_core)
    }

    // /// Creates a new null `Event`.
    // ///
    // /// ## Safety 
    // ///
    // /// Don't use unless you know what you're doing.
    // pub unsafe fn new_null() -> Event {
    //  Event(EventCore::null())
    // }

    /// Returns info about the event. 
    pub fn info(&self, info_kind: EventInfo) -> EventInfoResult {
        match core::get_event_info(&self.0, info_kind) {
            Ok(pi) => pi,
            Err(err) => EventInfoResult::Error(Box::new(err)),
        }
    }

    /// Returns info about the event. 
    pub fn profiling_info(&self, info_kind: ProfilingInfo) -> ProfilingInfoResult {
        match core::get_event_profiling_info(&self.0, info_kind) {
            Ok(pi) => pi,
            Err(err) => ProfilingInfoResult::Error(Box::new(err)),
        }
    }

    /// Returns a string containing a formatted list of event properties.
    pub fn to_string(&self) -> String {
        self.clone().into()
    }

    /// Returns the underlying `EventCore`.
    pub fn core_as_ref(&self) -> &EventCore {
        &self.0
    }

    /// Returns the underlying `EventCore`.
    pub fn core_as_mut(&mut self) -> &mut EventCore {
        &mut self.0
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Event")
            .field("CommandQueue", &self.info(EventInfo::CommandQueue))
            .field("CommandType", &self.info(EventInfo::CommandType))
            .field("ReferenceCount", &self.info(EventInfo::ReferenceCount))
            .field("CommandExecutionStatus", &self.info(EventInfo::CommandExecutionStatus))
            .field("Context", &self.info(EventInfo::Context))
            .finish()
    }
}

impl Into<String> for Event {
    fn into(self) -> String {
        format!("{}", self)
    }
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl AsRef<EventCore> for Event{
    fn as_ref(&self) -> &EventCore {
        &self.0
    }
}

impl Deref for Event {
    type Target = EventCore;

    fn deref(&self) -> &EventCore {
        &self.0
    }
}

impl DerefMut for Event {
    fn deref_mut(&mut self) -> &mut EventCore {
        &mut self.0
    }
}



//     // ##################################################
//     // ##################### EVENT ######################
//     // ##################################################

//     // [FIXME]: Complete this section.
//     // pub enum EventInfo {
//     //     CommandQueue = cl_h::CL_EVENT_COMMAND_QUEUE as isize,
//     //     CommandType = cl_h::CL_EVENT_COMMAND_TYPE as isize,
//     //     ReferenceCount = cl_h::CL_EVENT_REFERENCE_COUNT as isize,
//     //     CommandExecutionStatus = cl_h::CL_EVENT_COMMAND_EXECUTION_STATUS as isize,
//     //     Context = cl_h::CL_EVENT_CONTEXT as isize,
//     // }

//     println!("EventInfo:\n\
//          {t}CommandQueue: {}\n\
//             {t}CommandType: {}\n\
//             {t}ReferenceCount: {}\n\
//             {t}CommandExecutionStatus: {}\n\
//             {t}Context: {}\n\
//      ",
//      core::get_event_info(event, EventInfo::CommandQueue).unwrap(),
//         core::get_event_info(event, EventInfo::CommandType).unwrap(),
//         core::get_event_info(event, EventInfo::ReferenceCount).unwrap(),
//         core::get_event_info(event, EventInfo::CommandExecutionStatus).unwrap(),
//         core::get_event_info(event, EventInfo::Context).unwrap(),
//      t = util::TAB,
//  );

//     // ##################################################
//     // ################ EVENT PROFILING #################
//     // ##################################################

//     // [FIXME]: Complete this section.
//     // pub enum ProfilingInfo {
//     //     Queued = cl_h::CL_PROFILING_COMMAND_QUEUED as isize,
//     //     Submit = cl_h::CL_PROFILING_COMMAND_SUBMIT as isize,
//     //     Start = cl_h::CL_PROFILING_COMMAND_START as isize,
//     //     End = cl_h::CL_PROFILING_COMMAND_END as isize,
//     // }

//     println!("ProfilingInfo:\n\
//          {t}Queued: {}\n\
//          {t}Submit: {}\n\
//          {t}Start: {}\n\
//          {t}End: {}\n\
//      ",
//      core::get_event_profiling_info(event, ProfilingInfo::Queued).unwrap(),
//         core::get_event_profiling_info(event, ProfilingInfo::Submit).unwrap(),
//         core::get_event_profiling_info(event, ProfilingInfo::Start).unwrap(),
//         core::get_event_profiling_info(event, ProfilingInfo::End).unwrap(),
//      t = util::TAB,
//  );


//     print!("\n");
// }
