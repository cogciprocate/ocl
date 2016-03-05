//! An OpenCL event.

use std;
use std::ops::{Deref, DerefMut};
use std::convert::Into;
use cl_h;
use error::{Error as OclError, Result as OclResult};
use core::{self, Event as EventCore, EventInfo, EventInfoResult, ProfilingInfo, ProfilingInfoResult,
    ClEventPtrNew};


/// An event representing a command or user created event.
#[derive(Clone, Debug)]
pub struct Event(Option<EventCore>);

impl Event {
    /// Creates a new, empty event which must be filled by a newly initiated
    /// command.
    pub fn empty() -> Event {
        Event(None)
    }

    /// Creates a new `Event` from a `EventCore`.
    ///
    /// ## Safety 
    ///
    /// Not meant to be called directly.
    pub unsafe fn from_core(event_core: EventCore) -> Event {
        Event(Some(event_core))
    }

       /// Waits for all events in list to complete.
    pub fn wait(&self) -> OclResult<()> {
        assert!(!self.is_empty(), "ocl::Event::wait(): {}", self.err_empty());
        core::wait_for_event(self.0.as_ref().unwrap())
    }

    /// Returns info about the event. 
    pub fn info(&self, info_kind: EventInfo) -> EventInfoResult {
        match self.0 {
            Some(ref core) => { 
                match core::get_event_info(core, info_kind) {
                    Ok(pi) => pi,
                    Err(err) => EventInfoResult::Error(Box::new(err)),
                }
            },
            None => EventInfoResult::Error(Box::new(self.err_empty())),
        }
    }

    /// Returns info about the event. 
    pub fn profiling_info(&self, info_kind: ProfilingInfo) -> ProfilingInfoResult {
        match self.0 {
            Some(ref core) => {
                match core::get_event_profiling_info(core, info_kind) {
                    Ok(pi) => pi,
                    Err(err) => ProfilingInfoResult::Error(Box::new(err)),
                }
            },
            None => ProfilingInfoResult::Error(Box::new(self.err_empty())),
        }
    }

    /// Returns the underlying `EventCore`.
    pub fn core_as_ref(&self) -> Option<&EventCore> {
        self.0.as_ref()
    }

    /// Returns the underlying `EventCore`.
    pub fn core_as_mut(&mut self) -> Option<&mut EventCore> {
        self.0.as_mut()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    fn err_empty(&self) -> OclError {
        OclError::new("This `ocl::Event` is empty and cannot be used until \
            filled by a command.")
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

impl AsRef<EventCore> for Event {
    fn as_ref(&self) -> &EventCore {
        self.0.as_ref().ok_or(self.err_empty()).expect("ocl::Event::as_ref()")
    }
}

impl Deref for Event {
    type Target = EventCore;

    fn deref(&self) -> &EventCore {
        self.0.as_ref().ok_or(self.err_empty()).expect("ocl::Event::deref()")
    }
}

impl DerefMut for Event {
    fn deref_mut(&mut self) -> &mut EventCore {
        assert!(!self.is_empty(), "ocl::Event::deref_mut(): {}", self.err_empty());
        self.0.as_mut().unwrap()
    }
}

unsafe impl ClEventPtrNew for Event {
    fn ptr_mut_ptr_new(&mut self) -> OclResult<*mut cl_h::cl_event> {
        if !self.is_empty() {
            return OclError::err("ocl::Event: Attempting to use a non-empty event as a new event
                is not allowed. Please create a new, empty, event with ocl::Event::empty().");
        }

        unsafe { 
            self.0 = Some(EventCore::null());
            Ok(self.0.as_mut().unwrap().as_ptr_mut())
        }
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
