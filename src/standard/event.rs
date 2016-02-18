//! An OpenCL event.

// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::convert::Into;
use error::Result as OclResult;
use standard::Platform;
use raw::{self, EventRaw, EventInfo, EventInfoResult};
// use util;

#[derive(Copy, Clone, Debug)]
/// An event representing a command.
pub struct Event(EventRaw);

impl Event {
	/// Creates a new `Event` from a `EventRaw`.
	///
	/// ### Safety 
	///
	/// Not meant to be called unless you know what you're doing.
	pub unsafe fn new(id_raw: EventRaw) -> Event {
		Event(id_raw)
	}

	// pub fn list(platform: Platform, device_types: Option<EventType>) -> Vec<Event> {
	// 	let list_raw = raw::get_device_ids(platform.as_raw(), device_types)
	// 		.expect("Event::list: Error retrieving device list");

	// 	unsafe { list_raw.into_iter().map(|pr| Event::new(pr) ).collect() }
	// }

	/// Returns a string containing a formatted list of event properties.
	pub fn to_string(&self) -> String {
		self.clone().into()
	}

	/// Returns the underlying `EventRaw`.
	pub fn as_raw(&self) -> EventRaw {
		self.0
	}
}

impl Into<String> for Event {
	fn into(self) -> String {
		format!("{}", self)
	}
}

impl Into<EventRaw> for Event {
	fn into(self) -> EventRaw {
		self.as_raw()
	}
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		writeln!(f, "EVENT:\n\
				CommandQueue: {}\n\
	            CommandType: {}\n\
	            ReferenceCount: {}\n\
	            CommandExecutionStatus: {}\n\
	            Context: {}\n\
			",
			raw::get_event_info(self.0, EventInfo::CommandQueue).unwrap(),
	        raw::get_event_info(self.0, EventInfo::CommandType).unwrap(),
	        raw::get_event_info(self.0, EventInfo::ReferenceCount).unwrap(),
	        raw::get_event_info(self.0, EventInfo::CommandExecutionStatus).unwrap(),
	        raw::get_event_info(self.0, EventInfo::Context).unwrap(),
		)
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
// 			{t}CommandQueue: {}\n\
//             {t}CommandType: {}\n\
//             {t}ReferenceCount: {}\n\
//             {t}CommandExecutionStatus: {}\n\
//             {t}Context: {}\n\
// 		",
// 		raw::get_event_info(event, EventInfo::CommandQueue).unwrap(),
//         raw::get_event_info(event, EventInfo::CommandType).unwrap(),
//         raw::get_event_info(event, EventInfo::ReferenceCount).unwrap(),
//         raw::get_event_info(event, EventInfo::CommandExecutionStatus).unwrap(),
//         raw::get_event_info(event, EventInfo::Context).unwrap(),
// 		t = util::TAB,
// 	);

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
// 			{t}Queued: {}\n\
// 	    	{t}Submit: {}\n\
// 	    	{t}Start: {}\n\
// 	    	{t}End: {}\n\
// 		",
// 		raw::get_event_profiling_info(event, ProfilingInfo::Queued).unwrap(),
//         raw::get_event_profiling_info(event, ProfilingInfo::Submit).unwrap(),
//         raw::get_event_profiling_info(event, ProfilingInfo::Start).unwrap(),
//         raw::get_event_profiling_info(event, ProfilingInfo::End).unwrap(),
// 		t = util::TAB,
// 	);


//     print!("\n");
// }
