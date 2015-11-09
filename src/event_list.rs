use super::{ cl_h };

pub struct EventList {
	events: Vec<cl_h::cl_event>,
}

impl EventList {
	pub fn new() -> EventList {
		EventList { events: Vec::with_capacity(16) }
	}
}
