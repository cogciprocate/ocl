use std::ptr;

use super::{ cl_h };

pub struct EventList {
	events: Vec<cl_h::cl_event>,
}

impl EventList {
	pub fn new() -> EventList {
		EventList { events: Vec::with_capacity(16) }
	}

	pub fn allot(&mut self) -> &mut [cl_h::cl_event] {
		self.events.push(ptr::null_mut());
		let len = self.events.len();
		&mut self.events[(len - 1)..len]
	}

	pub fn events(&self) -> &[cl_h::cl_event] {
		&self.events[..]
	}

	pub fn count(&self) -> u32 {
		self.events.len() as u32
	}

	pub fn clear(&mut self) {
		self.events.clear();
	}
}
