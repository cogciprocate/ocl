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

	pub fn as_ptr(&self) -> *const cl_h::cl_event {
		self.events().as_ptr()
	}

	pub fn wait(&self) {
		if self.events.len() > 0 {
			super::wait_for_events(self);
		}
	}

	pub fn set_callback(&self) {
		if self.events.len() > 0 {
			super::set_event_callback(self.events[(self.events.len() - 1)])
		}
	}

	pub fn count(&self) -> u32 {
		self.events.len() as u32
	}

	pub fn clear(&mut self) {
		self.events.clear();
	}
}


// #[repr(C)]
// struct RustObject {
//     a: i32,
//     // other members
// }

// extern "C" fn callback(target: *mut RustObject, a: i32) {
//     println!("I'm called from C with value {0}", a);
//     unsafe {
//         // Update the value in RustObject with the value received from the callback
//         (*target).a = a;
//     }
// }

// #[link(name = "extlib")]
// extern {
//    fn register_callback(target: *mut RustObject,
//                         cb: extern fn(*mut RustObject, i32)) -> i32;
//    fn trigger_callback();
// }

// fn main() {
//     // Create the object that will be referenced in the callback
//     let mut rust_object = Box::new(RustObject { a: 5 });

//     unsafe {
//         register_callback(&mut *rust_object, callback);
//         trigger_callback();
//     }
// }
