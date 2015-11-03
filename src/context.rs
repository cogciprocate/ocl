
use formatting::MT;
use super::{ cl_h, cl_platform_id, cl_device_id, cl_context };

const DEFAULT_PLATFORM: usize = 0;

pub struct Context {
	//platforms: Vec<cl_platform_id>,
	platform: cl_platform_id,
	devices: Vec<cl_device_id>,
	context: cl_context,
}

impl Context {
	pub fn new(platform_idx: Option<usize>) -> Context {
		let platforms = super::get_platform_ids();
		if platforms.len() == 0 { panic!("\nNo OpenCL platforms found!\n"); }

		let platform = match platform_idx {
			Some(pf_idx) => platforms[pf_idx],
			None => platforms[DEFAULT_PLATFORM],
		};
		
		let devices: Vec<cl_device_id> = super::get_device_ids(platform);
		if devices.len() == 0 { panic!("\nNo OpenCL devices found!\n"); }

		println!("{}OCL::NEW(): device list: {:?}", MT, devices);

		let context: cl_context = super::create_context(&devices);

		Context {
			//platforms: platforms,
			platform: platform,
			devices: devices,
			context:  context,
		}
	}

	pub fn release_components(&mut self) {		
    	unsafe {
			cl_h::clReleaseContext(self.context);
		}
		print!("[platform]");
	}


	pub fn context(&self) -> cl_context {
		self.context
	}

	pub fn devices(&self) -> &Vec<cl_device_id> {
		&self.devices
	}

	pub fn platform(&self) -> cl_platform_id {
		self.platform
	}

	pub fn valid_device(&self, selected_idx: usize) -> cl_device_id {
		let valid_idx = selected_idx % self.devices.len();
		self.devices[valid_idx]
	}
}

