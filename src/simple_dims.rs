use super::{ ProQueue, EnvoyDims };

pub struct SimpleDims {
	d0: u32,
	d1: u32,
	d2: u32,
}

impl SimpleDims {
	pub fn new(d0: u32, d1: u32, d2: u32) -> SimpleDims {
		SimpleDims { d0: d0, d1: d1, d2: d2 }
	}
}

impl EnvoyDims for SimpleDims {
	fn padded_envoy_len(&self, pq: &ProQueue) -> u32 {
		let simple_len = self.d0 * self.d1 * self.d2;

		super::padded_len(simple_len, pq.get_max_work_group_size())

		// if len_mod == 0 {
		// 	Ok(simple_len)
		// } else {
		// 	let pad = physical_increment - len_mod;
		// 	let padded_envoy_len = simple_len + pad;
		// 	debug_assert_eq!(padded_envoy_len % phys_incr, 0);
		// 	Ok(padded_envoy_len)
		// }
	}
}
