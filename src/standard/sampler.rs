//! An image sampler.

use std;
use std::ops::{Deref, DerefMut};
use error::{Result as OclResult};
use core::{self, Sampler as SamplerCore, AddressingMode, FilterMode, SamplerInfo, SamplerInfoResult};
use standard::Context;

/// An image sampler.
pub struct Sampler(SamplerCore);

impl Sampler {
	/// Creates and returns a new sampler.
	///
	/// ### Enum Quick Reference
	///
	/// `addressing_mode`:
	///
	/// - AddressingMode::None
	/// - AddressingMode::ClampToEdge
	/// - AddressingMode::Clamp
	/// - AddressingMode::Repeat
	/// - AddressingMode::MirroredRepeat
	///
	/// `filter_mode`:
	///
	/// - FilterMode::Nearest
	/// - FilterMode::Linear
	///
	/// See [SDK Docs](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateSampler.html)
	/// for more information.
	///
	pub fn new(context: &Context, normalize_coords: bool, addressing_mode: AddressingMode,
            filter_mode: FilterMode) -> OclResult<Sampler> 
	{
		let sampler_core = try!(core::create_sampler(context, normalize_coords,
			addressing_mode, filter_mode));

		Ok(Sampler(sampler_core))
	}

	/// Returns various kinds of information about the sampler.
	pub fn info(&self, info_kind: SamplerInfo) -> SamplerInfoResult {
        match core::get_sampler_info(&self.0, info_kind) {
            Ok(res) => res,
            Err(err) => SamplerInfoResult::Error(Box::new(err)),
        }        
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("ReferenceCount", &self.info(SamplerInfo::ReferenceCount))
            .field("Context", &self.info(SamplerInfo::Context))
            .field("NormalizedCoords", &self.info(SamplerInfo::NormalizedCoords))
            .field("AddressingMode", &self.info(SamplerInfo::AddressingMode))
            .field("FilterMode", &self.info(SamplerInfo::FilterMode))
            .finish()
    }
}

impl std::fmt::Display for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl Deref for Sampler {
    type Target = SamplerCore;

    fn deref(&self) -> &SamplerCore {
        &self.0
    }
}

impl DerefMut for Sampler {
    fn deref_mut(&mut self) -> &mut SamplerCore {
        &mut self.0
    }
}
