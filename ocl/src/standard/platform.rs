//! An `OpenCL` platform identifier.
//!
//! Documentation copied from [https://www.khronos.org/registry/cl/sdk/1.2/doc
//! s/man/xhtml/clGetPlatformInfo.html](https://www.khronos.org/registry/cl/sd
//! k/1.2/docs/man/xhtml/clGetPlatformInfo.html)

use std;
use std::ops::{Deref, DerefMut};
use std::str::SplitWhitespace;
use ffi::cl_platform_id;
use core::{self, PlatformId as PlatformIdCore, PlatformInfo, PlatformInfoResult, ClPlatformIdPtr};
use error::{Error as OclError, Result as OclResult};


#[derive(Debug, Fail)]
pub enum PlatformError {
    #[fail(display = "No platforms found.")]
    NoPlatforms,
}


/// Extensions of a platform.
#[derive(Debug, Clone)]
pub struct Extensions {
    inner: String,
}

impl Extensions {
    /// Iterate over platform extensions, split at whitespace.
    pub fn iter(&self) -> SplitWhitespace {
        self.inner.split_whitespace()
    }

    pub fn as_str(&self) -> &str {
        &self.inner
    }
}


/// A platform identifier.
///
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Platform(PlatformIdCore);

impl Platform {
    /// Returns a list of all platforms avaliable on the host machine.
    pub fn list() -> Vec<Platform> {
        let list_core = core::get_platform_ids()
            .expect("Platform::list: Error retrieving platform list");

        list_core.into_iter().map(Platform::new).collect()
    }

    /// Returns the first available platform.
    ///
    /// This method differs from `Platform::default()` in two ways. First, it
    /// ignores the `OCL_DEFAULT_PLATFORM_IDX` environment variable
    /// (`Platform::default` always respects it). Second, this function will
    /// not panic if no platforms are available but will instead return an
    /// error.
    pub fn first() -> OclResult<Platform> {
        core::get_platform_ids()?
            .first()
            .map(|&p| Platform::new(p))
            .ok_or(PlatformError::NoPlatforms.into())
    }

    /// Creates a new `Platform` from a `PlatformIdCore`.
    ///
    /// ## Safety
    ///
    /// Not meant to be called unless you know what you're doing.
    ///
    /// Use list to get a list of platforms.
    pub fn new(id_core: PlatformIdCore) -> Platform {
        Platform(id_core)
    }

    /// Returns a list of `Platform`s from a list of `PlatformIdCore`s
    pub fn list_from_core(platforms: Vec<PlatformIdCore>) -> Vec<Platform> {
        platforms.into_iter().map(Platform::new).collect()
    }

    /// Returns info about the platform.
    pub fn info(&self, info_kind: PlatformInfo) -> OclResult<PlatformInfoResult> {
        core::get_platform_info(&self.0, info_kind).map_err(OclError::from)
    }

    /// Returns the platform profile as a string.
    ///
    /// Returns the profile name supported by the implementation. The profile
    /// name returned can be one of the following strings:
    ///
    /// * FULL_PROFILE - if the implementation supports the OpenCL
    ///   specification (functionality defined as part of the core
    ///   specification and does not require any extensions to be supported).
    ///
    /// * EMBEDDED_PROFILE - if the implementation supports the OpenCL
    ///   embedded profile. The embedded profile is defined to be a subset for
    ///   each version of OpenCL.
    ///
    pub fn profile(&self) -> OclResult<String> {
        core::get_platform_info(&self.0, PlatformInfo::Profile)
            .map(|r| r.into()).map_err(OclError::from)
    }

    /// Returns the platform driver version as a string.
    ///
    /// Returns the OpenCL version supported by the implementation. This
    /// version string has the following format:
    ///
    /// * OpenCL<space><major_version.minor_version><space><platform-specific
    ///   information>
    ///
    /// * The major_version.minor_version value returned will be '1.2'.
    ///
    /// * TODO: Convert this to new version system returning an `OpenclVersion`.
    pub fn version(&self) -> OclResult<String> {
        core::get_platform_info(&self.0, PlatformInfo::Version)
            .map(|r| r.into()).map_err(OclError::from)
    }

    /// Returns the platform name as a string.
    pub fn name(&self) -> OclResult<String> {
        core::get_platform_info(&self.0, PlatformInfo::Name)
            .map(|r| r.into()).map_err(OclError::from)
    }

    /// Returns the platform vendor as a string.
    pub fn vendor(&self) -> OclResult<String> {
        core::get_platform_info(&self.0, PlatformInfo::Vendor)
            .map(|r| r.into()).map_err(OclError::from)
    }

    /// Returns the list of platform extensions.
    ///
    /// Extensions defined here must be supported by all devices associated
    /// with this platform.
    pub fn extensions(&self) -> OclResult<Extensions> {
        let extensions = core::get_platform_info(&self.0, PlatformInfo::Extensions);
        extensions.map(|e| Extensions { inner: e.into() }).map_err(OclError::from)
    }

    /// Returns a reference to the underlying `PlatformIdCore`.
    pub fn as_core(&self) -> &PlatformIdCore {
        &self.0
    }

    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Platform")
            .field("Profile", &self.info(PlatformInfo::Profile))
            .field("Version", &self.info(PlatformInfo::Version))
            .field("Name", &self.info(PlatformInfo::Name))
            .field("Vendor", &self.info(PlatformInfo::Vendor))
            .field("Extensions", &self.info(PlatformInfo::Extensions))
            .finish()
    }
}

unsafe impl ClPlatformIdPtr for Platform {
    fn as_ptr(&self) -> cl_platform_id {
        self.0.as_ptr()
    }
}
// unsafe impl<'a> ClPlatformIdPtr for &'a Platform {}

impl Default for Platform {
    /// Returns the first (0th) platform available, or the platform specified
    /// by the `OCL_DEFAULT_PLATFORM_IDX` environment variable if it is set.
    ///
    /// ### Panics
    ///
    /// Panics upon any OpenCL API error.
    ///
    fn default() -> Platform {
        let dflt_plat_core = core::default_platform().expect("Platform::default()");
        Platform::new(dflt_plat_core)
    }
}

impl From<PlatformIdCore> for Platform {
    fn from(core: PlatformIdCore) -> Platform {
        Platform(core)
    }
}

impl From<Platform> for String {
    fn from(p: Platform) -> String {
        format!("{}", p)
    }
}

impl From<Platform> for PlatformIdCore {
    fn from(p: Platform) -> PlatformIdCore {
        p.0
    }
}

impl<'a> From<&'a Platform> for PlatformIdCore {
    fn from(p: &Platform) -> PlatformIdCore {
        p.0.clone()
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}


impl Deref for Platform {
    type Target = PlatformIdCore;

    fn deref(&self) -> &PlatformIdCore {
        &self.0
    }
}

impl DerefMut for Platform {
    fn deref_mut(&mut self) -> &mut PlatformIdCore {
        &mut self.0
    }
}
