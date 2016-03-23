//! An OpenCL platform identifier.
//!
//! Documentation copied from [https://www.khronos.org/registry/cl/sdk/1.2/doc
//! s/man/xhtml/clGetPlatformInfo.html](https://www.khronos.org/registry/cl/sd
//! k/1.2/docs/man/xhtml/clGetPlatformInfo.html)

// use std::fmt::{std::fmt::Display, std::fmt::Formatter, Result as std::fmt::Result};
use std;
use std::ops::{Deref, DerefMut};
use std::convert::Into;
use core::{self, PlatformId as PlatformIdCore, PlatformInfo, PlatformInfoResult, ClPlatformIdPtr};

#[derive(Clone, Copy, Debug)]
/// A platform identifier.
pub struct Platform(PlatformIdCore);

impl Platform {
    /// Returns a list of all platforms avaliable on the host machine.
    pub fn list() -> Vec<Platform> {
        let list_core = core::get_platform_ids()
            .expect("Platform::list: Error retrieving platform list");

        list_core.into_iter().map(|pr| Platform::new(pr) ).collect()
    }

    /// Returns the first available platform on the host machine.
    pub fn first() -> Platform {
        let list_core = core::get_platform_ids()
            .expect("Platform::default: Error retrieving platform");

        Platform::new(list_core[0].clone())
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
        platforms.into_iter().map(|p| Platform::new(p)).collect()
    }

    // /// Returns a string containing a formatted list of every platform property.
    // pub fn to_string(&self) -> String {
    //     // self.clone().into()
    //     format!("{}", self)
    // }

    /// Returns info about the platform. 
    pub fn info(&self, info_kind: PlatformInfo) -> PlatformInfoResult {
        // match core::get_platform_info(Some(self.0.clone()), info_kind) {
        //     Ok(pi) => pi,
        //     Err(err) => PlatformInfoResult::Error(Box::new(err)),
        // }
        core::get_platform_info(Some(self.0.clone()), info_kind)
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
    pub fn profile(&self) -> String {
        // match core::get_platform_info(Some(self.0.clone()), PlatformInfo::Profile) {
        //     Ok(pi) => pi.into(),
        //     Err(err) => err.into(),
        // }
        core::get_platform_info(Some(self.0.clone()), PlatformInfo::Profile).into()
    }

    /// Returns the platform driver version as a string.
    ///
    /// Returns the OpenCL version supported by the implementation. This
    /// version string has the following format:
    ///
    /// * OpenCL<space><major_version.minor_version><space><platform-specific
    ///   information>
    ///
    /// * The major_version.minor_version value returned will be 1.2.
    pub fn version(&self) -> String {
        // match core::get_platform_info(Some(self.0.clone()), PlatformInfo::Version) {
        //     Ok(pi) => pi.into(),
        //     Err(err) => err.into(),
        // }
        core::get_platform_info(Some(self.0.clone()), PlatformInfo::Version).into()
    }

    /// Returns the platform name as a string.
    pub fn name(&self) -> String {
        // match core::get_platform_info(Some(self.0.clone()), PlatformInfo::Name) {
        //     Ok(pi) => pi.into(),
        //     Err(err) => err.into(),
        // }
        core::get_platform_info(Some(self.0.clone()), PlatformInfo::Name).into()
    }

    /// Returns the platform vendor as a string.
    pub fn vendor(&self) -> String {
        // match core::get_platform_info(Some(self.0.clone()), PlatformInfo::Vendor) {
        //     Ok(pi) => pi.into(),
        //     Err(err) => err.into(),
        // }
        core::get_platform_info(Some(self.0.clone()), PlatformInfo::Vendor).into()
    }

    /// Returns the list of platform extensions as a string.
    ///
    /// Returns a space-separated list of extension names (the extension names
    /// themselves do not contain any spaces) supported by the platform.
    /// Extensions defined here must be supported by all devices associated
    /// with this platform.
    pub fn extensions(&self) -> String {
        // match core::get_platform_info(Some(self.0.clone()), PlatformInfo::Extensions) {
        //     Ok(pi) => pi.into(),
        //     Err(err) => err.into(),
        // }
        core::get_platform_info(Some(self.0.clone()), PlatformInfo::Extensions).into()
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

unsafe impl ClPlatformIdPtr for Platform {}

impl Default for Platform {
    fn default() -> Platform {
        Platform::first()
    }
}

impl Into<String> for Platform {
    fn into(self) -> String {
        format!("{}", self)
    }
}

impl Into<PlatformIdCore> for Platform {
    fn into(self) -> PlatformIdCore {
        self.0
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)
    }
}

impl AsRef<Platform> for Platform {
    fn as_ref(&self) -> &Platform {
        self
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
