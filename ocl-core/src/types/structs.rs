//! Rust implementations of various structs used by the OpenCL API.

use libc::c_void;
use std;
// use std::ptr;
// use std::slice;
use std::mem;
// use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use std::collections::HashMap;
use num::FromPrimitive;
use error::{Error as OclError, Result as OclResult};
use ffi::{self, cl_mem, cl_buffer_region, cl_context_properties, cl_platform_id};
use ::{Mem, MemObjectType, ImageChannelOrder, ImageChannelDataType, ContextProperty,
    PlatformId, OclPrm};


// Until everything can be implemented:
pub type TemporaryPlaceholderType = ();


/// Parsed OpenCL version in the layout `({major}, {minor})`.
///
/// ex.: 'OpenCL 1.2' -> `OpenclVersion(1, 2)`.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpenclVersion {
    ver: [u16; 2],
}

impl OpenclVersion {
    pub fn new(major: u16, minor: u16) -> OpenclVersion {
        OpenclVersion { ver: [major, minor] }
    }

    pub fn max(&self) -> OpenclVersion {
        OpenclVersion { ver: [u16::max_value(), u16::max_value()] }
    }

    /// Parse the string `ver` and return a dual-integer result as
    /// `OpenclVersion`.
    ///
    /// Looks for the sequence of chars, "OpenCL" (non-case-sensitive), then
    /// splits the word just after that (at '.') and parses the two results
    /// into integers (major and minor version numbers).
    pub fn from_info_str(ver: &str) -> OclResult<OpenclVersion> {
        let mut version_word_idx: Option<usize> = None;
        let mut version: Option<OpenclVersion> = None;

        for (word_idx, word) in ver.split_whitespace().enumerate() {
            if let Some(wi) = version_word_idx {
                assert!(wi == word_idx);
                let nums: Vec<_> = word.split('.').collect();

                if nums.len() == 2 {
                    let (major, minor) = (nums[0].parse::<u16>(), nums[1].parse::<u16>());

                    if major.is_ok() && minor.is_ok() {
                        version = Some(OpenclVersion::new(major.unwrap(), minor.unwrap()));
                    }
                }
                break;
            }

            for (ch_idx, ch) in word.chars().enumerate() {
                match ch_idx {
                    0 => if ch != 'O' && ch != 'o' { break; },
                    1 => if ch != 'P' && ch != 'p' { break; },
                    2 => if ch != 'E' && ch != 'e' { break; },
                    3 => if ch != 'N' && ch != 'n' { break; },
                    4 => if ch != 'C' && ch != 'c' { break; },
                    5 => if ch == 'L' || ch == 'l' {
                        version_word_idx = Some(word_idx + 1);
                        break;
                    },
                    _ => break,
                }
            }
        }

        match version {
            Some(cl_ver) => Ok(cl_ver),
            None => OclError::err_string(format!("DeviceInfoResult::as_opencl_version(): \
                Error parsing version from the string: '{}'.", ver)),
        }
    }
}

impl From<[u16; 2]> for OpenclVersion {
    fn from(ver: [u16; 2]) -> OpenclVersion {
        OpenclVersion { ver: ver }
    }
}

impl std::fmt::Display for OpenclVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}.{}", self.ver[0], self.ver[1])
    }
}


// cl_context_properties enum  Property value  Description
//
// CL_CONTEXT_PLATFORM cl_platform_id  Specifies the platform to use.
//
// CL_CONTEXT_INTEROP_USER_SYNC    cl_bool Specifies whether the user is
// responsible for synchronization between OpenCL and other APIs. Please refer
// to the specific sections in the OpenCL 1.2 extension specification that
// describe sharing with other APIs for restrictions on using this flag.
//
//    - If CL_CONTEXT_INTEROP_USER_ SYNC is not specified, a default of
//      CL_FALSE is assumed.
//
// CL_CONTEXT_D3D10_DEVICE_KHR ID3D10Device*   If the cl_khr_d3d10_sharing
// extension is enabled, specifies the ID3D10Device* to use for Direct3D 10
// interoperability. The default value is NULL.
//
// CL_GL_CONTEXT_KHR   0, OpenGL context handle    OpenGL context to
// associated the OpenCL context with (available if the cl_khr_gl_sharing
// extension is enabled)
//
// CL_EGL_DISPLAY_KHR  EGL_NO_DISPLAY, EGLDisplay handle   EGLDisplay an
// OpenGL context was created with respect to (available if the
// cl_khr_gl_sharing extension is enabled)
//
// CL_GLX_DISPLAY_KHR  None, X handle  X Display an OpenGL context was created
// with respect to (available if the cl_khr_gl_sharing extension is enabled)
//
// CL_CGL_SHAREGROUP_KHR   0, CGL share group handle   CGL share group to
// associate the OpenCL context with (available if the cl_khr_gl_sharing
// extension is enabled)
//
// CL_WGL_HDC_KHR  0, HDC handle   HDC an OpenGL context was created with
// respect to (available if the cl_khr_gl_sharing extension is enabled)
//
// CL_CONTEXT_ADAPTER_D3D9_KHR IDirect3DDevice9 *  Specifies an
// IDirect3DDevice9 to use for D3D9 interop (if the cl_khr_dx9_media_sharing
// extension is supported).
//
// CL_CONTEXT_ADAPTER_D3D9EX_KHR   IDirect3DDeviceEx*  Specifies an
// IDirect3DDevice9Ex to use for D3D9 interop (if the cl_khr_dx9_media_sharing
// extension is supported).
//
// CL_CONTEXT_ADAPTER_DXVA_KHR IDXVAHD_Device *    Specifies an IDXVAHD_Device
// to use for DXVA interop (if the cl_khr_dx9_media_sharing extension is
// supported).
//
// CL_CONTEXT_D3D11_DEVICE_KHR ID3D11Device *  Specifies the ID3D11Device * to
// use for Direct3D 11 interoperability. The default value is NULL.
//
#[derive(Clone, Debug)]
pub enum ContextPropertyValue {
    Platform(PlatformId),
    InteropUserSync(bool),
    // Not sure about this type:
    D3d10DeviceKhr(*mut ffi::cl_d3d10_device_source_khr),
    GlContextKhr(*mut c_void),
    EglDisplayKhr(ffi::CLeglDisplayKHR),
    // Not sure about this type:
    GlxDisplayKhr(*mut c_void),
    // Not sure about this type:
    CglSharegroupKhr(*mut c_void),
    // Not sure about this type:
    WglHdcKhr(*mut c_void),
    AdapterD3d9Khr(isize),
    AdapterD3d9exKhr(isize),
    AdapterDxvaKhr(isize),
    D3d11DeviceKhr(isize),
}


/// Context properties list.
///
/// [MINIMALLY TESTED]
///
/// TODO: Check for duplicate property assignments.
#[derive(Clone, Debug)]
pub struct ContextProperties {
    props: HashMap<ContextProperty, ContextPropertyValue>,
    contains_gl_context_or_sharegroup: bool,
}

impl ContextProperties {
    /// Returns an empty new list of context properties
    pub fn new() -> ContextProperties {
        ContextProperties {
            props: HashMap::with_capacity(16),
            contains_gl_context_or_sharegroup: false,
        }
    }

    /// Specifies a platform (builder-style).
    pub fn platform<P: Into<PlatformId>>(mut self, platform: P) -> ContextProperties {
        self.set_platform(platform);
        self
    }

    /// Specifies whether the user is responsible for synchronization between
    /// OpenCL and other APIs (builder-style).
    pub fn interop_user_sync(mut self, sync: bool) -> ContextProperties {
        self.set_interop_user_sync(sync);
        self
    }

    /// Specifies an OpenGL context handle (builder-style).
    pub fn gl_context(mut self, gl_ctx: *mut c_void) -> ContextProperties {
        self.set_gl_context(gl_ctx);
        self
    }

    /// Specifies a Display pointer for the GLX context (builder-style).
    pub fn glx_display(mut self, glx_disp: *mut c_void) -> ContextProperties {
        self.set_glx_display(glx_disp);
        self
    }

    /// Specifies a Display pointer for the WGL HDC (builder-style).
    pub fn wgl_hdc(mut self, wgl_hdc: *mut c_void) -> ContextProperties {
        self.set_wgl_hdc(wgl_hdc);
        self
    }

    /// Specifies an OpenGL context CGL share group to associate the OpenCL
    /// context with (builder-style).
    pub fn cgl_sharegroup(mut self, gl_sharegroup: *mut c_void) -> ContextProperties {
        self.set_cgl_sharegroup(gl_sharegroup);
        self
    }

    /// Specifies a pointer for the EGL display (builder-style).
    pub fn egl_display(mut self, egl_disp: *mut c_void) -> ContextProperties {
        self.set_egl_display(egl_disp);
        self
    }

    /// Pushes a `ContextPropertyValue` onto this list of properties
    /// (builder-style).
    pub fn property_value(mut self, prop: ContextPropertyValue) -> ContextProperties {
        self.set_property_value(prop);
        self
    }

    /// Specifies a platform.
    pub fn set_platform<P: Into<PlatformId>>(&mut self, platform: P) {
        self.props.insert(ContextProperty::Platform, ContextPropertyValue::Platform(platform.into()));
    }

    /// Specifies whether the user is responsible for synchronization between
    /// OpenCL and other APIs.
    pub fn set_interop_user_sync(&mut self, sync: bool) {
        self.props.insert(ContextProperty::InteropUserSync, ContextPropertyValue::InteropUserSync(sync));
    }

    /// Specifies an OpenGL context handle.
    pub fn set_gl_context(&mut self, gl_ctx: *mut c_void) {
        self.props.insert(ContextProperty::GlContextKhr, ContextPropertyValue::GlContextKhr(gl_ctx));
        self.contains_gl_context_or_sharegroup = true;
    }

    /// Specifies a Display pointer for the GLX context.
    pub fn set_glx_display(&mut self, glx_disp: *mut c_void) {
        self.props.insert(ContextProperty::GlxDisplayKhr, ContextPropertyValue::GlxDisplayKhr(glx_disp));
        self.contains_gl_context_or_sharegroup = true;
    }

    /// Specifies a Display pointer for the WGL HDC.
    pub fn set_wgl_hdc(&mut self, wgl_hdc: *mut c_void) {
        self.props.insert(ContextProperty::WglHdcKhr, ContextPropertyValue::WglHdcKhr(wgl_hdc));
        self.contains_gl_context_or_sharegroup = true;
    }

    /// Specifies an OpenGL context CGL share group to associate the OpenCL
    /// context with.
    pub fn set_cgl_sharegroup(&mut self, gl_sharegroup: *mut c_void) {
        self.props.insert(ContextProperty::CglSharegroupKhr, ContextPropertyValue::CglSharegroupKhr(gl_sharegroup));
        self.contains_gl_context_or_sharegroup = true;
    }

    /// Specifies a pointer for the EGL display.
    pub fn set_egl_display(&mut self, egl_disp: *mut c_void) {
        self.props.insert(ContextProperty::EglDisplayKhr, ContextPropertyValue::EglDisplayKhr(egl_disp));
        self.contains_gl_context_or_sharegroup = true;
    }

    /// Pushes a `ContextPropertyValue` onto this list of properties.
    pub fn set_property_value(&mut self, prop: ContextPropertyValue) {
        match prop {
            ContextPropertyValue::Platform(val) => {
                self.props.insert(ContextProperty::Platform, ContextPropertyValue::Platform(val));
            },
            ContextPropertyValue::InteropUserSync(val) => {
                self.props.insert(ContextProperty::InteropUserSync,
                    ContextPropertyValue::InteropUserSync(val));
            },
            ContextPropertyValue::GlContextKhr(val) => {
                self.props.insert(ContextProperty::GlContextKhr,
                    ContextPropertyValue::GlContextKhr(val));
                self.contains_gl_context_or_sharegroup = true;
            },
            ContextPropertyValue::GlxDisplayKhr(val) => {
                self.props.insert(ContextProperty::GlxDisplayKhr,
                    ContextPropertyValue::GlxDisplayKhr(val));
                self.contains_gl_context_or_sharegroup = true;
            },
            ContextPropertyValue::WglHdcKhr(val) => {
                self.props.insert(ContextProperty::WglHdcKhr,
                    ContextPropertyValue::WglHdcKhr(val));
                self.contains_gl_context_or_sharegroup = true;
            },
            ContextPropertyValue::CglSharegroupKhr(val) => {
                self.props.insert(ContextProperty::CglSharegroupKhr,
                    ContextPropertyValue::CglSharegroupKhr(val));
                self.contains_gl_context_or_sharegroup = true;
            },
            ContextPropertyValue::EglDisplayKhr(val) => {
                self.props.insert(ContextProperty::EglDisplayKhr,
                    ContextPropertyValue::EglDisplayKhr(val));
                self.contains_gl_context_or_sharegroup = true;
            },
            _ => panic!("'{:?}' is not yet a supported variant.", prop),
        }
    }

    /// Returns a platform id or none.
    pub fn get_platform(&self) -> Option<PlatformId> {
        match self.props.get(&ContextProperty::Platform) {
            Some(prop_val) => {
                if let ContextPropertyValue::Platform(ref plat) = *prop_val {
                    Some(plat.clone())
                } else {
                    panic!("Internal error returning platform.");
                }
            },
            None => None
        }
    }

    /// Returns a cgl_sharegroup id or none.
    #[deprecated(since="0.6.0", note="Use ::contains_gl_context_or_sharegroup.")]
    pub fn get_cgl_sharegroup(&self) -> Option<*mut c_void> {
        match self.props.get(&ContextProperty::CglSharegroupKhr) {
            Some(prop_val) => {
                if let ContextPropertyValue::CglSharegroupKhr(ref cgl_sharegroup) = *prop_val {
                    Some(cgl_sharegroup.clone())
                } else {
                    panic!("Internal error returning cgl_sharegroup.");
                }
            },
            None => None
        }
    }

    /// Returns true if this set of context properties specifies any OpenGL
    /// context or sharegroup to associate with.
    pub fn contains_gl_context_or_sharegroup(&self) -> bool {
        self.contains_gl_context_or_sharegroup
    }

    /// Converts this list into a packed-word representation as specified
    /// [here](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateContext.html).
    ///
    // [NOTE]: Meant to replace `::to_bytes`.
    //
    // Return type is `Vec<cl_context_properties>` => `Vec<isize>`
    //
    // [FIXME]: Change return type to `Vec<(cl_context_properties, isize)>`
    pub fn to_raw(&self) -> Vec<isize> {
        let mut props_raw = Vec::with_capacity(32);

        // For each property ...
        for (key, val) in self.props.iter() {
            // convert both the kind of property (a u32 originally) and
            // the value (variable type/size) to an isize:
            match *val {
                ContextPropertyValue::Platform(ref platform_id_core) => {
                    props_raw.push(key.clone() as isize);
                    props_raw.push(platform_id_core.as_ptr() as isize);
                },
                ContextPropertyValue::InteropUserSync(sync) => {
                    props_raw.push(key.clone() as isize);
                    props_raw.push(sync as isize);
                },
                ContextPropertyValue::GlContextKhr(sync) => {
                    props_raw.push(key.clone() as isize);
                    props_raw.push(sync as isize);
                },
                ContextPropertyValue::GlxDisplayKhr(sync) => {
                    props_raw.push(key.clone() as isize);
                    props_raw.push(sync as isize);
                },
                ContextPropertyValue::WglHdcKhr(sync) => {
                    props_raw.push(key.clone() as isize);
                    props_raw.push(sync as isize);
                },
                ContextPropertyValue::CglSharegroupKhr(sync) => {
                    props_raw.push(key.clone() as isize);
                    props_raw.push(sync as isize);
                },
                ContextPropertyValue::EglDisplayKhr(sync) => {
                    props_raw.push(key.clone() as isize);
                    props_raw.push(sync as isize);
                },
                _ => panic!("'{:?}' is not yet a supported variant.", key),
            };
        }

        // Add a terminating 0:
        props_raw.push(0);

        props_raw.shrink_to_fit();
        props_raw
    }

    /// Returns a single context property value.
    pub unsafe fn extract_property_from_raw(property: ContextProperty,
            raw_context_properties: &[isize]) -> Option<ContextPropertyValue>
    {
        // REMEMBER: It's null terminated;

        // The raw properties **should** be `(isize, isize)` pairs + isize (null) terminator.
        assert!(raw_context_properties.len() % 2 == 1);
        assert!(*raw_context_properties.last().unwrap() == 0);

        let pair_count = raw_context_properties.len() / 2;

        match property {
            ContextProperty::Platform => {
                for pair_idx in 0..pair_count {
                    let idz = pair_idx * 2;
                    let key_raw = *raw_context_properties.get_unchecked(idz);
                    let val_raw = *raw_context_properties.get_unchecked(idz + 1);

                    if key_raw == property as cl_context_properties {
                        return Some(ContextPropertyValue::Platform(
                            PlatformId::from_raw(val_raw as cl_platform_id)));
                    }
                }
            },
            _ => unimplemented!(),
        }

        None
    }


    /// Converts raw stuff into other stuff.
    ///
    ///
    #[allow(unused_variables, unused_mut)]
    pub unsafe fn from_raw(raw_context_properties: &[isize]) -> OclResult<ContextProperties> {
        // The raw properties **should** be `(isize, isize)` pairs + isize (null) terminator.
        assert!(mem::size_of::<cl_context_properties>() == mem::size_of::<isize>());
        assert!(raw_context_properties.len() % 2 == 1);
        assert!(*raw_context_properties.last().unwrap() == 0);

        let pair_count = raw_context_properties.len() / 2;
        let mut context_props = ContextProperties {
            props: HashMap::with_capacity(pair_count),
            contains_gl_context_or_sharegroup: false,
        };

        for pair_idx in 0..pair_count {
            let idz = pair_idx * 2;
            let key_raw = *raw_context_properties.get_unchecked(idz);
            let val_raw = *raw_context_properties.get_unchecked(idz + 1);

            let key = ContextProperty::from_isize(key_raw).ok_or(OclError::from(
                format!("ContextProperties::from_raw: Unable to convert '{}' using \
                    'ContextProperty::from_isize'.", key_raw)))?;

            match key {
                    ContextProperty::Platform => {
                        context_props.props.insert(ContextProperty::Platform,
                            ContextPropertyValue::Platform(PlatformId::from_raw(val_raw as cl_platform_id))
                        );
                    },
                    ContextProperty::InteropUserSync => {
                        context_props.props.insert(ContextProperty::InteropUserSync,
                            ContextPropertyValue::InteropUserSync(val_raw > 0),
                        );
                    },
                    ContextProperty::D3d10DeviceKhr => {
                        context_props.props.insert(ContextProperty::D3d10DeviceKhr,
                            ContextPropertyValue::D3d10DeviceKhr(val_raw as *mut ffi::cl_d3d10_device_source_khr),
                        );
                    },
                    ContextProperty::GlContextKhr => {
                        context_props.props.insert(ContextProperty::GlContextKhr,
                            ContextPropertyValue::GlContextKhr(val_raw as *mut c_void),
                        );
                        context_props.contains_gl_context_or_sharegroup = true;
                    },
                    ContextProperty::EglDisplayKhr => {
                        context_props.props.insert(ContextProperty::EglDisplayKhr,
                            ContextPropertyValue::EglDisplayKhr(val_raw as ffi::CLeglDisplayKHR),
                        );
                        context_props.contains_gl_context_or_sharegroup = true;
                    },
                    ContextProperty::GlxDisplayKhr => {
                        context_props.props.insert(ContextProperty::GlxDisplayKhr,
                            ContextPropertyValue::GlxDisplayKhr(val_raw as *mut c_void),
                        );
                        context_props.contains_gl_context_or_sharegroup = true;
                    },
                    ContextProperty::CglSharegroupKhr => {
                        context_props.props.insert(ContextProperty::CglSharegroupKhr,
                            ContextPropertyValue::CglSharegroupKhr(val_raw as *mut c_void),
                        );
                        context_props.contains_gl_context_or_sharegroup = true;
                    },
                    ContextProperty::WglHdcKhr => {
                        context_props.props.insert(ContextProperty::WglHdcKhr,
                            ContextPropertyValue::WglHdcKhr(val_raw as *mut c_void),
                        );
                        context_props.contains_gl_context_or_sharegroup = true;
                    },
                    ContextProperty::AdapterD3d9Khr => {
                        context_props.props.insert(ContextProperty::AdapterD3d9Khr,
                            ContextPropertyValue::AdapterD3d9Khr(val_raw),
                        );
                    },
                    ContextProperty::AdapterD3d9exKhr => {
                        context_props.props.insert(ContextProperty::AdapterD3d9exKhr,
                            ContextPropertyValue::AdapterD3d9exKhr(val_raw),
                        );
                    },
                    ContextProperty::AdapterDxvaKhr => {
                        context_props.props.insert(ContextProperty::AdapterDxvaKhr,
                            ContextPropertyValue::AdapterDxvaKhr(val_raw),
                        );
                    },
                    ContextProperty::D3d11DeviceKhr => {
                        context_props.props.insert(ContextProperty::D3d11DeviceKhr,
                            ContextPropertyValue::D3d11DeviceKhr(val_raw),
                        );
                    },
            }
        }

        Ok(context_props)
    }
}

// impl Into<Vec<isize>> for ContextProperties {
//     fn into(self) -> Vec<isize> {
//         self.to_raw()
//     }
// }

impl From<ContextProperties> for Vec<isize> {
    fn from(cp: ContextProperties) -> Vec<isize> {
        cp.to_raw()
    }
}



/// Defines a buffer region for creating a sub-buffer.
///
/// ### Info (from [SDK](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateSubBuffer.html))
///
/// (origin, size) defines the offset and size in bytes in buffer.
///
/// If buffer is created with CL_MEM_USE_HOST_PTR, the host_ptr associated with
/// the buffer object returned is host_ptr + origin.
///
/// The buffer object returned references the data store allocated for buffer and
/// points to a specific region given by (origin, size) in this data store.
///
/// CL_INVALID_VALUE is returned in errcode_ret if the region specified by
/// (origin, size) is out of bounds in buffer.
///
/// CL_INVALID_BUFFER_SIZE if size is 0.
///
/// CL_MISALIGNED_SUB_BUFFER_OFFSET is returned in errcode_ret if there are no
/// devices in context associated with buffer for which the origin value is
/// aligned to the CL_DEVICE_MEM_BASE_ADDR_ALIGN value.
///
pub struct BufferRegion<T> {
    origin: usize,
    len: usize,
    _data: PhantomData<T>,
}

impl<T: OclPrm> BufferRegion<T> {
    pub fn new(origin: usize, len: usize) -> BufferRegion<T> {
        BufferRegion {
            origin: origin,
            len: len,
            _data: PhantomData,
        }
    }

    pub fn to_bytes(&self) -> cl_buffer_region {
        cl_buffer_region {
            origin: self.origin * mem::size_of::<T>(),
            size: self.len * mem::size_of::<T>(),
        }
    }

    pub fn from_bytes(ffi_struct: cl_buffer_region) -> BufferRegion<T> {
        assert!(ffi_struct.origin % mem::size_of::<T>() == 0);
        assert!(ffi_struct.size % mem::size_of::<T>() == 0);

        BufferRegion::new(ffi_struct.origin / mem::size_of::<T>(),
            ffi_struct.size / mem::size_of::<T>())
    }
}


pub enum ImageFormatParseError {
    UnknownImageChannelOrder(ffi::cl_channel_order),
    UnknownImageChannelDataType(ffi::cl_channel_type),
}

impl ::std::fmt::Debug for ImageFormatParseError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            ImageFormatParseError::UnknownImageChannelOrder(ord) => {
                write!(f, "unknown image channel ordering: '{}'", ord)
            },
            ImageFormatParseError::UnknownImageChannelDataType(dt) => {
                write!(f, "unknown image channel data type: '{}'", dt)
            },
        }
    }
}

impl ::std::fmt::Display for ImageFormatParseError {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ::std::error::Error for ImageFormatParseError {
    fn description(&self) -> &str {
        match *self {
            ImageFormatParseError::UnknownImageChannelOrder(_) => "unknown image channel ordering",
            ImageFormatParseError::UnknownImageChannelDataType(_) => "unknown image channel data type",
        }
    }
}

pub type ImageFormatParseResult = Result<ImageFormat, ImageFormatParseError>;




/// Image format properties used by `Image`.
///
/// A structure that describes format properties of the image to be allocated. (from SDK)
///
/// # Examples (from SDK)
///
/// To specify a normalized unsigned 8-bit / channel RGBA image:
///    image_channel_order = CL_RGBA
///    image_channel_data_type = CL_UNORM_INT8
///
/// image_channel_data_type values of CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 and CL_UNORM_INT_101010 are special cases of packed image formats where the channels of each element are packed into a single unsigned short or unsigned int. For these special packed image formats, the channels are normally packed with the first channel in the most significant bits of the bitfield, and successive channels occupying progressively less significant locations. For CL_UNORM_SHORT_565, R is in bits 15:11, G is in bits 10:5 and B is in bits 4:0. For CL_UNORM_SHORT_555, bit 15 is undefined, R is in bits 14:10, G in bits 9:5 and B in bits 4:0. For CL_UNORM_INT_101010, bits 31:30 are undefined, R is in bits 29:20, G in bits 19:10 and B in bits 9:0.
/// OpenCL implementations must maintain the minimum precision specified by the number of bits in image_channel_data_type. If the image format specified by image_channel_order, and image_channel_data_type cannot be supported by the OpenCL implementation, then the call to clCreateImage will return a NULL memory object.
///
#[derive(Debug, Clone)]
pub struct ImageFormat {
    pub channel_order: ImageChannelOrder,
    pub channel_data_type: ImageChannelDataType,
}

impl ImageFormat {
    pub fn new(order: ImageChannelOrder, data_type: ImageChannelDataType) -> ImageFormat {
        ImageFormat {
            channel_order: order,
            channel_data_type: data_type,
        }
    }

    pub fn new_rgba() -> ImageFormat {
        ImageFormat {
            channel_order: ImageChannelOrder::Rgba,
            channel_data_type: ImageChannelDataType::SnormInt8,
        }
    }

    pub fn from_raw(fmt_raw: ffi::cl_image_format) -> ImageFormatParseResult {
        // Ok(ImageFormat {
        //     channel_order: try!(ImageChannelOrder::from_u32(raw.image_channel_order)
        //         .ok_or(OclError::from("Error converting to 'ImageChannelOrder'."))),
        //     channel_data_type: try!(ImageChannelDataType::from_u32(raw.image_channel_data_type)
        //         .ok_or(OclError::from("Error converting to 'ImageChannelDataType'."))),
        // })

        let channel_order = match ImageChannelOrder::from_u32(fmt_raw.image_channel_order) {
            Some(ord) => ord,
            None => return Err(ImageFormatParseError::UnknownImageChannelOrder(
                fmt_raw.image_channel_order)),
        };

        let channel_data_type = match ImageChannelDataType::from_u32(fmt_raw.image_channel_data_type) {
            Some(dt) => dt,
            None => return Err(ImageFormatParseError::UnknownImageChannelDataType(
                fmt_raw.image_channel_data_type)),
        };

        Ok(ImageFormat { channel_order, channel_data_type })
    }

    pub fn list_from_raw(list_raw: Vec<ffi::cl_image_format>)
            -> Vec<ImageFormatParseResult>
    {
        // let mut result_list = Vec::with_capacity(list_raw.len());

        // for fmt_raw in list_raw.into_iter() {
        //     result_list.push(ImageFormat::from_raw(fmt_raw));
        // }

        // result_list

        list_raw.into_iter().map(|fmt_raw| ImageFormat::from_raw(fmt_raw)).collect()
    }

    pub fn to_raw(&self) -> ffi::cl_image_format {
        ffi::cl_image_format {
            image_channel_order: self.channel_order as ffi::cl_channel_order,
            image_channel_data_type: self.channel_data_type as ffi::cl_channel_type,
        }
    }

    pub fn new_raw() -> ffi::cl_image_format {
        ffi::cl_image_format {
            image_channel_order: 0 as ffi::cl_channel_order,
            image_channel_data_type: 0 as ffi::cl_channel_type,
        }
    }

    /// Returns the size in bytes of a pixel using the format specified by this
    /// `ImageFormat`.
    ///
    /// TODO: Add a special case for Depth & DepthStencil
    /// (https://www.khronos.org/registry/cl/sdk/2.0/docs/man/xhtml/cl_khr_gl_depth_images.html).
    ///
    /// TODO: Validate combinations.
    /// TODO: Use `core::get_image_info` to check these with a test.
    ///
    pub fn pixel_bytes(&self) -> usize {
        let channel_count = match self.channel_order {
            ImageChannelOrder::R => 1,
            ImageChannelOrder::A => 1,
            ImageChannelOrder::Rg => 2,
            ImageChannelOrder::Ra => 2,
            // This format can only be used if channel data type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010:
            ImageChannelOrder::Rgb => 1,
            ImageChannelOrder::Rgba => 4,
            // This format can only be used if channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8:
            ImageChannelOrder::Bgra => 4,
            // This format can only be used if channel data type = CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8:
            ImageChannelOrder::Argb => 4,
            // This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT:
            ImageChannelOrder::Intensity => 4,
            // This format can only be used if channel data type = CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT:
            ImageChannelOrder::Luminance => 4,
            ImageChannelOrder::Rx => 2,
            ImageChannelOrder::Rgx => 4,
            // This format can only be used if channel data type = CL_UNORM_SHORT_565, CL_UNORM_SHORT_555 or CL_UNORM_INT101010:
            ImageChannelOrder::Rgbx => 4,
            // Depth => 1,
            // DepthStencil => 1,
            _ => 0,
        };

        let channel_size = match self.channel_data_type {
            // Each channel component is a normalized signed 8-bit integer value:
            ImageChannelDataType::SnormInt8 => 1,
            // Each channel component is a normalized signed 16-bit integer value:
            ImageChannelDataType::SnormInt16 => 2,
            // Each channel component is a normalized unsigned 8-bit integer value:
            ImageChannelDataType::UnormInt8 => 1,
            // Each channel component is a normalized unsigned 16-bit integer value:
            ImageChannelDataType::UnormInt16 => 2,
            // Represents a normalized 5-6-5 3-channel RGB image. The channel order must be CL_RGB or CL_RGBx:
            ImageChannelDataType::UnormShort565 => 2,
            // Represents a normalized x-5-5-5 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx:
            ImageChannelDataType::UnormShort555 => 2,
            // Represents a normalized x-10-10-10 4-channel xRGB image. The channel order must be CL_RGB or CL_RGBx:
            ImageChannelDataType::UnormInt101010 => 4,
            // Each channel component is an unnormalized signed 8-bit integer value:
            ImageChannelDataType::SignedInt8 => 1,
            // Each channel component is an unnormalized signed 16-bit integer value:
            ImageChannelDataType::SignedInt16 => 2,
            // Each channel component is an unnormalized signed 32-bit integer value:
            ImageChannelDataType::SignedInt32 => 4,
            // Each channel component is an unnormalized unsigned 8-bit integer value:
            ImageChannelDataType::UnsignedInt8 => 1,
            // Each channel component is an unnormalized unsigned 16-bit integer value:
            ImageChannelDataType::UnsignedInt16 => 2,
            // Each channel component is an unnormalized unsigned 32-bit integer value:
            ImageChannelDataType::UnsignedInt32 => 4,
            // Each channel component is a 16-bit half-float value:
            ImageChannelDataType::HalfFloat => 2,
            // Each channel component is a single precision floating-point value:
            ImageChannelDataType::Float => 4,
            // Each channel component is a normalized unsigned 24-bit integer value:
            // UnormInt24 => 3,
            _ => 0
        };

        channel_count * channel_size
    }
}


/// An image descriptor use in the creation of `Image`.
///
/// image_type
/// Describes the image type and must be either CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE1D_BUFFER, CL_MEM_OBJECT_IMAGE1D_ARRAY, CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE2D_ARRAY, or CL_MEM_OBJECT_IMAGE3D.
///
/// image_width
/// The width of the image in pixels. For a 2D image and image array, the image width must be ≤ CL_DEVICE_IMAGE2D_MAX_WIDTH. For a 3D image, the image width must be ≤ CL_DEVICE_IMAGE3D_MAX_WIDTH. For a 1D image buffer, the image width must be ≤ CL_DEVICE_IMAGE_MAX_BUFFER_SIZE. For a 1D image and 1D image array, the image width must be ≤ CL_DEVICE_IMAGE2D_MAX_WIDTH.
///
/// image_height
/// The height of the image in pixels. This is only used if the image is a 2D, 3D or 2D image array. For a 2D image or image array, the image height must be ≤ CL_DEVICE_IMAGE2D_MAX_HEIGHT. For a 3D image, the image height must be ≤ CL_DEVICE_IMAGE3D_MAX_HEIGHT.
///
/// image_depth
/// The depth of the image in pixels. This is only used if the image is a 3D image and must be a value ≥ 1 and ≤ CL_DEVICE_IMAGE3D_MAX_DEPTH.
///
/// image_array_size
/// The number of images in the image array. This is only used if the image is a 1D or 2D image array. The values for image_array_size, if specified, must be a value ≥ 1 and ≤ CL_DEVICE_IMAGE_MAX_ARRAY_SIZE.
///
/// Note that reading and writing 2D image arrays from a kernel with image_array_size = 1 may be lower performance than 2D images.
///
/// image_row_pitch
/// The scan-line pitch in bytes. This must be 0 if host_ptr is NULL and can be either 0 or ≥ image_width * size of element in bytes if host_ptr is not NULL. If host_ptr is not NULL and image_row_pitch = 0, image_row_pitch is calculated as image_width * size of element in bytes. If image_row_pitch is not 0, it must be a multiple of the image element size in bytes.
///
/// image_slice_pitch
/// The size in bytes of each 2D slice in the 3D image or the size in bytes of each image in a 1D or 2D image array. This must be 0 if host_ptr is NULL. If host_ptr is not NULL, image_slice_pitch can be either 0 or ≥ image_row_pitch * image_height for a 2D image array or 3D image and can be either 0 or ≥ image_row_pitch for a 1D image array. If host_ptr is not NULL and image_slice_pitch = 0, image_slice_pitch is calculated as image_row_pitch * image_height for a 2D image array or 3D image and image_row_pitch for a 1D image array. If image_slice_pitch is not 0, it must be a multiple of the image_row_pitch.
///
/// num_mip_level, num_samples
/// Must be 0.
///
/// buffer
/// Refers to a valid buffer memory object if image_type is CL_MEM_OBJECT_IMAGE1D_BUFFER. Otherwise it must be NULL. For a 1D image buffer object, the image pixels are taken from the buffer object's data store. When the contents of a buffer object's data store are modified, those changes are reflected in the contents of the 1D image buffer object and vice-versa at corresponding sychronization points. The image_width * size of element in bytes must be ≤ size of buffer object data store.
///
/// Note
/// Concurrent reading from, writing to and copying between both a buffer object and 1D image buffer object associated with the buffer object is undefined. Only reading from both a buffer object and 1D image buffer object associated with the buffer object is defined.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ImageDescriptor {
    pub image_type: MemObjectType,
    pub image_width: usize,
    pub image_height: usize,
    pub image_depth: usize,
    pub image_array_size: usize,
    pub image_row_pitch: usize,
    pub image_slice_pitch: usize,
    num_mip_levels: u32,
    num_samples: u32,
    pub buffer: Option<Mem>,
}

impl ImageDescriptor {
    pub fn new(image_type: MemObjectType, width: usize, height: usize, depth: usize,
                array_size: usize, row_pitch: usize, slc_pitch: usize, buffer: Option<Mem>,
                ) -> ImageDescriptor {
        ImageDescriptor {
            image_type: image_type,
            image_width: width,
            image_height: height,
            image_depth: depth,
            image_array_size: array_size,
            image_row_pitch: row_pitch,
            image_slice_pitch: slc_pitch,
            num_mip_levels: 0,
            num_samples: 0,
            buffer: buffer,
        }
    }

    pub fn to_raw(&self) -> ffi::cl_image_desc {
        ffi::cl_image_desc {
            image_type: self.image_type as u32,
            image_width: self.image_width,
            image_height: self.image_height,
            image_depth: self.image_depth,
            image_array_size: self.image_array_size,
            image_row_pitch: self.image_row_pitch,
            image_slice_pitch: self.image_slice_pitch,
            num_mip_levels: self.num_mip_levels,
            num_samples: self.num_mip_levels,
            buffer: match self.buffer {
                        Some(ref b) => b.as_ptr(),
                        None => 0 as cl_mem,
                    },
        }
    }
}

