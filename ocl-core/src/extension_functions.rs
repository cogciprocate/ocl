#![allow(non_snake_case)]

use crate::{
    get_extension_function_address_for_platform, get_platform_info, Error, PlatformId,
    PlatformInfo, PlatformInfoResult, Result,
};
use cl_sys::*;
use std::ffi::c_void;
use std::mem::transmute;

#[derive(Default, Clone)]
pub struct ExtensionFunctions {
    // OpenGL
    pub clGetGLContextInfoKHR: Option<clGetGLContextInfoKHR_fn>,

    // D3D11
    pub clGetDeviceIDsFromD3D11: Option<clGetDeviceIDsFromD3D11_fn>,
    pub clCreateFromD3D11Buffer: Option<clCreateFromD3D11Buffer_fn>,
    pub clCreateFromD3D11Texture2D: Option<clCreateFromD3D11Texture2D_fn>,
    pub clCreateFromD3D11Texture3D: Option<clCreateFromD3D11Texture3D_fn>,
    pub clEnqueueAcquireD3D11Objects: Option<clEnqueueAcquireD3D11Objects_fn>,
    pub clEnqueueReleaseD3D11Objects: Option<clEnqueueReleaseD3D11Objects_fn>,
}

impl ExtensionFunctions {
    pub fn resolve_all(platform: PlatformId) -> Result<Self> {
        let extensions = match get_platform_info(platform, PlatformInfo::Extensions) {
            Ok(PlatformInfoResult::Extensions(s)) => s,
            Ok(_) => {
                return Err(Error::EmptyInfoResult(
                    crate::EmptyInfoResultError::Platform,
                ));
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        let supports_khr_d3d11 = extensions.contains("cl_khr_d3d11_sharing");
        let supports_nv_d3d11 = extensions.contains("cl_nv_d3d11_sharing");

        let mut functions = Self::default();
        functions.clGetGLContextInfoKHR =
            get_pointer(&platform, "clGetGLContextInfoKHR", "")?.map(|p| unsafe { transmute(p) });

        if supports_nv_d3d11 || supports_khr_d3d11 {
            let suffix = if supports_nv_d3d11 { "NV" } else { "KHR" };
            functions.clGetDeviceIDsFromD3D11 =
                get_pointer(&platform, "clGetDeviceIDsFromD3D11", suffix)?
                    .map(|p| unsafe { transmute(p) });
            functions.clCreateFromD3D11Buffer =
                get_pointer(&platform, "clCreateFromD3D11Buffer", suffix)?
                    .map(|p| unsafe { transmute(p) });
            functions.clCreateFromD3D11Texture2D =
                get_pointer(&platform, "clCreateFromD3D11Texture2D", suffix)?
                    .map(|p| unsafe { transmute(p) });
            functions.clCreateFromD3D11Texture3D =
                get_pointer(&platform, "clCreateFromD3D11Texture3D", suffix)?
                    .map(|p| unsafe { transmute(p) });
            functions.clEnqueueAcquireD3D11Objects =
                get_pointer(&platform, "clEnqueueAcquireD3D11Objects", suffix)?
                    .map(|p| unsafe { transmute(p) });
            functions.clEnqueueReleaseD3D11Objects =
                get_pointer(&platform, "clEnqueueReleaseD3D11Objects", suffix)?
                    .map(|p| unsafe { transmute(p) });
        }
        Ok(functions)
    }
}

fn get_pointer(
    platform: &PlatformId,
    func_name: &str,
    suffix: &str,
) -> Result<Option<*mut c_void>> {
    unsafe {
        match get_extension_function_address_for_platform(
            platform,
            &format!("{}{}", func_name, suffix),
            None,
        ) {
            Ok(pointer) => Ok(Some(pointer)),
            Err(Error::ApiWrapper(_)) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

impl std::fmt::Debug for ExtensionFunctions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionFunctions")
            .field(
                "clGetGLContextInfoKHR",
                &self.clGetGLContextInfoKHR.map(|x| x as *mut c_void),
            )
            .field(
                "clGetDeviceIDsFromD3D11",
                &self.clGetDeviceIDsFromD3D11.map(|x| x as *mut c_void),
            )
            .field(
                "clCreateFromD3D11Buffer",
                &self.clCreateFromD3D11Buffer.map(|x| x as *mut c_void),
            )
            .field(
                "clCreateFromD3D11Texture2D",
                &self.clCreateFromD3D11Texture2D.map(|x| x as *mut c_void),
            )
            .field(
                "clCreateFromD3D11Texture3D",
                &self.clCreateFromD3D11Texture3D.map(|x| x as *mut c_void),
            )
            .field(
                "clEnqueueAcquireD3D11Objects",
                &self.clEnqueueAcquireD3D11Objects.map(|x| x as *mut c_void),
            )
            .field(
                "clEnqueueReleaseD3D11Objects",
                &self.clEnqueueReleaseD3D11Objects.map(|x| x as *mut c_void),
            )
            .finish()
    }
}
