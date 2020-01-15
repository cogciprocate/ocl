#[cfg(target_os = "macos")]
extern crate cgl;
extern crate ocl;

#[cfg(target_os = "linux")]
#[allow(improper_ctypes)]
mod glx {
    include!(concat!(env!("OUT_DIR"), "/glx_bindings.rs"));
}

#[cfg(target_os = "windows")]
#[allow(improper_ctypes)]
mod wgl {
    include!(concat!(env!("OUT_DIR"), "/wgl_bindings.rs"));
}

#[cfg(target_os = "android")]
#[allow(improper_ctypes)]
mod egl {
    include!(concat!(env!("OUT_DIR"), "/egl_bindings.rs"));
}

pub fn get_properties_list() -> ocl::builders::ContextProperties {
    let mut properties = ocl::builders::ContextProperties::new();

    #[cfg(target_os = "linux")]
    unsafe {
        properties.set_gl_context(glx::GetCurrentContext() as *mut _);
        properties.set_glx_display(glx::GetCurrentDisplay() as *mut _);
    }

    #[cfg(target_os = "windows")]
    unsafe {
        properties.set_gl_context(wgl::GetCurrentContext() as *mut _);
        properties.set_wgl_hdc(wgl::GetCurrentDC() as *mut _);
    }

    #[cfg(target_os = "macos")]
    unsafe {
        let gl_context = cgl::CGLGetCurrentContext();
        let share_group = cgl::CGLGetShareGroup(gl_context);
        properties = properties.cgl_sharegroup(share_group);
    }

    #[cfg(target_os = "android")]
    unsafe {
        properties.set_gl_display(egl::GetCurrentContext() as *mut _);
        properties.set_egl_display(egl::GetCurrentDisplay() as *mut _);
    }

    return properties;
}

pub fn get_context() -> std::option::Option<ocl::Context> {
    ocl::Platform::list()
        .iter()
        .map(|plat| {
            //println!("Plat: {}",plat);
            ocl::Device::list(plat, Some(ocl::flags::DeviceType::new().gpu()))
                .unwrap()
                .iter()
                .map(|dev| {
                    let ctx = ocl::Context::builder()
                        .properties(get_properties_list().platform(plat))
                        .platform(*plat)
                        .devices(dev)
                        .build();
                    //println!("- Dev: {:?} Ctx: {:?}",dev,ctx);
                    ctx
                })
                .find(|t| t.is_ok())
        })
        .find(|t| t.is_some())
        .map(|ctx| ctx.unwrap().unwrap())
}

#[cfg(test)]
mod tests {
    use get_properties_list;
    #[test]
    fn it_doesnt_crash() {
        get_properties_list();
    }
}
