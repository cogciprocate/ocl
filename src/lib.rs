extern crate ocl;
#[cfg(target_os="macos")]
extern crate cgl;
#[cfg(target_os="linux")]
#[allow(improper_ctypes)]
mod glx {
    include!(concat!(env!("OUT_DIR"), "/glx_bindings.rs"));
}
#[cfg(target_os="windows")]
#[allow(improper_ctypes)]
mod wgl {
    include!(concat!(env!("OUT_DIR"), "/wgl_bindings.rs"));
}
#[cfg(target_os="android")]
#[allow(improper_ctypes)]
mod egl {
    include!(concat!(env!("OUT_DIR"), "/egl_bindings.rs"));
}

pub fn get_properties_list() -> ocl::builders::ContextProperties {
    let mut properties = ocl::builders::ContextProperties::new();


    #[cfg(target_os="linux")]
    unsafe {
        properties.set_gl_context(glx::GetCurrentContext() as (*mut _));
        properties.set_glx_display(glx::GetCurrentDisplay() as (*mut _));
    }

    #[cfg(target_os="windows")]
    unsafe {
        properties.set_gl_display(wgl::GetCurrentContext() as (*mut _));
        properties.set_glx_display(wgl::GetCurrentDC() as (*mut _));
    }
    #[cfg(target_os="macos")]
    unsafe {
        #![warn("Untested on MacOS")]
        let gl_context = cgl::CGLGetCurrentContext();
        let share_group = cgl::CGLGetShareGroup(gl_context);
        properties.cgl_sharegroup(share_group);
    }
    #[cfg(target_os="android")]
    unsafe {
        #![warn("Untested on Android")]
            properties.set_gl_display(egl::GetCurrentContext() as (*mut _));
            properties.set_egl_display(egl::GetCurrentDisplay() as (*mut _));
    }
    return properties;
}


#[cfg(test)]
mod tests {
    use get_properties_list;
    #[test]
    fn it_doesnt_crash() {
        get_properties_list();
    }
}
