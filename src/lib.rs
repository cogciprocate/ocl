extern crate ocl;
#[cfg(target="macos")]
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
    return properties;
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_doesnt_crash() {
        use get_properties_list;
        get_properties_list();
    }
}
