extern crate gl_generator;

use std::env;
use std::fs::File;
use std::path::PathBuf;
use gl_generator::{Registry, Api, Profile, Fallbacks};

fn main() {
    let target = env::var("TARGET").unwrap();
    let dest = PathBuf::from(&env::var("OUT_DIR").unwrap());

    if target.contains("darwin") {
        //macos
        //IDK if this line is needed..
        println!("cargo:rustc-link-lib=framework=OpenGL");
    } else if target.contains("windows") {
        //windows
        let mut file = File::create(&dest.join("wgl_bindings.rs")).unwrap();
        Registry::new(Api::Wgl, (1, 0), Profile::Core, Fallbacks::All, [])
            .write_bindings(gl_generator::StaticGenerator, &mut file)
            .unwrap();
        println!("cargo:rustc-link-lib=opengl32");
    } else if target.contains("linux") && !target.contains("android") {
        //linux
        let mut file = File::create(&dest.join("glx_bindings.rs")).unwrap();
        Registry::new(Api::Glx, (1, 4), Profile::Core, Fallbacks::All, [])
            .write_bindings(gl_generator::StaticGenerator, &mut file)
            .unwrap();
        println!("cargo:rustc-link-lib=GL");
    } else {
        println!("warning=Unknown Platform.. maybe OpenGLES is available here, but GLES isn't supported by ocl-gl",);
    }
}
