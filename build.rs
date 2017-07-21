//! Currently does very little other than print a possible location for
//! OpenCL.lib.
//!
//! Something needs to be done with this to allow it to actually search for
//! and link either OpenCL.lib or OpenCL.dll (depending on gnu/msvc
//! toolchain).


/*
 * This build script needs your help!
 *
 * To get cl-sys working on _all_ platforms will be a group effort.
 * Namely GPU's are ungodly expensive. I don't know where _every_
 * OpenCL.lib file is installed to on Windows.
 *
 * So if you want to use this library on windows. Please patch in your
 * OpenCL.lib location.
 *
 * You will need to install the OpenCL SDK for your _vendor_
 *
 * In the future we _may_ want to add feature flag to determine _which_ vendor's OpenCL you are
 * using.
 */

fn main() {
    if cfg!(windows) {
        let known_sdk = [ 
            // E.g. "c:\Program Files (x86)\Intel\OpenCL SDK\lib\x86\"
            ("INTELOCLSDKROOT", "x64", "x86"),
            // E.g. "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\Win32\"
            ("CUDA_PATH", "x64", "Win32"),
            // E.g. "C:\Program Files (x86)\AMD APP SDK\3.0\lib\x86\"
            ("AMDAPPSDKROOT", "x64_64", "x86"),
        ];

        for info in known_sdk.iter() {
            if let Ok(sdk) = std::env::var(info.0) {
                let mut path = std::path::PathBuf::from(sdk);
                path.push("lib");
                path.push(if cfg!(target_arch="x86_64") { info.1 } else { info.2 });
                println!("cargo:rustc-link-search=native={}", path.display());
            }
        }
    }
}
