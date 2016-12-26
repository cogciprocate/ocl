
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
    
    //This is is the build script related to windows
    #[cfg(all(target_os= "windows", target_arch="x86", target_pointer_width="32"))]
    {

        //This is where the AMD OpenCL.lib file is located for RX480's
        println!("cargo:rustc-link-search={}",r"C:\Program Files (x86)\AMD APP SDK\3.0\lib\x86\");
    }
    #[cfg(all(target_os= "windows", target_arch="x86_64", target_pointer_width="64"))]
    {

        println!("cargo:rustc-link-search={}",r"C:\Program Files (x86)\AMD APP SDK\3.0\lib\x86_64");
    }

}
