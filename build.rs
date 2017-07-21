
fn main() {
	if cfg!(windows) {
		// E.g. "c:\Program Files (x86)\Intel\OpenCL SDK\lib\x86\"
		if let Ok(intel_sdk) = std::env::var("INTELOCLSDKROOT") {
			let mut path = std::path::PathBuf::from(intel_sdk);
			path.push("lib");
			path.push(if cfg!(target_arch="x86_64") { "x64" } else { "x86"});
			println!("cargo:rustc-link-search=native={}", path.display());
		}
		// E.g. "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\Win32\"
		if let Ok(intel_sdk) = std::env::var("CUDA_PATH") {
			let mut path = std::path::PathBuf::from(intel_sdk);
			path.push("lib");
			path.push(if cfg!(target_arch="x86_64") { "x64" } else { "Win32"});
			println!("cargo:rustc-link-search=native={}", path.display());
		}
	}
}
