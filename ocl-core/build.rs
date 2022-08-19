

extern crate rustc_version;
// use rustc_version::{version, version_matches, version_meta, Channel};

fn main() {
    assert!(rustc_version::version().unwrap() >= rustc_version::Version::parse("1.13.0").unwrap(), "This crate has been recently redesigned
        to take advantage of MIR and requires versions of the compiler which do not create hidden
        drop flag fields on #[repr(C)] structs.");
}