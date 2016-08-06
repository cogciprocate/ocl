use std::ffi::CString;

#[test]
#[should_panic]
#[allow(unused_variables)]
fn bad_kernel_variable_names() {
    let kernel = r#"
        kernel void multiply(global float* buffer, float coeff) {

            not_a_variable + im_with_not_a_variable;

            buffer[get_global_id(0)] *= coeff;
        }
    "#;

    let platforms = ::get_platform_ids().unwrap();
    let platform = platforms.first().unwrap().clone();

    let devices = ::get_device_ids(&platform, None, None).unwrap();
    let device = devices.first().unwrap();

    let context_properties = ::ContextProperties::new().platform(platform);
    let context = ::create_context(&Some(context_properties), &[device], None, None).unwrap();

    ::create_build_program(&context, &[CString::new(kernel).unwrap()], &CString::new("").unwrap(),
                           &[device]).unwrap();
}
