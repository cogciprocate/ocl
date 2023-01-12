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

    let platform_id = crate::default_platform().unwrap();
    let device_ids = crate::get_device_ids(&platform_id, None, None).unwrap();
    let device = device_ids[0];
    let context_properties = crate::ContextProperties::new().platform(platform_id);
    let context = crate::create_context(Some(&context_properties), &[device], None, None).unwrap();

    crate::create_build_program(
        &context,
        &[CString::new(kernel).unwrap()],
        None::<&[()]>,
        &CString::new("").unwrap(),
    )
    .unwrap();
}
