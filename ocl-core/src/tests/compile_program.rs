use std::ffi::CString;

#[test]
fn compile_program() {
    let header = r#"
        static void world() {
            printf("world\n");
        }
    "#;
    let kernel = r#"
        #include "world.cl"
        __kernel void hello() {
            printf("hello ");
            world();
        }
    "#;
    let kernel2 = r#"
        __kernel void hello2() {
            unsigned int gid = get_global_size(0);
            printf("hi\n");
        }
    "#;
    let platform_id = crate::default_platform().unwrap();
    let device_ids = crate::get_device_ids(&platform_id, None, None).unwrap();

    for device in device_ids {
        if device.version().unwrap() < [1, 2].into() {
            println!("Device version too low. Skipping test for this device.");
            continue;
        }

        let context_properties = crate::ContextProperties::new().platform(platform_id);
        let context =
            crate::create_context(Some(&context_properties), &[device], None, None).unwrap();

        let program =
            crate::create_program_with_source(&context, &[CString::new(kernel).unwrap()]).unwrap();
        let program2 =
            crate::create_program_with_source(&context, &[CString::new(kernel2).unwrap()]).unwrap();
        let header =
            crate::create_program_with_source(&context, &[CString::new(header).unwrap()]).unwrap();
        let options = CString::new("").unwrap();

        crate::compile_program(
            &program,
            Some(&[device]),
            &options,
            &[&header],
            &[CString::new("world.cl").unwrap()],
            None,
            None,
            None,
        )
        .unwrap();
        crate::compile_program(
            &program2,
            Some(&[device]),
            &options,
            &[],
            &[],
            None,
            None,
            None,
        )
        .unwrap();
        crate::link_program(
            &context,
            Some(&[device]),
            &options,
            &[&program, &program2],
            None,
            None,
            None,
        )
        .unwrap();
    }
}
