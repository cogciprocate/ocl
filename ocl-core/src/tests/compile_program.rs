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
    let platform_id = ::default_platform().unwrap();
    let device_ids = ::get_device_ids(&platform_id, None, None).unwrap();
    let device = [device_ids[0]];

    let context_properties = ::ContextProperties::new().platform(platform_id);
    let context = ::create_context(Some(&context_properties),
        &device, None, None).unwrap();

    let program = ::create_program_with_source(&context, &[CString::new(kernel).unwrap()]).unwrap();
    let program2 = ::create_program_with_source(&context, &[CString::new(kernel2).unwrap()]).unwrap();
    let header = ::create_program_with_source(&context, &[CString::new(header).unwrap()]).unwrap();
    let some_dev: Option<_> = Some(&device[..]);
    let options = CString::new("").unwrap();

    ::compile_program(&program, some_dev, &options, &[&header],  &[CString::new("world.cl").unwrap()],
         None, None).unwrap();
    ::compile_program(&program2, some_dev, &options, &[], &[], None, None).unwrap();
    ::link_program(&context, some_dev, &options, &[&program, &program2], None, None).unwrap();
}