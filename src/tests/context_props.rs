//! Test `Context', particularly pertaining to properties, Paul.

use standard::{Platform, Device, Context};
use core::{ContextPropertyValue};

// static SRC: &'static str = r#"
//     __kernel void multiply(__global float* buffer, float coeff) {
//         buffer[get_global_id(0)] *= coeff;
//     }
// "#;

#[test] 
fn test_context_props() {
    // let dims = [2048];
    let platforms = Platform::list();

    println!("Looping through each avaliable platform ({}):", platforms.len());

    // Loop through all avaliable platforms:
    for p_idx in 0..platforms.len() {
        let platform = &platforms[p_idx];

        let devices = Device::list_all(platform);

        // [NOTE]: A new context can also be created for each device if desired.
        let context = Context::builder()
            .platform(platform.clone())
            .property(ContextPropertyValue::Platform(platform.clone().into()))
            .devices(&devices)
            .build().unwrap();


        println!("{}", platform);
        println!("{}", context);        

        for device in devices.iter() {
            println!("Device {{ Name: {}, Vendor: {} }}", device.name(), device.vendor());
        }

        print!("\n\n");

        // // Loop through each device
        // for d_idx in 0..devices.len() {
        //     let device = devices[d_idx];

        //     let queue = Queue::new(&context, device).unwrap();
        //     let buffer = Buffer::<f32>::new(&queue, None, &dims, None).unwrap();
        //     let image = Image::<u8>::builder()
        //         .dims(dims)
        //         .build(&queue).unwrap();
        //     let sampler = Sampler::with_defaults(&context).unwrap();
        //     let program = Program::builder()
        //         .src(SRC)
        //         .devices(device)
        //         .build(&context).unwrap();
        //     let kernel = Kernel::new("multiply", &program, &queue).unwrap()
        //         .gws(&dims)
        //         .arg_buf(&buffer)
        //         .arg_scl(10.0f32);

        //     let mut event_list = EventList::new();
        //     kernel.cmd().enew(&mut event_list).enq().unwrap();
        //     event_list.wait().unwrap();

        //     let mut event = Event::empty();
        //     buffer.cmd().write(&vec![0.0; dims[0]]).enew(&mut event).enq().unwrap();
        //     event.wait().unwrap();
        // }
    }
}