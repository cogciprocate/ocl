#![allow(dead_code, unused_variables, unused_imports, unused_mut)]

extern crate ocl;

use ocl::{Platform, Device, Context, Queue, Program, Buffer, Kernel, SubBuffer, OclPrm,
    Event, EventList, FutureMappedMem, MappedMem};
use ocl::flags::{MemFlags, MapFlags, CommandQueueProperties};
use ocl::aliases::ClFloat4;

pub fn main() {
    let platforms = Platform::list();

    for platform in platforms {
        let devices = Device::list_all(&platform).unwrap();

        for device in devices {
            in_order_queues(platform, device);

            out_of_order_queues(platform, device);
        }
    }
}


pub fn in_order_queues(platform: Platform, device: Device) {
    println!("Platform: {}", platform.name());
    println!("Device: {} {}", device.vendor(), device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    // In order queues:
    let io_queue = Queue::new(&context, device, Some(CommandQueueProperties::profiling())).unwrap();
    let kern_queue = Queue::new(&context, device, Some(CommandQueueProperties::profiling())).unwrap();

}



pub fn out_of_order_queues(platform: Platform, device: Device) {
    println!("Platform: {}", platform.name());
    println!("Device: {} {}", device.vendor(), device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build().unwrap();

    // Out of order queues:
    let io_queue = Queue::new(&context, device, Some(CommandQueueProperties::out_of_order() |
       CommandQueueProperties::profiling())).unwrap();
    let kern_queue = Queue::new(&context, device, Some(CommandQueueProperties::out_of_order() |
       CommandQueueProperties::profiling())).unwrap();
}

