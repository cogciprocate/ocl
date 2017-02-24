#![allow(dead_code, unused_variables, unused_imports, unused_mut)]

use ocl::flags::{self, MemFlags, MapFlags, CommandQueueProperties};

pub struct Switches {
    pub device_check: bool,
    pub queue_flags: CommandQueueProperties,
    // pub futures: bool,
}


// pub static SWITCHES: Switches = Switches {
//     device_check: false,
//     queue_ordering: flags::QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
//     futures: false,
// };

lazy_static! {
    pub static ref SWITCHES: Switches = Switches {
        // device_check: false,
        device_check: true,

        queue_flags: CommandQueueProperties::out_of_order(),
        // queue_flags: CommandQueueProperties::out_of_order() | CommandQueueProperties::profiling(),

        // futures: false,
    };
}