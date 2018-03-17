//! This is more of a test than anything but put here as an example of how
//! you could use custom event callbacks.
//!
//! Due to buggy and/or intentionally crippled drivers, this example may not
//! work on NVIDIA hardware. Until NVIDIA's implementation is corrected this
//! example will likely fail on that platform.
//!

extern crate ocl;
extern crate ocl_extras;
extern crate find_folder;

use find_folder::Search;
use ocl::{core, ProQue, Program, Buffer, EventList};
use ocl::ffi::{cl_event, cl_int, c_void};

// How many iterations we wish to run:
const ITERATIONS: usize = 8;
// Whether or not to print:
const PRINT_DEBUG: bool = true;
// How many results to print from each iteration:
const RESULTS_TO_PRINT: usize = 5;


struct TestEventsStuff {
    seed_vec: *const [u32],
    result_vec: *const [u32],
    data_set_size: usize,
    addend: u32,
    itr: usize,
}


// Callback for `test_events()`.
//
// Yeah it's ugly.
extern fn _test_events_verify_result(event: cl_event, status: cl_int, user_data: *mut c_void) {
    let buncha_stuff = user_data as *const TestEventsStuff;

    unsafe {
        let seed_vec = (*buncha_stuff).seed_vec as *const [u32];
        let result_vec = (*buncha_stuff).result_vec as *const [u32];
        let data_set_size: usize = (*buncha_stuff).data_set_size;
        let addend: u32 = (*buncha_stuff).addend;
        let itr: usize = (*buncha_stuff).itr;

        if PRINT_DEBUG { println!("\nEvent: `{:?}` has completed with status: `{}`, data_set_size: '{}`, \
                 addend: {}, itr: `{}`.", event, status, data_set_size, addend, itr); }

        for idx in 0..data_set_size {
            assert_eq!((*result_vec)[idx],
                ((*seed_vec)[idx] + ((itr + 1) as u32) * addend));

            if PRINT_DEBUG && (idx < RESULTS_TO_PRINT) {
                let correct_result = (*seed_vec)[idx] + (((itr + 1) as u32) * addend);
                print!("correct_result: {}, result_vec[{idx}]:{}\n",
                    correct_result, (*result_vec)[idx], idx = idx);
            }
        }

        let mut errors_found = 0;

        for idx in 0..data_set_size {
            // [FIXME]: Reportedly failing on OSX:
            assert_eq!((*result_vec)[idx],
             ((*seed_vec)[idx] + ((itr + 1) as u32) * addend));

            if PRINT_DEBUG {
                let correct_result = (*seed_vec)[idx] + (((itr + 1) as u32) * addend);

                if (*result_vec)[idx] != correct_result {
                    print!("correct_result:{}, result_vec[{idx}]:{}\n",
                        correct_result, (*result_vec)[idx], idx = idx);

                    errors_found += 1;
                }
            }
        }

        if PRINT_DEBUG {
            if errors_found > 0 { print!("TOTAL ERRORS FOUND: {}\n", errors_found); }
        }
    }
}


fn event_callbacks() -> ocl::Result<()> {
    // Set up data set size and work dimensions:
    let dataset_len = 1 << 17;

    // Get a path for our program source:
    let src_file = Search::ParentsThenKids(3, 3).for_folder("examples").unwrap()
        .join("cl/kernel_file.cl");

    // Create a context, program, & queue:
    let mut pb = Program::builder();
    pb.src_file(src_file);
    let ocl_pq = ProQue::builder()
        .dims(dataset_len)
        .prog_bldr(pb)
        .build()?;

    // Create source and result buffers (our data containers):
    // let seed_buffer = Buffer::with_vec_scrambled((0u32, 500u32), &dims, &ocl_pq.queue());
    let seed_vec = ocl_extras::scrambled_vec((0u32, 500u32), dataset_len);
    let seed_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(core::MEM_READ_WRITE)
        .len(dataset_len)
        .copy_host_slice(&seed_vec)
        .build()?;

    let mut result_vec = vec![0; dataset_len];
    let result_buffer = Buffer::<u32>::builder()
        .queue(ocl_pq.queue().clone())
        .len(dataset_len)
        .build()?;

    // Our arbitrary addend:
    let addend = 11u32;

    // Create kernel with the source initially set to our seed values.
    let kernel = ocl_pq.kernel_builder("add_scalar")
        .global_work_size(dataset_len)
        .arg_named("src", Some(&seed_buffer))
        .arg(addend)
        .arg(&result_buffer)
        .build()?;

    // Create event list:
    let mut kernel_event = EventList::new();

    //#############################################################################################

    // Create storage for per-event data:
    let mut buncha_stuffs = Vec::<TestEventsStuff>::with_capacity(ITERATIONS);

    // Run our test:
    for itr in 0..ITERATIONS {
        // Store information for use by the result callback function into a vector
        // which will persist until all of the commands have completed (as long as
        // we are sure to allow the queue to finish before returning).
        buncha_stuffs.push(TestEventsStuff {
            seed_vec: seed_vec.as_slice() as *const [u32],
            result_vec: result_vec.as_slice() as *const [u32],
            data_set_size: dataset_len,
            addend: addend,
            itr: itr,
        });

        // Change the source buffer to the result after seed values have been copied.
        // Yes, this is far from optimal...
        // Should just copy the values in the first place but whatever.
        if itr != 0 {
            // kernel.set_arg("src", Some(&result_buffer))?;
            kernel.set_arg("src", &result_buffer)?;
        }

        if PRINT_DEBUG { println!("Enqueuing kernel [itr:{}]...", itr); }
        unsafe {
            kernel.cmd().enew(&mut kernel_event).enq()?;
        }

        let mut read_event = EventList::new();

        if PRINT_DEBUG { println!("Enqueuing read buffer [itr:{}]...", itr); }
        unsafe { result_buffer.cmd().read(&mut result_vec)
            .enew(&mut read_event).block(true).enq()?; }

        // Clone event list just for fun (test drop a bit):
        let read_event = read_event.clone();

        let last_idx = buncha_stuffs.len() - 1;

        unsafe {
            if PRINT_DEBUG { println!("Setting callback (verify_result, buncha_stuff[{}]) [i:{}]...",
                last_idx, itr); }
            read_event.last().unwrap().set_callback(_test_events_verify_result,
                &mut buncha_stuffs[last_idx] as *mut _ as *mut c_void)?;
        }
    }

    // Wait for all queued tasks to finish so that verify_result() will be
    // called before returning:
    ocl_pq.queue().finish()?;
    Ok(())
}

pub fn main() {
    match event_callbacks() {
        Ok(_) => (),
        Err(err) => println!("{}", err),
    }
}