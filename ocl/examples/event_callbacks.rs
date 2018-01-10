//! This is more of a test than anything but put here as an example of how
//! you could use custom event callbacks.
//!
//! Due to buggy and/or intentionally crippled drivers, this example may not
//! work on NVIDIA hardware. Until NVIDIA's implementation is corrected this
//! example will likely fail on that platform.
//!

extern crate libc;
extern crate ocl;
extern crate find_folder;

use libc::c_void;
use find_folder::Search;
use ocl::{util, core, ProQue, Program, Buffer, EventList};
use ocl::ffi::{cl_event, cl_int};

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


fn main() {
    // Set up data set size and work dimensions:
    let dataset_len = 1 << 17;

    // Get a path for our program source:
    let src_file = Search::ParentsThenKids(3, 3).for_folder("examples").unwrap().join("cl/kernel_file.cl");

    // Create a context, program, & queue:
    let ocl_pq = ProQue::builder()
        .dims(dataset_len)
        .prog_bldr(Program::builder().src_file(src_file))
        .build().unwrap();

    // Create source and result buffers (our data containers):
    // let seed_buffer = Buffer::with_vec_scrambled((0u32, 500u32), &dims, &ocl_pq.queue());
    let seed_vec = util::scrambled_vec((0u32, 500u32), dataset_len);
    let seed_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(core::MEM_READ_WRITE | core::MEM_COPY_HOST_PTR)
        .len(dataset_len)
        .host_data(&seed_vec)
        .build().unwrap();

    let mut result_vec = vec![0; dataset_len];
    let mut result_buffer = Buffer::<u32>::builder()
        .queue(ocl_pq.queue().clone())
        .len(dataset_len)
        .build().unwrap();

    // Our arbitrary addend:
    let addend = 11u32;

    // Create kernel with the source initially set to our seed values.
    let mut kernel = ocl_pq.create_kernel("add_scalar").unwrap()
        .gws(dataset_len)
        .arg_buf_named("src", Some(&seed_buffer))
        .arg_scl(addend)
        .arg_buf(&mut result_buffer);

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
            kernel.set_arg_buf_named("src", Some(&result_buffer)).unwrap();
        }

        if PRINT_DEBUG { println!("Enqueuing kernel [itr:{}]...", itr); }
        unsafe {
            kernel.cmd().enew(&mut kernel_event).enq().unwrap();
        }

        let mut read_event = EventList::new();

        if PRINT_DEBUG { println!("Enqueuing read buffer [itr:{}]...", itr); }
        unsafe { result_buffer.cmd().read(&mut result_vec)
            .enew(&mut read_event).block(true).enq().unwrap(); }

        // Clone event list just for fun (test drop a bit):
        let read_event = read_event.clone();

        let last_idx = buncha_stuffs.len() - 1;

        unsafe {
            if PRINT_DEBUG { println!("Setting callback (verify_result, buncha_stuff[{}]) [i:{}]...",
                last_idx, itr); }
            read_event.last().unwrap().set_callback(_test_events_verify_result,
                &mut buncha_stuffs[last_idx] as *mut _ as *mut c_void).unwrap();
        }
    }

    // Wait for all queued tasks to finish so that verify_result() will be
    // called before returning:
    ocl_pq.queue().finish().unwrap();
}

