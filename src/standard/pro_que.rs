//! A convenient wrapper for `Program` and `Queue`.

use std::convert::Into;
use std::ops::Deref;
use core::OclNum;
use standard::{Context, ProQueBuilder, Program, Queue, Kernel, Buffer,
    BufferDims, SimpleDims, WorkDims};
use error::{Result as OclResult, Error as OclError};

static DIMS_ERR_MSG: &'static str = "This 'ProQue' has not had any dimensions specified. Use 
    'ProQue::builder().dims(__)...' or 'my_pro_que.set_dims(__)' to specify.";


/// An all-in-one chimera of the `Program`, `Queue`, and (optionally) the 
/// `Context` and `SimpleDims` types.
///
/// Handy when creating only a single context, program, and queue or when
/// using a unique program build on each device.
///
/// All `ProQue` functionality is also provided separately by the `Context`, `Queue`, 
/// and `Program` types.
/// 
/// # Creation
/// There are two ways to create a `ProQue`:
///
/// 1. First call `::new` and pass a `Context` and device index then call 
///    `::build` and pass a `ProgramBuilder`.
/// 2. Call `::builder` [FIXME]: Complete description and give examples.
///
/// # Destruction
///
/// Handled automatically. Freely use, store, clone, discard, share among 
/// threads... put some on your toast... whatever.
///
#[derive(Clone, Debug)]
pub struct ProQue {
    context: Context,
    queue: Queue,
    program: Program,
    dims: Option<SimpleDims>,
}

impl ProQue {
    /// Returns a new `ProQueBuilder`.
    ///
    /// Calling `ProQueBuilder::build()` will return a new `ProQue`.
    pub fn builder() -> ProQueBuilder {
        ProQueBuilder::new()
    }


    // / Creates a new queue on the device with `device_idx` (see 
    // / [`Queue`](http://docs.cogciprocate.com/ocl/struct.Queue.html) 
    // / documentation) and returns a new Program/Queue hybrid.
    // /
    // / `::build_program` must be called before this ProQue can be used.
    // /
    // / [FIXME]: Elaborate upon the following:
    // /
    // / - device_idx wraps around (round robins)
    // / - one device only per ProQue
    // / - when is built-in Context used / destroyed
    // /
    // / [UNSTABLE]: Prefer using `ProQueBuilder`.
    // pub fn new(context: Context, device_idx: Option<usize>) -> ProQue {
    //     let queue = Queue::new_by_device_index(context, device_idx);

    //     ProQue {
    //         context: context,
    //         queue: queue,
    //         program: None,
    //         dims: None,
    //     }
    // }


    /// Creates a new ProQue from individual parts.
    ///
    /// Use builder unless you know what you're doing. Creating parts which are
    /// from different devices or contexts will cause errors later on.
    ///
    /// [FIXME] TODO: DEPRICATE BUILDING THROUGH PROQUE.
    pub fn new<D: Into<SimpleDims>>(context: Context, queue: Queue, program: Program,
                    dims: Option<D>) -> ProQue 
    {
        ProQue {
            context: context,
            queue: queue,
            program: program,
            dims: dims.map(|d| d.into()),
        }
    }


    // /// Builds and stores the program defined by `builder`.
    // ///
    // /// ## Panics
    // /// This `ProQue` must not already contain a program.
    // ///
    // /// `program_builder` must not have any device indexes configured (via its
    // /// `::device_idxs` method). `ProQue` will only build programs for the device
    // /// previously configured or the default device if none had been specified.
    // ///
    // /// ## Stability
    // ///
    // /// The usefulness of this method is questionable now that we have a builder. 
    // /// It may be depricated.
    // ///
    // /// [UNSTABLE]: Prefer using `ProQueBuilder`.
    // pub fn build_program(&mut self, builder: &ProgramBuilder) -> OclResult<()> {
    //     if self.program.is_some() { 
    //         return OclError::err("ProQue::build_program(): Pre-existing build detected. Use \
    //             '.clear_build()' first.");
    //     }

    //     if builder.get_devices().len() > 0 {
    //         return OclError::err("ProQue::build_program(): The 'ProgramBuilder' passed \
    //             may not have any device indexes set as they will be ignored. See 'ProQue' \
    //             documentation for more information.");
    //     }
        
    //     self.program = Some(try!(Program::from_parts(
    //         try!(builder.get_src_strings().map_err(|e| e.to_string())), 
    //         try!(builder.get_compiler_options().map_err(|e| e.to_string())), 
    //         self.queue.context_core_as_ref(), 
    //         &vec![self.queue.device().clone()],
    //     )));

    //     Ok(())
    // }


    // /// Clears the current program build. Any kernels created with the pre-existing program will continue to work but new kernels will require a new program to be built. This can occasionally be useful for creating different programs based on the same source but with different constants.
    // /// 
    // /// ## Stability
    // ///
    // /// [UNSTABLE]: Usefulness and safety questionable.
    // ///
    // pub fn clear_build(&mut self) {
    //     // match self.program {
    //     //     Some(ref mut program) => { 
    //     //         program.release();              
    //     //     },

    //     //     None => (),
    //     // }
    //     self.program = None;
    // }


    /// Creates a kernel with pre-assigned dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the contained program has not been created / built, if
    /// there is a problem creating the kernel, or if this `ProQue` has no
    /// pre-assigned dimensions.
    pub fn create_kernel(&self, name: &str) -> Kernel {
        let kernel = Kernel::new(name.to_string(), &self.program, &self.queue)
            .expect("ocl::ProQue::create_kernel");

        match self.dims {
            Some(d) => kernel.gws(d),
            None => kernel,
        }        
    }


    // /// Returns a new Kernel with name: `name` and global work size: `gws`.
    // ///
    // /// # Panics
    // ///
    // /// Panics if the contained program has not been created / built or if
    // /// there is a problem creating the kernel.
    // pub fn create_kernel_with_dims<D: Into<SimpleDims>>(&self, name: &str, gws: D) -> Kernel {
    //     let program = match self.program {
    //         Some(ref prg) => prg,
    //         None => {
    //             panic!("\nProQue::create_kernel(): Cannot add new kernel until \
    //             OpenCL program is built. Use: \
    //             '{{your_proque}}.build_program({{your_program_builder}});'.\n")
    //         },
    //     };

    //     Kernel::new(name.to_string(), &program, &self.queue, gws.into()).unwrap()
    // }


    /// Returns a new buffer
    ///
    /// `with_vec` determines whether or not the buffer is created with a
    /// built-in vector.
    ///
    /// The default dimensions for this `ProQue` will be used.
    ///
    /// # Panics
    ///
    /// This `ProQue` must have been pre-configured with default dimensions
    // to use this method. Otherwise, use Buffer::new(), etc.
    ///
    pub fn create_buffer<T: OclNum>(&self, with_vec: bool) -> Buffer<T> {
        let dims = self.dims_result().expect("ocl::ProQue::create_buffer");

        if with_vec {
            Buffer::with_vec(&dims, &self.queue)
        } else {
            Buffer::new(&dims, &self.queue)
        }
    }

    pub fn set_dims<S: Into<SimpleDims>>(&mut self, dims: S) {
        self.dims = Some(dims.into());
    }

    /// Returns the maximum workgroup size supported by the device associated
    /// with this `ProQue`.
    pub fn max_wg_size(&self) -> usize {
        self.queue.device().max_wg_size()
    }

    /// Returns a reference to the queue associated with this ProQue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the contained context, if any.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Returns the current program build, if any.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Returns the current `dims` or panics.
    pub fn dims(&self) -> &SimpleDims {
        self.dims_result().expect("ocl::ProQue::dims(): This ProQue has no dimensions set. 
            Use `::set_dims` to set some or set them during building with `::dims`.")
    }

    /// Returns the current `dims` or an error.
    pub fn dims_result(&self) -> OclResult<&SimpleDims> {
        match self.dims {
            Some(ref dims) => Ok(dims),
            None => OclError::err(DIMS_ERR_MSG),
        }
    }
}

impl BufferDims for ProQue {
    fn padded_buffer_len(&self, len: usize) -> usize {
        self.dims_result().expect("ProQue::padded_buffer_len").padded_buffer_len(len)
    }
}

impl WorkDims for ProQue {
    fn dim_count(&self) -> u32 {
        self.dims_result().expect("ProQue::dim_count").dim_count()
    }

    fn to_work_size(&self) -> Option<[usize; 3]> {
        self.dims_result().expect("ProQue::to_work_size").to_work_size()
    }

    fn to_work_offset(&self) -> Option<[usize; 3]> {
        self.dims_result().expect("ProQue::to_work_offset").to_work_offset()
    }
}

impl Deref for ProQue {
    type Target = SimpleDims;

    fn deref(&self) -> &SimpleDims {
        match self.dims {
            Some(ref dims) => dims,
            None => panic!(DIMS_ERR_MSG),
        }
    }
}
