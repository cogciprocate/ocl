//! An OpenCL Image.
//!
//
// * NOTE: `Image` does not have the latest command builders nor does it have
//   support for mapping yet. (TODO: implement)
//

use std;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use core::error::{Result as OclCoreResult};
use core::{self, OclPrm, Mem as MemCore, MemFlags, MemObjectType, ImageFormatParseResult,
    ImageFormat, ImageDescriptor, ImageInfo, ImageInfoResult, MemInfo, MemInfoResult,
    ImageChannelOrder, ImageChannelDataType, AsMem, MemCmdRw, MemCmdAll,
    MapFlags};
use standard::{Context, Queue, SpatialDims, ClNullEventPtrEnum, ClWaitListPtrEnum,
    QueCtx};
use ::MemMap;

#[cfg(not(feature="opencl_vendor_mesa"))]
use ffi::{cl_GLuint, cl_GLint};
#[cfg(not(feature="opencl_vendor_mesa"))]
use core::{GlTextureTarget};


/// The type of operation to be performed by a command.
#[derive(Debug)]
pub enum ImageCmdKind<'c, T: 'c> {
    Unspecified,
    Read { data: &'c mut [T] },
    Write { data: &'c [T] },
    Map,
    Fill { color: T },
    Copy { dst_image: &'c MemCore, dst_origin: [usize; 3] },
    CopyToBuffer { buffer: &'c MemCore, dst_origin: usize },
    GLAcquire,
    GLRelease,
}

impl<'c, T: 'c> ImageCmdKind<'c, T> {
    fn is_unspec(&'c self) -> bool {
        if let ImageCmdKind::Unspecified = *self {
            true
        } else {
            false
        }
    }
}

/// An image command builder for enqueuing reads, writes, fills, and copies.
///
/// ## Examples
///
/// ```text
///
/// // Copies one image to another:
/// src_image.cmd().copy(&dst_image, [0, 0, 0]).enq().unwrap();
///
/// // Writes from a vector to an image, waiting on an event:
/// image.write(&src_vec).ewait(&event).enq().unwrap();
///
/// // Reads from a image into a vector, waiting on an event list and
/// // filling a new empty event:
/// image.read(&dst_vec).ewait(&event_list).enew(&empty_event).enq().unwrap();
///
/// // Reads without blocking:
/// image.cmd().read_async(&dst_vec).enew(&empty_event).enq().unwrap();
///
/// ```
///
/// [FIXME]: Fills not yet implemented.
#[must_use = "commands do nothing unless enqueued"]
#[allow(dead_code)]
pub struct ImageCmd<'c, T: 'c> {
    queue: Option<&'c Queue>,
    obj_core: &'c MemCore,
    block: bool,
    origin: [usize; 3],
    region: [usize; 3],
    row_pitch_bytes: usize,
    slc_pitch_bytes: usize,
    kind: ImageCmdKind<'c, T>,
    ewait: Option<ClWaitListPtrEnum<'c>>,
    enew: Option<ClNullEventPtrEnum<'c>>,
    mem_dims: [usize; 3],
}

/// [UNSTABLE]: All methods still in a state of adjustifulsomeness.
impl<'c, T: 'c + OclPrm> ImageCmd<'c, T> {
    /// Returns a new image command builder associated with with the
    /// memory object `obj_core` along with a default `queue` and `to_len`
    /// (the length of the device side image).
    fn new(queue: Option<&'c Queue>, obj_core: &'c MemCore, dims: [usize; 3])
            -> ImageCmd<'c, T> {
        ImageCmd {
            queue: queue,
            obj_core: obj_core,
            block: true,
            origin: [0, 0, 0],
            region: dims,
            row_pitch_bytes: 0,
            slc_pitch_bytes: 0,
            kind: ImageCmdKind::Unspecified,
            ewait: None,
            enew: None,
            mem_dims: dims,
        }
    }

    /// Specifies that this command will be a blocking read operation.
    ///
    /// After calling this method, the blocking state of this command will
    /// be locked to true and a call to `::block` will cause a panic.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified.
    ///
    /// ### More Information
    ///
    /// See [SDK][read_image] docs for more details.
    ///
    /// [read_image]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueReadImage.html
    pub fn read<'d>(mut self, dst_data: &'d mut [T]) -> ImageCmd<'c, T>
            where 'd: 'c {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::read(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Read { data: dst_data };
        self.block = true;
        self
    }

    /// Specifies that this command will be a write operation.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    /// ### More Information
    ///
    /// See [SDK][read_buffer] docs for more details.
    ///
    /// [read_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueReadBuffer.html
    pub fn write<'d>(mut self, src_data: &'d [T]) -> ImageCmd<'c, T>
            where 'd: 'c {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::write(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Write { data: src_data };
        self
    }

    /// Specifies that this command will be a map operation.
    ///
    /// If `.block(..)` has been set it will be ignored. Non-blocking map
    /// commands are enqueued using `::enq_async`.
    ///
    /// ## Safety
    ///
    /// The caller must ensure that only one mapping of a particular memory
    /// region exists at a time.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    /// ### More Information
    ///
    /// See [SDK][map_image] docs for more details.
    ///
    /// [map_image]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueMapImage.html
    pub unsafe fn map(mut self) -> ImageMapCmd<'c, T> {
        assert!(self.kind.is_unspec(), "ocl::BufferCmd::write(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Map;

        unimplemented!();
        // ImageMapCmd { cmd: self }
    }

    /// Specifies that this command will be a copy operation.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Errors
    ///
    /// If this is a rectangular copy, `dst_origin` and `len` must be zero.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy<'d>(mut self, dst_image: &'d Image<T>, dst_origin: [usize; 3]) -> ImageCmd<'c, T>
            where 'd: 'c {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::copy(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Copy {
            dst_image: dst_image.as_core(),
            dst_origin: dst_origin,
        };
        self
    }

    /// Specifies that this command will be a copy to image.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn copy_to_buffer<'d>(mut self, buffer: &'d MemCore, dst_origin: usize) -> ImageCmd<'c, T>
            where 'd: 'c {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::copy_to_buffer(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::CopyToBuffer { buffer: buffer, dst_origin: dst_origin };
        self
    }

    /// Specifies that this command will acquire a GL buffer.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn gl_acquire(mut self) -> ImageCmd<'c, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::gl_acquire(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::GLAcquire;
        self
    }

    /// Specifies that this command will release a GL buffer.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn gl_release(mut self) -> ImageCmd<'c, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::gl_release(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::GLRelease;
        self
    }

    /// Specifies that this command will be a fill.
    ///
    /// If `.block(..)` has been set it will be ignored.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn fill(mut self, color: T) -> ImageCmd<'c, T> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::fill(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Fill { color: color };
        self
    }

    /// Specifies a queue to use for this call only.
    ///
    /// Overrides the image's default queue if one is set. If no default queue
    /// is set, this method **must** be called before enqueuing the command.
    pub fn queue(mut self, queue: &'c Queue) -> ImageCmd<'c, T> {
        self.queue = Some(queue);
        self
    }

    /// Specifies whether or not to block the current thread until completion.
    ///
    /// Ignored if this is not a read or write operation.
    ///
    /// Default is `block = true`.
    ///
    /// ## Safety
    ///
    /// When performing non-blocking reads or writes, the caller must ensure
    /// that the data being read from or written to is not accessed improperly
    /// until the command completes. Use events (`Event::wait_for`) or the
    /// command queue (`Queue::finish`) to synchronize.
    ///
    /// If possible, prefer instead to use [`::map`] with [`::enq_async`] for
    /// optimal performance and data integrity.
    ///
    /// [`::map`]: struct.ImageMapCmd.html
    /// [`::enq_async`]: struct.ImageMapCmd.html#method.enq_async
    ///
    pub unsafe fn block(mut self, block: bool) -> ImageCmd<'c, T> {
        self.block = block;
        self
    }

    /// Sets the three dimensional offset, the origin point, for an operation.
    ///
    /// Defaults to [0, 0, 0] if not set.
    ///
    /// ## Panics
    ///
    /// The 'shape' may not have already been set to rectangular by the
    /// `::rect` function.
    pub fn origin<D>(mut self, origin: D) -> ImageCmd<'c, T>
            where D: Into<SpatialDims> {
        self.origin = origin.into().to_offset().unwrap();
        self
    }

    /// Sets the region size for an operation.
    ///
    /// Defaults to the full region size of the image(s) as defined when first
    /// created if not set.
    ///
    /// ## Panics [TENATIVE]
    ///
    /// Panics if the region is out of range on any of the three dimensions.
    ///
    /// [FIXME]: Probably delay checking this until enq().
    ///
    pub fn region<D>(mut self, region: D) -> ImageCmd<'c, T>
        where D: Into<SpatialDims> {
        self.region = region.into().to_lens().unwrap();
        self
    }

    /// Sets the row and slice pitch for a read or write operation in bytes.
    ///
    /// `row_pitch_bytes`: Must be greater than or equal to the region width
    /// in bytes (region[0] * sizeof(T)).
    ///
    /// `slice_pitch: Must be greater than or equal to `row_pitch` * region
    /// height in bytes (region[1] * sizeof(T)).
    ///
    /// Only needs to be set if region has been set to something other than
    /// the (default) image buffer size.
    ///
    pub unsafe fn pitch_bytes(mut self, row_pitch_bytes: usize, slc_pitch_bytes: usize) -> ImageCmd<'c, T> {
        self.row_pitch_bytes = row_pitch_bytes;
        self.slc_pitch_bytes = slc_pitch_bytes;
        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> ImageCmd<'c, T>
            where 'e: 'c, Ewl: Into<ClWaitListPtrEnum<'e>> {
        self.ewait = Some(ewait.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew<'e, En>(mut self, enew: En) -> ImageCmd<'c, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>> {
        self.enew = Some(enew.into());
        self
    }

    /// Enqueues this command.
    ///
    /// * TODO: FOR COPY, FILL, AND COPYTOBUFFER -- ENSURE PITCHES ARE BOTH
    ///   UNSET.
    pub fn enq(self) -> OclCoreResult<()> {
        let queue = match self.queue {
            Some(q) => q,
            None => return Err("ImageCmd::enq: No queue set.".into()),
        };

        match self.kind {
            ImageCmdKind::Read { data } => {
                unsafe { core::enqueue_read_image(queue, self.obj_core, self.block,
                    self.origin, self.region, self.row_pitch_bytes, self.slc_pitch_bytes, data, self.ewait,
                    self.enew) }
            },
            ImageCmdKind::Write { data } => {
                unsafe {
                    core::enqueue_write_image(queue, self.obj_core, self.block,
                        self.origin, self.region, self.row_pitch_bytes, self.slc_pitch_bytes, data, self.ewait,
                        self.enew)
                }
            },
            ImageCmdKind::Copy { dst_image, dst_origin } => {
                core::enqueue_copy_image(queue, self.obj_core, dst_image, self.origin,
                    dst_origin, self.region, self.ewait, self.enew)
            },

            #[cfg(not(feature="opencl_vendor_mesa"))]
            ImageCmdKind::GLAcquire => {
                // core::enqueue_acquire_gl_buffer(queue, self.obj_core, self.ewait, self.enew)
                let buf_slc = unsafe { std::slice::from_raw_parts(self.obj_core, 1) };
                core::enqueue_acquire_gl_objects(queue, buf_slc, self.ewait, self.enew)
            },

            #[cfg(not(feature="opencl_vendor_mesa"))]
            ImageCmdKind::GLRelease => {
                // core::enqueue_release_gl_buffer(queue, self.obj_core, self.ewait, self.enew)
                let buf_slc = unsafe { std::slice::from_raw_parts(self.obj_core, 1) };
                core::enqueue_release_gl_objects(queue, buf_slc, self.ewait, self.enew)
            },

            ImageCmdKind::Unspecified => Err("ocl::ImageCmd::enq(): No operation \
                specified. Use '.read(...)', 'write(...)', etc. before calling '.enq()'.".into()),
            _ => unimplemented!(),
        }
    }
}



/// A buffer command builder used to enqueue maps.
///
/// See [SDK][map_buffer] docs for more details.
///
/// [map_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueMapBuffer.html

    // const size_t  * origin ,
    // const size_t  * region ,
    // size_t  *image_row_pitch ,
    // size_t  *image_slice_pitch ,

pub struct ImageMapCmd<'c, T> where T: 'c {
    cmd: ImageCmd<'c, T>,
    flags: Option<MapFlags>,
}

impl<'c, T> ImageMapCmd<'c, T> where T: OclPrm {
    /// Specifies the flags to be used with this map command.
    ///
    /// See [SDK] docs for more details.
    ///
    /// [SDK]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clEnqueueMapBuffer.html
    pub fn flags(mut self, flags: MapFlags) -> ImageMapCmd<'c, T> {
        self.flags = Some(flags);
        self
    }

    /// Sets the three dimensional offset, the origin point, for an operation.
    ///
    /// Defaults to [0, 0, 0] if not set.
    ///
    /// ## Panics
    ///
    /// The 'shape' may not have already been set to rectangular by the
    /// `::rect` function.
    pub fn origin(mut self, origin: [usize; 3]) -> ImageMapCmd<'c, T> {
        self.cmd.origin = origin;
        self
    }

    /// Sets the region size for an operation.
    ///
    /// Defaults to the full region size of the image(s) as defined when first
    /// created if not set.
    ///
    /// ## Panics [TENATIVE]
    ///
    /// Panics if the region is out of range on any of the three dimensions.
    ///
    /// [FIXME]: Probably delay checking this until enq().
    ///
    pub fn region(mut self, region: [usize; 3]) -> ImageMapCmd<'c, T> {
        self.cmd.region = region;
        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait<'e, Ewl>(mut self, ewait: Ewl) -> ImageMapCmd<'c, T>
            where 'e: 'c, Ewl: Into<ClWaitListPtrEnum<'e>> {
        self.cmd.ewait = Some(ewait.into());
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew<'e, En>(mut self, enew: En) -> ImageMapCmd<'c, T>
            where 'e: 'c, En: Into<ClNullEventPtrEnum<'e>> {
        self.cmd.enew = Some(enew.into());
        self
    }

    /// Enqueues this command.
    ///
    /// * TODO: FOR COPY, FILL, AND COPYTOBUFFER -- ENSURE PITCHES ARE BOTH UNSET.
    #[allow(unused_variables, unreachable_code)]
    pub fn enq(self) -> OclCoreResult<MemMap<T>> {
        let queue = match self.cmd.queue {
            Some(q) => q,
            None => return Err("ImageCmd::enq: No queue set.".into()),
        };

        let flags = self.flags.unwrap_or(MapFlags::empty());

        match self.cmd.kind {
            ImageCmdKind::Map => {
                // try!(check_len(self.cmd.to_len, data.len(), offset));

                let mut row_pitch_bytes = 0usize;
                let mut slc_pitch_bytes = 0usize;

                unsafe {
                    let mm_core = core::enqueue_map_image::<T, _, _, _>(
                        queue,
                        self.cmd.obj_core,
                        self.cmd.block,
                        flags,
                        self.cmd.origin,
                        self.cmd.region,
                        &mut row_pitch_bytes,
                        &mut slc_pitch_bytes,
                        self.cmd.ewait,
                        self.cmd.enew,
                    )?;

                    let len_bytes = if slc_pitch_bytes == 0 {
                        // 1D or 2D image.
                        unimplemented!();
                    } else {
                        // 1D image array, 2D image array, or 3D image.
                        unimplemented!();
                    };

                    // let unmap_event = None;

                    // * TODO: Create a special container for mapped images
                    // that can take into account row and slice pitch. It
                    // cannot deref into a &[T] as the size of rows (and
                    // slices) can vary with byte-sized precision.

                    // Ok(MemMap::new(mm_core, 0, unmap_event, self.cmd.obj_core.clone(),
                    //     queue.core().clone()))
                }
            },
            _ => unreachable!(),
        }
    }
}



/// A section of device memory which represents one or many images.
///
/// Use `::builder` for an easy way to create. [UNIMPLEMENTED]
///
#[derive(Clone, Debug)]
pub struct Image<T: OclPrm> {
    obj_core: MemCore,
    queue: Option<Queue>,
    dims: SpatialDims,
    pixel_element_len: usize,
    _pixel: PhantomData<T>
}

impl<T: OclPrm> Image<T> {
    /// Returns a list of supported image formats.
    pub fn supported_formats(context: &Context, flags: MemFlags, mem_obj_type: MemObjectType)
            -> OclCoreResult<Vec<ImageFormatParseResult>> {
        core::get_supported_image_formats(context, flags, mem_obj_type)
    }

    /// Returns an `ImageBuilder`. This is the recommended method to create
    /// a new `Image`.
    pub fn builder<'a>() -> ImageBuilder<'a, T> {
        ImageBuilder::new()
    }

    /// Returns a new `Image`.
    ///
    /// Prefer `::builder` to create a new image.
    // pub fn new(queue: Queue, flags: MemFlags, image_format: ImageFormat,
    //         image_desc: ImageDescriptor, host_data: Option<&[E]>) -> OclCoreResult<Image<E>>
    pub fn new<'o, Q>(que_ctx: Q, flags: MemFlags, image_format: ImageFormat,
            image_desc: ImageDescriptor, host_data: Option<&[T]>) -> OclCoreResult<Image<T>>
            where Q: Into<QueCtx<'o>> {
        let que_ctx = que_ctx.into();
        let context = que_ctx.context_cloned();
        let device_versions = context.device_versions()?;

        let obj_core = unsafe { core::create_image(
            &context,
            flags,
            &image_format,
            &image_desc,
            host_data,
            Some(&device_versions),
        )? };

        let pixel_element_len = match core::get_image_info(&obj_core, ImageInfo::ElementSize)? {
            ImageInfoResult::ElementSize(s) => s / mem::size_of::<T>(),
            _ => return Err("ocl::Image::element_len(): \
                Unexpected 'ImageInfoResult' variant.".into()),
        };

        let dims = [image_desc.image_width, image_desc.image_height, image_desc.image_depth].into();

        let new_img = Image {
            obj_core: obj_core,
            queue: que_ctx.into(),
            dims: dims,
            pixel_element_len: pixel_element_len,
            _pixel: PhantomData,
        };

        Ok(new_img)
    }

    /// Returns a new `Image` from an existant GL texture2D/3D.
    // [WORK IN PROGRESS]
    #[cfg(not(feature="opencl_vendor_mesa"))]
    pub fn from_gl_texture<'o, Q>(que_ctx: Q, flags: MemFlags, image_desc: ImageDescriptor,
            texture_target: GlTextureTarget, miplevel: cl_GLint, texture: cl_GLuint)
            -> OclCoreResult<Image<T>>
            where Q: Into<QueCtx<'o>>
    {
        let que_ctx = que_ctx.into();
        let context = que_ctx.context_cloned();
        let device_versions = context.device_versions()?;

        if texture_target == GlTextureTarget::GlTextureBuffer && miplevel != 0 {
            return Err("If texture_target is GL_TEXTURE_BUFFER, miplevel must be 0.\
                Implementations may return CL_INVALID_OPERATION for miplevel values > 0".into());
        }

        let obj_core = unsafe { core::create_from_gl_texture(
            &context,
            texture_target as u32,
            miplevel,
            texture,
            flags,
            Some(&device_versions),
        )? };

        // FIXME can I do this from a GLTexture ?
        let pixel_element_len = match core::get_image_info(&obj_core, ImageInfo::ElementSize)? {
            ImageInfoResult::ElementSize(s) => s / mem::size_of::<T>(),
            _ => return Err("ocl::Image::element_len(): Unexpected \
                'ImageInfoResult' variant.".into()),
        };

        let dims = [image_desc.image_width, image_desc.image_height, image_desc.image_depth].into();

        let new_img = Image {
            obj_core: obj_core,
            queue: que_ctx.into(),
            dims: dims,
            pixel_element_len: pixel_element_len,
            _pixel: PhantomData,
        };

        Ok(new_img)
    }

    /// Returns a new `Image` from an existant renderbuffer.
    // [WORK IN PROGRESS]
    #[cfg(not(feature="opencl_vendor_mesa"))]
    pub fn from_gl_renderbuffer<'o, Q>(que_ctx: Q, flags: MemFlags, image_desc: ImageDescriptor,
            renderbuffer: cl_GLuint) -> OclCoreResult<Image<T>>
            where Q: Into<QueCtx<'o>>
    {
        let que_ctx = que_ctx.into();
        let context = que_ctx.context_cloned();

        let obj_core = unsafe { core::create_from_gl_renderbuffer(
            &context,
            renderbuffer,
            flags,
        )? };

        // FIXME can I do this from a renderbuffer ?
        let pixel_element_len = match core::get_image_info(&obj_core, ImageInfo::ElementSize)? {
            ImageInfoResult::ElementSize(s) => s / mem::size_of::<T>(),
            _ => return Err("ocl::Image::element_len(): \
                Unexpected 'ImageInfoResult' variant.".into()),
        };

        let dims = [image_desc.image_width, image_desc.image_height].into();

        let new_img = Image {
            obj_core: obj_core,
            queue: que_ctx.into(),
            dims: dims,
            pixel_element_len: pixel_element_len,
            _pixel: PhantomData,
        };

        Ok(new_img)
    }

    /// Returns an image command builder used to read, write, copy, etc.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](struct.ImageCmd)
    /// for more details.
    pub fn cmd(&self) -> ImageCmd<T> {
        ImageCmd::new(self.queue.as_ref(), &self.obj_core,
            self.dims.to_lens().expect("ocl::Image::cmd"))
    }

    /// Returns an image command builder set to read.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](struct.ImageCmd#method.read)
    /// for more details.
    pub fn read<'c, 'd>(&'c self, data: &'d mut [T]) -> ImageCmd<'c, T>
        where 'd: 'c {
        self.cmd().read(data)
    }

    /// Returns an image command builder set to write.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// See the [command builder documentation](struct.ImageCmd#method.write)
    /// for more details.
    pub fn write<'c, 'd>(&'c self, data: &'d [T]) -> ImageCmd<'c, T>
        where 'd: 'c {
        self.cmd().write(data)
    }

    /// Returns a command builder used to map data for reading or writing.
    ///
    /// Call `.enq()` to enqueue the command.
    ///
    /// ## Safety
    ///
    /// The caller must ensure that only one mapping of a particular memory
    /// region exists at a time.
    ///
    /// See the [command builder documentation](struct.ImageCmd#method.map)
    /// for more details.
    ///
    #[inline]
    pub unsafe fn map<'c>(&'c self) -> ImageMapCmd<'c, T> {
        unimplemented!();
        // self.cmd().map()
    }

    // /// Specifies that this command will be a copy operation.
    // ///
    // /// Call `.enq()` to enqueue the command.
    // ///
    // /// See the [command builder documentation](struct.ImageCmd#method.copy)
    // /// for more details.
    // ///
    // #[inline]
    // pub fn copy<'c, M>(&'c self, dst_buffer: &'c M, dst_offset: Option<usize>, len: Option<usize>)
    //         -> BufferCmd<'c, T>
    //         where M: AsMem<T>
    // {
    //     self.cmd().copy(dst_buffer, dst_offset, len)
    // }

    /// Changes the default queue.
    ///
    /// Returns a ref for chaining i.e.:
    ///
    /// `image.set_default_queue(queue).write(....);`
    ///
    /// [NOTE]: Even when used as above, the queue is changed permanently,
    /// not just for the one call. Changing the queue is cheap so feel free
    /// to change as often as needed.
    ///
    /// The new queue must be associated with a valid device.
    ///
    pub fn set_default_queue<'a>(&'a mut self, queue: Queue) -> &'a mut Image<T> {
        // self.command_queue_obj_core = queue.core().clone();
        self.queue = Some(queue);
        self
    }

    /// Returns a reference to the default queue.
    pub fn default_queue(&self) -> Option<&Queue> {
        self.queue.as_ref()
    }

    /// Returns this image's dimensions.
    pub fn dims(&self) -> &SpatialDims {
        &self.dims
    }

    /// Returns the total number of pixels in this image.
    pub fn pixel_count(&self) -> usize {
        self.dims.to_len()
    }

    /// Returns the length of each pixel element.
    pub fn pixel_element_len(&self) -> usize {
        self.pixel_element_len
    }

    /// Returns the total number of pixel elements in this image. Equivalent to its length.
    pub fn element_count(&self) -> usize {
        self.pixel_count() * self.pixel_element_len()
    }

    /// Get information about this image.
    pub fn info(&self, info_kind: ImageInfo) -> OclCoreResult<ImageInfoResult> {
        // match core::get_image_info(&self.obj_core, info_kind) {
        //     Ok(res) => res,
        //     Err(err) => ImageInfoResult::Error(Box::new(err)),
        // }
        core::get_image_info(&self.obj_core, info_kind)
    }

    /// Returns info about this image's memory.
    pub fn mem_info(&self, info_kind: MemInfo) -> OclCoreResult<MemInfoResult> {
        // match core::get_mem_object_info(&self.obj_core, info_kind) {
        //     Ok(res) => res,
        //     Err(err) => MemInfoResult::Error(Box::new(err)),
        // }
        core::get_mem_object_info(&self.obj_core, info_kind)
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    #[inline]
    pub fn as_core(&self) -> &MemCore {
        &self.obj_core
    }


    /// Format image info.
    fn fmt_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("ElementSize", &self.info(ImageInfo::ElementSize))
            .field("RowPitch", &self.info(ImageInfo::RowPitch))
            .field("SlicePitch", &self.info(ImageInfo::SlicePitch))
            .field("Width", &self.info(ImageInfo::Width))
            .field("Height", &self.info(ImageInfo::Height))
            .field("Depth", &self.info(ImageInfo::Depth))
            .field("ArraySize", &self.info(ImageInfo::ArraySize))
            .field("Buffer", &self.info(ImageInfo::Buffer))
            .field("NumMipLevels", &self.info(ImageInfo::NumMipLevels))
            .field("NumSamples", &self.info(ImageInfo::NumSamples))
            .finish()
    }

    /// Format image mem info.
    fn fmt_mem_info(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Mem")
            .field("Type", &self.mem_info(MemInfo::Type))
            .field("Flags", &self.mem_info(MemInfo::Flags))
            .field("Size", &self.mem_info(MemInfo::Size))
            .field("HostPtr", &self.mem_info(MemInfo::HostPtr))
            .field("MapCount", &self.mem_info(MemInfo::MapCount))
            .field("ReferenceCount", &self.mem_info(MemInfo::ReferenceCount))
            .field("Context", &self.mem_info(MemInfo::Context))
            .field("AssociatedMemobject", &self.mem_info(MemInfo::AssociatedMemobject))
            .field("Offset", &self.mem_info(MemInfo::Offset))
            .finish()
    }
}

impl<T: OclPrm> std::fmt::Display for Image<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_info(f)?;
        write!(f, " ")?;
        self.fmt_mem_info(f)
    }
}

impl<T: OclPrm> Deref for Image<T> {
    type Target = MemCore;

    fn deref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<T: OclPrm> DerefMut for Image<T> {
    fn deref_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}

impl<T: OclPrm> AsMem<T> for Image<T> {
    fn as_mem(&self) -> &MemCore {
        &self.obj_core
    }
}

unsafe impl<'a, T> MemCmdRw for Image<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdRw for &'a Image<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdRw for &'a mut Image<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for Image<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for &'a Image<T> where T: OclPrm {}
unsafe impl<'a, T> MemCmdAll for &'a mut Image<T> where T: OclPrm {}



/// A builder for `Image`.
#[must_use = "builders do nothing unless '::build' is called"]
pub struct ImageBuilder<'a, T> where T: 'a {
    queue_option: Option<QueCtx<'a>>,
    flags: MemFlags,
    host_data: Option<&'a [T]>,
    image_format: ImageFormat,
    image_desc: ImageDescriptor,
    _pixel: PhantomData<T>,
}


impl<'a, T> ImageBuilder<'a, T> where T: 'a + OclPrm {
    /// Returns a new `ImageBuilder` with very basic defaults.
    ///
    /// ## Defaults
    ///
    /// * Flags:
    ///
    /// ```text
    /// ocl::MEM_READ_WRITE
    /// ```
    ///
    /// * Image Format:
    ///
    /// ```text
    /// ocl::ImageFormat {
    ///    channel_order: ocl::ImageChannelOrder::Rgba,
    ///    channel_data_type: ocl::ImageChannelDataType::SnormInt8,
    /// }
    /// ```
    ///
    /// * Descriptor (stores everything else - width, height, pitch, etc.):
    ///
    /// ```text
    /// ImageDescriptor::new(MemObjectType::Image1d, 0, 0, 0, 0, 0, 0, None)
    /// ```
    ///
    /// ## Reference
    ///
    /// See the [official SDK documentation] for more information.
    ///
    /// Some descriptions here are adapted from various SDK docs.
    ///
    /// [official SDK docs]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateImage.html
    pub fn new() -> ImageBuilder<'a, T> {
        ImageBuilder {
            queue_option: None,
            flags: core::MEM_READ_WRITE,
            host_data: None,
            image_format: ImageFormat::new_rgba(),
            image_desc: ImageDescriptor::new(MemObjectType::Image1d, 0, 0, 0, 0, 0, 0, None),
            _pixel: PhantomData,
            // host_data: None,
        }
    }

    /// Sets the context with which to associate the buffer.
    ///
    /// May not be used in combination with `::queue` (use one or the other).
    pub fn context<'o>(mut self, context: &'o Context) -> ImageBuilder<'a, T>
            where 'o: 'a {
        assert!(self.queue_option.is_none());
        self.queue_option = Some(QueCtx::Context(context));
        self
    }

    /// Sets the default queue.
    ///
    /// If this is set, the context associated with the `default_queue` will
    /// be used when creating the buffer (use one or the other).
    pub fn queue<'b>(mut self, default_queue: Queue) -> ImageBuilder<'a, T> {
        assert!(self.queue_option.is_none());
        self.queue_option = Some(QueCtx::Queue(default_queue));
        self
    }

    /// Sets the flags for the memory to be created.
    ///
    /// Setting this overwrites any previously set flags. To combine them,
    /// use the bitwise or operator (`|`), for example:
    ///
    /// ```text
    /// ocl::Image::builder().flags(ocl::MEM_WRITE_ONLY | ocl::MEM_COPY_HOST_PTR)...
    /// ```
    ///
    /// Defaults to `core::MEM_READ_WRITE` if not set.

    pub fn flags(mut self, flags: MemFlags) -> ImageBuilder<'a, T> {
        self.flags = flags;
        self
    }

    /// A slice use to designate a region of memory for use in combination of
    /// one of the two following flags:
    ///
    /// * `flags::MEM_USE_HOST_PTR` aka. `MemFlags::new().use_host_ptr()`:
    ///   * This flag is valid only if `host_data` is not `None`. If
    ///     specified, it indicates that the application wants the OpenCL
    ///     implementation to use memory referenced by `host_data` as the
    ///     storage bits for the memory object (buffer/image).
    ///   * OpenCL implementations are allowed to cache the buffer contents
    ///     pointed to by `host_data` in device memory. This cached copy can
    ///     be used when kernels are executed on a device.
    ///   * The result of OpenCL commands that operate on multiple buffer
    ///     objects created with the same `host_data` or overlapping host
    ///     regions is considered to be undefined.
    ///   * Refer to the [description of the alignment][align_rules] rules for
    ///     `host_data` for memory objects (buffer and images) created using
    ///     `MEM_USE_HOST_PTR`.
    ///   * `MEM_ALLOC_HOST_PTR` and `MEM_USE_HOST_PTR` are mutually exclusive.
    ///
    /// * `flags::MEM_COPY_HOST_PTR` aka. `MemFlags::new().copy_host_ptr()`
    ///   * This flag is valid only if `host_data` is not NULL. If specified, it
    ///     indicates that the application wants the OpenCL implementation to
    ///     allocate memory for the memory object and copy the data from
    ///     memory referenced by `host_data`.
    ///   * CL_MEM_COPY_HOST_PTR and CL_MEM_USE_HOST_PTR are mutually
    ///     exclusive.
    ///   * CL_MEM_COPY_HOST_PTR can be used with CL_MEM_ALLOC_HOST_PTR to
    ///     initialize the contents of the cl_mem object allocated using
    ///     host-accessible (e.g. PCIe) memory.
    ///
    /// Note: Descriptions adapted from:
    /// [https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html][create_buffer].
    ///
    ///
    /// [align_rules]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/dataTypes.html
    /// [create_buffer]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
    pub fn host_data<'d>(mut self, host_data: &'d [T]) -> ImageBuilder<'a, T>
            where 'd: 'a {
        self.host_data = Some(host_data);
        self
    }

    pub fn channel_order(mut self, order: ImageChannelOrder) -> ImageBuilder<'a, T> {
        self.image_format.channel_order = order;
        self
    }

    pub fn channel_data_type(mut self, data_type: ImageChannelDataType) -> ImageBuilder<'a, T> {
        self.image_format.channel_data_type = data_type;
        self
    }

    /// Sets the type of image (technically the type of memory buffer).
    ///
    /// Describes the image type and must be either `Image1d`, `Image1dBuffer`,
    /// `Image1dArray`, `Image2d`, `Image2dArray`, or `Image3d`.
    ///
    pub fn image_type(mut self, image_type: MemObjectType) -> ImageBuilder<'a, T> {
        self.image_desc.image_type = image_type;
        self
    }

    /// The width, height, and depth of an image or image array:
    ///
    /// Some notes adapted from SDK docs:
    ///
    /// ## Width
    ///
    /// The width of the image in pixels. For a 2D image and image array, the
    /// image width must be ≤ `DeviceInfo::Image2dMaxWidth`. For a 3D image, the
    /// image width must be ≤ `DeviceInfo::Image3dMaxWidth`. For a 1D image buffer,
    /// the image width must be ≤ `DeviceInfo::ImageMaxBufferSize`. For a 1D image
    /// and 1D image array, the image width must be ≤ `DeviceInfo::Image2dMaxWidth`.
    ///
    /// ## Height
    ///
    /// The height of the image in pixels. This is only used if the
    /// image is a 2D, 3D or 2D image array. For a 2D image or image array, the
    /// image height must be ≤ `DeviceInfo::Image2dMaxHeight`. For a 3D image, the
    /// image height must be ≤ `DeviceInfo::Image3dMaxHeight`.
    ///
    /// ## Depth
    ///
    /// image_depth The depth of the image in pixels. This is only used if the
    /// image is a 3D image and must be a value ≥ 1 and ≤
    /// `DeviceInfo::Image3dMaxDepth`.
    ///
    /// ## Examples
    ///
    /// * To set the dimensions of a 2d image use:
    ///   `SpatialDims::Two(width, height)`.
    /// * To set the dimensions of a 2d image array use:
    ///   `SpatialDims::Three(width, height, array_length)`.
    /// * To set the dimensions of a 3d image use:
    ///   `SpatialDims::Three(width, height, depth)`.
    ///
    pub fn dims<D>(mut self, dims: D) -> ImageBuilder<'a, T>
            where D: Into<SpatialDims> {
        let dims = dims.into().to_lens().unwrap();
        self.image_desc.image_width = dims[0];
        self.image_desc.image_height = dims[1];
        self.image_desc.image_depth = dims[2];
        self
    }

    /// Image array size.
    ///
    /// The number of images in the image array. This is only used if the image is
    /// a 1D or 2D image array. The values for image_array_size, if specified,
    /// must be a value ≥ 1 and ≤ `DeviceInfo::ImageMaxArraySize`.
    ///
    /// Note that reading and writing 2D image arrays from a kernel with
    /// image_array_size = 1 may be lower performance than 2D images.
    ///
    pub fn array_size(mut self, array_size: usize) -> ImageBuilder<'a, T> {
        self.image_desc.image_array_size = array_size;
        self
    }

    /// Image row pitch.
    ///
    /// The scan-line pitch in bytes. This must be 0 if host data is `None` and
    /// can be either 0 or ≥ image_width * size of element in bytes if host data
    /// is not `None`. If host data is not `None` and image_row_pitch = 0,
    /// image_row_pitch is calculated as image_width * size of element in bytes.
    /// If image_row_pitch is not 0, it must be a multiple of the image element
    /// size in bytes.
    ///
    pub fn row_pitch_bytes(mut self, row_pitch: usize) -> ImageBuilder<'a, T> {
        self.image_desc.image_row_pitch = row_pitch;
        self
    }

    /// Image slice pitch.
    ///
    /// The size in bytes of each 2D slice in the 3D image or the size in bytes of
    /// each image in a 1D or 2D image array. This must be 0 if host data is
    /// `None`. If host data is not `None`, image_slice_pitch can be either 0 or ≥
    /// image_row_pitch * image_height for a 2D image array or 3D image and can be
    /// either 0 or ≥ image_row_pitch for a 1D image array. If host data is not
    /// `None` and image_slice_pitch = 0, image_slice_pitch is calculated as
    /// image_row_pitch * image_height for a 2D image array or 3D image and
    /// image_row_pitch for a 1D image array. If image_slice_pitch is not 0, it
    /// must be a multiple of the image_row_pitch.
    ///
    pub fn slc_pitch_bytes(mut self, slc_pitch: usize) -> ImageBuilder<'a, T> {
        self.image_desc.image_slice_pitch = slc_pitch;
        self
    }

    /// Buffer synchronization.
    ///
    /// Refers to a valid buffer memory object if image_type is
    /// `MemObjectType::Image1dBuffer`. Otherwise it must be `None` (default).
    /// For a 1D image buffer object, the image pixels are taken from the buffer
    /// object's data store. When the contents of a buffer object's data store are
    /// modified, those changes are reflected in the contents of the 1D image
    /// buffer object and vice-versa at corresponding sychronization points. The
    /// image_width * size of element in bytes must be ≤ size of buffer object
    /// data store.
    ///
    pub fn buffer_sync(mut self, buffer: MemCore) -> ImageBuilder<'a, T> {
        self.image_desc.buffer = Some(buffer);
        self
    }

    /// Specifies the image pixel format.
    ///
    /// If unspecified, defaults to:
    ///
    /// ```text
    /// ImageFormat {
    ///    channel_order: ImageChannelOrder::Rgba,
    ///    channel_data_type: ImageChannelDataType::SnormInt8,
    /// }
    /// ```
    pub fn image_format(mut self, image_format: ImageFormat) -> ImageBuilder<'a, T> {
        self.image_format = image_format;
        self
    }

    /// Specifies the image descriptor containing a number of important settings.
    ///
    /// If unspecified (not recommended), defaults to:
    ///
    /// ```text
    /// ImageDescriptor {
    ///     image_type: MemObjectType::Image1d,
    ///     image_width: 0,
    ///     image_height: 0,
    ///     image_depth: 0,
    ///     image_array_size: 0,
    ///     image_row_pitch: 0,
    ///     image_slice_pitch: 0,
    ///     num_mip_levels: 0,
    ///     num_samples: 0,
    ///     buffer: None,
    /// }
    /// ```
    ///
    /// If you are unsure, just set the first four by using
    /// `ImageDescriptor::new`. Ex.:
    ///
    /// ```text
    /// ocl::Image::builder()
    ///    .image_desc(ocl::ImageDescriptor::new(
    ///       ocl::MemObjectType::Image2d, 1280, 800, 1))
    ///    ...
    ///    ...
    ///    .build()
    /// ```
    ///
    /// Setting this overwrites any previously set type, dimensions, array size, pitch, etc.
    ///
    pub unsafe fn image_desc(mut self, image_desc: ImageDescriptor) -> ImageBuilder<'a, T> {
        self.image_desc = image_desc;
        self
    }

    /// Builds with no host side image data memory specified and returns a
    /// new `Image`.
    pub fn build(self) -> OclCoreResult<Image<T>> {
        match self.queue_option {
            Some(qo) => {
                Image::new(qo, self.flags, self.image_format.clone(),
                    self.image_desc.clone(), self.host_data)
            },
            None => panic!("ocl::ImageBuilder::build: A context or default queue must be set \
                with '.context(...)' or '.queue(...)'."),
        }

    }
}