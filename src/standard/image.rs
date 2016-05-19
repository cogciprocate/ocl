//! An OpenCL Image.
//!

use std;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use std::convert::Into;
use error::{Error as OclError, Result as OclResult};
use core::{self, OclPrm, Mem as MemCore, MemFlags, MemObjectType, ImageFormat, ImageDescriptor,
    ImageInfo, ImageInfoResult, MemInfo, MemInfoResult, ClEventPtrNew, ClWaitList,
    ImageChannelOrder, ImageChannelDataType, GlTextureTarget};
use standard::{Context, Queue, MemLen, SpatialDims};
use ffi::{cl_GLuint, cl_GLint};

/// A builder for `Image`.
pub struct ImageBuilder<S: OclPrm> {
    flags: MemFlags,
    image_format: ImageFormat,
    image_desc: ImageDescriptor,
    _pixel: PhantomData<S>,
    // image_data: Option<&'a [S]>,
}

impl<S: OclPrm> ImageBuilder<S> {
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
    pub fn new() -> ImageBuilder<S> {
        ImageBuilder {
            flags: core::MEM_READ_WRITE,
            image_format: ImageFormat::new_rgba(),
            image_desc: ImageDescriptor::new(MemObjectType::Image1d, 0, 0, 0, 0, 0, 0, None),
            _pixel: PhantomData,
            // image_data: None,
        }
    }

    /// Builds with no host side image data memory specified and returns a
    /// new `Image`.
    pub fn build(&self, queue: &Queue) -> OclResult<Image<S>> {
        Image::new(queue, self.flags, self.image_format.clone(), self.image_desc.clone(),
            None)
    }

    /// Builds with the host side image data specified by `image_data`
    /// and returns a new `Image`.
    ///
    /// Useful with the `ocl::MEM_COPY_HOST_PTR` flag for initializing device
    /// memory by copying the contents of `image_data`.
    ///
    /// Also used with the `ocl::MEM_USE_HOST_PTR` and `ocl::ALLOC_HOST_PTR`
    /// flags. See the [official SDK docs] for more info.
    ///
    /// [official SDK docs]: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateImage.html
    pub fn build_with_data(&self, queue: &Queue, image_data: &[S]) -> OclResult<Image<S>> {
        Image::new(queue, self.flags, self.image_format.clone(), self.image_desc.clone(),
            Some(image_data))
    }

    pub fn channel_order(&mut self, order: ImageChannelOrder) -> &mut ImageBuilder<S> {
        self.image_format.channel_order = order;
        self
    }

    pub fn channel_data_type(&mut self, data_type: ImageChannelDataType) -> &mut ImageBuilder<S> {
        self.image_format.channel_data_type = data_type;
        self
    }


    /// Sets the type of image (technically the type of memory buffer).
    ///
    /// Describes the image type and must be either `Image1d`, `Image1dBuffer`,
    /// `Image1dArray`, `Image2d`, `Image2dArray`, or `Image3d`.
    ///
    pub fn image_type(&mut self, image_type: MemObjectType) -> &mut ImageBuilder<S> {
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
    pub fn dims<D: MemLen>(&mut self, dims: D) -> &mut ImageBuilder<S> {
        let dims = dims.to_lens();
        // let size = dims.to_lens().expect(&format!("ocl::ImageBuilder::dims(): Invalid image \
     //        dimensions: {:?}", dims));
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
    pub fn array_size(&mut self, array_size: usize) -> &mut ImageBuilder<S> {
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
    pub fn row_pitch_bytes(&mut self, row_pitch: usize) -> &mut ImageBuilder<S> {
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
    pub fn slc_pitch_bytes(&mut self, slc_pitch: usize) -> &mut ImageBuilder<S> {
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
    pub fn buffer_sync(&mut self, buffer: MemCore) -> &mut ImageBuilder<S> {
        self.image_desc.buffer = Some(buffer);
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
    pub fn flags(&mut self, flags: MemFlags) -> &mut ImageBuilder<S> {
        self.flags = flags;
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
    pub fn image_format(&mut self, image_format: ImageFormat) -> &mut ImageBuilder<S> {
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
    pub unsafe fn image_desc(&mut self, image_desc: ImageDescriptor) -> &mut ImageBuilder<S> {
        self.image_desc = image_desc;
        self
    }
}

/// The type of operation to be performed by a command.
#[derive(Debug)]
pub enum ImageCmdKind<'b, E: 'b> {
    Unspecified,
    Read { data: &'b mut [E] },
    Write { data: &'b [E] },
    Fill { color: E },
    Copy { dst_image: &'b MemCore, dst_origin: [usize; 3] },
    CopyToBuffer { buffer: &'b MemCore, dst_origin: usize },
    GLAcquire,
    GLRelease,
}

impl<'b, E: 'b> ImageCmdKind<'b, E> {
    fn is_unspec(&'b self) -> bool {
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
#[allow(dead_code)]
pub struct ImageCmd<'b, E: 'b + OclPrm> {
    queue: &'b Queue,
    obj_core: &'b MemCore,
    block: bool,
    lock_block: bool,
    origin: [usize; 3],
    region: [usize; 3],
    row_pitch: usize,
    slc_pitch: usize,
    kind: ImageCmdKind<'b, E>,
    ewait: Option<&'b ClWaitList>,
    enew: Option<&'b mut ClEventPtrNew>,
    mem_dims: [usize; 3],
}

/// [UNSTABLE]: All methods still in a state of adjustifulsomeness.
impl<'b, E: 'b + OclPrm> ImageCmd<'b, E> {
    /// Returns a new image command builder associated with with the
    /// memory object `obj_core` along with a default `queue` and `to_len`
    /// (the length of the device side image).
    pub fn new(queue: &'b Queue, obj_core: &'b MemCore, dims: [usize; 3])
            -> ImageCmd<'b, E>
    {
        ImageCmd {
            queue: queue,
            obj_core: obj_core,
            block: true,
            lock_block: false,
            origin: [0, 0, 0],
            region: dims,
            row_pitch: 0,
            slc_pitch: 0,
            kind: ImageCmdKind::Unspecified,
            ewait: None,
            enew: None,
            mem_dims: dims,
        }
    }

    /// Specifies a queue to use for this call only.
    pub fn queue(mut self, queue: &'b Queue) -> ImageCmd<'b, E> {
        self.queue = queue;
        self
    }

    /// Specifies whether or not to block thread until completion.
    ///
    /// Ignored if this is a copy, fill, or copy to image operation.
    ///
    /// ## Panics
    ///
    /// Will panic if `::read` has already been called. Use `::read_async`
    /// (unsafe) for a non-blocking read operation.
    ///
    pub fn block(mut self, block: bool) -> ImageCmd<'b, E> {
        if !block && self.lock_block {
            panic!("ocl::ImageCmd::block(): Blocking for this command has been disabled by \
                the '::read' method. For non-blocking reads use '::read_async'.");
        }
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
    pub fn origin(mut self, origin: [usize; 3]) -> ImageCmd<'b, E> {
        self.origin = origin;
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
    pub fn region(mut self, region: [usize; 3]) -> ImageCmd<'b, E> {
        self.region = region;
        self
    }

    /// [UNSTABLE] Sets the row and slice pitch for a read or write operation.
    ///
    /// `row_pitch`: Must be greater than or equal to the region width
    /// (region[0]).
    ///
    /// `slice_pitch: Must be greater than or equal to `row_pitch` * region
    /// height (region[1]).
    ///
    /// Only needs to be set if region has been set to something other than
    /// the (default) image buffer size.
    ///
    /// ## Stability
    ///
    /// Probably will be depricated unless I can think of a reason why you'd
    /// set the pitches to something other than the image dims.
    ///
    /// [FIXME]: Remove this or figure out if it's necessary at all.
    ///
    #[allow(unused_variables, unused_mut)]
    pub unsafe fn pitch(mut self, row_pitch: usize, slc_pitch: usize) -> ImageCmd<'b, E> {
        unimplemented!();
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
    pub fn read(mut self, dst_data: &'b mut [E]) -> ImageCmd<'b, E> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::read(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Read { data: dst_data };
        self.block = true;
        self.lock_block = true;
        self
    }

    /// Specifies that this command will be a non-blocking, asynchronous read
    /// operation.
    ///
    /// Sets the block mode to false automatically but it may still be freely
    /// toggled back. If set back to `true` this method call becomes equivalent
    /// to calling `::read`.
    ///
    /// ## Safety
    ///
    /// Caller must ensure that the container referred to by `dst_data` lives
    /// until the call completes.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub unsafe fn read_async(mut self, dst_data: &'b mut [E]) -> ImageCmd<'b, E> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::read(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Read { data: dst_data };
        self
    }

    /// Specifies that this command will be a write operation.
    ///
    /// ## Panics
    ///
    /// The command operation kind must not have already been specified
    ///
    pub fn write(mut self, src_data: &'b [E]) -> ImageCmd<'b, E> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::write(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Write { data: src_data };
        self
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
    pub fn copy(mut self, dst_image: &'b Image<E>, dst_origin: [usize; 3]) -> ImageCmd<'b, E> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::copy(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Copy {
            dst_image: dst_image.core_as_ref(),
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
    pub fn copy_to_buffer(mut self, buffer: &'b MemCore, dst_origin: usize) -> ImageCmd<'b, E> {
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
    pub fn gl_acquire(mut self) -> ImageCmd<'b, E> {
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
    pub fn gl_release(mut self) -> ImageCmd<'b, E> {
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
    pub fn fill(mut self, color: E) -> ImageCmd<'b, E> {
        assert!(self.kind.is_unspec(), "ocl::ImageCmd::fill(): Operation kind \
            already set for this command.");
        self.kind = ImageCmdKind::Fill { color: color };
        self
    }

    /// Specifies a list of events to wait on before the command will run.
    pub fn ewait(mut self, ewait: &'b ClWaitList) -> ImageCmd<'b, E> {
        self.ewait = Some(ewait);
        self
    }

    /// Specifies a list of events to wait on before the command will run or
    /// resets it to `None`.
    pub fn ewait_opt(mut self, ewait: Option<&'b ClWaitList>) -> ImageCmd<'b, E> {
        self.ewait = ewait;
        self
    }

    /// Specifies the destination for a new, optionally created event
    /// associated with this command.
    pub fn enew(mut self, enew: &'b mut ClEventPtrNew) -> ImageCmd<'b, E> {
        self.enew = Some(enew);
        self
    }

    /// Specifies a destination for a new, optionally created event
    /// associated with this command or resets it to `None`.
    pub fn enew_opt(mut self, enew: Option<&'b mut ClEventPtrNew>) -> ImageCmd<'b, E> {
        self.enew = enew;
        self
    }

    /// Enqueues this command.
    ///
    /// TODO: FOR COPY, FILL, AND COPYTOBUFFER -- ENSURE PITCHES ARE BOTH UNSET.
    pub fn enq(self) -> OclResult<()> {
        match self.kind {
            ImageCmdKind::Read { data } => {
                // try!(check_len(self.to_len, data.len(), offset));
                unsafe { core::enqueue_read_image(self.queue, self.obj_core, self.block,
                    self.origin, self.region, self.row_pitch, self.slc_pitch, data, self.ewait,
                    self.enew) }
            },
            ImageCmdKind::Write { data } => {
                core::enqueue_write_image(self.queue, self.obj_core, self.block,
                    self.origin, self.region, self.row_pitch, self.slc_pitch, data, self.ewait,
                    self.enew)
            },
            ImageCmdKind::Copy { dst_image, dst_origin } => {
                core::enqueue_copy_image::<E>(self.queue, self.obj_core, dst_image, self.origin,
                    dst_origin, self.region, self.ewait, self.enew)
            },
            ImageCmdKind::GLAcquire => {
                core::enqueue_acquire_gl_buffer(self.queue, self.obj_core, self.ewait, self.enew)
            },
            ImageCmdKind::GLRelease => {
                core::enqueue_release_gl_buffer(self.queue, self.obj_core, self.ewait, self.enew)
            },
            ImageCmdKind::Unspecified => OclError::err("ocl::ImageCmd::enq(): No operation \
                specified. Use '.read(...)', 'write(...)', etc. before calling '.enq()'."),
            _ => unimplemented!(),
        }
    }
}

/// A section of device memory which represents one or many images.
///
/// Use `::builder` for an easy way to create. [UNIMPLEMENTED]
///
#[derive(Clone, Debug)]
pub struct Image<E: OclPrm> {
    obj_core: MemCore,
    queue: Queue,
    dims: SpatialDims,
    pixel_element_len: usize,
    _pixel: PhantomData<E>
}

impl<E: OclPrm> Image<E> {
    /// Returns a list of supported image formats.
    pub fn supported_formats(context: &Context, flags: MemFlags, mem_obj_type: MemObjectType,
                ) -> OclResult<Vec<ImageFormat>> {
        core::get_supported_image_formats(context, flags, mem_obj_type)
    }

    /// Returns an `ImageBuilder`. This is the recommended method to create
    /// a new `Image`.
    pub fn builder() -> ImageBuilder<E> {
        ImageBuilder::new()
    }

    /// Returns a new `Image`.
    ///
    /// Prefer `::builder` to create a new image.
    pub fn new(queue: &Queue, flags: MemFlags, image_format: ImageFormat,
            image_desc: ImageDescriptor, image_data: Option<&[E]>) -> OclResult<Image<E>>
    {
        let obj_core = unsafe { try!(core::create_image(
            queue.context_core_as_ref(),
            flags,
            &image_format,
            &image_desc,
            image_data,
        )) };

        let pixel_element_len = match core::get_image_info(&obj_core, ImageInfo::ElementSize) {
            ImageInfoResult::ElementSize(s) => s / mem::size_of::<E>(),
            ImageInfoResult::Error(err) => return Err(*err),
            _ => return OclError::err("ocl::Image::element_len(): \
                Unexpected 'ImageInfoResult' variant."),
        };

        let dims = [image_desc.image_width, image_desc.image_height, image_desc.image_depth].into();

        let new_img = Image {
            obj_core: obj_core,
            queue: queue.clone(),
            dims: dims,
            pixel_element_len: pixel_element_len,
            _pixel: PhantomData,
        };

        Ok(new_img)
    }

    /// Returns a new `Image` from an existant GL texture2D/3D.
    // [WORK IN PROGRESS]
    pub fn from_gl_texture(queue: &Queue, flags: MemFlags, image_desc: ImageDescriptor,
            texture_target: GlTextureTarget, miplevel: cl_GLint, texture: cl_GLuint)
            -> OclResult<Image<E>>
    {
        if texture_target == GlTextureTarget::GlTextureBuffer && miplevel != 0 {
            return OclError::err("If texture_target is GL_TEXTURE_BUFFER, miplevel must be 0.\
                Implementations may return CL_INVALID_OPERATION for miplevel values > 0");
        }

        // let obj_core = match image_desc.image_depth {
        //     2 => unsafe { try!(core::create_from_gl_texture_2d(
        //                         queue.context_core_as_ref(),
        //                         texture_target,
        //                         miplevel,
        //                         texture,
        //                         flags)) },
        //     3 => unsafe { try!(core::create_from_gl_texture_3d(
        //                         queue.context_core_as_ref(),
        //                         texture_target,
        //                         miplevel,
        //                         texture,
        //                         flags)) },
        //     _ => unimplemented!() // FIXME: return an error ? or panic! ?
        // };
        let obj_core = unsafe { try!(core::create_from_gl_texture(
                                        queue.context_core_as_ref(),
                                        texture_target as u32,
                                        miplevel,
                                        texture,
                                        flags)) };

        // FIXME can I do this from a GLTexture ?
        let pixel_element_len = match core::get_image_info(&obj_core, ImageInfo::ElementSize) {
            ImageInfoResult::ElementSize(s) => s / mem::size_of::<E>(),
            ImageInfoResult::Error(err) => return Err(*err),
            _ => return OclError::err("ocl::Image::element_len(): \
                Unexpected 'ImageInfoResult' variant."),
        };

        let dims = [image_desc.image_width, image_desc.image_height, image_desc.image_depth].into();

        let new_img = Image {
            obj_core: obj_core,
            queue: queue.clone(),
            dims: dims,
            pixel_element_len: pixel_element_len,
            _pixel: PhantomData,
        };

        Ok(new_img)
    }

    /// Returns a new `Image` from an existant renderbuffer.
    // [WORK IN PROGRESS]
    pub fn from_gl_renderbuffer(queue: &Queue, flags: MemFlags, image_desc: ImageDescriptor,
            renderbuffer: cl_GLuint) -> OclResult<Image<E>>
    {
        let obj_core = unsafe { try!(core::create_from_gl_renderbuffer(
                                        queue.context_core_as_ref(),
                                        renderbuffer,
                                        flags)) };

        // FIXME can I do this from a renderbuffer ?
        let pixel_element_len = match core::get_image_info(&obj_core, ImageInfo::ElementSize) {
            ImageInfoResult::ElementSize(s) => s / mem::size_of::<E>(),
            ImageInfoResult::Error(err) => return Err(*err),
            _ => return OclError::err("ocl::Image::element_len(): \
                Unexpected 'ImageInfoResult' variant."),
        };

        let dims = [image_desc.image_width, image_desc.image_height].into();

        let new_img = Image {
            obj_core: obj_core,
            queue: queue.clone(),
            dims: dims,
            pixel_element_len: pixel_element_len,
            _pixel: PhantomData,
        };

        Ok(new_img)
    }

    /// Returns an image command builder used to read, write, copy, etc.
    ///
    /// Run `.enq()` to enqueue the command.
    ///
    pub fn cmd<'b>(&self) -> ImageCmd<E> {
        ImageCmd::new(&self.queue, &self.obj_core,
            self.dims.to_lens().expect("ocl::Image::cmd"))
    }

    /// Returns an image command builder set to read.
    ///
    /// Run `.enq()` to enqueue the command.
    ///
    pub fn read<'b>(&'b self, data: &'b mut [E]) -> ImageCmd<'b, E> {
        self.cmd().read(data)
    }

    /// Returns an image command builder set to write.
    ///
    /// Run `.enq()` to enqueue the command.
    ///
    pub fn write<'b>(&'b self, data: &'b [E]) -> ImageCmd<'b, E> {
        self.cmd().write(data)
    }

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
    pub fn set_default_queue<'a>(&'a mut self, queue: &Queue) -> &'a mut Image<E> {
        // self.command_queue_obj_core = queue.core_as_ref().clone();
        self.queue = queue.clone();
        self
    }

    /// Returns a reference to the default queue.
    pub fn default_queue(&self) -> &Queue {
        &self.queue
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
    pub fn info(&self, info_kind: ImageInfo) -> ImageInfoResult {
        // match core::get_image_info(&self.obj_core, info_kind) {
        //     Ok(res) => res,
        //     Err(err) => ImageInfoResult::Error(Box::new(err)),
        // }
        core::get_image_info(&self.obj_core, info_kind)
    }

    /// Returns info about this image's memory.
    pub fn mem_info(&self, info_kind: MemInfo) -> MemInfoResult {
        // match core::get_mem_object_info(&self.obj_core, info_kind) {
        //     Ok(res) => res,
        //     Err(err) => MemInfoResult::Error(Box::new(err)),
        // }
        core::get_mem_object_info(&self.obj_core, info_kind)
    }

    /// Returns a reference to the core pointer wrapper, usable by functions in
    /// the `core` module.
    pub fn core_as_ref(&self) -> &MemCore {
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

impl<E: OclPrm> std::fmt::Display for Image<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        try!(self.fmt_info(f));
        try!(write!(f, " "));
        self.fmt_mem_info(f)
    }
}

impl<E: OclPrm> Deref for Image<E> {
    type Target = MemCore;

    fn deref(&self) -> &MemCore {
        &self.obj_core
    }
}

impl<E: OclPrm> DerefMut for Image<E> {
    fn deref_mut(&mut self) -> &mut MemCore {
        &mut self.obj_core
    }
}



    // /// Reads from the device image buffer into `data`.
    // ///
    // /// Setting `queue` to `None` will use the default queue set during creation.
    // /// Otherwise, the queue passed will be used for this call only.
    // ///
    // /// ## Safety
    // ///
    // /// Caller must ensure that `data` lives until the read is complete. Use
    // /// the new event in `dest_list` to monitor it (use [`EventList::last_clone`]).
    // ///
    // ///
    // /// See the [SDK docs](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueReadImage.html)
    // /// for more detailed information.
    // /// [`EventList::get_clone`]: /ocl/ocl/struct.EventList.html#method.last_clone
    // ///
    // pub unsafe fn enqueue_read(&self, queue: Option<&Queue>, block: bool, origin: [usize; 3],
    //             region: [usize; 3], row_pitch: usize, slc_pitch: usize, data: &mut [E],
    //             wait_list: Option<&EventList>, dest_list: Option<&mut ClEventPtrNew>) -> OclResult<()>
    // {
    //     let command_queue = match queue {
    //         Some(q) => q,
    //         None => &self.queue,
    //     };

    //     core::enqueue_read_image(command_queue, &self.obj_core, block, origin, region,
    //         row_pitch, slc_pitch, data, wait_list, dest_list)
    // }

    // /// Writes from `data` to the device image buffer.
    // ///
    // /// Setting `queue` to `None` will use the default queue set during creation.
    // /// Otherwise, the queue passed will be used for this call only.
    // ///
    // /// See the [SDK docs](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clEnqueueWriteImage.html)
    // /// for more detailed information.
    // pub fn enqueue_write(&self, queue: Option<&Queue>, block: bool, origin: [usize; 3],
    //             region: [usize; 3], row_pitch: usize, slc_pitch: usize, data: &[E],
    //             wait_list: Option<&EventList>, dest_list: Option<&mut ClEventPtrNew>) -> OclResult<()>
    // {
    //     let command_queue = match queue {
    //         Some(q) => q,
    //         None => &self.queue,
    //     };

    //     core::enqueue_write_image(command_queue, &self.obj_core, block, origin, region,
    //         row_pitch, slc_pitch, data, wait_list, dest_list)
    // }

    // /// Reads the entire device image buffer into `data`, blocking until complete.
    // ///
    // /// `data` must be equal to the size of the device image buffer and must be
    // /// alligned without pitch or offset of any kind.
    // ///
    // /// Use `::enqueue_read` for the complete range of options.
    // pub fn read_old(&self, data: &mut [E]) -> OclResult<()> {
    //     // Safe because `block = true`:
    //     unsafe { self.enqueue_read(None, true, [0, 0, 0], try!(self.dims.to_lens()), 0, 0,  data, None, None) }
    // }

    // /// Writes from `data` to the device image buffer, blocking until complete.
    // ///
    // /// `data` must be equal to the size of the device image buffer and must be
    // /// alligned without pitch or offset of any kind.
    // ///
    // /// Use `::enqueue_write` for the complete range of options.
    // pub fn write_old(&self, data: &[E]) -> OclResult<()> {
    //     self.enqueue_write(None, true, [0, 0, 0], try!(self.dims.to_lens()), 0, 0,  data, None, None)
    // }
