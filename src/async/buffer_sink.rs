


/// Represents mapped memory and allows frames of data to be 'flushed'
/// (written) from host-accessible mapped memory region into its associated
/// device-visible `Buffer` in a repeated fashion.
///
/// This represents the absolute fastest method for writing data to an OpenCL
/// device.
pub struct BufferSink<T> {
    core: MemMapCore<T>,
}