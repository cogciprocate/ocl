


/// Represents mapped memory and allows frames of data to be 'flooded' (read)
/// from a device-visible `Buffer` into its associated host-accessible mapped
/// memory region in a repeated fashion.
///
/// This represents the absolute fastest method for reading data from an
/// OpenCL device.
pub struct BufferStream<T> {
    core: MemMapCore<T>,
}