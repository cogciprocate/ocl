pub fn to_bytes(&self) -> Vec<isize> {
        // let mut bytes: Vec<u8> = Vec::with_capacity(8);
        let mut props: Vec<isize> = Vec::with_capacity(32);

        unsafe {
            // For each property:
            for prop in self.0.iter() {
                // Convert both the kind of property (a u32) and the value (variable type/size)
                // into just a core byte vector (Vec<u8>):
                let (kind: isize, val: isize) = match prop {
                    &ContextProperty::Platform(ref platform_id_core) => (
                        // util::into_bytes(PropKind::Platform as cl_h::cl_uint),
                        // util::into_bytes(platform_id_core.as_ptr() as cl_h::cl_platform_id)
                        PropKind::Platform
                    ),
                    &ContextProperty::InteropUserSync(sync) => (
                        // util::into_bytes(PropKind::InteropUserSync as cl_h::cl_uint),
                        // util::into_bytes(sync as cl_h::cl_bool)
                    ),
                    // &ContextProperty::GlContextKhr(ctx) => (
                    //     util::into_bytes(PropKind::GlContextKhr as cl_h::cl_uint),
                    //     util::into_bytes(sync as cl_h::cl_bool)
                    // ),
                    _ => continue,
                };

                // Property Kind Enum:
                bytes.extend_from_slice(&kind);
                // 32 bits of padding:
                bytes.extend_from_slice(&util::into_bytes(0 as u32));
                // Value:
                bytes.extend_from_slice(&val);
                // 32 bits of padding:
                bytes.extend_from_slice(&util::into_bytes(0 as u32));
            }

            // Add a terminating 0:
            bytes.extend_from_slice(&util::into_bytes(0 as usize));
        }

        bytes.shrink_to_fit();
        bytes
    }