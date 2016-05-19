
/// Context properties list.
///
/// [MINIMALLY TESTED]
///
/// TODO: Check for duplicate property assignments.
#[derive(Clone, Debug)]
pub struct ContextProperties(Vec<(ContextProperty, ContextPropertyValue)>);

impl ContextProperties {
    /// Returns an empty new list of context properties
    pub fn new() -> ContextProperties {
        ContextProperties(Vec::with_capacity(4))
    }

    /// Specifies a platform (builder-style).
    pub fn platform<'a, P: Into<PlatformId>>(&'a mut self, platform: P) -> &'a mut ContextProperties {
        self.0.push((ContextProperty::Platform, ContextPropertyValue::Platform(platform.into())));
        self
    }

    /// Specifies whether the user is responsible for synchronization between
    /// OpenCL and other APIs (builder-style).
    pub fn interop_user_sync<'a>(&'a mut self, sync: bool) -> &'a mut ContextProperties {
        self.0.push((ContextProperty::InteropUserSync, ContextPropertyValue::InteropUserSync(sync)));
        self
    }

    /// Specifies an OpenGL context handle.
    pub fn gl_context<'a>(&'a mut self, gl_ctx: ffi::cl_GLuint) -> &'a mut ContextProperties {
        self.0.push((ContextProperty::GlContextKhr, ContextPropertyValue::GlContextKhr(gl_ctx)));
        self
    }    

    /// Pushes a `ContextPropertyValue` onto this list of properties.
    pub fn prop<'a>(&'a mut self, prop: ContextPropertyValue) -> &'a mut ContextProperties {
        match prop {
            ContextPropertyValue::Platform(val) => 
                    self.0.push((ContextProperty::Platform, ContextPropertyValue::Platform(val))),
            ContextPropertyValue::InteropUserSync(val) => 
                    self.0.push((ContextProperty::InteropUserSync, 
                        ContextPropertyValue::InteropUserSync(val))),
            ContextPropertyValue::GlContextKhr(val) => 
                    self.0.push((ContextProperty::GlContextKhr, 
                        ContextPropertyValue::GlContextKhr(val))),
            _ => panic!("'{:?}' is not yet a supported variant.", prop),
        }        
        self
    }

    /// Returns a platform id or none.
    pub fn get_platform(&self) -> Option<PlatformId> {
        let mut platform = None;

        for prop in self.0.iter() {
            if let &ContextPropertyValue::Platform(ref plat) = &prop.1 {
                platform = Some(plat.clone());
            }
        }

        platform
    }

    /// Converts this list into a packed-word representation as specified
    /// [here](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateContext.html).
    ///
    // [NOTE]: Meant to replace `::to_bytes`.
    //
    // Return type is `Vec<cl_context_properties>` => `Vec<isize>`
    pub fn to_raw(&self) -> Vec<isize> {
        let mut props_raw = Vec::with_capacity(32);

        unsafe {
            // For each property ...
            for prop in self.0.iter() {
                // convert both the kind of property (a u32 originally) and
                // the value (variable type/size) to an isize:
                match &prop.1 {
                    &ContextPropertyValue::Platform(ref platform_id_core) => {
                        props_raw.push(prop.0 as isize);
                        props_raw.push(platform_id_core.as_ptr() as isize);
                    },
                    &ContextPropertyValue::InteropUserSync(sync) => {
                        props_raw.push(prop.0 as isize);
                        props_raw.push(sync as isize);
                    },
                    // &ContextPropertyValue::GlContextKhr(ctx) => (
                    //     util::into_bytes(PropKind::GlContextKhr as cl_h::cl_uint),
                    //     util::into_bytes(sync as cl_h::cl_bool)
                    // ),
                    _ => panic!("'{:?}' is not yet a supported variant.", prop.0),
                };
            }

            // Add a terminating 0:
            props_raw.push(0);
        }

        props_raw.shrink_to_fit();
        props_raw
    }

    // /// [UNTESTED: Not properly tested]
    // /// Converts this list into a packed-byte representation as specified
    // /// [here](https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clCreateContext.html).
    // ///
    // /// TODO: Evaluate cleaner ways to do this.
    // pub fn to_bytes(&self) -> Vec<u8> {
    //     let mut bytes = Vec::with_capacity(128);

    //     unsafe {
    //         // For each property:
    //         for prop in self.0.iter() {
    //             // Convert both the kind of property (a u32) and the value (variable type/size)
    //             // into just a core byte vector (Vec<u8>):
    //             let (kind, val) = match prop {
    //                 &ContextPropertyValue::Platform(ref platform_id_core) => (
    //                     util::into_bytes(PropKind::Platform as cl_h::cl_uint),
    //                     util::into_bytes(platform_id_core.as_ptr() as cl_h::cl_platform_id)
    //                 ),
    //                 &ContextPropertyValue::InteropUserSync(sync) => (
    //                     util::into_bytes(PropKind::InteropUserSync as cl_h::cl_uint),
    //                     util::into_bytes(sync as cl_h::cl_bool)
    //                 ),
    //                 // &ContextPropertyValue::GlContextKhr(ctx) => (
    //                 //     util::into_bytes(PropKind::GlContextKhr as cl_h::cl_uint),
    //                 //     util::into_bytes(sync as cl_h::cl_bool)
    //                 // ),
    //                 _ => continue,
    //             };

    //             // Property Kind Enum:
    //             bytes.extend_from_slice(&kind);
    //             // 32 bits of padding:
    //             bytes.extend_from_slice(&util::into_bytes(0 as u32));
    //             // Value:
    //             bytes.extend_from_slice(&val);
    //             // 32 bits of padding:
    //             bytes.extend_from_slice(&util::into_bytes(0 as u32));
    //         }

    //         // Add a terminating 0:
    //         bytes.extend_from_slice(&util::into_bytes(0 as usize));
    //     }

    //     bytes.shrink_to_fit();
    //     bytes
    // }
}

impl Into<Vec<(ContextProperty, ContextPropertyValue)>> for ContextProperties {
    fn into(self) -> Vec<(ContextProperty, ContextPropertyValue)> {
        self.0
    }
}

// impl Into<Vec<u8>> for ContextProperties {
//     fn into(self) -> Vec<u8> {
//         self.to_bytes()
//     }
// }

impl Into<Vec<isize>> for ContextProperties {
    fn into(self) -> Vec<isize> {
        self.to_raw()
    }
}