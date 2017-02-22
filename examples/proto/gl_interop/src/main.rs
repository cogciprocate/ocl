//! This example is meant to demonstrate and test interoprability between
//! OpenCL and OpenGL.
//!
//! Work in progress
//!

extern crate ocl;
#[macro_use] extern crate glium;
use ocl::{Context, SpatialDims};
use ocl::enums::ContextPropertyValue;
use glium::{glutin, DisplayBuild, Surface};

const DIMS: [usize; 3] = [16, 16, 16];

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
implement_vertex!(Vertex, position);


fn main() {
    // ########## OpenGL Stuff ###########

    let display = glutin::WindowBuilder::new().build().unwrap();

    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [ 0.0,  0.5] };
    let vertex3 = Vertex { position: [ 0.5, -0.25] };
    let shape = vec![vertex1, vertex2, vertex3];

    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();


    // ########## OpenCL Stuff ###########

    let src = r#"
        __kernel void add(__global float* buffer, float scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    let context = Context::builder()
        .property(ContextPropertyValue::GlContextKhr(___))
        .gl_context(___)
        .build().unwrap();

    let dims: SpatialDims = DIMS.into();

    // let buffer = pro_que.create_buffer::<f32>().unwrap();

    // let kernel = pro_que.create_kernel("add").unwrap()
    //     .arg_buf(&buffer)
    //     .arg_scl(10.0f32);

    // kernel.enq().unwrap();

    let mut vec = vec![0.0f32; dims.to_len()];
    // buffer.read(&mut vec).enq().unwrap();

    // println!("The value at index [{}] is now '{}'!", 200007, vec[200007]);
}
