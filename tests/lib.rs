extern crate ocl_interop;
extern crate ocl;
extern crate gl;
#[macro_use]
extern crate glium;

extern crate sdl2;
extern crate glfw;
extern crate glutin;

use ocl_interop::get_properties_list;
use ocl::{util, ProQue, Buffer, MemFlags, Context};
use gl::types::*;

// Number of results to print out:
const RESULTS_TO_PRINT: usize = 20;

//3 triangles, 3 Vertices per triangle, 2 floats per vertex
const BUFFER_LENGTH: usize = 18;
const COEFF: f32 = 5432.1;
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 640;
const MAX_FRAME_COUNT: u32 = 600; //5 seconds

const KERNEL_SRC: &'static str = include_str!("kernels.cl");
const VERTEX_SRC: &'static str = include_str!("vertex.glsl");
const FRAGMENT_SRC: &'static str = include_str!("fragment.glsl");
//TODO:BetterName
trait TestContent{
    fn init(&mut self);
    fn render(&mut self);
    fn clean_up(&mut self);
}

struct CLMultiplyByScalar{
    gl_buff: GLuint,
}
impl CLMultiplyByScalar{
    fn new() -> CLMultiplyByScalar{
        return CLMultiplyByScalar{gl_buff:0};
    }
}
impl TestContent for CLMultiplyByScalar{
    fn init(&mut self){
        unsafe {
            gl::GenBuffers(1, &mut self.gl_buff);
            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            gl::BufferData(gl::ARRAY_BUFFER,
                           (BUFFER_LENGTH * std::mem::size_of::<f32>()) as isize,
                           std::ptr::null(),
                           gl::STATIC_DRAW);
        }
        //Create an OpenCL context with the GL interop enabled
        let context=ocl::Platform::list().iter().map(|plat|{
          ocl::Device::list(plat, Some(ocl::flags::DeviceType::new().gpu())).iter().map(|dev|{
            Context::builder()
                .properties(get_properties_list().platform(plat))
                .platform(*plat)
                .devices(dev)
                .build()
          }).find(|t|t.is_ok())
        }).find(|t|t.is_some()).expect("Cannot find GL's device in CL").unwrap().unwrap();
        //let context = Context::builder()
        //    .properties(get_properties_list())
        //    .build();
	/*println!("WOOO HEEE WE GONNA CRASH BOII");
        let context=ocl::Platform::list().iter().map(|plat|{
            let properties=get_properties_list().platform(plat);
            println!("Platform: {:?}",plat);
            // CRASH! get_gl_context_info_khr
            match ocl::core::get_gl_context_info_khr(&properties, ocl::core::GlContextInfo::CurrentDevice){
                ocl::core::GlContextInfoResult::CurrentDevice(dev)=>{
                    println!(" Device: {:?}",dev);
                    Some(Context::builder()
                        .properties(properties)
                        .devices(ocl::Device::from(dev))
                        .platform(*plat)
                        .build()
                        .unwrap())
                },
                ocl::core::GlContextInfoResult::Error(err)=>{
                    println!("Unable to get CL device to match GL context {}",err);
                    None
                },
                res=>{
                    panic!("Unexpected result {}",res);
                }
            }}).find(|t|t.is_some()).expect("Cannot find GL's device in CL").unwrap();
*/
        /*let properties=get_properties_list();
        match ocl::core::get_gl_context_info_khr(&properties, ocl::core::GlContextInfo::CurrentDevice){
            ocl::core::GlContextInfoResult::CurrentDevice(dev)=>{
                match ocl::core::get_device_info(dev, ocl::core::DeviceInfo::Platform){
                    ocl::core::DeviceInfoResult::Platform(plat)=>{
                                context=Context::builder()
                                    .properties(properties)
                                    .devices(ocl::Device::from(dev))
                                    .platform(ocl::Platform::from(plat))
                                    .build()
                                    .unwrap();
                    }
                    ocl::core::DeviceInfoResult::Error(err)=>{panic!("Unable to get CL platform to match GL device {}",err)}
                    _=>{panic!("Unexpected error")}
                }
            }
            ocl::core::GlContextInfoResult::Error(err)=>{panic!("Unable to get CL device to match GL context {}",err)}
            _=>{panic!("Unexpected error")}
        }*/

        // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
        let ocl_pq = ProQue::builder()
            .context(context)
            .src(KERNEL_SRC)
            .dims(BUFFER_LENGTH)
            .build()
            .expect("Build ProQue");
        let cl_buff : ocl::Buffer<f32> = ocl::Buffer::from_gl_buffer(ocl_pq.queue(), None, self.gl_buff)
            .unwrap();

        // Create a temporary init vector and the source buffer. Initialize them
        // with random floats between 0.0 and 20.0:
        let vec_source = util::scrambled_vec((0.0, 20.0), ocl_pq.dims().to_len());
        let source_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_write().copy_host_ptr())
            .dims(ocl_pq.dims().clone())
            .host_data(&vec_source)
            .build()
            .unwrap();



        // Create a kernel with arguments corresponding to those in the kernel:
        let kern = ocl_pq
            .create_kernel("multiply_by_scalar")
            .unwrap()
            .arg_scl(COEFF)
            .arg_buf(&source_buffer)
            .arg_buf(&cl_buff);

        println!("Kernel global work size: {:?}", kern.get_gws());

        //get GL Objects
        let mut acquire_globj_event: ocl::Event = ocl::Event::empty();
        ocl::builders::BufferCmd::<f32>::new(&cl_buff, Some(ocl_pq.queue()), BUFFER_LENGTH)
            .gl_acquire()
            .enew(&mut acquire_globj_event)
            .enq()
            .unwrap();


        // Enqueue kernel:
        let mut kernel_run_event: ocl::Event = ocl::Event::empty();
        unsafe{
        kern.cmd()
            .enew(&mut kernel_run_event)
            .ewait(&acquire_globj_event)
            .enq()
            .unwrap();
}


        // Create an empty vec and buffer (the quick way) for results. Note that
        // there is no need to initialize the buffer as we did above because we
        // will be writing to the entire buffer first thing, overwriting any junk
        // data that may be there.
        let mut vec_result = vec![0.0f32; BUFFER_LENGTH];
        assert!((BUFFER_LENGTH * std::mem::size_of::<f32>()) ==
                std::mem::size_of::<[f32; BUFFER_LENGTH]>());
        // Read results from the device into result_buffer's local vector:
        //result_buffer.read(&mut vec_result).enq().unwrap();
        let mut read_buffer_event: ocl::Event = ocl::Event::empty();
        cl_buff
            .read(&mut vec_result)
            .queue(ocl_pq.queue())
            .enew(&mut read_buffer_event)
            .ewait(&kernel_run_event)
            .enq()
            .unwrap();

        ocl::builders::BufferCmd::<f32>::new(&cl_buff, Some(ocl_pq.queue()), BUFFER_LENGTH)
            .gl_release()
            .ewait(&read_buffer_event)
            .enq()
            .unwrap();

        // Check results and print the first 20:
        for idx in 0..BUFFER_LENGTH {
            if idx < RESULTS_TO_PRINT {
                println!("source[{idx}]: {:.03}, \t coeff: {}, \tresult[{idx}]: {}",
                         vec_source[idx],
                         COEFF,
                         vec_result[idx],
                         idx = idx);
            }
            assert_eq!(vec_source[idx] * COEFF, vec_result[idx]);
        }
    }
    fn render(&mut self){
        unsafe{
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
    }
    fn clean_up(&mut self){
    }
}

struct CLGenVBO{
    gl_buff: GLuint,
    gl_program: GLuint,
    vertex_shader: GLuint,
    fragment_shader: GLuint,
}
impl CLGenVBO{
    fn new() -> CLGenVBO{
        return CLGenVBO{gl_buff:0,gl_program:0,vertex_shader:0,fragment_shader:0};
    }
}
impl TestContent for CLGenVBO{
    fn init(&mut self){
        //laziness
        unsafe {
            gl::Viewport(0, 0, 640, 480);
            //Create Program, create shaders, set source code, and compile.
            self.gl_program = gl::CreateProgram();
            self.vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
            self.fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            println!("PROGRAM: {} VERTEX: {} FRAG: {}",
                     self.gl_program,
                     self.vertex_shader,
                     self.fragment_shader);
            gl::ShaderSource(self.vertex_shader,
                             1,
                             &(VERTEX_SRC.as_ptr() as *const GLchar),
                             &(VERTEX_SRC.len() as GLint));
            gl::ShaderSource(self.fragment_shader,
                             1,
                             &(FRAGMENT_SRC.as_ptr() as *const GLchar),
                             &(FRAGMENT_SRC.len() as GLint));
            gl::CompileShader(self.vertex_shader);
            gl::CompileShader(self.fragment_shader);
            let mut gl_return: GLint = gl::FALSE as GLint;
            gl::GetShaderiv(self.vertex_shader, gl::COMPILE_STATUS, &mut gl_return);
            //TODO: Print shader logs
            if gl_return != gl::TRUE as GLint {
                panic!("Unable to compile vertex shader!");
            }
            gl_return = gl::FALSE as GLint;
            gl::GetShaderiv(self.fragment_shader, gl::COMPILE_STATUS, &mut gl_return);
            if gl_return != gl::TRUE as GLint {
                panic!("Unable to compile fragment shader!");
            }
            gl::AttachShader(self.gl_program, self.vertex_shader);
            gl::AttachShader(self.gl_program, self.fragment_shader);
            gl::LinkProgram(self.gl_program);

            let gl_enum_err = gl::GetError();
            if gl_enum_err != gl::NO_ERROR {
                panic!("Couldn't link gl_program!")
            }
            gl::UseProgram(self.gl_program);

            //Create Buffer, bind buffer
            gl::GenBuffers(1, &mut self.gl_buff);
            assert!(self.gl_buff != 0, "GL Buffers are never 0");
            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            gl::BufferData(gl::ARRAY_BUFFER,
                           (BUFFER_LENGTH * std::mem::size_of::<f32>()) as isize,
                           std::ptr::null(),
                           gl::STATIC_DRAW);
            println!("this line is fine {}", gl::GetError());
            const SHADER_ATTRIBUTE: GLuint = 0;
            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            println!("this line should be okay {}", gl::GetError());
            gl::VertexAttribPointer(SHADER_ATTRIBUTE,
                                    2,
                                    gl::FLOAT,
                                    gl::FALSE,
                                    0,
                                    std::ptr::null());
            println!("this line used to get an INVALID_OPERATION error [1282]. (#{})",
                     gl::GetError());
            gl::EnableVertexAttribArray(SHADER_ATTRIBUTE);
            println!("this line might be okay {}", gl::GetError());


            gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
        }
        //Create an OpenCL context with the GL interop enabled
        let context=ocl::Platform::list().iter().map(|plat|{
          println!("Plat: {}",plat);
          ocl::Device::list(plat, Some(ocl::flags::DeviceType::new().gpu())).iter().map(|dev|{
            let ctx = Context::builder()
                .properties(get_properties_list().platform(plat))
                .platform(*plat)
                .devices(dev)
                .build();
            println!("- Dev: {:?} Ctx: {:?}",dev,ctx);
            ctx
          }).find(|t|t.is_ok())
        }).find(|t|t.is_some()).expect("Cannot find GL's device in CL").unwrap().unwrap();

        //let context = Context::builder()
        //    .properties(get_properties_list())
        //    .build()
        //    .unwrap();
        // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
        let ocl_pq = ProQue::builder()
            .context(context)
            .src(KERNEL_SRC)
            .build()
            .expect("Build ProQue");
        let cl_buff = ocl::Buffer::<f32>::from_gl_buffer(ocl_pq.queue(), None, self.gl_buff)
            .unwrap();

        // Create a kernel with arguments corresponding to those in the kernel:
        let kern = ocl_pq
            .create_kernel("fill_vbo")
            .unwrap()
            .arg_buf(&cl_buff)
            .gws(BUFFER_LENGTH);

        //get GL Objects
        let mut acquire_globj_event: ocl::Event = ocl::Event::empty();
        ocl::builders::BufferCmd::<f32>::new(&cl_buff, Some(ocl_pq.queue()), BUFFER_LENGTH)
            .gl_acquire()
            .enew(&mut acquire_globj_event)
            .enq()
            .unwrap();


        // Enqueue kernel:
        let mut kernel_run_event: ocl::Event = ocl::Event::empty();
        unsafe{
        kern.cmd()
            .enew(&mut kernel_run_event)
            .ewait(&acquire_globj_event)
            .enq()
            .unwrap();
        }

        // Create an empty vec and buffer (the quick way) for results. Note that
        // there is no need to initialize the buffer as we did above because we
        // will be writing to the entire buffer first thing, overwriting any junk
        // data that may be there.
        let mut vec_result = vec![0.0f32; BUFFER_LENGTH];
        assert!((BUFFER_LENGTH * std::mem::size_of::<f32>()) ==
                std::mem::size_of::<[f32; BUFFER_LENGTH]>());
        // Read results from the device into result_buffer's local vector:
        //result_buffer.read(&mut vec_result).enq().unwrap();
        let mut read_buffer_event: ocl::Event = ocl::Event::empty();
        unsafe {
            cl_buff
                .read(&mut vec_result)
                .block(false)
                .queue(ocl_pq.queue())
                .enew(&mut read_buffer_event)
                .ewait(&kernel_run_event)
                .enq()
                .unwrap();
        }
        //Release GL OBJs
        ocl::builders::BufferCmd::<f32>::new(&cl_buff, Some(ocl_pq.queue()), BUFFER_LENGTH)
            .gl_release()
            //.ewait(&kernel_run_event)
            .ewait(&read_buffer_event)
            .enq()
            .unwrap();
        //Finish OpenCL Queue before starting to use the gl buffer in the main loop
        ocl_pq.queue().finish().unwrap();
    }
    fn render(&mut self){
        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::EnableVertexAttribArray(0);
            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            gl::DrawArrays(gl::TRIANGLES, 0, BUFFER_LENGTH as GLint / 2);
            gl::DisableVertexAttribArray(0);
        }
    }
    fn clean_up(&mut self){
        //TODO: Clean up more memory
        unsafe {
            gl::DeleteProgram(self.gl_program);
        }
    }
}

//Runs tests sequentially
#[test]
fn all_works(){
    glfw_works();
    glutin_works();
    sdl2_works();
}
fn sdl2_works() {
    use sdl2::keyboard::Keycode;

    fn find_sdl_gl_driver() -> Option<u32> {
        for (index, item) in sdl2::render::drivers().enumerate() {
            if item.name == "opengl" {
                return Some(index as u32);
            }
        }
        None
    }
    let sdl_context = sdl2::init().unwrap();

    let video_subsystem = sdl_context.video().unwrap();


    let window = video_subsystem
        .window("SDL2 Window", 800, 600)
        .position_centered()
        .opengl()
        .build()
        .unwrap();

    let mut canvas = window
        .into_canvas()
        .index(find_sdl_gl_driver().unwrap())
        .build()
        .expect("Couldn't make Canvas!");

    let gl_context = canvas.window().gl_create_context().unwrap();

    let thing:&mut TestContent=&mut CLGenVBO::new();
    //let thing:&mut TestContent=&mut CLMultiplyByScalar::new();

    gl::load_with(|name| video_subsystem.gl_get_proc_address(name) as *const _);
    canvas.window().gl_make_current(&gl_context).unwrap();
    canvas.window().gl_set_context_to_current().unwrap();

    let mut event_pump = sdl_context.event_pump().unwrap();
    thing.init();
    //canvas.set_draw_color(sdl2::pixels::Color::RGB(255, 41, 0));
    let mut frame_count = 0;
    'running: while frame_count < MAX_FRAME_COUNT {
        //println!("{}",frame_count);
        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. } |
                sdl2::event::Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running
                }
                _ => {}
            }
        }

        unsafe {
            //canvas.clear();
            canvas.window().gl_make_current(&gl_context).unwrap();
            //SDL resets clear color
            gl::ClearColor(1.0, 0.16, 0.0, 1.0);
            thing.render();
        }

        canvas.present();

        frame_count += 1;
        std::thread::sleep(std::time::Duration::from_millis(1000/60));
    }
    thing.clean_up();
}
fn glfw_works() {
    use glfw::{Action, Key};
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    // Create a windowed mode window and its OpenGL context
    let (mut window, events) = glfw.create_window(WINDOW_WIDTH,
                                                  WINDOW_HEIGHT,
                                                  "GLFW Window",
                                                  glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    gl::load_with(|name| window.get_proc_address(name) as *const _);
    glfw::Context::make_current(&mut window);
    window.set_key_polling(true);

    let thing:&mut TestContent=&mut CLGenVBO::new();
    unsafe {
        gl::ClearColor(0.0, 0.5, 1.0, 1.0);
        gl::Viewport(0, 0, 640, 480);
    }
    thing.init();

    let mut frame_count = 0;
    while !window.should_close() && frame_count < MAX_FRAME_COUNT {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            println!("{:?}", event);
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true)
                }
                _ => {}
            }
        }

        thing.render();

        glfw::Context::swap_buffers(&mut window);
        frame_count += 1;
    }
    thing.clean_up();
    //GLFW Window is closed when it goes out of scope.
}

fn glutin_works() {
    use glutin::GlContext;
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Glutin Window")
        .with_dimensions(WINDOW_WIDTH, WINDOW_HEIGHT);
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();
    //let thing:&mut TestContent=&mut CLGenVBO::new();
    let thing:&mut TestContent=&mut CLGenVBO::new();

    unsafe {
        //Activate the window's context
        gl_window.make_current().unwrap();

        //Load GL Functions
        gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);
        gl::ClearColor(0.0, 1.0, 0.333, 1.0);
    }
    thing.init();

    let mut frame_count = 0;
    let mut running = true;
    while running && frame_count < MAX_FRAME_COUNT {
        events_loop.poll_events(|event| match event {
                                    glutin::Event::WindowEvent { event, .. } => {
                                        match event {
                                            glutin::WindowEvent::Closed => running = false,
                                            glutin::WindowEvent::Resized(w, h) => {
                                                gl_window.resize(w, h)
                                            }
                                            _ => (),
                                        }
                                    }
                                    _ => (),
                                });


        thing.render();

        gl_window.swap_buffers().unwrap();
        frame_count += 1;
    }
    gl_window.hide();
    thing.clean_up();
}

fn glium_works(){

    use glium::Surface;

    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Glutin Window with glium");
    let context = glutin::ContextBuilder::new();
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
    }

    implement_vertex!(Vertex, position);

    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [ 0.0,  0.5] };
    let vertex3 = Vertex { position: [ 0.5, -0.25] };
    let shape = vec![vertex1, vertex2, vertex3];

    let vertex_buffer = glium::VertexBuffer::new(&display, &shape).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let vertex_shader_src = r#"
        #version 140

        in vec2 position;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        out vec4 color;

        void main() {
            color = vec4(0.0, 0.0, 0.0, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let mut running = true;
    let mut frame_count=0;
    while running && frame_count < MAX_FRAME_COUNT {
        let mut target = display.draw();
        target.clear_color(0.33333, 0.0, 1.0, 1.0);
        target.draw(&vertex_buffer, &indices, &program, &glium::uniforms::EmptyUniforms,
                    &Default::default()).unwrap();
        target.finish().unwrap();

        events_loop.poll_events(|event| {
            match event {
                glutin::Event::WindowEvent { event, .. } => match event {
                    glutin::WindowEvent::Closed => running = false,
                    _ => ()
                },
                _ => (),
            }
        });
        frame_count+=1;
    }
}
