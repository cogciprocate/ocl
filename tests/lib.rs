extern crate ocl_interop;
extern crate ocl;
extern crate gl;
extern crate glutin;

extern crate sdl2;
extern crate glfw;

use ocl_interop::get_properties_list;
use ocl::{ProQue, Context};
use gl::types::*;

//3 triangles, 3 Vertices per triangle, 2 floats per vertex
const BUFFER_LENGTH: usize = 18;
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 640;
const MAX_FRAME_COUNT: u32 = 300; //5 seconds

const KERNEL_SRC: &'static str = include_str!("kernels.cl");
const VERTEX_SRC: &'static str = include_str!("vertex.glsl");
const FRAGMENT_SRC: &'static str = include_str!("fragment.glsl");
//TODO:BetterName
trait TestContent{
    fn init(&mut self);
    fn render(&mut self);
    fn clean_up(&mut self);
}

struct CLGenVBO{
    gl_buff: GLuint,
    vao: GLuint,
    gl_program: GLuint,
}
impl CLGenVBO{
    fn new() -> CLGenVBO{
        return CLGenVBO{gl_buff:0, vao:0, gl_program: 0};
    }
}
impl TestContent for CLGenVBO{
    fn init(&mut self){
        unsafe {
            gl::Viewport(0, 0, 640, 480);
            //Create Program, create shaders, set source code, and compile.
            self.gl_program = gl::CreateProgram();
            let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
            let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            println!("PROGRAM: {} VERTEX: {} FRAG: {}",
                     self.gl_program,
                     vertex_shader,
                     fragment_shader);
            gl::ShaderSource(vertex_shader,
                             1,
                             &(VERTEX_SRC.as_ptr() as *const GLchar),
                             &(VERTEX_SRC.len() as GLint));
            gl::ShaderSource(fragment_shader,
                             1,
                             &(FRAGMENT_SRC.as_ptr() as *const GLchar),
                             &(FRAGMENT_SRC.len() as GLint));
            gl::CompileShader(vertex_shader);
            gl::CompileShader(fragment_shader);
            let mut gl_return: GLint = gl::FALSE as GLint;
            gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut gl_return);
            //TODO: Print shader logs
            if gl_return != gl::TRUE as GLint {
                panic!("Unable to compile vertex shader!");
            }
            gl_return = gl::FALSE as GLint;
            gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut gl_return);
            if gl_return != gl::TRUE as GLint {
                panic!("Unable to compile fragment shader!");
            }
            gl::AttachShader(self.gl_program, vertex_shader);
            gl::AttachShader(self.gl_program, fragment_shader);
            gl::LinkProgram(self.gl_program);
            gl::DetachShader(self.gl_program, vertex_shader);
          	gl::DetachShader(self.gl_program, fragment_shader);

          	gl::DeleteShader(vertex_shader);
          	gl::DeleteShader(fragment_shader);

            let gl_enum_err = gl::GetError();
            if gl_enum_err != gl::NO_ERROR {
                panic!("Couldn't link gl_program!")
            }
            gl::UseProgram(self.gl_program);

            //Create Buffer, bind buffer
            gl::GenBuffers(1, &mut self.gl_buff);
            gl::GenVertexArrays(1, &mut self.vao);
            gl::BindVertexArray(self.vao);

            assert!(self.gl_buff != 0, "GL Buffers are never 0");
            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            gl::BufferData(gl::ARRAY_BUFFER,
                           (BUFFER_LENGTH * std::mem::size_of::<f32>()) as isize,
                           std::ptr::null(),
                           gl::STATIC_DRAW);
            const SHADER_ATTRIBUTE: GLuint = 0;
            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            gl::VertexAttribPointer(SHADER_ATTRIBUTE,
                                    2,
                                    gl::FLOAT,
                                    gl::FALSE,
                                    0,
                                    std::ptr::null());
            gl::EnableVertexAttribArray(SHADER_ATTRIBUTE);


            gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
        }
        //Create an OpenCL context with the GL interop enabled
        let context=ocl::Platform::list().iter().map(|plat|{
          println!("Plat: {}",plat);
          ocl::Device::list(plat, Some(ocl::flags::DeviceType::new().gpu())).unwrap().iter().map(|dev|{
            let ctx = Context::builder()
                .properties(get_properties_list().platform(plat))
                .platform(*plat)
                .devices(dev)
                .build();
            println!("- Dev: {:?} Ctx: {:?}",dev,ctx);
            ctx
          }).find(|t|t.is_ok())
        }).find(|t|t.is_some()).expect("Cannot find GL's device in CL").unwrap().unwrap();

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
            gl::DeleteVertexArrays(1,&self.vao);
        }
    }
}

//Runs tests sequentially
#[test]
fn all_works(){
    //Red
    sdl2_works();
    //Green
    glutin_works();
    //Blue
    glfw_works();
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


        //Activate the window's context
        unsafe {
          gl_window.make_current().unwrap();
        }
        thing.render();

        gl_window.swap_buffers().unwrap();
        frame_count += 1;
    }
    gl_window.hide();
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
