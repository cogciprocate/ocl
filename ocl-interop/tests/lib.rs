extern crate gl;
extern crate glutin;
extern crate ocl;
extern crate ocl_interop;
extern crate glfw;
extern crate sdl2;

use ocl::ProQue;
use gl::types::*;

// 3 triangles, 3 Vertices per triangle, 2 floats per vertex
const BUFFER_LENGTH: usize = 18;
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 640;
const MAX_FRAME_COUNT: u32 = 300; // 5 seconds

const KERNEL_SRC: &'static str = include_str!("kernels.cl");
const VERTEX_SRC: &'static str = include_str!("vertex.glsl");
const FRAGMENT_SRC: &'static str = include_str!("fragment.glsl");

// TODO:BetterName
trait TestContent {
    fn init(&mut self);
    fn render(&mut self);
    fn clean_up(&mut self);
}

struct CLGenVBO {
    gl_buff: GLuint,
    vao: GLuint,
    gl_program: GLuint,
}

impl CLGenVBO {
    fn new() -> CLGenVBO {
        return CLGenVBO {
            gl_buff: 0,
            vao: 0,
            gl_program: 0,
        };
    }
}

impl TestContent for CLGenVBO {
    fn init(&mut self) {
        unsafe {
            gl::Viewport(0, 0, 640, 480);
            // Create Program, create shaders, set source code, and compile.
            self.gl_program = gl::CreateProgram();
            let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
            let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);

            println!(
                "PROGRAM: {} VERTEX: {} FRAG: {}",
                self.gl_program, vertex_shader, fragment_shader
            );

            gl::ShaderSource(
                vertex_shader,
                1,
                &(VERTEX_SRC.as_ptr() as *const GLchar),
                &(VERTEX_SRC.len() as GLint),
            );

            gl::ShaderSource(
                fragment_shader,
                1,
                &(FRAGMENT_SRC.as_ptr() as *const GLchar),
                &(FRAGMENT_SRC.len() as GLint),
            );

            gl::CompileShader(vertex_shader);
            gl::CompileShader(fragment_shader);
            let mut gl_return: GLint = gl::FALSE as GLint;
            gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut gl_return);

            // TODO: Print shader logs

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

            // Create Buffer, bind buffer
            gl::GenBuffers(1, &mut self.gl_buff);
            gl::GenVertexArrays(1, &mut self.vao);
            gl::BindVertexArray(self.vao);

            assert!(self.gl_buff != 0, "GL Buffers are never 0");
            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (BUFFER_LENGTH * std::mem::size_of::<f32>()) as isize,
                std::ptr::null(),
                gl::STATIC_DRAW,
            );

            const SHADER_ATTRIBUTE: GLuint = 0;

            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            gl::VertexAttribPointer(
                SHADER_ATTRIBUTE,
                2,
                gl::FLOAT,
                gl::FALSE,
                0,
                std::ptr::null(),
            );
            gl::EnableVertexAttribArray(SHADER_ATTRIBUTE);

            gl::PolygonMode(gl::FRONT_AND_BACK, gl::FILL);
        }

        // Create an OpenCL context with the GL interop enabled
        let context = ocl_interop::get_context().expect("Cannot find GL's device in CL");

        // Create a big ball of OpenCL-ness (see ProQue and ProQueBuilder docs for info):
        let ocl_pq = ProQue::builder()
            .context(context)
            .src(KERNEL_SRC)
            .build()
            .expect("Build ProQue");
        let cl_buff =
            ocl::Buffer::<f32>::from_gl_buffer(ocl_pq.queue(), None, self.gl_buff).unwrap();

        // Create a kernel with arguments corresponding to those in the kernel:
        let kern = ocl_pq
            .kernel_builder("fill_vbo")
            .global_work_size(BUFFER_LENGTH)
            .arg(&cl_buff)
            .build()
            .unwrap();

        // get GL Objects
        let mut acquire_globj_event: ocl::Event = ocl::Event::empty();
        cl_buff.cmd()
            .gl_acquire()
            .enew(&mut acquire_globj_event)
            .enq()
            .unwrap();

        // Enqueue kernel:
        let mut kernel_run_event: ocl::Event = ocl::Event::empty();
        unsafe {
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

        assert!(
            (BUFFER_LENGTH * std::mem::size_of::<f32>())
                == std::mem::size_of::<[f32; BUFFER_LENGTH]>()
        );

        // Read results from the device into result_buffer's local vector:
        // result_buffer.read(&mut vec_result).enq().unwrap();
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

        // Release GL OBJs
        cl_buff.cmd()
            .gl_release()
            // .ewait(&kernel_run_event)
            .ewait(&read_buffer_event)
            .enq()
            .unwrap();

        // Finish OpenCL Queue before starting to use the gl buffer in the main loop
        ocl_pq.queue().finish().unwrap();
    }
    fn render(&mut self) {
        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::EnableVertexAttribArray(0);
            gl::BindBuffer(gl::ARRAY_BUFFER, self.gl_buff);
            gl::DrawArrays(gl::TRIANGLES, 0, BUFFER_LENGTH as GLint / 2);
            gl::DisableVertexAttribArray(0);
        }
    }
    fn clean_up(&mut self) {
        // TODO: Clean up more memory
        unsafe {
            gl::DeleteProgram(self.gl_program);
            gl::DeleteVertexArrays(1, &self.vao);
        }
    }
}

// Runs tests sequentially
#[test]
fn all_works() {
    // Red
    sdl2_works();
    // Blue
    glfw_works();
    // Green
    glutin_works(); // needs to be last because the event loop never returns
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

    let thing: &mut dyn TestContent = &mut CLGenVBO::new();
    // let thing:&mut TestContent=&mut CLMultiplyByScalar::new();

    gl::load_with(|name| video_subsystem.gl_get_proc_address(name) as *const _);
    canvas.window().gl_make_current(&gl_context).unwrap();
    canvas.window().gl_set_context_to_current().unwrap();

    let mut event_pump = sdl_context.event_pump().unwrap();
    thing.init();
    // canvas.set_draw_color(sdl2::pixels::Color::RGB(255, 41, 0));
    let mut frame_count = 0;

    'running: while frame_count < MAX_FRAME_COUNT {
        // println!("{}",frame_count);
        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. }
                | sdl2::event::Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        unsafe {
            // canvas.clear();
            canvas.window().gl_make_current(&gl_context).unwrap();
            // SDL resets clear color
            gl::ClearColor(1.0, 0.16, 0.0, 1.0);
            thing.render();
        }

        canvas.present();

        frame_count += 1;
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    thing.clean_up();
}

fn glutin_works() {
    use glutin::event::{Event, WindowEvent};
    use glutin::event_loop::{ControlFlow, EventLoopBuilder};
    use glutin::window::WindowBuilder;
    use glutin::ContextBuilder;
    use glutin::dpi::*;

    #[cfg(target_os = "windows")]
    use glutin::platform::windows::EventLoopBuilderExtWindows;
    #[cfg(target_os = "linux")]
    use glutin::platform::unix::EventLoopBuilderExtUnix;

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    let events_loop = EventLoopBuilder::new().with_any_thread(true).build();

    #[cfg(target_os = "macos")]
    let events_loop = EventLoopBuilder::new().build();

    let window = WindowBuilder::new()
        .with_title("Glutin Window")
        .with_inner_size(PhysicalSize { width: WINDOW_WIDTH, height: WINDOW_HEIGHT });
    let gl_window = ContextBuilder::new().with_vsync(true).build_windowed(window, &events_loop).unwrap();

    let mut thing = CLGenVBO::new();

    // Activate the window's context
    let gl_window = unsafe { gl_window.make_current().unwrap() };

    gl::load_with(|symbol| gl_window.context().get_proc_address(symbol) as *const _);
    unsafe { gl::ClearColor(0.0, 1.0, 0.333, 1.0); }

    thing.init();

    let mut frame_count = 0;
    events_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => { *control_flow = ControlFlow::Exit; },
                WindowEvent::Resized(physical_size) => gl_window.resize(physical_size),
                _ => (),
            },
            Event::RedrawRequested(_) => {
                thing.render();
                gl_window.swap_buffers().unwrap();
                frame_count += 1;
                if frame_count >= MAX_FRAME_COUNT {
                    *control_flow = ControlFlow::Exit;
                }
            },
            Event::MainEventsCleared => {
                gl_window.window().request_redraw();
            },
            Event::LoopDestroyed => {
                thing.clean_up();
            },
            _ => (),
        }
    });
}

fn glfw_works() {
    use glfw::{Action, Key};
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    // Create a windowed mode window and its OpenGL context
    let (mut window, events) = glfw.create_window(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        "GLFW Window",
        glfw::WindowMode::Windowed,
    ).expect("Failed to create GLFW window.");

    gl::load_with(|name| window.get_proc_address(name) as *const _);
    glfw::Context::make_current(&mut window);
    window.set_key_polling(true);

    let thing: &mut dyn TestContent = &mut CLGenVBO::new();

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
        std::thread::sleep(std::time::Duration::from_millis(5));
        frame_count += 1;
    }

    thing.clean_up();
    // GLFW Window is closed when it goes out of scope.
}
