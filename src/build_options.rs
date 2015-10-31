pub struct BuildOptions {
	options: Vec<BuildOption>,
	string: String,
	kernel_file_names: Vec<String>,
}

impl BuildOptions {
	pub fn new(cl_options: &'static str) -> BuildOptions {
		let bo = BuildOptions {
			options: Vec::with_capacity(1 << 5),
			string: String::with_capacity(1 << 11),
			kernel_file_names: Vec::with_capacity(20),
		};

		bo.str(cl_options)
	}

	fn str(mut self, st: &'static str) -> BuildOptions {
		self.string.push_str(st);
		self
	}

	pub fn opt(mut self, name: &'static str, val: i32) -> BuildOptions {
		self.options.push(BuildOption::new(name, val));
		self
	}

	pub fn add_opt(mut self, bo: BuildOption) -> BuildOptions {
		self.options.push(bo);
		self
	}

	pub fn kern_file(mut self, file_name: String) -> BuildOptions {
		self.add_kern_file(file_name);
		self
	}

	pub fn add_kern_file(&mut self, file_name: String) {
		self.kernel_file_names.push(file_name);
	}

	pub fn to_build_string(mut self) -> String {
		for option in self.options.iter() {
			if !option.val_is_string {
				self.string.push_str(&option.as_preprocessor_option_string());
			}
		}
		//println!("\n\tBuildOptions::as_slc(): length: {}, \n \tstring: {}", self.string.len(), self.string);
		self.string
	}

	pub fn cl_file_header(&self) -> Vec<u8> {
		let mut header = String::with_capacity(300);

		header.push_str("\n");

		for option in self.options.iter() {
			if option.val_is_string {
				header.push_str(&option.as_define_directive_string());
			}
		}

		header.push_str("\n");

		header.into_bytes()
	}

	pub fn kernel_file_names(&self) -> &Vec<String> {
		&self.kernel_file_names
	}

	// pub fn as_str(&mut self) -> &str {
	// 	&self.string
	// }
}



pub struct BuildOption {
	name: &'static str,
	val: String,
	//string: String,
	val_is_string: bool,
}

impl BuildOption {
	pub fn new(name: &'static str, val: i32) -> BuildOption {
		BuildOption {
			name: name,
			val: val.to_string(),
			//string: String::with_capacity(name.len()),
			val_is_string: false,
		}
	}

	pub fn with_str_val(name: &'static str, val: String) -> BuildOption {
		BuildOption {
			name: name,
			val: val,
			//string: String::with_capacity(name.len()),
			val_is_string: true,
		}
	}

	pub fn as_preprocessor_option_string(&self) -> String {
		//self.string = format!(" -D{}={}", self.name, self.val);

		// if self.val_is_string {
		// 	self.string = format!(" -D{}={}", self.name, self.val);
		// } else {
		// 	self.string = format!(" -D{}={}", self.name, self.val);
		// }

		//&self.string
		format!(" -D{}={}", self.name, self.val)
	}

	pub fn as_define_directive_string(&self) -> String {
		format!("#define {}  {}\n", self.name, self.val)
	}
}
