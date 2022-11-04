#include "GLViewer.hpp"
#include <random>


#if defined(_DEBUG) && defined(_WIN32)
//#error "This sample should not be built in Debug mode, use RelWithDebInfo if you want to do step by step."
#endif

#define FADED_RENDERING
const float grid_size = 10.0f;

GLchar* VERTEX_SHADER =
"#version 330 core\n"
"layout(location = 0) in vec3 in_Vertex;\n"
"layout(location = 1) in vec4 in_Color;\n"
"uniform mat4 u_mvpMatrix;\n"
"out vec4 b_color;\n"
"void main() {\n"
"   b_color = in_Color;\n"
"	gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
"}";

GLchar* FRAGMENT_SHADER =
"#version 330 core\n"
"in vec4 b_color;\n"
"layout(location = 0) out vec4 out_Color;\n"
"void main() {\n"
"   out_Color = b_color;\n"
"}";

void addVert(Simple3DObject &obj, float i_f, float limit, float height, sl::float4 &clr) {
	auto p1 = sl::float3(i_f, height, -limit);
	auto p2 = sl::float3(i_f, height, limit);
	auto p3 = sl::float3(-limit, height, i_f);
	auto p4 = sl::float3(limit, height, i_f);

	obj.addLine(p1, p2, clr);
	obj.addLine(p3, p4, clr);
}

GLViewer* currentInstance_ = nullptr;

GLViewer::GLViewer() : available(false) {
	currentInstance_ = this;
	mouseButton_[0] = mouseButton_[1] = mouseButton_[2] = false;
	clearInputs();
	previousMouseMotion_[0] = previousMouseMotion_[1] = 0;
}

GLViewer::~GLViewer() {}

void GLViewer::exit() {
	if (available) {
		available = false;
		pointCloud_.close();
		image_handler.close();
	}
}

bool GLViewer::isAvailable() {
	if (available)
		glutMainLoopEvent();
	return available;
}

Simple3DObject createFrustum(sl::CameraParameters param) {

	// Create 3D axis
	Simple3DObject it(sl::Translation(0, 0, 0), true);

	float Z_ = -150;
	sl::float3 cam_0(0, 0, 0);
	sl::float3 cam_1, cam_2, cam_3, cam_4;

	float fx_ = 1.f / param.fx;
	float fy_ = 1.f / param.fy;

	cam_1.z = Z_;
	cam_1.x = (0 - param.cx) * Z_ *fx_;
	cam_1.y = (0 - param.cy) * Z_ *fy_;

	cam_2.z = Z_;
	cam_2.x = (param.image_size.width - param.cx) * Z_ *fx_;
	cam_2.y = (0 - param.cy) * Z_ *fy_;

	cam_3.z = Z_;
	cam_3.x = (param.image_size.width - param.cx) * Z_ *fx_;
	cam_3.y = (param.image_size.height - param.cy) * Z_ *fy_;

	cam_4.z = Z_;
	cam_4.x = (0 - param.cx) * Z_ *fx_;
	cam_4.y = (param.image_size.height - param.cy) * Z_ *fy_;

	sl::float4 clr(0.8f, 0.5f, 0.2f, 1.0f);
	it.addTriangle(cam_0, cam_1, cam_2, clr);
	it.addTriangle(cam_0, cam_2, cam_3, clr);
	it.addTriangle(cam_0, cam_3, cam_4, clr);
	it.addTriangle(cam_0, cam_4, cam_1, clr);

	it.setDrawingType(GL_TRIANGLES);
	return it;
}

void CloseFunc(void) {
	if (currentInstance_)
		currentInstance_->exit();
}

void GLViewer::init(int argc, char **argv, sl::CameraParameters &param) {
	glutInit(&argc, argv);
	int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
	int wnd_h = glutGet(GLUT_SCREEN_HEIGHT);

	glutInitWindowSize(1200, 700);
	glutInitWindowPosition(wnd_w * 0.05, wnd_h * 0.05);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow("ZED Object Detection");

	GLenum err = glewInit();
	if (GLEW_OK != err)
		std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	pointCloud_.initialize(param.image_size);


	bool status_ = image_handler.initialize(param.image_size);
	if (!status_)
		std::cout << "ERROR: Failed to initialized Image Renderer" << std::endl;

	image_handler.reshape(wnd_w, wnd_h);



	floorFind = false;

	// Compile and create the shader
	shader.it = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
	shader.MVP_Mat = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

	shaderLine.it = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
	shaderLine.MVP_Mat = glGetUniformLocation(shaderLine.it.getProgramId(), "u_mvpMatrix");

	// Create the camera
	camera_ = CameraGL(sl::Translation(0, 0, 1000), sl::Translation(0, 0, -100));

	frustum = createFrustum(param);
	frustum.pushToGPU();

	bckgrnd_clr = sl::float4(0.2f, 0.19f, 0.2f, 1.0f);

	floor_grid = Simple3DObject(sl::Translation(0, 0, 0), true);
	floor_grid.setDrawingType(GL_LINES);

	float limit = 20.f;
	sl::float4 clr_grid(80, 80, 80, 255);
	clr_grid /= 255.f;
	for (int i = (int)(-limit); i <= (int)(limit); i++)
		addVert(floor_grid, i * 1000, limit * 1000,0, clr_grid);
	floor_grid.pushToGPU();

	world_ref = Simple3DObject(sl::Translation(0, 0, 0), true);
	world_ref.setDrawingType(GL_LINES);
	sl::float3 w_ref(0, 0, 0);
	world_ref.addLine(w_ref, sl::float3(1 * 500, 0, 0), sl::float4(0.8, 0.1, 0.1, 1));
	world_ref.addLine(w_ref, sl::float3(0, 1 * 500, 0), sl::float4(0.1, 0.8, 0.1, 1));
	world_ref.addLine(w_ref, sl::float3(0, 0, 1 * 500), sl::float4(0.1, 0.1, 0.8, 1));
	world_ref.pushToGPU();

	// Map glut function on this class methods
	glutDisplayFunc(GLViewer::drawCallback);
	glutMouseFunc(GLViewer::mouseButtonCallback);
	glutMotionFunc(GLViewer::mouseMotionCallback);
	glutReshapeFunc(GLViewer::reshapeCallback);
	glutKeyboardFunc(GLViewer::keyPressedCallback);
	glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
	glutCloseFunc(CloseFunc);

	available = true;
}

void GLViewer::render() {
	if (available) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(bckgrnd_clr.b, bckgrnd_clr.g, bckgrnd_clr.r, bckgrnd_clr.a);
		update();
		draw();
		printText();
		glutSwapBuffers();
		glutPostRedisplay();
	}
}

void GLViewer::updateData(sl::Mat &matXYZRGBA, sl::Transform& pose) {
	const std::lock_guard<std::mutex> lock(mtx);
	pointCloud_.pushNewPC(matXYZRGBA);
	cam_pose = pose;
	floorFind = true;
}

void GLViewer::updateImage(sl::Mat& image) {
	image_handler.pushNewImage(image);
}

void GLViewer::update() {
	if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP) {
		currentInstance_->exit();
		return;
	}

	if (keyStates_['r'] == KEY_STATE::UP || keyStates_['R'] == KEY_STATE::UP) {
		camera_.setPosition(sl::Translation(0.0f, 0.0f, 1500.0f));
		//camera_.setOffsetFromPosition(sl::Translation(0.0f, 0.0f, 1500.0f));
		camera_.setDirection(sl::Translation(0.0f, 0.0f, 1.0f), sl::Translation(0.0f, 1.0f, 0.0f));
	}

	if (keyStates_['t'] == KEY_STATE::UP || keyStates_['T'] == KEY_STATE::UP) {
		camera_.setPosition(sl::Translation(0.0f, 0.0f, 1500.0f));
		camera_.setOffsetFromPosition(sl::Translation(0.0f, 0.0f, 6000.0f));
		camera_.translate(sl::Translation(0.0f, 1500.0f, -4000.0f));
		camera_.setDirection(sl::Translation(0.0f, -1.0f, 0.0f), sl::Translation(0.0f, 1.0f, 0.0f));
	}

	// Rotate camera with mouse
	if (mouseButton_[MOUSE_BUTTON::LEFT]) {
		camera_.rotate(sl::Rotation((float)mouseMotion_[1] * MOUSE_R_SENSITIVITY, camera_.getRight()));
		camera_.rotate(sl::Rotation((float)mouseMotion_[0] * MOUSE_R_SENSITIVITY, camera_.getVertical() * -1.f));
	}

	// Translate camera with mouse
	if (mouseButton_[MOUSE_BUTTON::RIGHT]) {
		camera_.translate(camera_.getUp() * (float)mouseMotion_[1] * MOUSE_T_SENSITIVITY * 1000);
		camera_.translate(camera_.getRight() * (float)mouseMotion_[0] * MOUSE_T_SENSITIVITY * 1000);
	}

	// Zoom in with mouse wheel
	if (mouseWheelPosition_ != 0) {
		//float distance = sl::Translation(camera_.getOffsetFromPosition()).norm();
		if (mouseWheelPosition_ > 0 /* && distance > camera_.getZNear()*/) { // zoom
			camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * 1000 * -1);
		}
		else if (/*distance < camera_.getZFar()*/ mouseWheelPosition_ < 0) {// unzoom
			//camera_.setOffsetFromPosition(camera_.getOffsetFromPosition() * MOUSE_DZ_SENSITIVITY);
			camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * 1000);
		}
	}

	camera_.update();
	mtx.lock();
	// Update point cloud buffers
	pointCloud_.update();
	mtx.unlock();
	clearInputs();
}

void GLViewer::draw() {
	glEnable(GL_DEPTH_TEST);
	sl::Transform vpMatrix = camera_.getViewProjectionMatrix();
	glUseProgram(shaderLine.it.getProgramId());
	glUniformMatrix4fv(shaderLine.MVP_Mat, 1, GL_TRUE, vpMatrix.m);
	glLineWidth(1.f);
	floor_grid.draw();
	world_ref.draw();
	glUseProgram(0);

	if (floorFind) {
		glPointSize(1.f);

		pointCloud_.draw(vpMatrix);

		glUseProgram(shader.it.getProgramId());
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		// Apply IMU Rotation compensation
		vpMatrix = vpMatrix * cam_pose;

		glUniformMatrix4fv(shader.MVP_Mat, 1, GL_TRUE, vpMatrix.m);
		glLineWidth(2.f);
		frustum.draw();
		glUseProgram(0);
	}

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	image_handler.draw();
}

sl::float2 compute3Dprojection(sl::float3 &pt, const sl::Transform &cam, sl::Resolution wnd_size) {
	sl::float4 pt4d(pt.x, pt.y, pt.z, 1.);
	auto proj3D_cam = pt4d * cam;
	proj3D_cam.y += 1000.f;
	sl::float2 proj2D;
	proj2D.x = ((proj3D_cam.x / pt4d.w) * wnd_size.width) / (2.f * proj3D_cam.w) + wnd_size.width / 2.f;
	proj2D.y = ((proj3D_cam.y / pt4d.w) * wnd_size.height) / (2.f * proj3D_cam.w) + wnd_size.height / 2.f;
	return proj2D;
}

static void safe_glutBitmapString(void* font, const char* str) {
	for (size_t x = 0; x < strlen(str); ++x)
		glutBitmapCharacter(font, str[x]);
}

void GLViewer::printText() {
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	int w_wnd = glutGet(GLUT_WINDOW_WIDTH);
	int h_wnd = glutGet(GLUT_WINDOW_HEIGHT);
	glOrtho(0, w_wnd, 0, h_wnd, -1.0f, 1.0f);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	int start_w = 20;
	int start_h = h_wnd - 40;

	floorFind ? glColor3f(0.2f, 0.65f, 0.2f) : glColor3f(0.85f, 0.2f, 0.2f);
	glRasterPos2i(start_w, start_h);
	std::string _str = floorFind ? std::string("FLOOR PLANE FOUND, WORLD REFERENCE IS SET") : std::string("LOOKING FOR FLOOR PLANE, move around");
	safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, _str.c_str());

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void GLViewer::clearInputs() {
	mouseMotion_[0] = mouseMotion_[1] = 0;
	mouseWheelPosition_ = 0;
	for (unsigned int i = 0; i < 256; ++i)
		if (keyStates_[i] != KEY_STATE::DOWN)
			keyStates_[i] = KEY_STATE::FREE;
}

void GLViewer::drawCallback() {
	currentInstance_->render();
}

void GLViewer::mouseButtonCallback(int button, int state, int x, int y) {
	if (button < 5) {
		if (button < 3) {
			currentInstance_->mouseButton_[button] = state == GLUT_DOWN;
		}
		else {
			currentInstance_->mouseWheelPosition_ += button == MOUSE_BUTTON::WHEEL_UP ? 1 : -1;
		}
		currentInstance_->mouseCurrentPosition_[0] = x;
		currentInstance_->mouseCurrentPosition_[1] = y;
		currentInstance_->previousMouseMotion_[0] = x;
		currentInstance_->previousMouseMotion_[1] = y;
	}
}

void GLViewer::mouseMotionCallback(int x, int y) {
	currentInstance_->mouseMotion_[0] = x - currentInstance_->previousMouseMotion_[0];
	currentInstance_->mouseMotion_[1] = y - currentInstance_->previousMouseMotion_[1];
	currentInstance_->previousMouseMotion_[0] = x;
	currentInstance_->previousMouseMotion_[1] = y;
}

void GLViewer::reshapeCallback(int width, int height) {
	glViewport(0, 0, width, height);
	currentInstance_->image_handler.reshape(width, height);
	float hfov = (180.0f / M_PI) * (2.0f * atan(width / (2.0f * 500)));
	float vfov = (180.0f / M_PI) * (2.0f * atan(height / (2.0f * 500)));
	currentInstance_->camera_.setProjection(hfov, vfov, currentInstance_->camera_.getZNear(), currentInstance_->camera_.getZFar());
}

void GLViewer::keyPressedCallback(unsigned char c, int x, int y) {
	currentInstance_->keyStates_[c] = KEY_STATE::DOWN;
}

void GLViewer::keyReleasedCallback(unsigned char c, int x, int y) {
	currentInstance_->keyStates_[c] = KEY_STATE::UP;
}

void GLViewer::idle() {
	glutPostRedisplay();
}

Simple3DObject::Simple3DObject(sl::Translation position, bool isStatic) : isStatic_(isStatic) {
	vaoID_ = 0;
	drawingType_ = GL_TRIANGLES;
	position_ = position;
	rotation_.setIdentity();
}

Simple3DObject::~Simple3DObject() {
	if (vaoID_ != 0) {
		glDeleteBuffers(3, vboID_);
		glDeleteVertexArrays(1, &vaoID_);
	}
}

void Simple3DObject::addPt(sl::float3 pt) {
	vertices_.push_back(pt.x);
	vertices_.push_back(pt.y);
	vertices_.push_back(pt.z);
}

void Simple3DObject::addClr(sl::float4 clr) {
	colors_.push_back(clr.b);
	colors_.push_back(clr.g);
	colors_.push_back(clr.r);
	colors_.push_back(clr.a);
}

void Simple3DObject::addPoint(sl::float3 pt, sl::float4 clr) {
	addPt(pt);
	addClr(clr);
	indices_.push_back((int)indices_.size());
}

void Simple3DObject::addLine(sl::float3 p1, sl::float3 p2, sl::float4 clr) {
	addPoint(p1, clr);
	addPoint(p2, clr);
}

void Simple3DObject::addTriangle(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float4 clr) {
	addPoint(p1, clr);
	addPoint(p2, clr);
	addPoint(p3, clr);
}

void Simple3DObject::pushToGPU() {
	if (!isStatic_ || vaoID_ == 0) {
		if (vaoID_ == 0) {
			glGenVertexArrays(1, &vaoID_);
			glGenBuffers(3, vboID_);
		}

		if (vertices_.size() > 0) {
			glBindVertexArray(vaoID_);
			glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
			glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
			glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
		}

		if (colors_.size() > 0) {
			glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
			glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(float), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
			glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);
		}

		if (indices_.size() > 0) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int), &indices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
		}

		glBindVertexArray(0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void Simple3DObject::clear() {
	vertices_.clear();
	colors_.clear();
	indices_.clear();
}

void Simple3DObject::setDrawingType(GLenum type) {
	drawingType_ = type;
}

void Simple3DObject::draw() {
	if (indices_.size() && vaoID_) {
		glBindVertexArray(vaoID_);
		glDrawElements(drawingType_, (GLsizei)indices_.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}
}

void Simple3DObject::translate(const sl::Translation& t) {
	position_ = position_ + t;
}

void Simple3DObject::setPosition(const sl::Translation& p) {
	position_ = p;
}

void Simple3DObject::setRT(const sl::Transform& mRT) {
	position_ = mRT.getTranslation();
	rotation_ = mRT.getOrientation();
}

void Simple3DObject::rotate(const sl::Orientation& rot) {
	rotation_ = rot * rotation_;
}

void Simple3DObject::rotate(const sl::Rotation& m) {
	this->rotate(sl::Orientation(m));
}

void Simple3DObject::setRotation(const sl::Orientation& rot) {
	rotation_ = rot;
}

void Simple3DObject::setRotation(const sl::Rotation& m) {
	this->setRotation(sl::Orientation(m));
}

const sl::Translation& Simple3DObject::getPosition() const {
	return position_;
}

sl::Transform Simple3DObject::getModelMatrix() const {
	sl::Transform tmp;
	tmp.setOrientation(rotation_);
	tmp.setTranslation(position_);
	return tmp;
}

Shader::Shader(GLchar* vs, GLchar* fs) {
	if (!compile(verterxId_, GL_VERTEX_SHADER, vs)) {
		std::cout << "ERROR: while compiling vertex shader" << std::endl;
	}
	if (!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)) {
		std::cout << "ERROR: while compiling fragment shader" << std::endl;
	}

	programId_ = glCreateProgram();

	glAttachShader(programId_, verterxId_);
	glAttachShader(programId_, fragmentId_);

	glBindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_vertex");
	glBindAttribLocation(programId_, ATTRIB_COLOR_POS, "in_texCoord");

	glLinkProgram(programId_);

	GLint errorlk(0);
	glGetProgramiv(programId_, GL_LINK_STATUS, &errorlk);
	if (errorlk != GL_TRUE) {
		std::cout << "ERROR: while linking Shader :" << std::endl;
		GLint errorSize(0);
		glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &errorSize);

		char *error = new char[errorSize + 1];
		glGetShaderInfoLog(programId_, errorSize, &errorSize, error);
		error[errorSize] = '\0';
		std::cout << error << std::endl;

		delete[] error;
		glDeleteProgram(programId_);
	}
}

Shader::~Shader() {
	if (verterxId_ != 0)
		glDeleteShader(verterxId_);
	if (fragmentId_ != 0)
		glDeleteShader(fragmentId_);
	if (programId_ != 0)
		glDeleteShader(programId_);
}

GLuint Shader::getProgramId() {
	return programId_;
}

bool Shader::compile(GLuint &shaderId, GLenum type, GLchar* src) {
	shaderId = glCreateShader(type);
	if (shaderId == 0) {
		std::cout << "ERROR: shader type (" << type << ") does not exist" << std::endl;
		return false;
	}
	glShaderSource(shaderId, 1, (const char**)&src, 0);
	glCompileShader(shaderId);

	GLint errorCp(0);
	glGetShaderiv(shaderId, GL_COMPILE_STATUS, &errorCp);
	if (errorCp != GL_TRUE) {
		std::cout << "ERROR: while compiling Shader :" << std::endl;
		GLint errorSize(0);
		glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

		char *error = new char[errorSize + 1];
		glGetShaderInfoLog(shaderId, errorSize, &errorSize, error);
		error[errorSize] = '\0';
		std::cout << error << std::endl;

		delete[] error;
		glDeleteShader(shaderId);
		return false;
	}
	return true;
}

GLchar* POINTCLOUD_VERTEX_SHADER =
"#version 330 core\n"
"layout(location = 0) in vec4 in_VertexRGBA;\n"
"uniform mat4 u_mvpMatrix;\n"
"out vec4 b_color;\n"
"void main() {\n"
// Decompose the 4th channel of the XYZRGBA buffer to retrieve the color of the point (1float to 4uint)
"   uint vertexColor = floatBitsToUint(in_VertexRGBA.w); \n"
"   vec3 clr_int = vec3((vertexColor & uint(0x000000FF)), (vertexColor & uint(0x0000FF00)) >> 8, (vertexColor & uint(0x00FF0000)) >> 16);\n"
"   b_color = vec4(clr_int.r / 255.0f, clr_int.g / 255.0f, clr_int.b / 255.0f, 1.f);"
"	gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n"
"}";

GLchar* POINTCLOUD_FRAGMENT_SHADER =
"#version 330 core\n"
"in vec4 b_color;\n"
"layout(location = 0) out vec4 out_Color;\n"
"void main() {\n"
"   out_Color = b_color;\n"
"}";

PointCloud::PointCloud() : hasNewPCL_(false) {
}

PointCloud::~PointCloud() {
	close();
}

void checkError(cudaError_t err) {
	if (err != cudaSuccess)
		std::cerr << "Error: (" << err << "): " << cudaGetErrorString(err) << std::endl;
}

void PointCloud::close() {
	if (matGPU_.isInit()) {
		matGPU_.free();
		checkError(cudaGraphicsUnmapResources(1, &bufferCudaID_, 0));
		glDeleteBuffers(1, &bufferGLID_);
	}
}

void PointCloud::initialize(sl::Resolution res) {
	glGenBuffers(1, &bufferGLID_);
	glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
	glBufferData(GL_ARRAY_BUFFER, res.area() * 4 * sizeof(float), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkError(cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsWriteDiscard));

	shader.it = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
	shader.MVP_Mat = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

	matGPU_.alloc(res, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);

	checkError(cudaGraphicsMapResources(1, &bufferCudaID_, 0));
	checkError(cudaGraphicsResourceGetMappedPointer((void**)&xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_));
}

void PointCloud::pushNewPC(sl::Mat &matXYZRGBA) {
	if (matGPU_.isInit()) {
		matGPU_.setFrom(matXYZRGBA, sl::COPY_TYPE::GPU_GPU);
		hasNewPCL_ = true;
	}
}

void PointCloud::update() {
	if (hasNewPCL_ && matGPU_.isInit()) {
		checkError(cudaMemcpy(xyzrgbaMappedBuf_, matGPU_.getPtr<sl::float4>(sl::MEM::GPU), numBytes_, cudaMemcpyDeviceToDevice));
		hasNewPCL_ = false;
	}
}

void PointCloud::draw(const sl::Transform& vp) {
	if (matGPU_.isInit()) {
		glUseProgram(shader.it.getProgramId());
		glUniformMatrix4fv(shader.MVP_Mat, 1, GL_TRUE, vp.m);

		glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
		glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

		glDrawArrays(GL_POINTS, 0, matGPU_.getResolution().area());
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glUseProgram(0);
	}
}

const sl::Translation CameraGL::ORIGINAL_FORWARD = sl::Translation(0, 0, 1);
const sl::Translation CameraGL::ORIGINAL_UP = sl::Translation(0, 1, 0);
const sl::Translation CameraGL::ORIGINAL_RIGHT = sl::Translation(1, 0, 0);

CameraGL::CameraGL(sl::Translation position, sl::Translation direction, sl::Translation vertical) {
	this->position_ = position;
	setDirection(direction, vertical);

	offset_ = sl::Translation(0, 0, 0);
	view_.setIdentity();
	updateView();
	setProjection(70, 70, 200.f, 50000.f);
	updateVPMatrix();
}

CameraGL::~CameraGL() {}

void CameraGL::update() {
	if (sl::Translation::dot(vertical_, up_) < 0)
		vertical_ = vertical_ * -1.f;
	updateView();
	updateVPMatrix();
}

void CameraGL::setProjection(float horizontalFOV, float verticalFOV, float znear, float zfar) {
	horizontalFieldOfView_ = horizontalFOV;
	verticalFieldOfView_ = verticalFOV;
	znear_ = znear;
	zfar_ = zfar;

	float fov_y = verticalFOV * M_PI / 180.f;
	float fov_x = horizontalFOV * M_PI / 180.f;

	projection_.setIdentity();
	projection_(0, 0) = 1.0f / tanf(fov_x * 0.5f);
	projection_(1, 1) = 1.0f / tanf(fov_y * 0.5f);
	projection_(2, 2) = -(zfar + znear) / (zfar - znear);
	projection_(3, 2) = -1;
	projection_(2, 3) = -(2.f * zfar * znear) / (zfar - znear);
	projection_(3, 3) = 0;
}

const sl::Transform& CameraGL::getViewProjectionMatrix() const {
	return vpMatrix_;
}

float CameraGL::getHorizontalFOV() const {
	return horizontalFieldOfView_;
}

float CameraGL::getVerticalFOV() const {
	return verticalFieldOfView_;
}

void CameraGL::setOffsetFromPosition(const sl::Translation& o) {
	offset_ = o;
}

const sl::Translation& CameraGL::getOffsetFromPosition() const {
	return offset_;
}

void CameraGL::setDirection(const sl::Translation& direction, const sl::Translation& vertical) {
	sl::Translation dirNormalized = direction;
	dirNormalized.normalize();
	this->rotation_ = sl::Orientation(ORIGINAL_FORWARD, dirNormalized * -1.f);
	updateVectors();
	this->vertical_ = vertical;
	if (sl::Translation::dot(vertical_, up_) < 0)
		rotate(sl::Rotation(M_PI, ORIGINAL_FORWARD));
}

void CameraGL::translate(const sl::Translation& t) {
	position_ = position_ + t;
}

void CameraGL::setPosition(const sl::Translation& p) {
	position_ = p;
}

void CameraGL::rotate(const sl::Orientation& rot) {
	rotation_ = rot * rotation_;
	updateVectors();
}

void CameraGL::rotate(const sl::Rotation& m) {
	this->rotate(sl::Orientation(m));
}

void CameraGL::setRotation(const sl::Orientation& rot) {
	rotation_ = rot;
	updateVectors();
}

void CameraGL::setRotation(const sl::Rotation& m) {
	this->setRotation(sl::Orientation(m));
}

const sl::Translation& CameraGL::getPosition() const {
	return position_;
}

const sl::Translation& CameraGL::getForward() const {
	return forward_;
}

const sl::Translation& CameraGL::getRight() const {
	return right_;
}

const sl::Translation& CameraGL::getUp() const {
	return up_;
}

const sl::Translation& CameraGL::getVertical() const {
	return vertical_;
}

float CameraGL::getZNear() const {
	return znear_;
}

float CameraGL::getZFar() const {
	return zfar_;
}

void CameraGL::updateVectors() {
	forward_ = ORIGINAL_FORWARD * rotation_;
	up_ = ORIGINAL_UP * rotation_;
	right_ = sl::Translation(ORIGINAL_RIGHT * -1.f) * rotation_;
}

void CameraGL::updateView() {
	sl::Transform transformation(rotation_, (offset_ * rotation_) + position_);
	view_ = sl::Transform::inverse(transformation);
}

void CameraGL::updateVPMatrix() {
	vpMatrix_ = projection_ * view_;
}


GLchar* IMAGE_VERTEX_SHADER =
"#version 330\n"
"layout(location = 0) in vec2 vert;\n"
"layout(location = 1) in vec3 vert_tex;\n"
"out vec2 UV;"
"void main() {\n"
"	UV = vert;\n"
"	gl_Position = vec4(vert_tex, 1);\n"
"}\n";

GLchar* IMAGE_FRAGMENT_SHADER =
"#version 330 core\n"
" in vec2 UV;\n"
" out vec4 color;\n"
" uniform sampler2D texImage;\n"
" void main() {\n"
"	vec2 scaler = vec2(UV.x, UV.y);\n"
"	color = vec4(texture(texImage, scaler).zyxw);\n"
"}";

ImageHandler::ImageHandler():init(false) {}

ImageHandler::~ImageHandler() {
	close();
}

void ImageHandler::close() {
	if (init) {
		init = false;
		glDeleteTextures(1, &imageTex);
		glDeleteBuffers(3, vboID_);
		glDeleteVertexArrays(1, &vaoID_);
	}
}

bool ImageHandler::initialize(sl::Resolution res) {
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	shader = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);
	texID = glGetUniformLocation(shader.getProgramId(), "texImage");

	glGenVertexArrays(1, &vaoID_);
	glGenBuffers(3, vboID_);
	{
		float w_0 = 0.f;
		float w_1 = 1.f;
		float h_0 = 1.f;
		float h_1 = 0.f;

		static const GLfloat g_quad_vertex_buffer_data[] = {
			w_0, h_0,
			w_1, h_0,
			w_0, h_1,
			w_1, h_1 };

		glBindVertexArray(vaoID_);
		glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
		glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
		glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
	}

	{
		float w_0 = -1.f;
		float w_1 = 1.f;
		float h_0 = -1.f;
		float h_1 = 1.f;

		static const GLfloat g_quad_vertex_buffer_data[] = {
			w_0, h_0, 0.f,
			w_1, h_0, 0.f,
			w_0, h_1, 0.f,
			w_1, h_1, 0.f };
		glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
		glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), g_quad_vertex_buffer_data, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	std::vector<unsigned int> indices_; // 4 corners
	for (int i = 0; i < 4; i++) indices_.push_back(i);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int), &indices_[0], GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenTextures(1, &imageTex);
	glBindTexture(GL_TEXTURE_2D, imageTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.width, res.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_gl_ressource, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	if (err) std::cout << "err alloc " << err << " " << cudaGetErrorString(err) << "\n";
	glDisable(GL_TEXTURE_2D);
		
	init = true;
	return (err == cudaSuccess);
}

void ImageHandler::reshape(int width, int height) {
	const float w_0 = -1.f;
	const float w_1 = -0.5f;
	float h_0 = -1.f;
	const float h_1 = h_0 + (1.f / (16.f / 9.f));

	static const GLfloat g_quad_vertex_buffer_data[] = {
		w_0, h_0, 0.f,
		w_1, h_0, 0.f,
		w_0, h_1, 0.f,
		w_1, h_1, 0.f };

	glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
	glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), g_quad_vertex_buffer_data, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ImageHandler::pushNewImage(sl::Mat& image) {
	if (!init)  return;

	glEnable(GL_TEXTURE_2D);
	cudaArray_t ArrIm;
	auto err = cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
	if (err) std::cout << "err 0 " << err << " " << cudaGetErrorString(err) << "\n";
	err = cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
	if (err) std::cout << "err 1 " << err << " " << cudaGetErrorString(err) << "\n";
	err = cudaMemcpy2DToArray(ArrIm, 0, 0, image.getPtr<sl::uchar1>(sl::MEM::GPU), image.getStepBytes(sl::MEM::GPU), image.getPixelBytes() * image.getWidth(), image.getHeight(), cudaMemcpyDeviceToDevice);
	if (err) std::cout << "err 2 " << err << " " << cudaGetErrorString(err) << "\n";
	err = cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);
	if (err) std::cout << "err 3 " << err << " " << cudaGetErrorString(err) << "\n";
	glDisable(GL_TEXTURE_2D);	
}

void ImageHandler::draw() {
	if (!init)  return;
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);

	const auto id_shade = shader.getProgramId();
	glUseProgram(id_shade);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, imageTex);
	glUniform1i(texID, 0);

	glBindVertexArray(vaoID_);
	glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)4, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	glUseProgram(0);
	glDisable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
}
