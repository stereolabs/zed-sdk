#include "GLViewer.hpp"

GLchar* VERTEX_SHADER =
"#version 330 core\n"
"layout(location = 0) in vec3 in_Vertex;\n"
"layout(location = 1) in vec3 in_Color;\n"
"uniform mat4 u_mvpMatrix;\n"
"out vec3 b_color;\n"
"void main() {\n"
"   b_color = in_Color;\n"
"	gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
"}";

GLchar* FRAGMENT_SHADER =
"#version 330 core\n"
"in vec3 b_color;\n"
"layout(location = 0) out vec4 out_Color;\n"
"void main() {\n"
"   out_Color = vec4(b_color, 1);\n"
"}";

using namespace sl;

GLViewer* GLViewer::currentInstance_ = nullptr;

void getColor(int num_segments, int i, float &c1, float &c2, float &c3) {
    float r = fabs(1.f - (float(i)*2.f) / float(num_segments));
    c1 = (0.1f * r);
    c2 = (0.3f * r);
    c3 = (0.8f * r);
}

GLViewer::GLViewer(): initialized_(false) {
    if (currentInstance_ != nullptr) {
        delete currentInstance_;
    }
    currentInstance_ = this;

    wnd_w = 1000;
    wnd_h = 1000;

    cb = 0.847058f;
    cg = 0.596078f;
    cr = 0.203921f;

    mouseButton_[0] = mouseButton_[1] = mouseButton_[2] = false;

    clearInputs();
    previousMouseMotion_[0] = previousMouseMotion_[1] = 0;
    ended_ = true;
}

GLViewer::~GLViewer() {}

void GLViewer::exit() {
    if (initialized_) {
        pointCloud_.close();
        ended_ = true;
        glutLeaveMainLoop();
    }
}

bool GLViewer::isEnded() {
    return ended_;
}

void GLViewer::init(int w, int h) {
    res.width = w;
    res.height = h;
    // Get current CUDA context (created by the ZED) for CUDA - OpenGL interoperability
    cuCtxGetCurrent(&ctx);
    initialize();
    // Wait for OpenGL to initialize
    while (!isInitialized()) sl::sleep_ms(1);
}

void GLViewer::initialize() {
    char *argv[1];
    argv[0] = '\0';
    int argc = 1;
    glutInit(&argc, argv);
    glutInitWindowSize(wnd_w, wnd_h);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED 3D Viewer");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

    glEnable(GL_DEPTH_TEST);

    pointCloud_.initialize((int) res.width, (int) res.height, ctx);

    // Compile and create the shader
    shader_ = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");

    // Create the camera
    camera_ = CameraGL(sl::Translation(0, 0, 0), sl::Translation(0, 0, -1));
    camera_.setOffsetFromPosition(sl::Translation(0, 0, 4));

    // Create 3D axis
    sl::Translation posStart(0, 0, 0);
    axis_X = Simple3DObject(posStart, true);
    axis_Y = Simple3DObject(posStart, true);
    axis_Z = Simple3DObject(posStart, true);

    int num_segments = 60;
    float rad = 0.10f;
    float fade = 0.5f;

    for (int ii = 0; ii < num_segments; ii++) {
        float c1 = (cr * fade);
        float c2 = (cg * fade);
        float c3 = (cb * fade);

        float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);
        axis_X.addPoint(rad * cosf(theta), rad * sinf(theta), 0, c1, c2, c3);

        getColor(num_segments, ii, c1, c2, c3);
        axis_Y.addPoint(0, rad * sinf(theta), rad * cosf(theta), c3, c2, c2);

        theta = 2.0f * M_PI * (float(ii) + float(num_segments) / 4.f) / float(num_segments);
        theta = theta > (2.f * M_PI) ? theta - (2.f * M_PI) : theta;
        getColor(num_segments, ii, c1, c2, c3);
        axis_Z.addPoint(rad * cosf(theta), 0, rad * sinf(theta), c2, c3, c1);
    }

    axis_X.setDrawingType(GL_LINE_LOOP);
    axis_X.pushToGPU();
    axis_Y.setDrawingType(GL_LINE_LOOP);
    axis_Y.pushToGPU();
    axis_Z.setDrawingType(GL_LINE_LOOP);
    axis_Z.pushToGPU();


    // Map glut function on this class methods
    glutDisplayFunc(GLViewer::drawCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutMotionFunc(GLViewer::mouseMotionCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardFunc(GLViewer::keyPressedCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);

    initialized_ = true;
    ended_ = false;
}

void GLViewer::render() {
    if (!ended_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glClearColor(0.12f, 0.12f, 0.12f, 1.0f);
        glLineWidth(2.f);
        glPointSize(2.f);
        update();
        draw();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

bool GLViewer::isInitialized() {
    return initialized_;
}

void GLViewer::updatePointCloud(sl::Mat &matXYZRGBA) {
    pointCloud_.mutexData.lock();
    pointCloud_.pushNewPC(matXYZRGBA);
    pointCloud_.mutexData.unlock();
}

void GLViewer::update() {
    if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP) {
        currentInstance_->exit();
        return;
    }

    // Rotate camera with mouse
    if (mouseButton_[MOUSE_BUTTON::LEFT]) {
        camera_.rotate(sl::Rotation((float) mouseMotion_[1] * MOUSE_R_SENSITIVITY, camera_.getRight()));
        camera_.rotate(sl::Rotation((float) mouseMotion_[0] * MOUSE_R_SENSITIVITY, camera_.getVertical() * -1.f));
    }

    // Translate camera with mouse
    if (mouseButton_[MOUSE_BUTTON::RIGHT]) {
        camera_.translate(camera_.getUp() * (float) mouseMotion_[1] * MOUSE_T_SENSITIVITY);
        camera_.translate(camera_.getRight() * (float) mouseMotion_[0] * MOUSE_T_SENSITIVITY);
    }

    // Zoom in with mouse wheel
    if (mouseWheelPosition_ != 0) {
        float distance = sl::Translation(camera_.getOffsetFromPosition()).norm();
        if (mouseWheelPosition_ > 0 && distance > camera_.getZNear()) { // zoom
            camera_.setOffsetFromPosition(camera_.getOffsetFromPosition() * MOUSE_UZ_SENSITIVITY);
        } else if (distance < camera_.getZFar()) {// unzoom
            camera_.setOffsetFromPosition(camera_.getOffsetFromPosition() * MOUSE_DZ_SENSITIVITY);
        }
    }

    // Translate camera with keyboard
    if (keyStates_['u'] == KEY_STATE::DOWN) {
        camera_.translate((camera_.getForward()*-1.f) * KEY_T_SENSITIVITY);
    }
    if (keyStates_['j'] == KEY_STATE::DOWN) {
        camera_.translate(camera_.getForward() * KEY_T_SENSITIVITY);
    }
    if (keyStates_['h'] == KEY_STATE::DOWN) {
        camera_.translate(camera_.getRight() * KEY_T_SENSITIVITY);
    }
    if (keyStates_['k'] == KEY_STATE::DOWN) {
        camera_.translate((camera_.getRight()*-1.f) * KEY_T_SENSITIVITY);
    }

    // Update point cloud buffers
    pointCloud_.mutexData.lock();
    pointCloud_.update();
    pointCloud_.mutexData.unlock();
    camera_.update();
    clearInputs();
}

void GLViewer::draw() {
    const sl::Transform vpMatrix = camera_.getViewProjectionMatrix();

    // Simple 3D shader for simple 3D objects
    glUseProgram(shader_.getProgramId());
    // Axis
    glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_FALSE, sl::Transform::transpose(vpMatrix * axis_X.getModelMatrix()).m);
    axis_X.draw();
    axis_Y.draw();
    axis_Z.draw();
    glUseProgram(0);

    // Draw point cloud with its own shader
    pointCloud_.draw(sl::Transform::transpose(vpMatrix));
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
        } else {
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
    glutPostRedisplay();
}

void GLViewer::reshapeCallback(int width, int height) {
    glViewport(0, 0, width, height);
    float hfov = currentInstance_->camera_.getHorizontalFOV();
    currentInstance_->camera_.setProjection(hfov, hfov * (float) height / (float) width, currentInstance_->camera_.getZNear(), currentInstance_->camera_.getZFar());
}

void GLViewer::keyPressedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::DOWN;
    glutPostRedisplay();
}

void GLViewer::keyReleasedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::UP;
}

void GLViewer::idle() {
    glutPostRedisplay();
}

Simple3DObject::Simple3DObject(Translation position, bool isStatic): isStatic_(isStatic) {
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

void Simple3DObject::addPoint(float x, float y, float z, float r, float g, float b) {
    vertices_.push_back(x);
    vertices_.push_back(y);
    vertices_.push_back(z);
    colors_.push_back(r);
    colors_.push_back(g);
    colors_.push_back(b);
    indices_.push_back((int) indices_.size());
}

void Simple3DObject::pushToGPU() {
    if (!isStatic_ || vaoID_ == 0) {
        if (vaoID_ == 0) {
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(3, vboID_);
        }
        glBindVertexArray(vaoID_);
        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(float), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int), &indices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);

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
    glBindVertexArray(vaoID_);
    glDrawElements(drawingType_, (GLsizei) indices_.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Simple3DObject::translate(const Translation& t) {
    position_ = position_ + t;
}

void Simple3DObject::setPosition(const Translation& p) {
    position_ = p;
}

void Simple3DObject::setRT(const Transform& mRT) {
    position_ = mRT.getTranslation();
    rotation_ = mRT.getOrientation();
}

void Simple3DObject::rotate(const Orientation& rot) {
    rotation_ = rot * rotation_;
}

void Simple3DObject::rotate(const Rotation& m) {
    this->rotate(sl::Orientation(m));
}

void Simple3DObject::setRotation(const Orientation& rot) {
    rotation_ = rot;
}

void Simple3DObject::setRotation(const Rotation& m) {
    this->setRotation(sl::Orientation(m));
}

const Translation& Simple3DObject::getPosition() const {
    return position_;
}

Transform Simple3DObject::getModelMatrix() const {
    Transform tmp = Transform::identity();
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
    glShaderSource(shaderId, 1, (const char**) &src, 0);
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
"out vec3 b_color;\n"
"vec4 decomposeFloat(const in float value)\n"
"{\n"
"   uint rgbaInt = floatBitsToUint(value);\n"
"	uint bIntValue = (rgbaInt / 256U / 256U) % 256U;\n"
"	uint gIntValue = (rgbaInt / 256U) % 256U;\n"
"	uint rIntValue = (rgbaInt) % 256U; \n"
"	return vec4(rIntValue / 255.0f, gIntValue / 255.0f, bIntValue / 255.0f, 1.0); \n"
"}\n"
"void main() {\n"
// Decompose the 4th channel of the XYZRGBA buffer to retrieve the color of the point (1float to 4uint)
"   b_color = decomposeFloat(in_VertexRGBA.a).xyz;\n"
"	gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n"
"}";

GLchar* POINTCLOUD_FRAGMENT_SHADER =
"#version 330 core\n"
"in vec3 b_color;\n"
"layout(location = 0) out vec4 out_Color;\n"
"void main() {\n"
"   out_Color = vec4(b_color, 1);\n"
"}";

PointCloud::PointCloud():
    hasNewPCL_(false), initialized_(false) {}

PointCloud::~PointCloud() {
    close();
}

void PointCloud::close() {
    if (initialized_) {
        initialized_ = false;
        matGPU_.free();
        glDeleteBuffers(1, &bufferGLID_);
    }
}

void PointCloud::initialize(unsigned int width, unsigned int height, CUcontext ctx) {
    width_ = width;
    height_ = height;
    cuda_zed_ctx = ctx;
    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, width_ * height_ * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set current Cuda context, as this function is called in a thread which doesn't handle the Cuda context
    cuCtxSetCurrent(cuda_zed_ctx);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess)
        std::cerr << "Error: CUDA - OpenGL Interop failed (" << err << ")" << std::endl;

    shader_ = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");

    matGPU_.alloc(width_, height_, sl::MAT_TYPE_32F_C4, sl::MEM_GPU);
    initialized_ = true;
}

void PointCloud::pushNewPC(sl::Mat &matXYZRGBA) {
    if (initialized_) {
        cuCtxSetCurrent(cuda_zed_ctx);
        matGPU_.setFrom(matXYZRGBA, sl::COPY_TYPE_GPU_GPU);
        hasNewPCL_ = true;
    }
}

void PointCloud::update() {
    if (hasNewPCL_ && initialized_) {
        cudaError_t err = cudaGraphicsMapResources(1, &bufferCudaID_, 0);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA MapResources (" << err << ")" << std::endl;

        err = cudaGraphicsResourceGetMappedPointer((void**) &xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA GetMappedPointer (" << err << ")" << std::endl;

        err = cudaMemcpy(xyzrgbaMappedBuf_, matGPU_.getPtr<sl::float4>(sl::MEM_GPU), numBytes_, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA MemCpy (" << err << ")" << std::endl;

        err = cudaGraphicsUnmapResources(1, &bufferCudaID_, 0);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA UnmapResources (" << err << ")" << std::endl;

        hasNewPCL_ = false;
    }
}

void PointCloud::draw(const sl::Transform& vp) {
    if (initialized_) {
        glUseProgram(shader_.getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_FALSE, vp.m);

        glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glDrawArrays(GL_POINTS, 0, width_ * height_);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
    }
}

unsigned int PointCloud::getWidth() {
    return width_;
}

unsigned int PointCloud::getHeight() {
    return height_;
}

const sl::Translation CameraGL::ORIGINAL_FORWARD = sl::Translation(0, 0, 1);
const sl::Translation CameraGL::ORIGINAL_UP = sl::Translation(0, 1, 0);
const sl::Translation CameraGL::ORIGINAL_RIGHT = sl::Translation(1, 0, 0);

CameraGL::CameraGL(Translation position, Translation direction, Translation vertical) {
    this->position_ = position;
    setDirection(direction, vertical);

    offset_ = sl::Translation(0, 0, 0);
    view_.setIdentity();
    updateView();
    setProjection(60, 60, 0.01f, 100.f);
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
