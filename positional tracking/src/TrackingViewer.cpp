#include "TrackingViewer.hpp"

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

GLViewer::GLViewer() {
    if (currentInstance_ != nullptr) {
        delete currentInstance_;
    }
    currentInstance_ = this;

    mouseButton_[0] = mouseButton_[1] = mouseButton_[2] = false;

    clearInputs();
    previousMouseMotion_[0] = previousMouseMotion_[1] = 0;
    ended_ = true;
}

GLViewer::~GLViewer() {}

void GLViewer::exit() {
    ended_ = true;
    glutLeaveMainLoop();
}

bool GLViewer::isEnded() {
    return ended_;
}

void fillZED(int nb_tri, float *vertices, int *triangles, sl::float3 color, Simple3DObject *zed_camera) {
    for (int p = 0; p < nb_tri * 3; p = p + 3) {
        int index = triangles[p] - 1;
        zed_camera->addPoint(vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2], color.r, color.g, color.b);
        index = triangles[p + 1] - 1;
        zed_camera->addPoint(vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2], color.r, color.g, color.b);
        index = triangles[p + 2] - 1;
        zed_camera->addPoint(vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2], color.r, color.g, color.b);
    }
}

void GLViewer::init(sl::MODEL camera_model) {
    char *argv[1];
    argv[0] = '\0';
    int argc = 1;
    glutInit(&argc, argv);

    int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = glutGet(GLUT_SCREEN_HEIGHT) *0.9;
    glutInitWindowSize(wnd_w*0.9, wnd_h*0.9);
    glutInitWindowPosition(wnd_w*0.05, wnd_h*0.05);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED Positional Tracking");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

    glEnable(GL_DEPTH_TEST);

    // Compile and create the shader
    shader_ = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");

    // Create the camera
    camera_ = CameraGL(sl::Translation(0, 0.3, -0.15), sl::Translation(0.7, -0.25, -6.5));
    camera_.setOffsetFromPosition(sl::Translation(0, 0, 4));

    sl::float3 c1(13.f / 255.f, 17.f / 255.f, 20.f / 255.f);
    sl::float3 c2(213.f / 255.f, 207.f / 255.f, 200.f / 255.f);
    float span = 20.f;
    for (int i = (int) -span; i <= (int) span; i++) {
        grill.addPoint(sl::float3(i, 0, -span), c1);
        grill.addPoint(sl::float3(i, 0, span), c2);
        float clr = (i + span) / (span * 2);
        sl::float3 c3(clr, clr, clr);
        grill.addPoint(sl::float3(-span, 0, i), c3);
        grill.addPoint(sl::float3(span, 0, i), c3);
    }
    grill.setDrawingType(GL_LINES);
    grill.pushToGPU();

    zedPath.setDrawingType(GL_LINES);

    Model3D *model;
    if (camera_model == sl::MODEL_ZED)
        model = new Model3D_ZED;
    else
        model = new Model3D_ZED_M;

    for (int p = 0; p < model->part.size(); p++)
        fillZED(model->part[p].nb_triangles, model->vertices, model->part[p].triangles, model->part[p].color, &zedModel);

    zedModel.pushToGPU();

    updateZEDposition = false;

    // Map glut function on this class methods
    glutDisplayFunc(GLViewer::drawCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutMotionFunc(GLViewer::mouseMotionCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardFunc(GLViewer::keyPressedCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);

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

    if (keyStates_['p'] == KEY_STATE::DOWN) {
        std::cout << "Camera position " << camera_.getPosition() << std::endl;
    }

    // Update point cloud buffers  
    camera_.update();
    clearInputs();
    mtx.lock();
    if (updateZEDposition) {
        zedPath.clear();
        sl::float3 clr(0.1f, 0.5f, 0.9f);
        for (int i = 1; i < vecPath.size(); i++) {
            float fade = (i*1.f) / vecPath.size();
            sl::float3 new_color = clr * fade;
            zedPath.addPoint(vecPath[i - 1], new_color);
            zedPath.addPoint(vecPath[i], new_color);
        }
        zedPath.pushToGPU();
        updateZEDposition = false;
    }
    mtx.unlock();
}

void GLViewer::draw() {
    const sl::Transform vpMatrix = camera_.getViewProjectionMatrix();

    // Simple 3D shader for simple 3D objects
    glUseProgram(shader_.getProgramId());
    glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);

    grill.draw();

    zedPath.draw();

    // Move the ZED 3D model to correct position
    glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_FALSE, (sl::Transform::transpose(zedModel.getModelMatrix()) *  sl::Transform::transpose(vpMatrix)).m);
    zedModel.draw();

    glUseProgram(0);

    printText();
}

void GLViewer::updateZEDPosition(sl::Transform zed_rt) {
    mtx.lock();
    vecPath.emplace_back();
    vecPath.back().x = zed_rt.getTranslation().x;
    vecPath.back().y = zed_rt.getTranslation().y;
    vecPath.back().z = zed_rt.getTranslation().z;
    zedModel.setRT(zed_rt);
    updateZEDposition = true;
    mtx.unlock();
}

void GLViewer::updateText(std::string str_t, std::string str_r, sl::TRACKING_STATE state) {
    txtT = str_t;
    txtR = str_r;
    trackState = state;
}

static void safe_glutBitmapString(void *font, const char *str) {
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

    (trackState == sl::TRACKING_STATE_OK) ? glColor3f(0.2f, 0.65f, 0.2f) : glColor3f(0.85f, 0.2f, 0.2f);
    glRasterPos2i(start_w, start_h);
    std::string track_str = (str_tracking + sl::toString(trackState).c_str());
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, track_str.c_str());

    glColor3f(0.9255f, 0.9412f, 0.9451f);
    glRasterPos2i(start_w, start_h - 25);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, "Translation (m) :");

    glColor3f(0.4980f, 0.5490f, 0.5529f);
    glRasterPos2i(155, start_h - 25);

    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, txtT.c_str());

    glColor3f(0.9255f, 0.9412f, 0.9451f);
    glRasterPos2i(start_w, start_h - 50);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, "Rotation   (rad) :");

    glColor3f(0.4980f, 0.5490f, 0.5529f);
    glRasterPos2i(155, start_h - 50);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, txtR.c_str());

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

Simple3DObject::Simple3DObject(): isStatic_(false) {
    vaoID_ = 0;
    drawingType_ = GL_TRIANGLES;
    position_ = sl::float3(0, 0, 0);
    rotation_.setIdentity();
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

void Simple3DObject::addPoint(sl::float3 position, sl::float3 color) {
    addPoint(position.x, position.y, position.z, color.r, color.g, color.b);
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