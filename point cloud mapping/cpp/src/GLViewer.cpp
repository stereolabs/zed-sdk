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

GLchar* FPC_VERTEX_SHADER =
    "#version 330 core\n"
    "layout(location = 0) in vec4 in_VertexRGBA;\n"
    "uniform mat4 u_mvpMatrix;\n"
    "uniform float pointsize;\n"
    "out vec3 b_color;\n"
    "void main() {\n"
    "   uint vertexColor = floatBitsToUint(in_VertexRGBA.w); \n"
    "   b_color = vec3(((vertexColor & uint(0x00FF0000)) >> 16) / 255.f, ((vertexColor & uint(0x0000FF00)) >> 8) / 255.f, (vertexColor & uint(0x000000FF)) / 255.f);\n"
    "	gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n"
    "   gl_PointSize = pointsize;\n"
    "}";

GLchar* FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 out_Color;\n"
        "void main() {\n"
        "   out_Color = vec4(b_color, 1);\n"
        "}";

GLViewer* currentInstance_ = nullptr;

GLViewer::GLViewer() : available(false){
    currentInstance_ = this;
    mouseButton_[0] = mouseButton_[1] = mouseButton_[2] = false;
    clearInputs();
    previousMouseMotion_[0] = previousMouseMotion_[1] = 0;
}

GLViewer::~GLViewer() {}

void GLViewer::exit() {
    if (currentInstance_) 
        available = false;
}

bool GLViewer::isAvailable() {
    if(available)
        glutMainLoopEvent();
    return available;
}

void CloseFunc(void) { if(currentInstance_)  currentInstance_->exit();}

void fillZED(int nb_tri, float *vertices, int *triangles, sl::float3 color, Simple3DObject *zed_camera) {
    for (int p = 0; p < nb_tri * 3; p = p + 3) {
        int index = triangles[p] - 1;
        zed_camera->addPoint(sl::float3(vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2]) * 1000, sl::float3(color.r, color.g, color.b));
        index = triangles[p + 1] - 1;
        zed_camera->addPoint(sl::float3(vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2]) * 1000, sl::float3(color.r, color.g, color.b));
        index = triangles[p + 2] - 1;
        zed_camera->addPoint(sl::float3(vertices[index * 3], vertices[index * 3 + 1], vertices[index * 3 + 2]) * 1000, sl::float3(color.r, color.g, color.b));
    }
}

GLenum GLViewer::init(int argc, char **argv, 
sl::CameraParameters param, sl::FusedPointCloud* ptr, sl::MODEL zed_model) {

    glutInit(&argc, argv);
    int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = glutGet(GLUT_SCREEN_HEIGHT) *0.9;
    glutInitWindowSize(1280, 720);
    glutInitWindowPosition(wnd_w*0.05, wnd_h*0.05);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED PointCloud Fusion");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        return err;

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    p_fpc = ptr;
    
    // Compile and create the shader
    mainShader.it = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
    mainShader.MVP_Mat = glGetUniformLocation(mainShader.it.getProgramId(), "u_mvpMatrix");

    pcf_shader.it = Shader(FPC_VERTEX_SHADER, FRAGMENT_SHADER);
    pcf_shader.MVP_Mat = glGetUniformLocation(pcf_shader.it.getProgramId(), "u_mvpMatrix");

    // Create the camera
    camera_ = CameraGL(sl::Translation(0, 0, 1000), sl::Translation(0, 0, -100));
    camera_.setOffsetFromPosition(sl::Translation(0, 0, 1500));

    // change background color
    bckgrnd_clr = sl::float3(37, 42, 44);
    bckgrnd_clr /= 255.f;

    zedPath.setDrawingType(GL_LINE_STRIP);
    zedModel.setDrawingType(GL_TRIANGLES);
    Model3D *model;
    switch (zed_model)
    {
    case sl::MODEL::ZED: model = new Model3D_ZED(); break;
    case sl::MODEL::ZED2: model = new Model3D_ZED2(); break;
    case sl::MODEL::ZED_M: model = new Model3D_ZED_M(); break;
    }
    for (auto it : model->part)
        fillZED(it.nb_triangles, model->vertices, it.triangles, it.color, &zedModel);
    delete model;

    zedModel.pushToGPU();
    updateZEDposition = false;

    // Map glut function on this class methods
    glutDisplayFunc(GLViewer::drawCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutMotionFunc(GLViewer::mouseMotionCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardFunc(GLViewer::keyPressedCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
    glutCloseFunc(CloseFunc);

    available = true;
    
    // ready to start
    chunks_pushed = true;

    return err;
}

void GLViewer::render() {
    if (available) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(bckgrnd_clr.r, bckgrnd_clr.g, bckgrnd_clr.b, 1.f);
        update();
        draw();
        printText();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

void GLViewer::update() {
    if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP) {    
        currentInstance_->exit();
        return;
    }

    if (keyStates_['f'] == KEY_STATE::UP || keyStates_['F'] == KEY_STATE::UP) {
        followCamera = !followCamera;
        if(followCamera)
            camera_.setOffsetFromPosition(sl::Translation(0, 0, 1500));        
    }

    // Rotate camera with mouse
    if(!followCamera){
        if (mouseButton_[MOUSE_BUTTON::LEFT]) {
            camera_.rotate(sl::Rotation((float) mouseMotion_[1] * MOUSE_R_SENSITIVITY, camera_.getRight()));
            camera_.rotate(sl::Rotation((float) mouseMotion_[0] * MOUSE_R_SENSITIVITY, camera_.getVertical() * -1.f));
        }

        // Translate camera with mouse
        if (mouseButton_[MOUSE_BUTTON::RIGHT]) {
            camera_.translate(camera_.getUp() * (float) mouseMotion_[1] * MOUSE_T_SENSITIVITY);
            camera_.translate(camera_.getRight() * (float) mouseMotion_[0] * MOUSE_T_SENSITIVITY);
        }
    }

    // Zoom in with mouse wheel
    if (mouseWheelPosition_ != 0) {
        sl::Translation cur_offset = camera_.getOffsetFromPosition();
        bool zoom_ = mouseWheelPosition_ > 0;
        sl::Translation new_offset = cur_offset * (zoom_? MOUSE_UZ_SENSITIVITY : MOUSE_DZ_SENSITIVITY);
        if (zoom_) {
            if(followCamera) {
                if((new_offset.tz<500.f))
                    new_offset.tz = 500.f;                
            } else {
                if((new_offset.tz<50.f) ) 
                    new_offset.tz = 50.f;                
            }
        } else {
            if(followCamera) {
                if(new_offset.tz>5000.f)
                    new_offset.tz = 5000.f;                
            }
        }
        camera_.setOffsetFromPosition(new_offset);
    }
    
    // Update point cloud buffers    
    camera_.update();
    clearInputs();
    mtx.lock();
    if (updateZEDposition) {
        sl::float3 clr(0.1f, 0.5f, 0.9f);
        for(auto it: vecPath)
            zedPath.addPoint(it, clr);
        zedPath.pushToGPU();
        vecPath.clear();
        updateZEDposition = false;
    }
    
    if (new_chunks) {
        const int nb_c = p_fpc->chunks.size();
        if (nb_c > sub_maps.size()) {
            const float step = 500.f;
            size_t new_size = ((nb_c / step) + 1) * step;
            sub_maps.resize(new_size);
        }
        int c = 0;
        for (auto& it : sub_maps) {
            if ((c < nb_c) && p_fpc->chunks[c].has_been_updated)
                it.update(p_fpc->chunks[c]);
            c++;
        }

        new_chunks = false;
        chunks_pushed = true;
    }

    mtx.unlock();
}

void GLViewer::draw() {
    const sl::Transform vpMatrix = camera_.getViewProjectionMatrix();

    glUseProgram(mainShader.it.getProgramId());
    glUniformMatrix4fv(mainShader.MVP_Mat, 1, GL_TRUE, vpMatrix.m);

    glLineWidth(1.f);
    zedPath.draw();

    glUniformMatrix4fv(mainShader.MVP_Mat, 1, GL_FALSE, (sl::Transform::transpose(zedModel.getModelMatrix()) *  sl::Transform::transpose(vpMatrix)).m);
 
    zedModel.draw();
    glUseProgram(0);

    if(sub_maps.size()){
        glPointSize(2.f);
        glUseProgram(pcf_shader.it.getProgramId());
        glUniformMatrix4fv(pcf_shader.MVP_Mat, 1, GL_TRUE, vpMatrix.m);

        for (auto &it: sub_maps)
            it.draw();
        glUseProgram(0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
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

void printGL(float x, float y, const char *string) {
    glRasterPos2f(x, y);
    int len = (int) strlen(string);
    for(int i = 0; i < len; i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, string[i]);
    }
}

void GLViewer::printText() {
    if(available) {
        glColor3f(0.85f, 0.86f, 0.83f);
        printGL(-0.99f, 0.90f, "Press 'F' to un/follow the camera");

        std::string positional_tracking_state_str();
        // Show mapping state
        if ((tracking_state == sl::POSITIONAL_TRACKING_STATE::OK))
            glColor3f(0.25f, 0.99f, 0.25f);
        else
            glColor3f(0.99f, 0.25f, 0.25f);        
        std::string state_str("POSITIONAL TRACKING STATE : ");
        state_str += sl::toString(tracking_state).c_str();
        printGL(-0.99f, 0.95f, state_str.c_str());
    }
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
    float hfov = (180.0f / M_PI) * (2.0f * atan(width / (2.0f * 500)));
    float vfov = (180.0f / M_PI) * (2.0f * atan(height / (2.0f * 500)));
    currentInstance_->camera_.setProjection(hfov, vfov, currentInstance_->camera_.getZNear(), currentInstance_->camera_.getZFar());
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

void GLViewer::updatePose(sl::Pose pose_, sl::POSITIONAL_TRACKING_STATE state) {
    mtx.lock();
    pose = pose_;
    tracking_state = state;
    vecPath.push_back(pose.getTranslation());
    zedModel.setRT(pose.pose_data);
    updateZEDposition = true;
    mtx.unlock();
    if(followCamera)   {
        camera_.setPosition(pose.getTranslation());
        sl::Rotation rot = pose.getRotationMatrix();
        camera_.setRotation(rot);
    }
}

Simple3DObject::Simple3DObject() : isStatic_(false) {
    vaoID_ = 0;
    drawingType_ = GL_TRIANGLES;
    position_ = sl::float3(0, 0, 0);
    rotation_.setIdentity();
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
    indices_.push_back((int)indices_.size());
}

void Simple3DObject::addLine(sl::float3 p1, sl::float3 p2, sl::float3 clr) {
    vertices_.push_back(p1.x);
    vertices_.push_back(p1.y);
    vertices_.push_back(p1.z);

    vertices_.push_back(p2.x);
    vertices_.push_back(p2.y);
    vertices_.push_back(p2.z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);

    indices_.push_back((int)indices_.size());
    indices_.push_back((int)indices_.size());
}

void Simple3DObject::pushToGPU() {
    if (!isStatic_ || vaoID_ == 0) {
        if (vaoID_ == 0) {
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(3, vboID_);
        }
        glBindVertexArray(vaoID_);
        if (vertices_.size()) {
            glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
            glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
            glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
        }

        if (colors_.size()) {
            glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
            glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(float), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
            glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);
        }

        if (indices_.size()) {
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
    sl::Transform tmp = sl::Transform::identity();
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

const sl::Translation CameraGL::ORIGINAL_FORWARD = sl::Translation(0, 0, 1);
const sl::Translation CameraGL::ORIGINAL_UP = sl::Translation(0, 1, 0);
const sl::Translation CameraGL::ORIGINAL_RIGHT = sl::Translation(1, 0, 0);

CameraGL::CameraGL(sl::Translation position, sl::Translation direction, sl::Translation vertical) {
    this->position_ = position;
    setDirection(direction, vertical);

    offset_ = sl::Translation(0, 0, 0);
    view_.setIdentity();
    updateView();
    setProjection(80, 80, 100.f, 900000.f);
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

SubMapObj::SubMapObj() {
    current_fc = 0;
    vaoID_ = 0;
}

SubMapObj::~SubMapObj() {
    current_fc = 0;
    if(vaoID_) {
        glDeleteBuffers(2, vboID_);
        glDeleteVertexArrays(1, &vaoID_);
    }
}

void SubMapObj::update(sl::PointCloudChunk &chunk) {
    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(2, vboID_);
    }

    glShadeModel(GL_SMOOTH);

	const auto nb_v = chunk.vertices.size();
    index.resize(nb_v);
    for (int c = 0; c < nb_v; c++) index[c] = c;
    
    glBindVertexArray(vaoID_);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[Shader::ATTRIB_VERTICES_POS]);
    glBufferData(GL_ARRAY_BUFFER, chunk.vertices.size() * sizeof(sl::float4), &chunk.vertices[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index.size() * sizeof(sl::uint1), &index[0], GL_DYNAMIC_DRAW);
    current_fc = (int)index.size();
    
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void SubMapObj::draw() {
    if(current_fc && vaoID_) {
        glBindVertexArray(vaoID_);
        glDrawElements(GL_POINTS, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}
