#include "GLViewer.hpp"

const GLchar* MESH_VERTEX_SHADER =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "layout(location = 1) in vec3 in_Color;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "out vec3 b_color;\n"
        "void main() {\n"
        "   b_color = in_Color.bgr;\n"
        "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
        "}";

const GLchar* MESH_FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "   color = vec4(b_color, 0.95);\n"
        "}";

const GLchar* POINTCLOUD_VERTEX_SHADER =
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
        "	return vec4(bIntValue / 255.0f, gIntValue / 255.0f, rIntValue / 255.0f, 1.0); \n"
        "}\n"
        "void main() {\n"
        // Decompose the 4th channel of the XYZRGBA buffer to retrieve the color of the point (1float to 4uint)
        "   b_color = decomposeFloat(in_VertexRGBA.a).xyz;\n"
        "	gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n"
        "}";

const GLchar* POINTCLOUD_FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 out_Color;\n"
        "void main() {\n"
        "   out_Color = vec4(b_color, 0.9);\n"
        "}";

const GLchar* VERTEX_SHADER_TEXTURE =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "layout(location = 1) in vec2 in_UVs;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "out vec2 UV;\n"
        "void main() {\n"
        "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
        "    UV = in_UVs;\n"
        "}\n";

const GLchar* FRAGMENT_SHADER_TEXTURE =
        "#version 330 core\n"
        "in vec2 UV;\n"
        "uniform sampler2D texture_sampler;\n"
        "void main() {\n"
        "    gl_FragColor = vec4(texture(texture_sampler, UV).bgr, 1.0);\n"
        "}\n";


using namespace sl;

GLViewer* GLViewer::currentInstance_ = nullptr;

GLViewer::GLViewer() : available(false) {
    if (currentInstance_ != nullptr) {
        delete currentInstance_;
    }
    currentInstance_ = this;

    mouseButton_[0] = mouseButton_[1] = mouseButton_[2] = false;

    clearInputs();
    previousMouseMotion_[0] = previousMouseMotion_[1] = 0;
}

GLViewer::~GLViewer() {
}

bool GLViewer::isAvailable() {
    glutMainLoopEvent();
    return available;
}

void GLViewer::exit() {
    if (available)
        glutLeaveMainLoop();

    available = false;
}

void GLViewer::init(int argc, char **argv, sl::Mat &image, sl::Mat & pointcloud, CUstream stream) {

    glutInit(&argc, argv);

    int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = glutGet(GLUT_SCREEN_HEIGHT);
    glutInitWindowSize(wnd_w * 0.8, wnd_h * 0.8);
    glutInitWindowPosition(wnd_w * 0.1, wnd_h * 0.1);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED Mapping");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

    glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);

    strm = stream;
   
    pc_render.initialize(pointcloud);
    camera_viewer.initialize(image);

    shader.it.set((GLchar*) MESH_VERTEX_SHADER, (GLchar*) MESH_FRAGMENT_SHADER);
    shader.MVPM = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

    shader_mesh.it.set((GLchar*) MESH_VERTEX_SHADER, (GLchar*) MESH_FRAGMENT_SHADER);
    shader_mesh.MVPM = glGetUniformLocation(shader_mesh.it.getProgramId(), "u_mvpMatrix");

    shader_fpc.it.set((GLchar*) POINTCLOUD_VERTEX_SHADER, (GLchar*) POINTCLOUD_FRAGMENT_SHADER);
    shader_fpc.MVPM = glGetUniformLocation(shader_fpc.it.getProgramId(), "u_mvpMatrix");

    camera_ = CameraGL(sl::Translation(0, 2, 2.000), sl::Translation(0, 0, -0.1));
    sl::Rotation rot;
    rot.setEulerAngles(sl::float3(-45,0,0), false);
    camera_.setRotation(rot);

    draw_mesh_as_wire = false;
    draw_live_point_cloud = true;
    dark_background = true;

    // Map glut function on this class methods
    glutDisplayFunc(GLViewer::drawCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutMotionFunc(GLViewer::mouseMotionCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardFunc(GLViewer::keyPressedCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);

    available = true;
}

void GLViewer::render() {
    if (available) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if(dark_background)
            glClearColor(59 / 255.f, 63 / 255.f, 69 / 255.f, 1.0f);
        else
            glClearColor(211 / 255.f, 220 / 255.f, 232 / 255.f, 1.0f);

        update();
        draw();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

void GLViewer::updateCameraPose(sl::Transform p, sl::POSITIONAL_TRACKING_STATE state_) {
    cam_pose = p;
    pc_render.pushNewPC(strm);
    camera_viewer.pushNewImage(strm);
}

void GLViewer::updateMap(sl::FusedPointCloud &fpc){
    map_fpc.resize(fpc.chunks.size());
    int c_id = 0;
    for (auto& it : map_fpc) {
        auto& chunk = fpc.chunks[c_id++];
        if (chunk.has_been_updated)
            it.add(chunk.vertices);
    }
}

void GLViewer::updateMap(sl::Mesh &mesh) {
    map_mesh.resize(mesh.chunks.size());
    int c_id = 0;
    for (auto& it : map_mesh) {
        auto& chunk = mesh.chunks[c_id++];
        if (chunk.has_been_updated)
            it.add(chunk.vertices, chunk.triangles, chunk.colors);
    }
}

void GLViewer::update() {

    if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP) {
        currentInstance_->exit();
        return;
    }

    if (keyStates_['o'] == KEY_STATE::UP || keyStates_['O'] == KEY_STATE::UP) 
        point_size = point_size - 0.2;    

    if (keyStates_['p'] == KEY_STATE::UP || keyStates_['P'] == KEY_STATE::UP) 
        point_size = point_size + 0.2;
    
    if (keyStates_['w'] == KEY_STATE::UP || keyStates_['W'] == KEY_STATE::UP) 
        draw_mesh_as_wire = !draw_mesh_as_wire;
    
    if (keyStates_['l'] == KEY_STATE::UP || keyStates_['L'] == KEY_STATE::UP) 
        draw_live_point_cloud = !draw_live_point_cloud;

    if (keyStates_['d'] == KEY_STATE::UP || keyStates_['D'] == KEY_STATE::UP) 
        dark_background = !dark_background;
       
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
        if (mouseWheelPosition_ > 0 /* && distance > camera_.getZNear()*/) // zoom
            camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * -1);

        else if (/*distance < camera_.getZFar()*/ mouseWheelPosition_ < 0) // unzoom
            camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY);
    }

    for (auto &it : map_mesh)
        it.pushToGPU();

    for (auto& it : map_fpc)
        it.pushToGPU();

    //pointCloud_.mutexData.unlock();
    camera_.update();
    clearInputs();
}

void GLViewer::draw() {
    sl::Transform vpMatrix = camera_.getViewProjectionMatrix();

    glPolygonMode(GL_FRONT, GL_LINE);
    glPolygonMode(GL_BACK, GL_LINE);
    glLineWidth(2.f);
    glPointSize(point_size);

    glUseProgram(shader.it.getProgramId());

    sl::Transform pose_ = vpMatrix * cam_pose;
    glUniformMatrix4fv(shader.MVPM, 1, GL_FALSE, sl::Transform::transpose(pose_).m);
    camera_viewer.frustum.draw();
    glUseProgram(0);

    if(map_fpc.size()){
        glUseProgram(shader_fpc.it.getProgramId());
        glUniformMatrix4fv(shader_fpc.MVPM, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);
        for (auto &it: map_fpc)
            it.draw();
        glUseProgram(0);
    }

    if(map_mesh.size()){
        glUseProgram(shader_mesh.it.getProgramId());
        glUniformMatrix4fv(shader_mesh.MVPM, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);
        for (auto &it: map_mesh)
            it.draw(draw_mesh_as_wire);
        glUseProgram(0);
    }

    if(draw_live_point_cloud)
        pc_render.draw(pose_);
    camera_viewer.draw(pose_);
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

Simple3DObject::Simple3DObject() {
    vaoID_ = 0;
    drawingType_ = GL_TRIANGLES;
    isStatic_ = need_update = false;
}

Simple3DObject::~Simple3DObject() {
    clear();
    if (vaoID_ != 0) {
        glDeleteBuffers(3, vboID_);
        glDeleteVertexArrays(1, &vaoID_);
    }
}

void Simple3DObject::addPoint(sl::float3 pt, sl::float3 clr){
    vertices_.push_back(pt);
    colors_.push_back(clr);
    indices_.push_back((int) indices_.size());
    need_update = true;
}
void Simple3DObject::addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float3 clr){
    addPoint(p1, clr);
    addPoint(p2, clr);
    addPoint(p3, clr);
}

void Simple3DObject::pushToGPU() {
    if(!need_update) return;

    if (!isStatic_ || vaoID_ == 0) {
        if (vaoID_ == 0) {
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(3, vboID_);
        }
        glBindVertexArray(vaoID_);
        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(sl::float3), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(sl::float3), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof (unsigned int), &indices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        need_update = false;
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



// ____________

FpcObj::FpcObj() {
    vaoID_ = 0;
    need_update = false;
}

FpcObj::~FpcObj() {
    clear();
    if (vaoID_ != 0) {
        glDeleteBuffers(3, vboID_);
        glDeleteVertexArrays(1, &vaoID_);
    }
}

void FpcObj::add(std::vector<sl::float4>&pts) {
    clear();
    vertices_ = pts;
    for (int i = 0; i < vertices_.size(); i++)
        indices_.push_back(i);
    need_update = true;
}

void FpcObj::pushToGPU() {
    if (!need_update) return;

    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(2, vboID_);
    }
    glBindVertexArray(vaoID_);
    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, vertices_.size() * 4 * sizeof (float), &vertices_[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof (unsigned int), &indices_[0], GL_DYNAMIC_DRAW);
    nb_v = indices_.size();
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    need_update = false;
}

void FpcObj::clear() {
    vertices_.clear();
    indices_.clear();
    nb_v = 0;
}

void FpcObj::draw() {
    if (nb_v && vaoID_) {
        glDisable(GL_BLEND);
        glBindVertexArray(vaoID_);
        glDrawElements(GL_POINTS, (GLsizei) nb_v, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glEnable(GL_BLEND);
    }
}
// ____________

MeshObject::MeshObject() {
    current_fc = 0;
    vaoID_ = 0;
    need_update = false;
}

MeshObject::~MeshObject() {
}

void MeshObject::add(std::vector<sl::float3> &vertices, std::vector<sl::uint3> &triangles, std::vector<sl::uchar3> &colors) {
    vert = vertices;
    clr = colors;
    faces = triangles;
    need_update = true;
}

void MeshObject::pushToGPU() {
    if (!need_update) return;
    if (faces.empty()) return;

    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(3, vboID_);
    }

    glBindVertexArray(vaoID_);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof (vert[0]), &vert[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ARRAY_BUFFER, clr.size() * sizeof (clr[0]), &clr[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
    current_fc = faces.size() * sizeof (faces[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, current_fc, &faces[0], GL_DYNAMIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    need_update = false;
}

void MeshObject::draw(bool draw_wire) {
    if ((current_fc > 0) && (vaoID_ != 0)) {
        auto type = draw_wire ? GL_LINE : GL_FILL;
        glPolygonMode(GL_FRONT, type);
        glPolygonMode(GL_BACK, type);
            
        glBindVertexArray(vaoID_);
        glDrawElements(GL_TRIANGLES, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}

Shader::Shader(const GLchar* vs, const GLchar* fs) {
    set(vs, fs);
}

void Shader::set(const GLchar* vs, const GLchar* fs) {
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
    glBindAttribLocation(programId_, ATTRIB_COLOR_POS, "in_Color");

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
    if (verterxId_ != 0 && glIsShader(verterxId_))
        glDeleteShader(verterxId_);
    if (fragmentId_ != 0 && glIsShader(fragmentId_))
        glDeleteShader(fragmentId_);
    if (programId_ != 0 && glIsProgram(programId_))
        glDeleteProgram(programId_);
}

GLuint Shader::getProgramId() {
    return programId_;
}

bool Shader::compile(GLuint &shaderId, GLenum type, const GLchar* src) {
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

PointCloud::PointCloud() {

}

PointCloud::~PointCloud() {
    close();
}

void PointCloud::close() {
    if (refMat.isInit()) {
        auto err = cudaGraphicsUnmapResources(1, &bufferCudaID_, 0);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA UnmapResources (" << err << ")" << std::endl;
        glDeleteBuffers(1, &bufferGLID_);
    }
}

void PointCloud::initialize(sl::Mat &ref) {
    refMat = ref;

    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, refMat.getResolution().area() * 4 * sizeof (float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess)
        std::cerr << "Error: CUDA - OpenGL Interop failed (" << err << ")" << std::endl;

    err = cudaGraphicsMapResources(1, &bufferCudaID_, 0);
    if (err != cudaSuccess)
        std::cerr << "Error: CUDA MapResources (" << err << ")" << std::endl;

    err = cudaGraphicsResourceGetMappedPointer((void**) &xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_);
    if (err != cudaSuccess)
        std::cerr << "Error: CUDA GetMappedPointer (" << err << ")" << std::endl;

    shader_.set(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");
}

void PointCloud::pushNewPC(CUstream strm) {
    if (refMat.isInit())
        cudaMemcpyAsync(xyzrgbaMappedBuf_, refMat.getPtr<sl::float4>(sl::MEM::GPU), numBytes_, cudaMemcpyDeviceToDevice, strm);
}

void PointCloud::draw(const sl::Transform& vp) {
    if (refMat.isInit()) {
        glDisable(GL_BLEND);
        glUseProgram(shader_.getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_TRUE, vp.m);

        glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glDrawArrays(GL_POINTS, 0, refMat.getResolution().area());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
        glEnable(GL_BLEND);
    }
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
    setProjection(90, 90, 0.5f, 50.f);
    updateVPMatrix();
}

CameraGL::~CameraGL() {
}

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
    updateVPMatrix();
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

CameraViewer::CameraViewer() {

}

CameraViewer::~CameraViewer() {
	close();
}

void CameraViewer::close() {
	if (ref.isInit()) {	
        
        auto err = cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);
	    if (err) std::cout << "err 3 " << err << " " << cudaGetErrorString(err) << "\n";

		glDeleteTextures(1, &texture);
		glDeleteBuffers(3, vboID_);
		glDeleteVertexArrays(1, &vaoID_);
	}
}

bool CameraViewer::initialize(sl::Mat &im) {

    // Create 3D axis
    float fx,fy,cx,cy;
    fx = fy = 1400;
    float width, height;
    width = 2208;
    height = 1242;
    cx = width /2;
    cy = height /2;
        
    float Z_ = .5f;
    sl::float3 toOGL(1,-1,-1);
    sl::float3 cam_0(0, 0, 0);
    sl::float3 cam_1, cam_2, cam_3, cam_4;

    float fx_ = 1.f / fx;
    float fy_ = 1.f / fy;

    cam_1.z = Z_;
    cam_1.x = (0 - cx) * Z_ *fx_;
    cam_1.y = (0 - cy) * Z_ *fy_ ;
    cam_1 *= toOGL;

    cam_2.z = Z_;
    cam_2.x = (width - cx) * Z_ *fx_;
    cam_2.y = (0 - cy) * Z_ *fy_;
    cam_2 *= toOGL;

    cam_3.z = Z_;
    cam_3.x = (width - cx) * Z_ *fx_;
    cam_3.y = (height - cy) * Z_ *fy_;
    cam_3 *= toOGL;

    cam_4.z = Z_;
    cam_4.x = (0 - cx) * Z_ *fx_;
    cam_4.y = (height - cy) * Z_ *fy_;
    cam_4 *= toOGL;

    sl::float3 clr(0.1,0.5,0.7);

    frustum.addFace(cam_0, cam_1, cam_2, clr);
    frustum.addFace(cam_0, cam_2, cam_3, clr);
    frustum.addFace(cam_0, cam_3, cam_4, clr);
    frustum.addFace(cam_0, cam_4, cam_1, clr);    
    frustum.setDrawingType(GL_TRIANGLES);
    frustum.pushToGPU();
    
    vert.push_back(cam_1);
    vert.push_back(cam_2);
    vert.push_back(cam_3);
    vert.push_back(cam_4);

    uv.push_back(sl::float2(0,0));
    uv.push_back(sl::float2(1,0));
    uv.push_back(sl::float2(1,1));
    uv.push_back(sl::float2(0,1));
    
    faces.push_back(sl::uint3(0,1,2));
    faces.push_back(sl::uint3(0,2,3));

    ref = im;
	shader.set(VERTEX_SHADER_TEXTURE, FRAGMENT_SHADER_TEXTURE);
    shMVPMatrixLocTex_ = glGetUniformLocation(shader.getProgramId(), "u_mvpMatrix");

	glGenVertexArrays(1, &vaoID_);
	glGenBuffers(3, vboID_);

    glBindVertexArray(vaoID_);
    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof(sl::float3), &vert[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof(sl::float2), &uv[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(sl::uint3), &faces[0], GL_DYNAMIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    auto res = ref.getResolution();
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.width, res.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_gl_ressource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	if (err) std::cout << "err alloc " << err << " " << cudaGetErrorString(err) << "\n";
	glDisable(GL_TEXTURE_2D);
	
	err = cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
	if (err) std::cout << "err 0 " << err << " " << cudaGetErrorString(err) << "\n";
	err = cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
	if (err) std::cout << "err 1 " << err << " " << cudaGetErrorString(err) << "\n";

	return (err == cudaSuccess);
}

void CameraViewer::pushNewImage(CUstream strm) {
	if (!ref.isInit())  return;
	auto err = cudaMemcpy2DToArrayAsync(ArrIm, 0, 0, ref.getPtr<sl::uchar1>(sl::MEM::GPU), ref.getStepBytes(sl::MEM::GPU), ref.getPixelBytes() * ref.getWidth(), ref.getHeight(), cudaMemcpyDeviceToDevice, strm);
	if (err) std::cout << "err 2 " << err << " " << cudaGetErrorString(err) << "\n";
}

void CameraViewer::draw(sl::Transform vpMatrix) {
	if (!ref.isInit())  return;

    glUseProgram(shader.getProgramId());
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    
    glUniformMatrix4fv(shMVPMatrixLocTex_, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);
    glBindTexture(GL_TEXTURE_2D, texture);
        
    glBindVertexArray(vaoID_);
    glDrawElements(GL_TRIANGLES, (GLsizei)faces.size()*3, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
    glUseProgram(0);
}
