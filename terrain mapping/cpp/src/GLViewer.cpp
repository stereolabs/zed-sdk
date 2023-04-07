#include "GLViewer.hpp"

GLchar* MESH_VERTEX_SHADER =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "uniform vec3 u_color;\n"
        "out vec3 b_color;\n"
        "void main() {\n"
        "   b_color = u_color;\n"
        "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
        "}";

GLchar* MESH_FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "   color = vec4(b_color, 0.8);\n"
        "}";

GLchar* MESH_VERTEX_SHADER_LIGHT =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "layout(location = 1) in vec3 in_Color;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "out vec3 b_color;\n"
        "void main() {\n"
        "   b_color = in_Color;\n"
        "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
        "}";

GLchar* MESH_FRAGMENT_SHADER_LIGHT =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "   color = vec4(b_color, 0.95);\n"
        "}";


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
        "	return vec4(bIntValue / 255.0f, gIntValue / 255.0f, rIntValue / 255.0f, 1.0); \n"
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
        "   out_Color = vec4(b_color, 0.9);\n"
        "}";

using namespace sl;

GLViewer* GLViewer::currentInstance_ = nullptr;

GLViewer::GLViewer() : available(false) {
    if (currentInstance_ != nullptr) {
        delete currentInstance_;
    }
    currentInstance_ = this;

    cb = 0.847058f;
    cg = 0.596078f;
    cr = 0.203921f;

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

void GLViewer::init(sl::Mat &ref) {

    char *argv[1];
    argv[0] = "\0";
    int argc = 1;
    glutInit(&argc, argv);

    int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = glutGet(GLUT_SCREEN_HEIGHT);
    glutInitWindowSize(wnd_w * 0.8, wnd_h * 0.8);
    glutInitWindowPosition(wnd_w * 0.1, wnd_h * 0.1);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("Terrain");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

    glEnable(GL_DEPTH_TEST);

    const int nb_r = 5;
    const int num_segments = 48;
    floor_grid.resize(nb_r);
    float rad = 0.05f;
    float sacler = 1.f / nb_r;
    sl::float4 clrP(0.f / 255.f, 174.f / 255.f, 236.f / 255.f, 1.f);
    sl::float4 clrc;
    for (int i = 0; i < nb_r; i++) {
        floor_grid[i].setStatic(true);
        for (int ii = 0; ii < num_segments; ii++) {
            float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);
            floor_grid[i].addPoint(rad * cosf(theta), 0, rad * sinf(theta), clrP.r, clrP.g, clrP.b);
        }
        if (i == 0)
            rad = 1.f;
        else
            rad += 1.f;
        floor_grid[i].setDrawingType(GL_LINE_LOOP);
        floor_grid[i].pushToGPU();
    }

    axis.setStatic(true);
    axis.setDrawingType(GL_LINES);

    axis.addPoint(0, 0, 0, 0.7, 0.7, 0.7);
    axis.addPoint(1, 0, 0, 1, 0, 0);
    axis.addPoint(0, 0, 0, 0.7, 0.7, 0.7);
    axis.addPoint(0, 1, 0, 0, 1, 0);
    axis.addPoint(0, 0, 0, 0.7, 0.7, 0.7);
    axis.addPoint(0, 0, -1, 0, 0, 1);
    axis.pushToGPU();

    pc.initialize(ref);

    shader_clr_fix.it = Shader((GLchar*) MESH_VERTEX_SHADER, (GLchar*) MESH_FRAGMENT_SHADER);
    shader_clr_fix.MVPM = glGetUniformLocation(shader_clr_fix.it.getProgramId(), "u_mvpMatrix");
    shColorLoc = glGetUniformLocation(shader_clr_fix.it.getProgramId(), "u_color");

    shader.it = Shader((GLchar*) MESH_VERTEX_SHADER_LIGHT, (GLchar*) MESH_FRAGMENT_SHADER_LIGHT);
    shader.MVPM = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

    shader_fpc.it = Shader((GLchar*) POINTCLOUD_VERTEX_SHADER, (GLchar*) POINTCLOUD_FRAGMENT_SHADER);
    shader_fpc.MVPM = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

    camera_ = CameraGL(sl::Translation(0.0f, 4.0f, 4.0f), sl::Translation(0, 0, -0.1));
    sl::Rotation rot;
    rot.setEulerAngles(sl::float3(-0.75, 0, 0));
    camera_.setRotation(rot);


    bkgd_col = sl::float3(59, 63, 69) / 255.f;

    // Map glut function on this class methods
    glutDisplayFunc(GLViewer::drawCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutMotionFunc(GLViewer::mouseMotionCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardFunc(GLViewer::keyPressedCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    available = true;
}

void GLViewer::render() {
    if (available) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(bkgd_col.b, bkgd_col.g, bkgd_col.r, 1.0f);
        glLineWidth(2.f);
        update();
        draw();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

void GLViewer::updateCameraPose(sl::Transform p) {
    pose = p;
    vecPath.push_back(p.getTranslation());
}

void GLViewer::updatePc(sl::FusedPointCloud&pc) {
    map_fpc.resize(pc.chunks.size());
    int c_id = 0;
    for (auto& it : map_fpc) {
        auto& chunk = pc.chunks[c_id++];
        if (chunk.has_been_updated)
            it.add(chunk.vertices);
    }
}

void GLViewer::updateMesh(sl::TerrainMesh &mesh, sl::REFERENCE_FRAME ref) {
    map_mesh.resize(mesh.chunks.size());
    int c_id = 0;
    for (auto& it : map_mesh) {
        auto& chunk = mesh.chunks[c_id++];
        if (chunk.has_been_updated)
            it.addMesh(chunk.vertices, chunk.faces, chunk.colors);
    }

    camera_centric_terrain = (ref == REFERENCE_FRAME::CAMERA);
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

    if (keyStates_['w'] == KEY_STATE::UP || keyStates_['W'] == KEY_STATE::UP) {
        draw_wire_frame = !draw_wire_frame;
    }
    if (keyStates_['o'] == KEY_STATE::UP || keyStates_['O'] == KEY_STATE::UP) {
        point_size = point_size - 0.2;
    }
    if (keyStates_['p'] == KEY_STATE::UP || keyStates_['P'] == KEY_STATE::UP) {
        point_size = point_size + 0.2;
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

    pc.pushNewPC();

    //pointCloud_.mutexData.unlock();
    camera_.update();
    clearInputs();
}

void GLViewer::draw() {
    sl::Transform vpMatrix = camera_.getViewProjectionMatrix();

    glUseProgram(shader.it.getProgramId());

    // Draw triangles
    glUniformMatrix4fv(shader.MVPM, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);

    //axis.draw();

    for (auto& it : floor_grid)
        it.draw();

    sl::Transform pose_no_rot = pose;
    auto eul = pose_no_rot.getEulerAngles();
    eul.x = 0;
    eul.z = 0;
    pose_no_rot.setEulerAngles(eul);

    sl::Transform vpM_pose, vp_map_pose;
    if(!camera_centric_terrain) {
        vpM_pose = vpMatrix * pose_no_rot;
        vp_map_pose = vpMatrix;
    } else {
        vpM_pose = vpMatrix;
        vp_map_pose = vpMatrix * sl::Transform::inverse(pose);
    }

    glUniformMatrix4fv(shader.MVPM, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);

    for (auto& it : map_mesh)
        it.draw(1);

    glUseProgram(0);

    if (draw_wire_frame) {
        // Draw wireframe
        glUseProgram(shader_clr_fix.it.getProgramId());
        //if (camera_centric_terrain)
            glUniformMatrix4fv(shader_clr_fix.MVPM, 1, GL_FALSE, sl::Transform::transpose(vpM_pose).m);
        sl::float3 clr_wire(0.1, 0.1, 0.1);
        glUniform3fv(shColorLoc, 1, clr_wire.v);
        for (auto& it : map_mesh)
            it.draw(0);
        glUseProgram(0);
    }

    glPointSize(point_size * 2);
    glUseProgram(shader_fpc.it.getProgramId());
    glUniformMatrix4fv(shader_fpc.MVPM, 1, GL_FALSE, sl::Transform::transpose(vp_map_pose).m);
    for (auto& it : map_fpc)
        it.draw();
    glUseProgram(0);

    glPointSize(point_size);
    pc.draw(vpMatrix);
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
#if 0
    float hfov = (180.0f / M_PI) * (2.0f * atan(width / (2.0f * 500)));
    float vfov = (180.0f / M_PI) * (2.0f * atan(height / (2.0f * 500)));
    currentInstance_->camera_.setProjection(hfov, vfov * (float) height / (float) width, currentInstance_->camera_.getZNear(), currentInstance_->camera_.getZFar());
#endif
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
    isStatic_ = false;
}

Simple3DObject::~Simple3DObject() {
    clear();
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
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof (float), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof (float), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof (unsigned int), &indices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);

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
        glBindVertexArray(vaoID_);
        glDrawElements(GL_POINTS, (GLsizei) nb_v, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
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

void MeshObject::addMesh(std::vector<sl::float3> &vertices, std::vector<sl::uint4> &triangles, std::vector<sl::float3> &colors) {
    vert = vertices;
    clr = colors;
    faces = triangles;
    need_update = true;
}

void MeshObject::pushToGPU() {
    if (!need_update) return;
    if (faces.empty()) return;

    need_update = false;

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
    glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
    current_fc = faces.size() * sizeof (faces[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, current_fc, &faces[0], GL_DYNAMIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MeshObject::draw(bool fill) {

    if ((current_fc > 0) && (vaoID_ != 0)) {

        if (fill) {
            glPolygonMode(GL_FRONT, GL_FILL);
            glPolygonMode(GL_BACK, GL_FILL);
        } else {
            glPolygonMode(GL_FRONT, GL_LINE);
            glPolygonMode(GL_BACK, GL_LINE);
            glLineWidth(2.f);
        }

        glBindVertexArray(vaoID_);
        glDrawElements(GL_QUADS, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
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
    width_ = refMat.getWidth();
    height_ = refMat.getHeight();

    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, width_ * height_ * 4 * sizeof (float), 0, GL_DYNAMIC_DRAW);
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

    shader_ = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");
}

void PointCloud::pushNewPC() {
    if (refMat.isInit())
        cudaMemcpy(xyzrgbaMappedBuf_, refMat.getPtr<sl::float4>(sl::MEM::GPU), numBytes_, cudaMemcpyDeviceToDevice);
}

void PointCloud::draw(const sl::Transform& vp) {
    if (refMat.isInit()) {
        glUseProgram(shader_.getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_TRUE, vp.m);

        glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glDrawArrays(GL_POINTS, 0, width_ * height_);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
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
    setProjection(70, 70, 0.5f, 50.f);
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
