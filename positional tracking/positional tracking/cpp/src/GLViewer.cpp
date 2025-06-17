#include "GLViewer.hpp"

const GLchar* MESH_VERTEX_SHADER_ =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "layout(location = 1) in vec4 in_Color;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "out vec4 b_color;\n"
        "void main() {\n"
        "   b_color = in_Color;\n"
        "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
        "}";

const GLchar* MESH_FRAGMENT_SHADER_ =
        "#version 330 core\n"
        "in vec4 b_color;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "   color = b_color;\n"
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

const GLchar* IMAGE_VERTEX_SHADER =
        "#version 330\n"
        "layout(location = 0) in vec2 vert;\n"
        "layout(location = 1) in vec3 vert_tex;\n"
        "out vec2 UV;"
        "void main() {\n"
        "	UV = vert;\n"
        "	gl_Position = vec4(vert_tex, 1);\n"
        "}\n";
        
const GLchar* IMAGE_FRAGMENT_SHADER =
        "#version 330 core\n"
        " in vec2 UV;\n"
        " out vec4 color;\n"
        " uniform sampler2D texImage;\n"
        " void main() {\n"
        "	vec2 scaler = vec2(UV.x, UV.y);\n"
        "	color = vec4(texture(texImage, scaler).zyxw);\n"
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

using namespace sl;

float const to_f = 1.f/ 255.f;
const sl::float4 clr_lime(217*to_f,255*to_f,66*to_f, 1.f);
const sl::float4 clr_iron(194*to_f,194*to_f,194*to_f, 0.2f);
const sl::float4 clr_pearl(242*to_f,242*to_f,242*to_f, 0.7f);

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

    wnd_size = sl::Resolution(wnd_w * 0.8, wnd_h * 0.8);

    glutInitWindowSize(wnd_size.width, wnd_size.height);
    glutInitWindowPosition(wnd_w * 0.1, wnd_h * 0.1);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED Positional Tracking");

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

    shader.it.set((GLchar*) MESH_VERTEX_SHADER_, (GLchar*) MESH_FRAGMENT_SHADER_);
    shader.MVPM = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

    ZED_path.setDrawingType(GL_LINE_STRIP);

    lms.setDrawingType(GL_POINTS);
    lms_tracked.setDrawingType(GL_LINES);

    origin_axis.setStatic(true);
    origin_axis.setDrawingType(GL_LINES);
    origin_axis.addPoint(sl::float3(0,0,0), sl::float4(1,0,0,1));
    origin_axis.addPoint(sl::float3(1,0,0), sl::float4(1,0,0,1));

    origin_axis.addPoint(sl::float3(0,0,0), sl::float4(0,1,0,1));
    origin_axis.addPoint(sl::float3(0,1,0), sl::float4(0,1,0,1));

    origin_axis.addPoint(sl::float3(0,0,0), sl::float4(0,0,1,1));
    origin_axis.addPoint(sl::float3(0,0,1), sl::float4(0,0,1,1));
    origin_axis.pushToGPU();

    camera_ = CameraGL(sl::Translation(0, 2, 5.000), sl::Translation(0, 0, -0.1));

    sl::Rotation rot;
    rot.setEulerAngles(sl::float3(-25,0,0), false);
    camera_.setRotation(rot);

    draw_live_point_cloud = true;
    dark_background = true;
    draw_landmark = false;
    follow_cam = false;

    // Map glut function on this class methods
    glutDisplayFunc(GLViewer::drawCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutMotionFunc(GLViewer::mouseMotionCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardFunc(GLViewer::keyPressedCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
	glutSpecialFunc(GLViewer::specialKeyReleasedCallback);

    available = true;
}

void GLViewer::render() {
    if (available) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if(dark_background)
            glClearColor(59 / 255.f, 63 / 255.f, 69 / 255.f, 1.f);
        else
            glClearColor(211 / 255.f, 220 / 255.f, 232 / 255.f, 1.f);

        update();
        draw();
        printText();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

inline
void safe_glutBitmapString(void *font, const char *str) {
    for (size_t x = 0; x < strlen(str); ++x)
        glutBitmapCharacter(font, str[x]);
}

inline
std::string setTxt(sl::float3 value) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

void GLViewer::printText() {
    glViewport(0, 0, wnd_size.width, wnd_size.height);
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

    float dark_clr = 0.12f;
    std::string odom_status = "POSITIONAL TRACKING STATUS: ";
    
    glColor3f(dark_clr, dark_clr, dark_clr);
    glRasterPos2i(start_w, start_h);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, odom_status.c_str());

    (trackState.tracking_fusion_status != sl::POSITIONAL_TRACKING_FUSION_STATUS::UNAVAILABLE) ? glColor3f(0.2f, 0.65f, 0.2f) : glColor3f(0.85f, 0.2f, 0.2f);
    std::string track_str = (sl::toString(trackState.tracking_fusion_status).c_str());
    glRasterPos2i(start_w + 300, start_h);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, track_str.c_str());
    
    glColor3f(dark_clr, dark_clr, dark_clr);
    glRasterPos2i(start_w, start_h - 20);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, "Translation (m) :");

    glColor3f(0.4980f, 0.5490f, 0.5529f);
    glRasterPos2i(155, start_h - 20);

    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, setTxt(cam_pose.getTranslation()).c_str());

    glColor3f(dark_clr, dark_clr, dark_clr);
    glRasterPos2i(start_w, start_h - 40);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, "Rotation   (rad) :");

    glColor3f(0.4980f, 0.5490f, 0.5529f);
    glRasterPos2i(155, start_h - 40);
    safe_glutBitmapString(GLUT_BITMAP_HELVETICA_18, setTxt(cam_pose.getRotationVector()).c_str());

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void GLViewer::updateCameraPose(sl::Transform p, sl::PositionalTrackingStatus state_) {
    cam_pose = p;
    ZED_path.addPoint(cam_pose.getTranslation(), clr_pearl);
    if(draw_live_point_cloud)
        pc_render.pushNewPC(strm);
    camera_viewer.pushNewImage(strm);
    trackState = state_;
}

void GLViewer::pushTrackedLM(std::vector<sl::float3> &lm){
    lms_tracked.clear();
    sl::float3 cam_p = cam_pose.getTranslation();
    for(auto &it:lm){
        lms_tracked.addPoint(cam_p, clr_iron);
        lms_tracked.addPoint(it, clr_iron);
    }
}

void GLViewer::pushLM(std::map<uint64_t, sl::Landmark> &lm){
    lms.clear();
    for(auto &it:lm)
        lms.addPoint(it.second.position, clr_lime);
}

constexpr float MOUSE_RT_SENSITIVITY = 0.05f;
constexpr float MOUSE_UZ_SENSITIVITY = 2.f;

void GLViewer::update() {

    if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP) {
        currentInstance_->exit();
        return;
    }
    
    if (keyStates_['l'] == KEY_STATE::UP || keyStates_['L'] == KEY_STATE::UP) 
        draw_live_point_cloud = !draw_live_point_cloud;

    if (keyStates_['d'] == KEY_STATE::UP || keyStates_['D'] == KEY_STATE::UP) 
        dark_background = !dark_background;
       
    if(keyStates_[32] == KEY_STATE::UP) 
        currentInstance_->SPLIT_DISPLAY = !currentInstance_->SPLIT_DISPLAY;

    if (keyStates_['f'] == KEY_STATE::UP || keyStates_['F'] == KEY_STATE::UP) 
        follow_cam = !follow_cam;
        
    if (keyStates_['m'] == KEY_STATE::UP || keyStates_['M'] == KEY_STATE::UP) 
        draw_landmark = !draw_landmark;

    // Rotate camera with mouse
    if (mouseButton_[MOUSE_BUTTON::LEFT]) {
        camera_.rotate(sl::Rotation((float) mouseMotion_[1] * MOUSE_RT_SENSITIVITY, camera_.getRight()));
        camera_.rotate(sl::Rotation((float) mouseMotion_[0] * MOUSE_RT_SENSITIVITY, camera_.getVertical() * -1.f));
    }

    // Translate camera with mouse
    if (mouseButton_[MOUSE_BUTTON::RIGHT]) {
        camera_.translate(camera_.getUp() * (float) mouseMotion_[1] * MOUSE_RT_SENSITIVITY * control_magnitude);
        camera_.translate(camera_.getRight() * (float) mouseMotion_[0] * MOUSE_RT_SENSITIVITY * control_magnitude);
    }
    
    // Zoom in with mouse wheel
    if (mouseWheelPosition_ != 0) {
        if (mouseWheelPosition_ > 0 /* && distance > camera_.getZNear()*/) // zoom
            camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * -1 * control_magnitude);

        else if (/*distance < camera_.getZFar()*/ mouseWheelPosition_ < 0) // unzoom
            camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * control_magnitude);
    }

    ZED_path.pushToGPU();

    if(draw_landmark){
        lms.pushToGPU();
        lms_tracked.pushToGPU();
    }

    camera_.update();
    clearInputs();
}

void GLViewer::draw() {

    if(SPLIT_DISPLAY){

        glPolygonMode(GL_FRONT, GL_FILL);

        int half_w = wnd_size.width *.5f;

        // DRAW 2D image part
        int h = half_w / camera_viewer.aspect_ratio;
        int y = (wnd_size.height - h) / 2.f;

        glViewport(0, y, half_w, h);
        camera_viewer.draw2D();
        
        // DRAW 3D
        glViewport(half_w, 0, half_w, wnd_size.height);
    }else
        glViewport(0, 0, wnd_size.width, wnd_size.height);

    sl::Transform vpMatrix = camera_.getViewProjectionMatrix();

    glPolygonMode(GL_FRONT, GL_LINE);
    glPolygonMode(GL_BACK, GL_LINE);
    glLineWidth(3.f);

    glUseProgram(shader.it.getProgramId());

    if(follow_cam){
        sl::Transform pose_w_t = cam_pose;
        auto ea = pose_w_t.getEulerAngles();
        ea.x = 0;
        //ea.y = 0;
        ea.z = 0;
        pose_w_t.setEulerAngles(ea);
        vpMatrix = vpMatrix * sl::Transform::inverse(pose_w_t);
    }
    
    glUniformMatrix4fv(shader.MVPM, 1, GL_TRUE, vpMatrix.m);
    origin_axis.draw();

    ZED_path.draw();

    if(draw_landmark){
        glPointSize(3.f);
        lms.draw();
        lms_tracked.draw();
    }

    glPointSize(2.f);

    sl::Transform pose_ = vpMatrix * cam_pose;
    glUniformMatrix4fv(shader.MVPM, 1, GL_TRUE, pose_.m);
    camera_viewer.frustum.draw();
    glUseProgram(0);

    if(draw_live_point_cloud)
        pc_render.draw(pose_);
}

void GLViewer::clearInputs() {
    mouseMotion_[0] = mouseMotion_[1] = 0;
    mouseWheelPosition_ = 0;
    for (unsigned int i = 0; i < 256; ++i)
        if (keyStates_[i] != KEY_STATE::FREE)
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
    currentInstance_->wnd_size = sl::Resolution(width, height);
}

void GLViewer::keyPressedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::DOWN;
}

void GLViewer::keyReleasedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::UP;
    glutPostRedisplay();
}

void GLViewer::specialKeyReleasedCallback(int c, int x, int y){
    switch (c)
    {
    case 112: /* Shift */ 
        std::cout<<"switch to soft control"<<std::endl;
        currentInstance_->control_magnitude = 0.2f; 
        break;
    case 114: /* Ctrl */
        std::cout<<"switch to strong control"<<std::endl;
        currentInstance_->control_magnitude = 5.f;
        break;
    case 116: /* Alt */ 
        std::cout<<"switch to regular control"<<std::endl;
        currentInstance_->control_magnitude = 1.f;
        break;
    }
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

void Simple3DObject::addPoint(sl::float3 pt, sl::float4 clr){
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
        glBindBuffer(GL_ARRAY_BUFFER, vboID_[Shader::ATTRIB_VERTICES_POS]);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(sl::float3), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[Shader::ATTRIB_COLOR_POS]);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(sl::float4), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
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
    if(vaoID_){
        glBindVertexArray(vaoID_);
        glDrawElements(drawingType_, (GLsizei) indices_.size(), GL_UNSIGNED_INT, 0);
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
        //glDisable(GL_BLEND);
        glUseProgram(shader_.getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_TRUE, vp.m);

        glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glDrawArrays(GL_POINTS, 0, refMat.getResolution().area());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
        //glEnable(GL_BLEND);
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
    setProjection(90, 90, 1.f, 1000.f);
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
        
    float Z_ = .15f;
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

    frustum.addPoint(cam_0, clr_lime);
    frustum.addPoint(cam_1, clr_lime);

    frustum.addPoint(cam_0, clr_lime);
    frustum.addPoint(cam_2, clr_lime);

    frustum.addPoint(cam_0, clr_lime);
    frustum.addPoint(cam_3, clr_lime);
    
    frustum.addPoint(cam_0, clr_lime);
    frustum.addPoint(cam_4, clr_lime);

    frustum.setDrawingType(GL_LINES);
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

    shader_im.set(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);

    {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(3, vboID_);
        glBindVertexArray(vaoID_);
        {
            float w_0 = 0.f;
            float w_1 = 1.f;
            float h_0 = 1.f;
            float h_1 = 0.f;

            GLfloat g_quad_vertex_buffer_data[] = {
                w_0, h_0,
                w_1, h_0,
                w_0, h_1,
                w_1, h_1};

            glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
            glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
            glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 2, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        {
            float w_0 = -1.f;
            float w_1 = 1.f;
            float h_0 = -1.f;
            float h_1 = 1.f;

            GLfloat g_quad_vertex_buffer_data[] = {
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
    }

    auto res = ref.getResolution();
    aspect_ratio = res.width / (1.f*res.height);
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


void CameraViewer::draw2D() {
	if (!ref.isInit())  return;

    glDisable(GL_DEPTH_TEST);
    glUseProgram(shader_im.getProgramId());
    glBindTexture(GL_TEXTURE_2D, texture);

    glBindVertexArray(vaoID_);
    glDrawElements(GL_TRIANGLE_STRIP, (GLsizei)4, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    glUseProgram(0);
    glEnable(GL_DEPTH_TEST);
}
