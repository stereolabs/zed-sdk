#include "GLViewer.hpp"

const GLchar* VERTEX_SHADER =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "layout(location = 1) in vec3 in_Color;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "out vec3 b_color;\n"
        "void main() {\n"
        "   b_color = in_Color.bgr;\n"
        "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
        "}";

const GLchar* FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "   color = vec4(b_color, 0.95);\n"
        "}";


const GLchar* POINTCLOUD_VERTEX_SHADER =
        "#version 330 core\n"
        "layout(location = 0) in vec4 in_VertexRGBA;\n"
        "out vec4 b_color;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "uniform vec3 u_color;\n"
        "uniform bool u_drawFlat;\n"
        "void main() {\n"
        "   if(u_drawFlat)\n"
        "       b_color = vec4(u_color.bgr, .85f);\n"
        "else{"
        // Decompose the 4th channel of the XYZRGBA buffer to retrieve the color of the point (1float to 4uint)
        "       uint vertexColor = floatBitsToUint(in_VertexRGBA.w); \n"
        "       vec3 clr_int = vec3((vertexColor & uint(0x000000FF)), (vertexColor & uint(0x0000FF00)) >> 8, (vertexColor & uint(0x00FF0000)) >> 16);\n"
        "       b_color = vec4(clr_int.b / 255.0f, clr_int.g / 255.0f, clr_int.r / 255.0f, .85f);\n"
        "   }"
        "   gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n"
        "}";

const GLchar* POINTCLOUD_FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec4 b_color;\n"
        "layout(location = 0) out vec4 out_Color;\n"
        "void main() {\n"
        "   out_Color = b_color;\n"
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


GLViewer* currentInstance_ = nullptr;

GLViewer::GLViewer() : available(false) {
    currentInstance_ = this;
    mouseButton_[0] = mouseButton_[1] = mouseButton_[2] = false;
    clearInputs();
    previousMouseMotion_[0] = previousMouseMotion_[1] = 0;
}

GLViewer::~GLViewer() {
}

void GLViewer::exit() {
    if (currentInstance_) {
        available = false;
        for (auto& pc : point_clouds)
        {
            pc.second.close();
        }
    }
}

bool GLViewer::isAvailable() {
    if (currentInstance_ && available) {
        glutMainLoopEvent();
    }
    return available;
}

void CloseFunc(void) {
    if (currentInstance_) currentInstance_->exit();
}

void addVert(Simple3DObject &obj, float i_f, float limit, float height, sl::float4 &clr) {
    auto p1 = sl::float3(i_f, height, -limit);
    auto p2 = sl::float3(i_f, height, limit);
    auto p3 = sl::float3(-limit, height, i_f);
    auto p4 = sl::float3(limit, height, i_f);

    obj.addLine(p1, p2, clr);
    obj.addLine(p3, p4, clr);
}

void GLViewer::init(int argc, char **argv) {

    glutInit(&argc, argv);
    int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = glutGet(GLUT_SCREEN_HEIGHT);

    glutInitWindowSize(1200, 700);
    glutInitWindowPosition(wnd_w * 0.05, wnd_h * 0.05);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    glutCreateWindow("ZED| 3D View");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        std::cout << "ERROR: glewInit failed: " << glewGetErrorString(err) << "\n";

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

#ifndef JETSON_STYLE
    glEnable(GL_POINT_SMOOTH);
#endif

    // Compile and create the shader for 3D objects
    shader.it.set(VERTEX_SHADER, FRAGMENT_SHADER);
    shader.MVP_Mat = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

    // Create the camera
    camera_ = CameraGL(sl::Translation(0, 2, 10), sl::Translation(0, 0, -1));

    // Create the skeletons objects
    skeletons.setDrawingType(GL_LINES);
    floor_grid.setDrawingType(GL_LINES);

    // Set background color (black)
    bckgrnd_clr = sl::float4(0.2f, 0.19f, 0.2f, 1.0f);


    float limit = 20.0f;
    sl::float4 clr_grid(80, 80, 80, 255);
    clr_grid /= 255.f;

    float grid_height = -0;
    for (int i = (int) (-limit); i <= (int) (limit); i++)
        addVert(floor_grid, i, limit, grid_height, clr_grid);

    floor_grid.pushToGPU();

    std::random_device dev;
    rng = std::mt19937(dev());
    rng.seed(42);
    uint_dist360 = std::uniform_int_distribution<uint16_t>(0, 360);

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

sl::float3 newColor(float hh) {
    float s = 0.75f;
    float v = 0.9f;

    sl::float3 clr;
    int i = (int)hh;
    float ff = hh - i;
    float p = v * (1.f - s);
    float q = v * (1.f - (s * ff));
    float t = v * (1.f - (s * (1.f - ff)));
    switch (i) {
    case 0:
        clr.r = v;
        clr.g = t;
        clr.b = p;
        break;
    case 1:
        clr.r = q;
        clr.g = v;
        clr.b = p;
        break;
    case 2:
        clr.r = p;
        clr.g = v;
        clr.b = t;
        break;

    case 3:
        clr.r = p;
        clr.g = q;
        clr.b = v;
        break;
    case 4:
        clr.r = t;
        clr.g = p;
        clr.b = v;
        break;
    case 5:
    default:
        clr.r = v;
        clr.g = p;
        clr.b = q;
        break;
    }
    return clr;
}

sl::float3 GLViewer::getColor(int id, bool for_skeleton){
    const std::lock_guard<std::mutex> lock(mtx_clr);
    if(for_skeleton){
        if (colors_sk.find(id) == colors_sk.end()) {
            float hh = uint_dist360(rng) / 60.f;
            colors_sk[id] = newColor(hh);
        }
        return colors_sk[id];
    }else{
        if (colors.find(id) == colors.end()) {
            int h_ = uint_dist360(rng);
            float hh =  h_ / 60.f;
            colors[id] = newColor(hh);
        }
        return colors[id];
    }
}

void GLViewer::updateCamera(int id, sl::Mat &view, sl::Mat &pc){
    const std::lock_guard<std::mutex> lock(mtx);
    auto clr = getColor(id, false);
    if(view.isInit() && viewers.find(id) == viewers.end())
        viewers[id].initialize(view, clr);

    if(pc.isInit() && point_clouds.find(id) == point_clouds.end())
        point_clouds[id].initialize(pc, clr);
    
}

void GLViewer::updateCamera(sl::Mat &pc){
    const std::lock_guard<std::mutex> lock(mtx);
    int id = 0;
    auto clr = getColor(id, false);
    
    // we need to release old pc and initialize new one because fused point cloud don't have the same number of points for each process
    // I used close but it crashed in draw. Not yet investigated
    point_clouds[id].initialize(pc, clr);
}


void GLViewer::setRenderCameraProjection(sl::CameraParameters params, float znear, float zfar) {
    // Just slightly up the ZED camera FOV to make a small black border
    float fov_y = (params.v_fov + 0.5f) * M_PI / 180.f;
    float fov_x = (params.h_fov + 0.5f) * M_PI / 180.f;

    projection_(0, 0) = 1.0f / tanf(fov_x * 0.5f);
    projection_(1, 1) = 1.0f / tanf(fov_y * 0.5f);
    projection_(2, 2) = -(zfar + znear) / (zfar - znear);
    projection_(3, 2) = -1;
    projection_(2, 3) = -(2.f * zfar * znear) / (zfar - znear);
    projection_(3, 3) = 0;

    projection_(0, 0) = 1.0f / tanf(fov_x * 0.5f); //Horizontal FoV.
    projection_(0, 1) = 0;
    projection_(0, 2) = 2.0f * ((params.image_size.width - 1.0f * params.cx) / params.image_size.width) - 1.0f; //Horizontal offset.
    projection_(0, 3) = 0;

    projection_(1, 0) = 0;
    projection_(1, 1) = 1.0f / tanf(fov_y * 0.5f); //Vertical FoV.
    projection_(1, 2) = -(2.0f * ((params.image_size.height - 1.0f * params.cy) / params.image_size.height) - 1.0f); //Vertical offset.
    projection_(1, 3) = 0;

    projection_(2, 0) = 0;
    projection_(2, 1) = 0;
    projection_(2, 2) = -(zfar + znear) / (zfar - znear); //Near and far planes.
    projection_(2, 3) = -(2.0f * zfar * znear) / (zfar - znear); //Near and far planes.

    projection_(3, 0) = 0;
    projection_(3, 1) = 0;
    projection_(3, 2) = -1;
    projection_(3, 3) = 0.0f;
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

void GLViewer::setCameraPose(int id, sl::Transform pose) {
    const std::lock_guard<std::mutex> lock(mtx);
    getColor(id, false);
    poses[id] = pose;
}

inline bool renderBody(const sl::BodyData& i, const bool isTrackingON) {
    if (isTrackingON)
        return (i.tracking_state == sl::OBJECT_TRACKING_STATE::OK);
    else
        return (i.tracking_state == sl::OBJECT_TRACKING_STATE::OK || i.tracking_state == sl::OBJECT_TRACKING_STATE::OFF);
}

template<typename T>
void createSKPrimitive(sl::BodyData& body, const std::vector<std::pair<T, T>>& map, Simple3DObject& skp, sl::float3 clr_id, bool raw) {
    const float cylinder_thickness = raw ? 0.01f : 0.025f;

    for (auto& limb : map) {
        sl::float3 kp_1 = body.keypoint[getIdx(limb.first)];
        sl::float3 kp_2 = body.keypoint[getIdx(limb.second)];
        if (std::isfinite(kp_1.norm()) && std::isfinite(kp_2.norm()))
            skp.addLine(kp_1, kp_2, clr_id);
    }
}

void GLViewer::addSKeleton(sl::BodyData& obj, Simple3DObject& simpleObj, sl::float3 clr_id, bool raw) {
    switch (obj.keypoint.size()) {
    case 18:
        createSKPrimitive(obj, sl::BODY_18_BONES, simpleObj, clr_id, raw);
        break;
    case 34:
        createSKPrimitive(obj, sl::BODY_34_BONES, simpleObj, clr_id, raw);
        break;
    case 38:
        createSKPrimitive(obj, sl::BODY_38_BONES, simpleObj, clr_id, raw);
        break;
    }
}

void GLViewer::updateBodies(sl::Bodies &bodies, std::map<sl::CameraIdentifier, sl::Bodies>& singledata, sl::FusionMetrics& metrics) {
    const std::lock_guard<std::mutex> lock(mtx);

    if (bodies.is_new) {
        skeletons.clear();
        for(auto &it:bodies.body_list) {
            auto clr = getColor(it.id, true);
            if (renderBody(it, bodies.is_tracked))
                addSKeleton(it, skeletons, clr, false);
        }
    }

    fusionStats.clear();
    int id = 0;
        
    ObjectClassName obj_str;
    obj_str.name_lineA = "Publishers :" + std::to_string(metrics.mean_camera_fused);
    obj_str.name_lineB = "Sync :" + std::to_string(metrics.mean_stdev_between_camera * 1000.f);
    obj_str.color = sl::float4(0.9,0.9,0.9,1);
    obj_str.position = sl::float3(10, (id * 30), 0);
    fusionStats.push_back(obj_str);

    for (auto &it : singledata) {
        auto clr = getColor(it.first.sn, false);
        id++;
        if (it.second.is_new) 
        {
            auto& sk_r = skeletons_raw[it.first.sn];
            sk_r.clear();
            sk_r.setDrawingType(GL_LINES);

            for (auto& sk : it.second.body_list) {
                if(renderBody(sk, it.second.is_tracked))
                    addSKeleton(sk, sk_r, clr, true);
            }
        }
            
        ObjectClassName obj_str;
        obj_str.name_lineA = "CAM: " + std::to_string(it.first.sn) + " FPS: " + std::to_string(metrics.camera_individual_stats[it.first].received_fps);
        obj_str.name_lineB = "Ratio Detection :" + std::to_string(metrics.camera_individual_stats[it.first].ratio_detection) + " Delta " + std::to_string(metrics.camera_individual_stats[it.first].delta_ts * 1000.f);
        obj_str.color = clr;
        obj_str.position = sl::float3(10, (id * 30), 0);
        fusionStats.push_back(obj_str);
    }
}

void GLViewer::update() {
    
    if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP) {
        currentInstance_->exit();
        return;
    }

    if (keyStates_['r'] == KEY_STATE::UP)
        currentInstance_->show_raw = !currentInstance_->show_raw;

    if (keyStates_['c'] == KEY_STATE::UP)
        currentInstance_->draw_flat_color = !currentInstance_->draw_flat_color;

    if (keyStates_['s'] == KEY_STATE::UP)
        currentInstance_->show_pc = !currentInstance_->show_pc;

    if (keyStates_['p'] == KEY_STATE::UP || keyStates_['P'] == KEY_STATE::UP || keyStates_[32] == KEY_STATE::UP)
        play = !play;

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
        //float distance = sl::Translation(camera_.getOffsetFromPosition()).norm();
        if (mouseWheelPosition_ > 0 /* && distance > camera_.getZNear()*/) { // zoom
            camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * 0.5f * -1);
        } else if (/*distance < camera_.getZFar()*/ mouseWheelPosition_ < 0) {// unzoom
            //camera_.setOffsetFromPosition(camera_.getOffsetFromPosition() * MOUSE_DZ_SENSITIVITY);
            camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * 0.5f);
        }
    }

    camera_.update();
    const std::lock_guard<std::mutex> lock(mtx);
    // Update point cloud buffers
    skeletons.pushToGPU();
    for(auto &it: skeletons_raw)
        it.second.pushToGPU();

    // Update point cloud buffers
    for(auto &it: point_clouds)
        it.second.pushNewPC();

    for(auto &it: viewers)
        it.second.pushNewImage();
    clearInputs();
}


void GLViewer::draw() {

    glPolygonMode(GL_FRONT, GL_LINE);
    glPolygonMode(GL_BACK, GL_LINE);
    glLineWidth(2.f);
    glPointSize(1.f);

    sl::Transform vpMatrix = camera_.getViewProjectionMatrix();
    glUseProgram(shader.it.getProgramId());
    glUniformMatrix4fv(shader.MVP_Mat, 1, GL_TRUE, vpMatrix.m);

    floor_grid.draw();
    skeletons.draw();

    if (show_raw)
        for (auto& it : skeletons_raw)
            it.second.draw();

    for (auto& it : viewers) {
        sl::Transform pose_ = vpMatrix * poses[it.first];
        glUniformMatrix4fv(shader.MVP_Mat, 1, GL_FALSE, sl::Transform::transpose(pose_).m);
        it.second.frustum.draw();
    }

    glUseProgram(0);

    for (auto& it : poses) {
        sl::Transform vpMatrix_world = vpMatrix * it.second;

        if(show_pc)
            if(point_clouds.find(it.first) != point_clouds.end())
                point_clouds[it.first].draw(vpMatrix_world, draw_flat_color);

        if (viewers.find(it.first) != viewers.end())
            viewers[it.first].draw(vpMatrix_world);
    }
}

sl::float2 compute3Dprojection(sl::float3 &pt, const sl::Transform &cam, sl::Resolution wnd_size) {
    sl::float4 pt4d(pt.x, pt.y, pt.z, 1.);
    auto proj3D_cam = pt4d * cam;
    sl::float2 proj2D;
    proj2D.x = ((proj3D_cam.x / pt4d.w) * wnd_size.width) / (2.f * proj3D_cam.w) + wnd_size.width / 2.f;
    proj2D.y = ((proj3D_cam.y / pt4d.w) * wnd_size.height) / (2.f * proj3D_cam.w) + wnd_size.height / 2.f;
    return proj2D;
}

void GLViewer::printText() {

    sl::Resolution wnd_size(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
    for (auto &it : fusionStats) {
#if 0
        auto pt2d = compute3Dprojection(it.position, projection_, wnd_size);
#else
        sl::float2 pt2d(it.position.x, it.position.y);
#endif
        glColor4f(it.color.b, it.color.g, it.color.r, .85f);
        const auto *string = it.name_lineA.c_str();
        glWindowPos2f(pt2d.x, pt2d.y + 15);
        int len = (int) strlen(string);
        for (int i = 0; i < len; i++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, string[i]);

        string = it.name_lineB.c_str();
        glWindowPos2f(pt2d.x, pt2d.y);
        len = (int) strlen(string);
        for (int i = 0; i < len; i++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, string[i]);
    }
}

void GLViewer::clearInputs() {
    mouseMotion_[0] = mouseMotion_[1] = 0;
    mouseWheelPosition_ = 0;
    for (unsigned int i = 0; i < 256; ++i) {
        if (keyStates_[i] == KEY_STATE::UP)
            last_key = i;
        if (keyStates_[i] != KEY_STATE::DOWN)
            keyStates_[i] = KEY_STATE::FREE;
    }
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
}

void GLViewer::reshapeCallback(int width, int height) {
    glViewport(0, 0, width, height);
    float hfov = (180.0f / M_PI) * (2.0f * atan(width / (2.0f * 500)));
    float vfov = (180.0f / M_PI) * (2.0f * atan(height / (2.0f * 500)));
    currentInstance_->camera_.setProjection(hfov, vfov, currentInstance_->camera_.getZNear(), currentInstance_->camera_.getZFar());
}

void GLViewer::keyPressedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::DOWN;
    currentInstance_->lastPressedKey = c;
    //glutPostRedisplay();
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

void Simple3DObject::addLine(sl::float3 pt1, sl::float3 pt2, sl::float3 clr){
    addPoint(pt1, clr);
    addPoint(pt2, clr);
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

const GLchar* IMAGE_FRAGMENT_SHADER =
        "#version 330 core\n"
        " in vec2 UV;\n"
        " out vec4 color;\n"
        " uniform sampler2D texImage;\n"
        " void main() {\n"
        "	vec2 scaler  =vec2(UV.x,1.f - UV.y);\n"
        "	vec3 rgbcolor = vec3(texture(texImage, scaler).zyx);\n"
        "	vec3 color_rgb = pow(rgbcolor, vec3(1.65f));\n"
        "	color = vec4(color_rgb,1);\n"
        "}";

const GLchar* IMAGE_VERTEX_SHADER =
        "#version 330\n"
        "layout(location = 0) in vec3 vert;\n"
        "out vec2 UV;"
        "void main() {\n"
        "	UV = (vert.xy+vec2(1,1))* .5f;\n"
        "	gl_Position = vec4(vert, 1);\n"
        "}\n";


PointCloud::PointCloud(): numBytes_(0), xyzrgbaMappedBuf_(nullptr) {

}

PointCloud::~PointCloud() {
    close();
}

void PointCloud::close() {
    if (refMat.isInit()) {
        auto err = cudaGraphicsUnmapResources(1, &bufferCudaID_, 0);
        if (err) std::cerr << "Error CUDA " << cudaGetErrorString(err) << std::endl;
        err = cudaGraphicsUnregisterResource(bufferCudaID_);
        if (err) std::cerr << "Error CUDA " << cudaGetErrorString(err) << std::endl;
        glDeleteBuffers(1, &bufferGLID_);
        refMat.free();
    }
}

void PointCloud::initialize(sl::Mat& ref, sl::float3 clr_) 
{
    refMat = ref;
    clr = clr_;
}

void PointCloud::pushNewPC() {
    if (refMat.isInit()) {

        int sizebytes = refMat.getResolution().area() * sizeof(sl::float4);
        if (numBytes_ != sizebytes) {

            if (numBytes_ == 0) {
                glGenBuffers(1, &bufferGLID_);

                shader_.set(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
                shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");
                shColor = glGetUniformLocation(shader_.getProgramId(), "u_color");
                shDrawColor = glGetUniformLocation(shader_.getProgramId(), "u_drawFlat");
            }
            else {
                cudaGraphicsUnmapResources(1, &bufferCudaID_, 0);
                cudaGraphicsUnregisterResource(bufferCudaID_);
            }

            glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
            glBufferData(GL_ARRAY_BUFFER, sizebytes, 0, GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);

            cudaError_t err = cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsNone);
            if (err) std::cerr << "Error CUDA " << cudaGetErrorString(err) << std::endl;

            err = cudaGraphicsMapResources(1, &bufferCudaID_, 0);
            if (err) std::cerr << "Error CUDA " << cudaGetErrorString(err) << std::endl;

            err = cudaGraphicsResourceGetMappedPointer((void**)&xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_);
            if (err) std::cerr << "Error CUDA " << cudaGetErrorString(err) << std::endl;
        }

        cudaMemcpy(xyzrgbaMappedBuf_, refMat.getPtr<sl::float4>(sl::MEM::CPU), numBytes_, cudaMemcpyHostToDevice);
    }
}

void PointCloud::draw(const sl::Transform& vp, bool draw_flat) {
    if (refMat.isInit()) {
#ifndef JETSON_STYLE
        glDisable(GL_BLEND);
#endif

        glUseProgram(shader_.getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_TRUE, vp.m);

        glUniform3fv(shColor, 1, clr.v);
        glUniform1i(shDrawColor, draw_flat);

        glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glDrawArrays(GL_POINTS, 0, refMat.getResolution().area());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);

#ifndef JETSON_STYLE
        glEnable(GL_BLEND);
#endif
    }
}


CameraViewer::CameraViewer() {

}

CameraViewer::~CameraViewer() {
	close();
}

void CameraViewer::close() {
	if (ref.isInit()) {	
        
        auto err = cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);
        if (err) std::cerr << "Error CUDA " << cudaGetErrorString(err) << std::endl;
        err = cudaGraphicsUnregisterResource(cuda_gl_ressource);
        if (err) std::cerr << "Error CUDA " << cudaGetErrorString(err) << std::endl;

		glDeleteTextures(1, &texture);
		glDeleteBuffers(3, vboID_);
		glDeleteVertexArrays(1, &vaoID_);
        ref.free();
	}
}

bool CameraViewer::initialize(sl::Mat &im, sl::float3 clr) {

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
    glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof(sl::float3), &vert[0], GL_STATIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof(sl::float2), &uv[0], GL_STATIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * sizeof(sl::uint3), &faces[0], GL_STATIC_DRAW);

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

void CameraViewer::pushNewImage() {
	if (!ref.isInit())  return;
	auto err = cudaMemcpy2DToArray(ArrIm, 0, 0, ref.getPtr<sl::uchar1>(sl::MEM::CPU), ref.getStepBytes(sl::MEM::CPU), ref.getPixelBytes() * ref.getWidth(), ref.getHeight(), cudaMemcpyHostToDevice);
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


const sl::Translation CameraGL::ORIGINAL_FORWARD = sl::Translation(0, 0, 1);
const sl::Translation CameraGL::ORIGINAL_UP = sl::Translation(0, 1, 0);
const sl::Translation CameraGL::ORIGINAL_RIGHT = sl::Translation(1, 0, 0);

CameraGL::CameraGL(sl::Translation position, sl::Translation direction, sl::Translation vertical) {
    this->position_ = position;
    setDirection(direction, vertical);

    offset_ = sl::Translation(0, 0, 0);
    view_.setIdentity();
    updateView();
    setProjection(70, 70, 0.200f, 50.f);
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
