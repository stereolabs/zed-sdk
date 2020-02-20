#include "GLViewer.hpp"
#include <random>


#if defined(_DEBUG) && defined(_WIN32)
//#error "This sample should not be built in Debug mode, use RelWithDebInfo if you want to do step by step."
#endif

#define FADED_RENDERING
const float grid_size = 8.0f;

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

void addVert(Simple3DObject &obj, float i_f, float limit, float height, sl::float3 &clr) {
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
    }
}

bool GLViewer::isAvailable() {
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

    sl::float4 clr(0.2f, 0.5f, 0.8f, 1.0f);

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

    // Compile and create the shader
    shader.it = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
    shader.MVP_Mat = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");
    
    shaderLine.it = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
    shaderLine.MVP_Mat = glGetUniformLocation(shaderLine.it.getProgramId(), "u_mvpMatrix");

    // Create the camera
    camera_ = CameraGL(sl::Translation(0, 0, 1000), sl::Translation(0, 0, -100));
    camera_.setOffsetFromPosition(sl::Translation(0, 0, 1500));

    // FOV
    camera_.setHorizontalFOV((180.0f / M_PI) * (2.0f * atan(param.image_size.width / (2.0f * param.fx))));
    camera_.setVerticalFOV((180.0f / M_PI) * (2.0f * atan(param.image_size.height / (2.0f * param.fy))));

    frustum = createFrustum(param);
    frustum.pushToGPU();

    BBox_edges = Simple3DObject(sl::Translation(0, 0, 0), false);
    BBox_edges.setDrawingType(GL_LINES);
    
    BBox_faces = Simple3DObject(sl::Translation(0, 0, 0), false);
    BBox_faces.setDrawingType(GL_QUADS);

    bckgrnd_clr = sl::float3(50, 50, 50) / 255.f;

    floor_grid = Simple3DObject(sl::Translation(0, 0, 0), true);
    floor_grid.setDrawingType(GL_LINES);

    float limit = 20.f;
    sl::float3 clr_grid(108, 122, 137);
    clr_grid /= 255.f;
    float height = -3; //-1.5;
    for (int i = (int) (-limit); i <= (int) (limit); i++) {
        addVert(floor_grid, i * 1000, limit * 1000, height * 1000, clr_grid);
    }
    floor_grid.pushToGPU();
    
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
        glClearColor(bckgrnd_clr.r, bckgrnd_clr.g, bckgrnd_clr.b, 1.f);
        update();
        draw();
        printText();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> dist(0.1, 1);
std::vector<sl::float4> color_vect;

inline sl::float4 generateColorID(unsigned int idx) {
    if (idx < 0)
        return sl::float4(dist(e2), dist(e2), dist(e2), 0.5);
    else {
        while (color_vect.size() <= idx) color_vect.emplace_back(dist(e2), dist(e2), dist(e2), 0.5);
        return color_vect.at(idx);
    }
}

void GLViewer::updateData(sl::Mat &matXYZRGBA, std::vector<sl::ObjectData> &objs) {
    mtx.lock();
    //if (mtx.try_lock()) {
        pointCloud_.pushNewPC(matXYZRGBA);
        BBox_edges.clear();
        BBox_faces.clear();
        objectsName.clear();

        for (unsigned int i = 0; i < objs.size(); i++) {
            if (objs[i].tracking_state == sl::OBJECT_TRACKING_STATE::OK || objs[i].tracking_state == sl::OBJECT_TRACKING_STATE::OFF) {
                auto bb_ = objs[i].bounding_box;
                if (!bb_.empty()) {
                    auto clr_class = getColorClass((unsigned int) objs[i].label);
                    auto clr_id = generateColorID_GR(objs[i].id) / 255.0f;

                    if (objs[i].tracking_state != sl::OBJECT_TRACKING_STATE::OK)
                        clr_id = clr_class;
                    else
                        createIDRendering(bb_, clr_id, objs[i].id);

#ifdef FADED_RENDERING
                    createBboxRendering(bb_, clr_id);
#else
                    BBox_edges.addBBox(bb_, clr_id);
                    BBox_faces.addFaces(bb_, clr_id);
#endif
                }
            }
        }
    mtx.unlock();
    //}
}

void GLViewer::createBboxRendering(std::vector<sl::float3> &bbox, sl::float3 bbox_clr) {
    // First create top and bottom full edges
    BBox_edges.addFullEdges(bbox, bbox_clr);

    // Add faded vertical edges
    BBox_edges.addVerticalEdges(bbox, bbox_clr);

    // Add top face
    BBox_faces.addTopFace(bbox, bbox_clr);

    // Add faces
    BBox_faces.addVerticalFaces(bbox, bbox_clr);
}

void GLViewer::createIDRendering(std::vector<sl::float3> &bbox, sl::float3 clr, unsigned int id) {
    ObjectClassName tmp;
    std::string name_ = std::to_string(id);
    tmp.name = "ID: " + name_;
    tmp.color = sl::float3(clr.x, clr.y, clr.z);
    tmp.position = (bbox[0] + bbox[3]) / 2.0f; // Reference point
    //tmp.position.x += 100.f;
    tmp.position.y += 100.f;
    objectsName.push_back(tmp);
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

    camera_.update();
    mtx.lock();
    // Update point cloud buffers
    BBox_edges.pushToGPU();
    BBox_faces.pushToGPU();
    pointCloud_.update();
    mtx.unlock();
    clearInputs();
}

void GLViewer::draw() {
    const sl::Transform vpMatrix = camera_.getViewProjectionMatrix();

    glUseProgram(shaderLine.it.getProgramId());
    glUniformMatrix4fv(shaderLine.MVP_Mat, 1, GL_TRUE, vpMatrix.m);
    glLineWidth(1.f);
    floor_grid.draw();
    glUseProgram(0);

    glPointSize(1.f);
    pointCloud_.draw(vpMatrix);

    glUseProgram(shader.it.getProgramId());
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glUniformMatrix4fv(shader.MVP_Mat, 1, GL_TRUE, vpMatrix.m);
    glLineWidth(2.f);
    frustum.draw();
    glLineWidth(1.5f);
    BBox_edges.draw();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    BBox_faces.draw();
    glUseProgram(0);
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
    const sl::Transform vpMatrix = camera_.getViewProjectionMatrix();
    sl::Resolution wnd_size(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
    for (auto it : objectsName) {
        auto pt2d = compute3Dprojection(it.position, vpMatrix, wnd_size);
        glColor4f(it.color.r, it.color.g, it.color.b, 1.f);
        const auto *string = it.name.c_str();
        glWindowPos2f(pt2d.x, pt2d.y);
        int len = (int) strlen(string);
        for (int i = 0; i < len; i++)
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, string[i]);
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
    float hfov = currentInstance_->camera_.getHorizontalFOV();
    float vfov = currentInstance_->camera_.getVerticalFOV();
    currentInstance_->camera_.setProjection(hfov, vfov, currentInstance_->camera_.getZNear(), currentInstance_->camera_.getZFar());
}

void GLViewer::keyPressedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::DOWN;
}

void GLViewer::keyReleasedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::UP;
}

void GLViewer::idle() {
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

void Simple3DObject::addBBox(std::vector<sl::float3> &pts, sl::float3 clr) {
    int start_id = vertices_.size() / 3;

    float transparency_top = 0.05f, transparency_bottom = 0.75f;
    for (unsigned int i = 0; i < pts.size(); i++) {
        vertices_.push_back(pts[i].x);
        vertices_.push_back(pts[i].y);
        vertices_.push_back(pts[i].z);

        colors_.push_back(clr.r);
        colors_.push_back(clr.g);
        colors_.push_back(clr.b);
        colors_.push_back(i < 4 ? transparency_top : transparency_bottom);
    }

    const std::vector<int> boxLinks = {4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7};

    for (unsigned int i = 0; i < boxLinks.size(); i += 2) {
        indices_.push_back(start_id + boxLinks[i]);
        indices_.push_back(start_id + boxLinks[i + 1]);
    }
}

void Simple3DObject::addFaces(std::vector<sl::float3> &pts, sl::float3 clr) {
    std::vector<std::vector<int>> faces
    {
        {
            0, 3, 7, 4
        }, // front face
        {
            3, 2, 6, 7
        }, // right face
        {
            2, 1, 5, 6
        }, // back face
        {
            1, 0, 4, 5
        } // left face
    };
    for (const auto order : faces) {
        // Top points
        vertices_.push_back(pts[order[0]].x);
        vertices_.push_back(pts[order[0]].y);
        vertices_.push_back(pts[order[0]].z);

        colors_.push_back(clr.r);
        colors_.push_back(clr.g);
        colors_.push_back(clr.b);
        colors_.push_back(0.025f);

        vertices_.push_back(pts[order[1]].x);
        vertices_.push_back(pts[order[1]].y);
        vertices_.push_back(pts[order[1]].z);

        colors_.push_back(clr.r);
        colors_.push_back(clr.g);
        colors_.push_back(clr.b);
        colors_.push_back(0.025f);

        // Bottom points
        vertices_.push_back(pts[order[2]].x);
        vertices_.push_back(pts[order[2]].y);
        vertices_.push_back(pts[order[2]].z);

        colors_.push_back(clr.r);
        colors_.push_back(clr.g);
        colors_.push_back(clr.b);
        colors_.push_back(0.2f);

        vertices_.push_back(pts[order[3]].x);
        vertices_.push_back(pts[order[3]].y);
        vertices_.push_back(pts[order[3]].z);

        colors_.push_back(clr.r);
        colors_.push_back(clr.g);
        colors_.push_back(clr.b);
        colors_.push_back(0.2f);

        indices_.push_back((int) indices_.size());
        indices_.push_back((int) indices_.size());
        indices_.push_back((int) indices_.size());
        indices_.push_back((int) indices_.size());
    }
}

void Simple3DObject::addPoint(sl::float3 pt, sl::float3 clr) {
    vertices_.push_back(pt.x);
    vertices_.push_back(pt.y);
    vertices_.push_back(pt.z);
    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(1.f);
    indices_.push_back((int) indices_.size());
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
    colors_.push_back(1.f);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(1.f);

    indices_.push_back((int) indices_.size());
    indices_.push_back((int) indices_.size());
}

void Simple3DObject::addTriangle(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float4 clr) {
    vertices_.push_back(p1.x);
    vertices_.push_back(p1.y);
    vertices_.push_back(p1.z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(clr.a);

    vertices_.push_back(p2.x);
    vertices_.push_back(p2.y);
    vertices_.push_back(p2.z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(clr.a);

    vertices_.push_back(p3.x);
    vertices_.push_back(p3.y);
    vertices_.push_back(p3.z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(clr.a);

    indices_.push_back((int) indices_.size());
    indices_.push_back((int) indices_.size());
    indices_.push_back((int) indices_.size());
}

void Simple3DObject::addFullEdges(std::vector<sl::float3> &pts, sl::float3 clr) {
    int start_id = vertices_.size() / 3;

    for (unsigned int i = 0; i < pts.size(); i++) {
        vertices_.push_back(pts[i].x);
        vertices_.push_back(pts[i].y);
        vertices_.push_back(pts[i].z);

        colors_.push_back(clr.r);
        colors_.push_back(clr.g);
        colors_.push_back(clr.b);
        colors_.push_back(0.75f);
    }

    const std::vector<int> boxLinksTop = {0, 1, 1, 2, 2, 3, 3, 0};
    for (unsigned int i = 0; i < boxLinksTop.size(); i += 2) {
        indices_.push_back(start_id + boxLinksTop[i]);
        indices_.push_back(start_id + boxLinksTop[i + 1]);
    }

    const std::vector<int> boxLinksBottom = {4, 5, 5, 6, 6, 7, 7, 4};
    for (unsigned int i = 0; i < boxLinksBottom.size(); i += 2) {
        indices_.push_back(start_id + boxLinksBottom[i]);
        indices_.push_back(start_id + boxLinksBottom[i + 1]);
    }
}

void Simple3DObject::addVerticalEdges(std::vector<sl::float3> &pts, sl::float3 clr) {
    auto addSingleVerticalLine = [&](sl::float3 top_pt, sl::float3 bot_pt) {
        std::vector<sl::float3> current_pts{
            top_pt,
            ((grid_size - 1.0f) * top_pt + bot_pt) / grid_size,
            ((grid_size - 2.0f) * top_pt + bot_pt * 2.0f) / grid_size,
            (2.0f * top_pt + bot_pt * (grid_size - 2.0f)) / grid_size,
            (top_pt + bot_pt * (grid_size - 1.0f)) / grid_size,
            bot_pt};

        int start_id = vertices_.size() / 3;
        for (unsigned int i = 0; i < current_pts.size(); i++) {
            vertices_.push_back(current_pts[i].x);
            vertices_.push_back(current_pts[i].y);
            vertices_.push_back(current_pts[i].z);

            colors_.push_back(clr.r);
            colors_.push_back(clr.g);
            colors_.push_back(clr.b);
            colors_.push_back((i == 2 || i == 3) ? 0.0f : 0.75f);
        }

        const std::vector<int> boxLinks = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5};
        for (unsigned int i = 0; i < boxLinks.size(); i += 2) {
            indices_.push_back(start_id + boxLinks[i]);
            indices_.push_back(start_id + boxLinks[i + 1]);
        }
    };

    addSingleVerticalLine(pts[0], pts[4]);
    addSingleVerticalLine(pts[1], pts[5]);
    addSingleVerticalLine(pts[2], pts[6]);
    addSingleVerticalLine(pts[3], pts[7]);
}

void Simple3DObject::addTopFace(std::vector<sl::float3> &pts, sl::float3 clr) {
    float alpha = 0.3f;

    vertices_.push_back(pts[0].x);
    vertices_.push_back(pts[0].y);
    vertices_.push_back(pts[0].z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(alpha);

    vertices_.push_back(pts[1].x);
    vertices_.push_back(pts[1].y);
    vertices_.push_back(pts[1].z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(alpha);

    vertices_.push_back(pts[2].x);
    vertices_.push_back(pts[2].y);
    vertices_.push_back(pts[2].z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(alpha);

    vertices_.push_back(pts[3].x);
    vertices_.push_back(pts[3].y);
    vertices_.push_back(pts[3].z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    colors_.push_back(alpha);

    indices_.push_back((int) indices_.size());
    indices_.push_back((int) indices_.size());
    indices_.push_back((int) indices_.size());
    indices_.push_back((int) indices_.size());
}

void Simple3DObject::addVerticalFaces(std::vector<sl::float3> &pts, sl::float3 clr) {
    auto addQuad = [&](std::vector<sl::float3> quad_pts, float alpha1, float alpha2) { // To use only with 4 points
        for (unsigned int i = 0; i < quad_pts.size(); ++i) {
            vertices_.push_back(quad_pts[i].x);
            vertices_.push_back(quad_pts[i].y);
            vertices_.push_back(quad_pts[i].z);

            colors_.push_back(clr.r);
            colors_.push_back(clr.g);
            colors_.push_back(clr.b);
            colors_.push_back(i < 2 ? alpha1 : alpha2);
        }

        indices_.push_back((int) indices_.size());
        indices_.push_back((int) indices_.size());
        indices_.push_back((int) indices_.size());
        indices_.push_back((int) indices_.size());
    };

    // For each face, we need to add 4 quads (the first 2 indexes are always the top points of the quad)
    std::vector<std::vector<int>> quads
    {
        {
            0, 3, 7, 4
        }, // front face
        {
            3, 2, 6, 7
        }, // right face
        {
            2, 1, 5, 6
        }, // back face
        {
            1, 0, 4, 5
        } // left face
    };
    float alpha = 0.3f;

    for (const auto quad : quads) {
        // Top full quad
        std::vector<sl::float3> quad_pts_1{
            pts[quad[0]],
            pts[quad[1]],
            ((grid_size - 1.0f) * pts[quad[1]] + pts[quad[2]]) / grid_size,
            ((grid_size - 1.0f) * pts[quad[0]] + pts[quad[3]]) / grid_size};
        addQuad(quad_pts_1, alpha, alpha);

        // Top faded quad
        std::vector<sl::float3> quad_pts_2{
            ((grid_size - 1.0f) * pts[quad[0]] + pts[quad[3]]) / grid_size,
            ((grid_size - 1.0f) * pts[quad[1]] + pts[quad[2]]) / grid_size,
            ((grid_size - 2.0f) * pts[quad[1]] + 2.0f * pts[quad[2]]) / grid_size,
            ((grid_size - 2.0f) * pts[quad[0]] + 2.0f * pts[quad[3]]) / grid_size};
        addQuad(quad_pts_2, alpha, 0.0f);

        // Bottom faded quad
        std::vector<sl::float3> quad_pts_3{
            (2.0f * pts[quad[0]] + (grid_size - 2.0f) * pts[quad[3]]) / grid_size,
            (2.0f * pts[quad[1]] + (grid_size - 2.0f) * pts[quad[2]]) / grid_size,
            (pts[quad[1]] + (grid_size - 1.0f) * pts[quad[2]]) / grid_size,
            (pts[quad[0]] + (grid_size - 1.0f) * pts[quad[3]]) / grid_size};
        addQuad(quad_pts_3, 0.0f, alpha);

        // Bottom full quad
        std::vector<sl::float3> quad_pts_4{
            (pts[quad[0]] + (grid_size - 1.0f) * pts[quad[3]]) / grid_size,
            (pts[quad[1]] + (grid_size - 1.0f) * pts[quad[2]]) / grid_size,
            pts[quad[2]],
            pts[quad[3]]};
        addQuad(quad_pts_4, alpha, alpha);
    }
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
            glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof (float), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
            glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
        }

        if (colors_.size() > 0) {
            glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
            glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof (float), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
            glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);
        }

        if (indices_.size() > 0) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof (unsigned int), &indices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
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
        glDrawElements(drawingType_, (GLsizei) indices_.size(), GL_UNSIGNED_INT, 0);
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
    glBufferData(GL_ARRAY_BUFFER, res.area() * 4 * sizeof (float), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkError(cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsWriteDiscard));

    shader.it = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    shader.MVP_Mat = glGetUniformLocation(shader.it.getProgramId(), "u_mvpMatrix");

    matGPU_.alloc(res, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);

    checkError(cudaGraphicsMapResources(1, &bufferCudaID_, 0));
    checkError(cudaGraphicsResourceGetMappedPointer((void**) &xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_));
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
    usePerspective_ = false;
    setDirection(direction, vertical);

    offset_ = sl::Translation(0, 0, 0);
    view_.setIdentity();
    updateView();
    setProjection(60, 60, 100.f, 200000.f);
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

void CameraGL::setPerspective(bool enable) {
    if (enable != this->usePerspective_) {
        if (!enable) {
            this->setPosition(sl::float3(0, 0, 0));
            this->setRotation(sl::Rotation());
            this->setOffsetFromPosition(sl::Translation(0, 0, 0));
        } else {
            setProjection(horizontalFieldOfView_, verticalFieldOfView_, 100.f, 200000.f);
            this->setOffsetFromPosition(sl::Translation(0, 0, 1000));
        }
        this->usePerspective_ = enable;
    }
}

void CameraGL::setProjection(float horizontalFOV, float verticalFOV, float znear, float zfar) {
    horizontalFieldOfView_ = horizontalFOV;
    verticalFieldOfView_ = verticalFOV;
    znear_ = znear;
    zfar_ = zfar;

    float fov_y = verticalFOV * M_PI / 180.f;
    float fov_x = horizontalFOV * M_PI / 180.f;

    projection_.setIdentity();
    if (usePerspective_) {
        projection_(0, 0) = 1.0f / tanf(fov_x * 0.5f);
        projection_(1, 1) = 1.0f / tanf(fov_y * 0.5f);
        projection_(2, 2) = -(zfar + znear) / (zfar - znear);
        projection_(3, 2) = -1;
        projection_(2, 3) = -(2.f * zfar * znear) / (zfar - znear);
        projection_(3, 3) = 0;
    } else {
        float x_max = 1.777777;
        float y_max = -1;

        projection_(0, 0) = 2 / x_max;
        projection_(1, 1) = -2 / y_max;
        projection_(2, 2) = 2 / (zfar - znear);
        projection_(3, 2) = -1;
        projection_(2, 3) = (zfar + znear) / (znear - zfar);
        projection_(3, 3) = 0;
    }
}

const sl::Transform& CameraGL::getViewProjectionMatrix() const {
    return vpMatrix_;
}

float CameraGL::getHorizontalFOV() const {
    return horizontalFieldOfView_;
}

void CameraGL::setHorizontalFOV(float hfov) {
    horizontalFieldOfView_ = hfov;
}

float CameraGL::getVerticalFOV() const {
    return verticalFieldOfView_;
}

void CameraGL::setVerticalFOV(float vfov) {
    verticalFieldOfView_ = vfov;
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
