#include "GLViewer.hpp"

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    cout << "[Sample]";
    if (err_code > sl::ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else if (err_code < sl::ERROR_CODE::SUCCESS)
        cout << "[Warning] ";
    else
        cout << " ";
    cout << msg_prefix << " ";
    if (err_code > sl::ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

const GLchar* VERTEX_SHADER = "#version 330 core\n"
                              "layout(location = 0) in vec3 in_Vertex;\n"
                              "layout(location = 1) in vec3 in_Color;\n"
                              "uniform mat4 u_mvpMatrix;\n"
                              "out vec3 b_color;\n"
                              "void main() {\n"
                              "   b_color = in_Color;\n"
                              "	gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
                              "}";

const GLchar* FRAGMENT_SHADER = "#version 330 core\n"
                                "in vec3 b_color;\n"
                                "layout(location = 0) out vec4 out_Color;\n"
                                "void main() {\n"
                                "   out_Color = vec4(b_color, 1);\n"
                                "}";

GLViewer* currentInstance_ = nullptr;

GLViewer::GLViewer()
    : available(false) {
    currentInstance_ = this;
    mouseButton_[0] = mouseButton_[1] = mouseButton_[2] = false;
    clearInputs();
    previousMouseMotion_[0] = previousMouseMotion_[1] = 0;
}

GLViewer::~GLViewer() { }

void GLViewer::exit() {
    if (currentInstance_) {
        // pointCloud_.close();
        available = false;
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
    cam_1.x = (0 - param.cx) * Z_ * fx_;
    cam_1.y = (0 - param.cy) * Z_ * fy_;

    cam_2.z = Z_;
    cam_2.x = (param.image_size.width - param.cx) * Z_ * fx_;
    cam_2.y = (0 - param.cy) * Z_ * fy_;

    cam_3.z = Z_;
    cam_3.x = (param.image_size.width - param.cx) * Z_ * fx_;
    cam_3.y = (param.image_size.height - param.cy) * Z_ * fy_;

    cam_4.z = Z_;
    cam_4.x = (0 - param.cx) * Z_ * fx_;
    cam_4.y = (param.image_size.height - param.cy) * Z_ * fy_;

    float const to_f = 1.f / 255.f;
    const sl::float4 clr_lime(217 * to_f, 255 * to_f, 66 * to_f, 1.f);

    it.addPoint(cam_0, clr_lime);
    it.addPoint(cam_1, clr_lime);

    it.addPoint(cam_0, clr_lime);
    it.addPoint(cam_2, clr_lime);

    it.addPoint(cam_0, clr_lime);
    it.addPoint(cam_3, clr_lime);

    it.addPoint(cam_0, clr_lime);
    it.addPoint(cam_4, clr_lime);

    it.setDrawingType(GL_LINES);
    return it;
}

void CloseFunc(void) {
    if (currentInstance_)
        currentInstance_->exit();
}

GLenum
GLViewer::init(int argc, char** argv, sl::CameraParameters param, CUstream strm_, sl::Resolution image_size, sl::Resolution preview_size) {

    glutInit(&argc, argv);
    int wnd_w = glutGet(GLUT_SCREEN_WIDTH);
    int wnd_h = glutGet(GLUT_SCREEN_HEIGHT) * 0.9;
    glutInitWindowSize(wnd_w * 0.9, wnd_h * 0.9);
    glutInitWindowPosition(wnd_w * 0.05, wnd_h * 0.05);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED Voxel Point Cloud");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        return err;

    windowW_ = wnd_w * 0.9;
    windowH_ = wnd_h * 0.9;

    pointCloud_.initialize(image_size, strm_);
    imageHandler_.initialize(preview_size);

    // Compile and create the shader
    shader_.set(VERTEX_SHADER, FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");

    // Create the camera
    camera_ = CameraGL(sl::Translation(0, 2000, 3000), sl::Translation(0, 0, -100));

    sl::Rotation rot;
    rot.setEulerAngles(sl::float3(-25, 0, 0), false);
    camera_.setRotation(rot);

    frustum = createFrustum(param);
    frustum.pushToGPU();

    bckgrnd_clr = sl::float3(25, 25, 25); // sl_soil
    bckgrnd_clr /= 255.f;

    // Map glut function on this class methods
    glutDisplayFunc(GLViewer::drawCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutMotionFunc(GLViewer::mouseMotionCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardFunc(GLViewer::keyPressedCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);
    glutSpecialFunc(GLViewer::specialKeyCallback);
    glutCloseFunc(CloseFunc);

    glEnable(GL_DEPTH_TEST);
#ifndef JETSON_STYLE
    glEnable(GL_LINE_SMOOTH);
#endif

    available = true;
    return err;
}

void GLViewer::render() {
    if (available) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(bckgrnd_clr.r, bckgrnd_clr.g, bckgrnd_clr.b, 1.f);
        glLineWidth(2.f);
        glPointSize(useVoxels_ ? pointSize_ : std::min(pointSize_, 2.f));
        update();
        draw();
        glutSwapBuffers();
        glutPostRedisplay();
    }
}

void GLViewer::updatePointCloud(sl::Mat& matXYZRGBA) {
    pointCloud_.mutexData.lock();
    pointCloud_.pushNewPC(matXYZRGBA);
    pointCloud_.mutexData.unlock();
}

void GLViewer::updateImage(sl::Mat& image) {
    imageHandler_.pushNewImage(image);
}

bool GLViewer::shouldSaveData() {
    bool out = shouldSaveData_;
    shouldSaveData_ = false;
    return out;
}

void GLViewer::update() {
    if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP) {
        pointCloud_.close();
        currentInstance_->exit();
        return;
    }

    if (keyStates_['s'] == KEY_STATE::UP || keyStates_['S'] == KEY_STATE::UP)
        currentInstance_->shouldSaveData_ = true;

    if (keyStates_['v'] == KEY_STATE::UP || keyStates_['V'] == KEY_STATE::UP)
        currentInstance_->useVoxels_ = !currentInstance_->useVoxels_;

    if (keyStates_[' '] == KEY_STATE::UP)
        currentInstance_->paused_ = !currentInstance_->paused_;

    // Voxel parameter controls
    if (keyStates_['+'] == KEY_STATE::UP || keyStates_['='] == KEY_STATE::UP)
        currentInstance_->voxelParams_.voxel_size *= 1.25f;

    if (keyStates_['-'] == KEY_STATE::UP)
        currentInstance_->voxelParams_.voxel_size = std::max(5.f, currentInstance_->voxelParams_.voxel_size * 0.8f);

    // Point size
    if (keyStates_['p'] == KEY_STATE::UP)
        currentInstance_->pointSize_ = std::max(0.5f, currentInstance_->pointSize_ - 0.2f);
    if (keyStates_['P'] == KEY_STATE::UP)
        currentInstance_->pointSize_ = std::min(20.f, currentInstance_->pointSize_ + 0.2f);

    if (keyStates_['1'] == KEY_STATE::UP)
        currentInstance_->voxelParams_.resolution_mode = sl::VOXELIZATION_MODE::FIXED;
    if (keyStates_['2'] == KEY_STATE::UP)
        currentInstance_->voxelParams_.resolution_mode = sl::VOXELIZATION_MODE::STEREO_UNCERTAINTY;
    if (keyStates_['3'] == KEY_STATE::UP)
        currentInstance_->voxelParams_.resolution_mode = sl::VOXELIZATION_MODE::LINEAR;

    if (keyStates_['.'] == KEY_STATE::UP || keyStates_['>'] == KEY_STATE::UP)
        currentInstance_->voxelParams_.resolution_scale = std::min(3.f, currentInstance_->voxelParams_.resolution_scale * 1.25f);

    if (keyStates_[','] == KEY_STATE::UP || keyStates_['<'] == KEY_STATE::UP)
        currentInstance_->voxelParams_.resolution_scale = std::max(0.01f, currentInstance_->voxelParams_.resolution_scale * 0.8f);

    if (keyStates_['c'] == KEY_STATE::UP || keyStates_['C'] == KEY_STATE::UP)
        currentInstance_->voxelParams_.centroid = !currentInstance_->voxelParams_.centroid;

    // Depth confidence
    if (keyStates_['d'] == KEY_STATE::UP)
        currentInstance_->confidenceThreshold_ = std::max(1, currentInstance_->confidenceThreshold_ - 5);
    if (keyStates_['D'] == KEY_STATE::UP)
        currentInstance_->confidenceThreshold_ = std::min(100, currentInstance_->confidenceThreshold_ + 5);

    if (keyStates_['r'] == KEY_STATE::UP || keyStates_['R'] == KEY_STATE::UP) {
        currentInstance_->voxelParams_.voxel_size = 50.f;
        currentInstance_->voxelParams_.centroid = true;
        currentInstance_->voxelParams_.resolution_mode = sl::VOXELIZATION_MODE::STEREO_UNCERTAINTY;
        currentInstance_->voxelParams_.resolution_scale = 0.2f;
        currentInstance_->pointSize_ = 3.f;
        currentInstance_->confidenceThreshold_ = 50;
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
        // float distance = sl::Translation(camera_.getOffsetFromPosition()).norm();
        if (mouseWheelPosition_ > 0 /* && distance > camera_.getZNear()*/) { // zoom
            camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * 1000 * -1);
        } else if (/*distance < camera_.getZFar()*/ mouseWheelPosition_ < 0) { // unzoom
            // camera_.setOffsetFromPosition(camera_.getOffsetFromPosition() * MOUSE_DZ_SENSITIVITY);
            camera_.translate(camera_.getForward() * MOUSE_UZ_SENSITIVITY * 1000);
        }
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

    // Full-window 3D viewport
    glViewport(0, 0, windowW_, windowH_);

    // Simple 3D shader for simple 3D objects
    glUseProgram(shader_.getProgramId());

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // Axis
    glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_FALSE, sl::Transform::transpose(vpMatrix * frustum.getModelMatrix()).m);
    frustum.draw();
    glBindVertexArray(0);
    glUseProgram(0);

    // Draw point cloud with its own shader
    pointCloud_.draw(vpMatrix);

    // Picture-in-picture: left camera image in bottom-left corner (25% of window)
    int pipW = windowW_ / 4;
    int pipH = windowH_ / 4;
    glViewport(10, 10, pipW, pipH);
    glDisable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    imageHandler_.draw();
    glEnable(GL_DEPTH_TEST);

    // Restore full viewport
    glViewport(0, 0, windowW_, windowH_);

    // HUD overlay
    drawHUD();
}

static void drawString(float x, float y, const char* str, void* font = GLUT_BITMAP_HELVETICA_12) {
    glWindowPos2f(x, y);
    while (*str)
        glutBitmapCharacter(font, *str++);
}

static void drawRect(float x, float y, float w, float h) {
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();
}

// ── HUD layout constants ──
static const float HUD_LEFT = 10.f;
static const float HUD_BTN_H = 22.f;
static const float HUD_GAP = 5.f;
static const float HUD_SM_W = 28.f;   // small button (- / +)
static const float HUD_MODE_W = 70.f; // mode button
static const float HUD_TOG_W = 100.f; // toggle button

static void drawButton(float x, float y, float w, float h, const char* label, bool active, bool highlight = false) {
    // Background — Stereolabs brand: sl_lime active, sl_charcoal normal, sl_steel highlight
    if (active)
        glColor4f(217 / 255.f, 255 / 255.f, 66 / 255.f, 0.9f); // sl_lime
    else if (highlight)
        glColor4f(60 / 255.f, 60 / 255.f, 60 / 255.f, 0.85f); // sl_charcoal
    else
        glColor4f(45 / 255.f, 45 / 255.f, 45 / 255.f, 0.8f); // sl_steel
    drawRect(x, y, w, h);

    // Border
    glColor4f(137 / 255.f, 137 / 255.f, 137 / 255.f, 0.6f); // sl_ash
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();

    // Label — dark text on lime, light on dark buttons
    if (active)
        glColor4f(25 / 255.f, 25 / 255.f, 25 / 255.f, 1.f); // sl_soil
    else
        glColor4f(242 / 255.f, 242 / 255.f, 242 / 255.f, 1.f); // sl_pearl
    float tx = x + 4.f;
    if (strlen(label) <= 2)
        tx = x + w * 0.5f - 4.f; // center short labels like "-" "+"
    drawString(tx, y + 6.f, label, GLUT_BITMAP_HELVETICA_10);
}

// Row positions (from top), computed relative to window height
struct HUDRow {
    float y;      // GL y (bottom-up)
    float labelX; // label x position
};

void GLViewer::drawHUD() {
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, windowW_, 0, windowH_, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    auto& vp = voxelParams_;
    const char* modeNames[] = {"FIXED", "STEREO", "LINEAR"};
    int modeIdx = static_cast<int>(vp.resolution_mode);
    if (modeIdx < 0 || modeIdx > 2)
        modeIdx = 0;

    float rowH = HUD_BTN_H + HUD_GAP;
    float topY = windowH_ - 15.f;

    // ── Row 0: Title ──
    glColor4f(217 / 255.f, 255 / 255.f, 66 / 255.f, 1.f); // sl_lime
    drawString(HUD_LEFT, topY, "Voxel Point Cloud", GLUT_BITMAP_HELVETICA_12);
    topY -= rowH;

    // ── Row 0b: Voxel / Full PC toggle ──
    {
        drawButton(HUD_LEFT, topY, 80.f, HUD_BTN_H, "Voxel", useVoxels_);
        drawButton(HUD_LEFT + 80.f + HUD_GAP, topY, 80.f, HUD_BTN_H, "Full PC", !useVoxels_);
    }
    topY -= rowH;

    // Grey out voxel-specific controls when in Full PC mode
    bool voxelControlsEnabled = useVoxels_;

    // ── Row 1: Voxel Size ── [ - ] value [ + ]
    {
        float a = voxelControlsEnabled ? 1.f : 0.5f;
        char buf[64];
        snprintf(buf, sizeof(buf), "Size: %.1f mm", vp.voxel_size);
        glColor4f(194 / 255.f * a, 194 / 255.f * a, 194 / 255.f * a, a); // sl_iron
        drawString(HUD_LEFT, topY + 6.f, buf, GLUT_BITMAP_HELVETICA_10);
        float bx = HUD_LEFT + 110.f;
        if (voxelControlsEnabled) {
            drawButton(bx, topY, HUD_SM_W, HUD_BTN_H, "-", false);
            drawButton(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W, HUD_BTN_H, "+", false);
        } else {
            glColor4f(45 / 255.f, 45 / 255.f, 45 / 255.f, 0.5f); // sl_steel disabled
            drawRect(bx, topY, HUD_SM_W, HUD_BTN_H);
            drawRect(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W, HUD_BTN_H);
        }
    }
    topY -= rowH;

    // ── Row 2: Resolution Scale ── [ - ] value [ + ] (disabled when FIXED or Full PC)
    {
        bool scaleEnabled = voxelControlsEnabled && (vp.resolution_mode != sl::VOXELIZATION_MODE::FIXED);
        char buf[64];
        snprintf(buf, sizeof(buf), "Scale: %.2f", vp.resolution_scale);
        float sa = scaleEnabled ? 1.f : 0.5f;
        glColor4f(194 / 255.f * sa, 194 / 255.f * sa, 194 / 255.f * sa, sa); // sl_iron
        drawString(HUD_LEFT, topY + 6.f, buf, GLUT_BITMAP_HELVETICA_10);
        float bx = HUD_LEFT + 110.f;
        if (scaleEnabled) {
            drawButton(bx, topY, HUD_SM_W, HUD_BTN_H, "-", false);
            drawButton(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W, HUD_BTN_H, "+", false);
        } else {
            // Greyed out buttons
            glColor4f(45 / 255.f, 45 / 255.f, 45 / 255.f, 0.5f); // sl_steel disabled
            drawRect(bx, topY, HUD_SM_W, HUD_BTN_H);
            drawRect(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W, HUD_BTN_H);
            glColor4f(137 / 255.f, 137 / 255.f, 137 / 255.f, 0.4f); // sl_ash
            drawString(bx + HUD_SM_W * 0.5f - 4.f, topY + 6.f, "-", GLUT_BITMAP_HELVETICA_10);
            drawString(bx + HUD_SM_W + HUD_GAP + HUD_SM_W * 0.5f - 4.f, topY + 6.f, "+", GLUT_BITMAP_HELVETICA_10);
        }
    }
    topY -= rowH;

    // ── Row 3: Mode buttons ── [FIXED] [STEREO] [LINEAR]
    {
        float a = voxelControlsEnabled ? 1.f : 0.5f;
        glColor4f(194 / 255.f * a, 194 / 255.f * a, 194 / 255.f * a, a); // sl_iron
        drawString(HUD_LEFT, topY + 6.f, "Mode:", GLUT_BITMAP_HELVETICA_10);
        float bx = HUD_LEFT + 50.f;
        if (voxelControlsEnabled) {
            for (int i = 0; i < 3; i++)
                drawButton(bx + i * (HUD_MODE_W + HUD_GAP), topY, HUD_MODE_W, HUD_BTN_H, modeNames[i], i == modeIdx);
        } else {
            for (int i = 0; i < 3; i++) {
                glColor4f(45 / 255.f, 45 / 255.f, 45 / 255.f, 0.5f); // sl_steel disabled
                drawRect(bx + i * (HUD_MODE_W + HUD_GAP), topY, HUD_MODE_W, HUD_BTN_H);
            }
        }
    }
    topY -= rowH;

    // ── Row 4: Centroid toggle ── [Centroid / Grid Center]
    {
        float a = voxelControlsEnabled ? 1.f : 0.5f;
        glColor4f(194 / 255.f * a, 194 / 255.f * a, 194 / 255.f * a, a); // sl_iron
        drawString(HUD_LEFT, topY + 6.f, "Pos:", GLUT_BITMAP_HELVETICA_10);
        float bx = HUD_LEFT + 50.f;
        if (voxelControlsEnabled) {
            drawButton(bx, topY, HUD_TOG_W, HUD_BTN_H, "Centroid", vp.centroid);
            drawButton(bx + HUD_TOG_W + HUD_GAP, topY, HUD_TOG_W, HUD_BTN_H, "Grid Center", !vp.centroid);
        } else {
            glColor4f(45 / 255.f, 45 / 255.f, 45 / 255.f, 0.5f); // sl_steel disabled
            drawRect(bx, topY, HUD_TOG_W, HUD_BTN_H);
            drawRect(bx + HUD_TOG_W + HUD_GAP, topY, HUD_TOG_W, HUD_BTN_H);
        }
    }
    topY -= rowH;

    // ── Row 5: Point Size ── [ - ] value [ + ]
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "Pt size: %.1f px", currentInstance_->pointSize_);
        glColor4f(194 / 255.f, 194 / 255.f, 194 / 255.f, 1.f); // sl_iron
        drawString(HUD_LEFT, topY + 6.f, buf, GLUT_BITMAP_HELVETICA_10);
        float bx = HUD_LEFT + 110.f;
        drawButton(bx, topY, HUD_SM_W, HUD_BTN_H, "-", false);
        drawButton(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W, HUD_BTN_H, "+", false);
    }
    topY -= rowH;

    // ── Row 6: Depth Confidence ── [ - ] value [ + ]
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "Depth Conf: %d", currentInstance_->confidenceThreshold_);
        glColor4f(194 / 255.f, 194 / 255.f, 194 / 255.f, 1.f); // sl_iron
        drawString(HUD_LEFT, topY + 6.f, buf, GLUT_BITMAP_HELVETICA_10);
        float bx = HUD_LEFT + 110.f;
        drawButton(bx, topY, HUD_SM_W, HUD_BTN_H, "-", false);
        drawButton(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W, HUD_BTN_H, "+", false);
    }
    topY -= rowH;

    // ── Row 7: Save + Reset ──
    {
        drawButton(HUD_LEFT, topY, 80.f, HUD_BTN_H, "Save PLY", false, true);
        drawButton(HUD_LEFT + 85.f + HUD_GAP, topY, 65.f, HUD_BTN_H, "Reset", false);
    }

    // ── Keyboard shortcuts (top-right) ──
    float tx = windowW_ - 270.f;
    float ty = windowH_ - 25.f;
    glColor4f(194 / 255.f, 194 / 255.f, 194 / 255.f, 0.6f); // sl_iron
    drawString(tx, ty, "+/-  Size    ,/.  Scale", GLUT_BITMAP_HELVETICA_10);
    drawString(tx, ty - 14, "1/2/3 Mode   c Centroid", GLUT_BITMAP_HELVETICA_10);
    drawString(tx, ty - 28, "d/D Depth Conf  p/P Pt size", GLUT_BITMAP_HELVETICA_10);
    drawString(tx, ty - 42, "v Toggle  Space Pause  </>/< Seek", GLUT_BITMAP_HELVETICA_10);
    drawString(tx, ty - 56, "s Save  r Reset  q Quit", GLUT_BITMAP_HELVETICA_10);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glEnable(GL_DEPTH_TEST);
}

bool GLViewer::handleButtonClick(int x, int y) {
    int gy = windowH_ - y; // GLUT y → GL y

    float rowH = HUD_BTN_H + HUD_GAP;
    float topY = windowH_ - 15.f - rowH; // skip title row

    auto hitTest = [&](float bx, float by, float bw) -> bool {
        return x >= bx && x <= bx + bw && gy >= by && gy <= by + HUD_BTN_H;
    };

    // Row 0b: Voxel / Full PC toggle
    {
        if (hitTest(HUD_LEFT, topY, 80.f)) {
            useVoxels_ = true;
            return true;
        }
        if (hitTest(HUD_LEFT + 80.f + HUD_GAP, topY, 80.f)) {
            useVoxels_ = false;
            return true;
        }
    }
    topY -= rowH;

    // Voxel-specific rows only clickable when voxel mode is on
    if (useVoxels_) {

        // Row 1: Voxel Size  [ - ] [ + ]
        {
            float bx = HUD_LEFT + 110.f;
            if (hitTest(bx, topY, HUD_SM_W)) {
                voxelParams_.voxel_size = std::max(5.f, voxelParams_.voxel_size * 0.8f);
                return true;
            }
            if (hitTest(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W)) {
                voxelParams_.voxel_size *= 1.25f;
                return true;
            }
        }
        topY -= rowH;

        // Row 2: Resolution Scale  [ - ] [ + ] (disabled when FIXED)
        {
            bool scaleEnabled = (voxelParams_.resolution_mode != sl::VOXELIZATION_MODE::FIXED);
            if (scaleEnabled) {
                float bx = HUD_LEFT + 110.f;
                if (hitTest(bx, topY, HUD_SM_W)) {
                    voxelParams_.resolution_scale = std::max(0.01f, voxelParams_.resolution_scale * 0.8f);
                    return true;
                }
                if (hitTest(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W)) {
                    voxelParams_.resolution_scale = std::min(3.f, voxelParams_.resolution_scale * 1.25f);
                    return true;
                }
            }
        }
        topY -= rowH;

        // Row 3: Mode buttons [FIXED] [STEREO] [LINEAR]
        {
            float bx = HUD_LEFT + 50.f;
            for (int i = 0; i < 3; i++) {
                if (hitTest(bx + i * (HUD_MODE_W + HUD_GAP), topY, HUD_MODE_W)) {
                    voxelParams_.resolution_mode = static_cast<sl::VOXELIZATION_MODE>(i);
                    return true;
                }
            }
        }
        topY -= rowH;

        // Row 4: Centroid toggle [Centroid] [Grid Center]
        {
            float bx = HUD_LEFT + 50.f;
            if (hitTest(bx, topY, HUD_TOG_W)) {
                voxelParams_.centroid = true;
                return true;
            }
            if (hitTest(bx + HUD_TOG_W + HUD_GAP, topY, HUD_TOG_W)) {
                voxelParams_.centroid = false;
                return true;
            }
        }
        topY -= rowH;

    } else {
        // Skip 4 rows when in Full PC mode
        topY -= rowH * 4;
    }

    // Row 5: Point Size  [ - ] [ + ]
    {
        float bx = HUD_LEFT + 110.f;
        if (hitTest(bx, topY, HUD_SM_W)) {
            pointSize_ = std::max(0.5f, pointSize_ - 0.2f);
            return true;
        }
        if (hitTest(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W)) {
            pointSize_ = std::min(20.f, pointSize_ + 0.2f);
            return true;
        }
    }
    topY -= rowH;

    // Row 6: Depth Confidence  [ - ] [ + ]
    {
        float bx = HUD_LEFT + 110.f;
        if (hitTest(bx, topY, HUD_SM_W)) {
            confidenceThreshold_ = std::max(1, confidenceThreshold_ - 5);
            return true;
        }
        if (hitTest(bx + HUD_SM_W + HUD_GAP, topY, HUD_SM_W)) {
            confidenceThreshold_ = std::min(100, confidenceThreshold_ + 5);
            return true;
        }
    }
    topY -= rowH;

    // Row 7: Save + Reset
    {
        if (hitTest(HUD_LEFT, topY, 80.f)) {
            shouldSaveData_ = true;
            return true;
        }
        if (hitTest(HUD_LEFT + 85.f + HUD_GAP, topY, 65.f)) {
            voxelParams_.voxel_size = 50.f;
            voxelParams_.centroid = true;
            voxelParams_.resolution_mode = sl::VOXELIZATION_MODE::STEREO_UNCERTAINTY;
            voxelParams_.resolution_scale = 0.2f;
            pointSize_ = 3.f;
            confidenceThreshold_ = 50;
            return true;
        }
    }

    return false;
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
    // Check HUD button clicks first
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        if (currentInstance_->handleButtonClick(x, y))
            return;
    }
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
    currentInstance_->windowW_ = width;
    currentInstance_->windowH_ = height;
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

void GLViewer::specialKeyCallback(int key, int x, int y) {
    if (key == GLUT_KEY_RIGHT)
        currentInstance_->seekOffset_ += 200;
    else if (key == GLUT_KEY_LEFT)
        currentInstance_->seekOffset_ -= 200;
}

void GLViewer::idle() {
    glutPostRedisplay();
}

Simple3DObject::Simple3DObject()
    : vaoID_(0) { }

Simple3DObject::Simple3DObject(sl::Translation position, bool isStatic)
    : isStatic_(isStatic) {
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

void Simple3DObject::addPoint(sl::float3 pt, sl::float3 clr) {
    vertices_.push_back(pt.x);
    vertices_.push_back(pt.y);
    vertices_.push_back(pt.z);
    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);
    indices_.push_back((int)indices_.size());
}

void Simple3DObject::addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float3 clr) {
    vertices_.push_back(p1.x);
    vertices_.push_back(p1.y);
    vertices_.push_back(p1.z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);

    vertices_.push_back(p2.x);
    vertices_.push_back(p2.y);
    vertices_.push_back(p2.z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);

    vertices_.push_back(p3.x);
    vertices_.push_back(p3.y);
    vertices_.push_back(p3.z);

    colors_.push_back(clr.r);
    colors_.push_back(clr.g);
    colors_.push_back(clr.b);

    indices_.push_back((int)indices_.size());
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
        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(float), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            indices_.size() * sizeof(unsigned int),
            &indices_[0],
            isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW
        );

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
    glDrawElements(drawingType_, (GLsizei)indices_.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
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

Shader::Shader(const GLchar* vs, const GLchar* fs) {
    set(vs, fs);
}

void Shader::set(const GLchar* vs, const GLchar* fs) {
    if (!compile(verterxId_, GL_VERTEX_SHADER, vs)) {
        print("ERROR: while compiling vertex shader");
    }
    if (!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)) {
        print("ERROR: while compiling fragment shader");
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
        print("ERROR: while linking shader : ");
        GLint errorSize(0);
        glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &errorSize);

        char* error = new char[errorSize + 1];
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

bool Shader::compile(GLuint& shaderId, GLenum type, const GLchar* src) {
    shaderId = glCreateShader(type);
    if (shaderId == 0) {
        return false;
    }
    glShaderSource(shaderId, 1, (const char**)&src, 0);
    glCompileShader(shaderId);

    GLint errorCp(0);
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &errorCp);
    if (errorCp != GL_TRUE) {
        print("ERROR: while compiling shader : ");
        GLint errorSize(0);
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

        char* error = new char[errorSize + 1];
        glGetShaderInfoLog(shaderId, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteShader(shaderId);
        return false;
    }
    return true;
}

const GLchar* POINTCLOUD_VERTEX_SHADER
    = "#version 330 core\n"
      "layout(location = 0) in vec4 in_VertexRGBA;\n"
      "uniform mat4 u_mvpMatrix;\n"
      "out vec4 b_color;\n"
      "void main() {\n"
      // Decompose the 4th channel of the XYZRGBA buffer to retrieve the color of the point (1float to 4uint)
      "   uint vertexColor = floatBitsToUint(in_VertexRGBA.w); \n"
      "   vec3 clr_int = vec3((vertexColor & uint(0x000000FF)), (vertexColor & uint(0x0000FF00)) >> 8, (vertexColor & uint(0x00FF0000)) >> "
      "16);\n"
      "   b_color = vec4(clr_int.r / 255.0f, clr_int.g / 255.0f, clr_int.b / 255.0f, 1.f);"
      "	gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n"
      "}";

const GLchar* POINTCLOUD_FRAGMENT_SHADER = "#version 330 core\n"
                                           "in vec4 b_color;\n"
                                           "layout(location = 0) out vec4 out_Color;\n"
                                           "void main() {\n"
                                           "   out_Color = b_color;\n"
                                           "}";

PointCloud::PointCloud()
    : hasNewPCL_(false)
    , mapped_(false)
    , xyzrgbaMappedBuf_(nullptr)
    , bufferGLID_(0)
    , bufferCudaID_(nullptr)
    , numBytes_(0)
    , capacity_(0)
    , currentCount_(0)
    , pendingPtr_(nullptr)
    , pendingCount_(0) { }

PointCloud::~PointCloud() {
    close();
}

void checkError(cudaError_t err) {
    if (err != cudaSuccess)
        std::cerr << "Error: (" << err << "): " << cudaGetErrorString(err) << std::endl;
}

void PointCloud::close() {
    if (bufferGLID_ != 0) {
        if (mapped_) {
            checkError(cudaGraphicsUnmapResources(1, &bufferCudaID_, 0));
            mapped_ = false;
        }
        checkError(cudaGraphicsUnregisterResource(bufferCudaID_));
        glDeleteBuffers(1, &bufferGLID_);
        bufferGLID_ = 0;
        bufferCudaID_ = nullptr;
        xyzrgbaMappedBuf_ = nullptr;
        capacity_ = 0;
        currentCount_ = 0;
    }
}

void PointCloud::initialize(sl::Resolution res, CUstream strm_) {
    strm = strm_;
    capacity_ = res.area();

    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, capacity_ * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    checkError(cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsNone));

    shader_.set(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");

    checkError(cudaGraphicsMapResources(1, &bufferCudaID_, 0));
    checkError(cudaGraphicsResourceGetMappedPointer((void**)&xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_));
    mapped_ = true;
}

void PointCloud::resize(size_t newCapacity) {
    if (mapped_) {
        checkError(cudaGraphicsUnmapResources(1, &bufferCudaID_, 0));
        mapped_ = false;
    }
    checkError(cudaGraphicsUnregisterResource(bufferCudaID_));
    bufferCudaID_ = nullptr;
    glDeleteBuffers(1, &bufferGLID_);
    bufferGLID_ = 0;
    xyzrgbaMappedBuf_ = nullptr;

    capacity_ = newCapacity;
    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, capacity_ * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess) {
        std::cerr << "PointCloud::resize — cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(err) << std::endl;
        glDeleteBuffers(1, &bufferGLID_);
        bufferGLID_ = 0;
        bufferCudaID_ = nullptr;
        capacity_ = 0;
        currentCount_ = 0;
        return;
    }

    err = cudaGraphicsMapResources(1, &bufferCudaID_, 0);
    if (err != cudaSuccess) {
        std::cerr << "PointCloud::resize — cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnregisterResource(bufferCudaID_);
        glDeleteBuffers(1, &bufferGLID_);
        bufferGLID_ = 0;
        bufferCudaID_ = nullptr;
        capacity_ = 0;
        currentCount_ = 0;
        return;
    }

    err = cudaGraphicsResourceGetMappedPointer((void**)&xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_);
    if (err != cudaSuccess) {
        std::cerr << "PointCloud::resize — cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &bufferCudaID_, 0);
        cudaGraphicsUnregisterResource(bufferCudaID_);
        glDeleteBuffers(1, &bufferGLID_);
        bufferGLID_ = 0;
        bufferCudaID_ = nullptr;
        xyzrgbaMappedBuf_ = nullptr;
        capacity_ = 0;
        currentCount_ = 0;
        return;
    }

    mapped_ = true;
}

void PointCloud::pushNewPC(sl::Mat& matXYZRGBA) {
    if (matXYZRGBA.isInit()) {
        pendingPtr_ = matXYZRGBA.getPtr<sl::float4>(sl::MEM::GPU);
        pendingCount_ = matXYZRGBA.getResolution().area();
        hasNewPCL_ = true;
    }
}

void PointCloud::update() {
    if (hasNewPCL_ && pendingPtr_ != nullptr && bufferGLID_ != 0) {
        if (pendingCount_ > capacity_)
            resize(pendingCount_);
        checkError(cudaMemcpyAsync(xyzrgbaMappedBuf_, pendingPtr_, pendingCount_ * 4 * sizeof(float), cudaMemcpyDeviceToDevice, strm));
        currentCount_ = pendingCount_;
        hasNewPCL_ = false;
    }
}

void PointCloud::draw(const sl::Transform& vp) {
    if (bufferGLID_ != 0 && currentCount_ > 0) {
        glUseProgram(shader_.getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_TRUE, vp.m);

        glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(currentCount_));
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
    setProjection(90, 90, 200.f, 50000.f);
    updateVPMatrix();
}

CameraGL::~CameraGL() { }

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

// ─── ImageHandler ──────────────────────────────────────────────────────────

const GLchar* IMAGE_VERTEX_SHADER = "#version 330\n"
                                    "layout(location = 0) in vec3 vert;\n"
                                    "out vec2 UV;"
                                    "void main() {\n"
                                    "   UV = (vert.xy+vec2(1,1))/2;\n"
                                    "   gl_Position = vec4(vert, 1);\n"
                                    "}\n";

const GLchar* IMAGE_FRAGMENT_SHADER = "#version 330 core\n"
                                      "in vec2 UV;\n"
                                      "out vec4 color;\n"
                                      "uniform sampler2D texImage;\n"
                                      "void main() {\n"
                                      "   vec2 scaler = vec2(UV.x, 1.0 - UV.y);\n"
                                      "   vec3 rgb = texture(texImage, scaler).zyx;\n" // BGR → RGB
                                      "   float gamma = 1.0/1.65;\n"
                                      "   color = vec4(pow(rgb, vec3(1.0/gamma)), 1);\n"
                                      "}\n";

ImageHandler::ImageHandler()
    : imageTex(0)
    , cuda_gl_ressource(nullptr)
    , quad_vao(0)
    , quad_vb(0) { }
ImageHandler::~ImageHandler() {
    close();
}

void ImageHandler::close() {
    if (imageTex != 0) {
        glDeleteTextures(1, &imageTex);
        imageTex = 0;
    }
    if (quad_vao != 0) {
        glDeleteVertexArrays(1, &quad_vao);
        quad_vao = 0;
    }
}

bool ImageHandler::initialize(sl::Resolution res) {
    shaderImage.set(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);
    texID = glGetUniformLocation(shaderImage.getProgramId(), "texImage");

    static const GLfloat quad[] = {-1.f, -1.f, 0.f, 1.f, -1.f, 0.f, -1.f, 1.f, 0.f, -1.f, 1.f, 0.f, 1.f, -1.f, 0.f, 1.f, 1.f, 0.f};

    glGenVertexArrays(1, &quad_vao);
    glBindVertexArray(quad_vao);

    glGenBuffers(1, &quad_vb);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.width, res.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_gl_ressource, imageTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    return (err == cudaSuccess);
}

void ImageHandler::pushNewImage(sl::Mat& image) {
    if (!cuda_gl_ressource)
        return;
    cudaArray_t ArrIm;
    cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
    cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
    cudaMemcpy2DToArray(
        ArrIm,
        0,
        0,
        image.getPtr<sl::uchar1>(sl::MEM::GPU),
        image.getStepBytes(sl::MEM::GPU),
        image.getPixelBytes() * image.getWidth(),
        image.getHeight(),
        cudaMemcpyDeviceToDevice
    );
    cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);
}

void ImageHandler::draw() {
    if (imageTex == 0)
        return;
    // Unbind any active VAO to avoid state leaks from prior draws
    glBindVertexArray(0);
    glUseProgram(shaderImage.getProgramId());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glUniform1i(texID, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, quad_vb);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}
