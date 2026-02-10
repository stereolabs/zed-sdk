#include "GlViewer.h"
#include <iostream>
#include <cuda_gl_interop.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// ─── Shaders ─────────────────────────────────────────────────────────────────

static const char* VERTEX_SHADER = "#version 330 core\n"
                                   "layout(location = 0) in vec3 in_Vertex;\n"
                                   "layout(location = 1) in vec3 in_Color;\n"
                                   "uniform mat4 u_mvpMatrix;\n"
                                   "uniform float pointsize;\n"
                                   "out vec3 b_color;\n"
                                   "void main() {\n"
                                   "   b_color = in_Color;\n"
                                   "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1.0);\n"
                                   "   gl_PointSize = pointsize;\n"
                                   "}\n";

static const char* FRAGMENT_SHADER = "#version 330 core\n"
                                     "in vec3 b_color;\n"
                                     "out vec4 out_Color;\n"
                                     "void main() { out_Color = vec4(b_color, 1.0); }\n";

static const char* PC_VERTEX_SHADER = "#version 330 core\n"
                                      "layout(location = 0) in vec4 in_VertexRGBA;\n"
                                      "out vec4 b_color;\n"
                                      "uniform mat4 u_mvpMatrix;\n"
                                      "uniform vec3 u_color;\n"
                                      "uniform bool u_KeepColor;\n"
                                      "uniform bool u_isLidar;\n"
                                      "uniform float pointsize;\n"
                                      "void main() {\n"
                                      "   if (u_isLidar) {\n"
                                      "       // Height-based blue-to-green gradient\n"
                                      "       float h = clamp((in_VertexRGBA.y + 2.0) / 4.0, 0.0, 1.0);\n"
                                      "       vec3 blue  = vec3(0.1, 0.3, 0.9);\n"
                                      "       vec3 green = vec3(0.1, 0.9, 0.4);\n"
                                      "       b_color = vec4(mix(blue, green, h), 1.0);\n"
                                      "   } else {\n"
                                      "       uint vertexColor = floatBitsToUint(in_VertexRGBA.w);\n"
                                      "       vec3 clr_int = vec3((vertexColor & uint(0x000000FF)),\n"
                                      "                           (vertexColor & uint(0x0000FF00)) >> 8,\n"
                                      "                           (vertexColor & uint(0x00FF0000)) >> 16);\n"
                                      "       vec3 clrf = clr_int / 255.0;\n"
                                      "       if (u_KeepColor) {\n"
                                      "           b_color = vec4(clrf, 1.0);\n"
                                      "       } else {\n"
                                      "           float grey = (clrf.r + clrf.g + clrf.b) / 3.0;\n"
                                      "           b_color = vec4(u_color * grey, 1.0);\n"
                                      "       }\n"
                                      "   }\n"
                                      "   gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1.0);\n"
                                      "   gl_PointSize = pointsize;\n"
                                      "}\n";

static const char* PC_FRAGMENT_SHADER = "#version 330 core\n"
                                        "in vec4 b_color;\n"
                                        "out vec4 out_Color;\n"
                                        "void main() { out_Color = b_color; }\n";

// ═══════════════════════════════════════════════════════════════════════════════
// Simple3DObject
// ═══════════════════════════════════════════════════════════════════════════════

Simple3DObject::Simple3DObject()
    : QOpenGLFunctions() {
    // defer initializeOpenGLFunctions() to pushToGPU, when GL context is guaranteed
}

Simple3DObject::~Simple3DObject() {
    canDraw_ = false;
    if (vao_.isCreated()) {
        glDeleteBuffers(3, vbo_);
        vao_.destroy();
    }
}

void Simple3DObject::addPoint(float x, float y, float z, float r, float g, float b) {
    vertices_.insert(vertices_.end(), {x, y, z});
    colors_.insert(colors_.end(), {r, g, b});
    indices_.push_back((unsigned int)indices_.size());
    needUpdate_ = true;
}

void Simple3DObject::addLine(sl::float3 p1, sl::float3 p2, sl::float3 clr) {
    addPoint(p1.x, p1.y, p1.z, clr.r, clr.g, clr.b);
    addPoint(p2.x, p2.y, p2.z, clr.r, clr.g, clr.b);
}

void Simple3DObject::addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float3 clr0, sl::float3 clr) {
    addPoint(p1.x, p1.y, p1.z, clr0.r, clr0.g, clr0.b);
    addPoint(p2.x, p2.y, p2.z, clr.r, clr.g, clr.b);
    addPoint(p3.x, p3.y, p3.z, clr.r, clr.g, clr.b);
}

void Simple3DObject::pushToGPU() {
    if (!isStatic_ || !vao_.isCreated() || needUpdate_) {
        canDraw_ = false;
        if (!vao_.isCreated()) {
            initializeOpenGLFunctions();
            vao_.create();
            glGenBuffers(3, vbo_);
        }
        vao_.bind();

        GLenum usage = isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW;
        glBindBuffer(GL_ARRAY_BUFFER, vbo_[0]);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), vertices_.data(), usage);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_[1]);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(float), colors_.data(), usage);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int), indices_.data(), usage);

        vao_.release();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        canDraw_ = true;
        needUpdate_ = false;
    }
}

void Simple3DObject::clear() {
    canDraw_ = false;
    vertices_.clear();
    colors_.clear();
    indices_.clear();
}

void Simple3DObject::draw() {
    if (canDraw_) {
        vao_.bind();
        glDrawElements(drawingType_, (GLsizei)indices_.size(), GL_UNSIGNED_INT, 0);
        vao_.release();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PointCloud
// ═══════════════════════════════════════════════════════════════════════════════

PointCloud::PointCloud()
    : QOpenGLFunctions() { }

PointCloud::~PointCloud() {
    if (initialized_) {
        std::lock_guard<std::mutex> lk(mtx_);
        initialized_ = false;
        if (matGPU_.isInit() && ownsRef_)
            matGPU_.free();
        if (glReady_) {
            cudaGraphicsUnmapResources(1, &bufferCuda_, 0);
            cudaGraphicsUnregisterResource(bufferCuda_);
            glDeleteBuffers(1, &bufferGL_);
            glReady_ = false;
        }
    }
}

void PointCloud::ensureGLInit() {
    if (glReady_ || width_ == 0)
        return;
    initializeOpenGLFunctions();

    glGenBuffers(1, &bufferGL_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGL_);
    glBufferData(GL_ARRAY_BUFFER, width_ * height_ * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&bufferCuda_, bufferGL_, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsMapResources(1, &bufferCuda_, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&mappedBuf_, &numBytes_, bufferCuda_);
    glReady_ = true;
}

void PointCloud::initFromRef(sl::Mat& ref, CUstream stream) {
    std::lock_guard<std::mutex> lk(mtx_);
    strm_ = stream;
    matGPU_ = ref;
    ownsRef_ = false;
    width_ = ref.getWidth();
    height_ = ref.getHeight();
    initialized_ = true;
    // GL resources created lazily in ensureGLInit()
}

void PointCloud::initSized(int width, int height) {
    std::lock_guard<std::mutex> lk(mtx_);
    strm_ = nullptr;
    ownsRef_ = false;
    width_ = width;
    height_ = height;
    initialized_ = true;
    // GL resources created lazily in ensureGLInit()
}

void PointCloud::upload(sl::Mat& gpuMat) {
    if (!initialized_)
        return;
    std::lock_guard<std::mutex> lk(mtx_);
    ensureGLInit();
    if (!glReady_)
        return;
    int w = gpuMat.getWidth();
    int h = gpuMat.getHeight();
    // Resize if needed
    if (w * h != width_ * height_) {
        cudaGraphicsUnmapResources(1, &bufferCuda_, 0);
        cudaGraphicsUnregisterResource(bufferCuda_);
        glDeleteBuffers(1, &bufferGL_);

        width_ = w;
        height_ = h;
        glGenBuffers(1, &bufferGL_);
        glBindBuffer(GL_ARRAY_BUFFER, bufferGL_);
        glBufferData(GL_ARRAY_BUFFER, width_ * height_ * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        cudaGraphicsGLRegisterBuffer(&bufferCuda_, bufferGL_, cudaGraphicsRegisterFlagsNone);
        cudaGraphicsMapResources(1, &bufferCuda_, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&mappedBuf_, &numBytes_, bufferCuda_);
    }
    size_t bytes = w * h * 4 * sizeof(float);
    if (bytes <= numBytes_)
        cudaMemcpy(mappedBuf_, gpuMat.getPtr<sl::float4>(sl::MEM::GPU), bytes, cudaMemcpyDeviceToDevice);
}

void PointCloud::ingest() {
    if (!initialized_)
        return;
    std::lock_guard<std::mutex> lk(mtx_);
    ensureGLInit();
    if (!glReady_)
        return;
    cudaMemcpyAsync(mappedBuf_, matGPU_.getPtr<sl::float4>(sl::MEM::GPU), numBytes_, cudaMemcpyDeviceToDevice, strm_);
}

void PointCloud::draw() {
    if (!initialized_)
        return;
    std::lock_guard<std::mutex> lk(mtx_);
    ensureGLInit();
    if (!glReady_)
        return;
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    glBindBuffer(GL_ARRAY_BUFFER, bufferGL_);
    f->glEnableVertexAttribArray(0);
    f->glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_POINTS, 0, width_ * height_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CameraGL
// ═══════════════════════════════════════════════════════════════════════════════

const sl::Translation CameraGL::ORIGINAL_FORWARD = sl::Translation(0, 0, 1);
const sl::Translation CameraGL::ORIGINAL_UP = sl::Translation(0, 1, 0);
const sl::Translation CameraGL::ORIGINAL_RIGHT = sl::Translation(1, 0, 0);

CameraGL::CameraGL(sl::Translation pos, sl::Translation dir, sl::Translation vert) {
    position_ = pos;
    setDirection(dir, vert);
    offset_ = sl::Translation(0, 0, 1);
    view_.setIdentity();
    updateView();
    updateVP();
}

void CameraGL::update() {
    if (sl::Translation::dot(vertical_, up_) < 0)
        vertical_ = vertical_ * -1.f;
    updateView();
    updateVP();
}

void CameraGL::setProjection(float hFov, float vFov, float znear, float zfar) {
    hFov_ = hFov;
    vFov_ = vFov;
    znear_ = znear;
    zfar_ = zfar;
    projection_.setIdentity();
    float sx = 1.f / tan(hFov_ * 0.5f * M_PI / 180.f);
    float sy = 1.f / tan(vFov_ * 0.5f * M_PI / 180.f);
    projection_(0, 0) = sx;
    projection_(1, 1) = sy;
    projection_(2, 2) = -zfar_ / (zfar_ - znear_);
    projection_(2, 3) = -zfar_ * znear_ / (zfar_ - znear_);
    projection_(3, 2) = -1;
    projection_(3, 3) = 0;
}

const sl::Transform& CameraGL::getViewProjectionMatrix() const {
    return vp_;
}

void CameraGL::setOffsetFromPosition(const sl::Translation& o) {
    offset_ = o;
}

void CameraGL::setDirection(const sl::Translation& dir, const sl::Translation& vert) {
    sl::Translation d = dir;
    d.normalize();
    rotation_ = sl::Orientation(ORIGINAL_FORWARD, d * -1.f);
    updateVectors();
    vertical_ = vert;
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
    rotate(sl::Orientation(m));
}

void CameraGL::updateVectors() {
    forward_ = ORIGINAL_FORWARD * rotation_;
    up_ = ORIGINAL_UP * rotation_;
    right_ = sl::Translation(ORIGINAL_RIGHT * -1.f) * rotation_;
}

void CameraGL::updateView() {
    sl::Transform t(rotation_, (offset_ * rotation_) + position_);
    view_ = sl::Transform::inverse(t);
}

void CameraGL::updateVP() {
    vp_ = projection_ * view_;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GlViewer
// ═══════════════════════════════════════════════════════════════════════════════

static void addGridLine(Simple3DObject* obj, float i, float limit, sl::float3& clr, float h) {
    obj->addLine(sl::float3(i, h, -limit), sl::float3(i, h, limit), clr);
    obj->addLine(sl::float3(-limit, h, i), sl::float3(limit, h, i), clr);
}

GlViewer::GlViewer(QWidget* parent)
    : QOpenGLWidget(parent)
    , QOpenGLFunctions() {
    camera_ = new CameraGL();
    camera_->setProjection(70, 70, 0.05f, 200.f);
    resetView();
    setFocusPolicy(Qt::StrongFocus);
}

GlViewer::~GlViewer() {
    canDraw_ = false;
    makeCurrent();
    sensors_.clear();
    delete shaderSimple_;
    delete shaderPC_;
    delete camera_;
    delete grid_;
}

void GlViewer::createShader(const char* vs, const char* fs, QOpenGLShaderProgram** out) {
    auto* v = new QOpenGLShader(QOpenGLShader::Vertex, this);
    v->compileSourceCode(vs);
    auto* f = new QOpenGLShader(QOpenGLShader::Fragment, this);
    f->compileSourceCode(fs);
    if (*out)
        delete *out;
    *out = new QOpenGLShaderProgram(this);
    (*out)->addShader(v);
    (*out)->addShader(f);
    (*out)->link();
    (*out)->bind();
    (*out)->release();
}

void GlViewer::buildGrid() {
    grid_ = new Simple3DObject();
    grid_->setStatic(true);
    grid_->setDrawingType(GL_LINES);

    float limit = 20.f;
    sl::float3 clr1(187, 190, 191);
    clr1 /= 255.f;
    sl::float3 clr2(73, 74, 74);
    clr2 /= 255.f;

    for (int i = (int)-limit; i <= (int)limit; i++) {
        addGridLine(grid_, (float)i, limit, (i % 5 == 0) ? clr2 : clr1, 0.f);
    }
    // Axis lines
    grid_->addLine({0, 0, 0}, {1, 0, 0}, {0.9f, 0, 0});
    grid_->addLine({0, 0, 0}, {0, 1, 0}, {0, 0.9f, 0});
    grid_->addLine({0, 0, 0}, {0, 0, 1}, {0, 0, 0.9f});
    grid_->pushToGPU();
}

void GlViewer::initializeGL() {
    initializeOpenGLFunctions();
    makeCurrent();
    createShader(VERTEX_SHADER, FRAGMENT_SHADER, &shaderSimple_);
    createShader(PC_VERTEX_SHADER, PC_FRAGMENT_SHADER, &shaderPC_);
    buildGrid();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

void GlViewer::paintGL() {
    makeCurrent();
    glClearColor(0.94f, 0.95f, 0.95f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (!canDraw_)
        return;

    // Update frustums
    for (auto& [id, s] : sensors_)
        s.frustum.pushToGPU();

    camera_->update();
    QMatrix4x4 vp(camera_->getViewProjectionMatrix().m);

    // Draw grid + frustums
    shaderSimple_->bind();
    shaderSimple_->setUniformValue("u_mvpMatrix", vp);
    shaderSimple_->setUniformValue("pointsize", ptSize_);
    grid_->draw();

    for (auto& [id, s] : sensors_) {
        if (!s.canDraw)
            continue;
        QMatrix4x4 pose(s.pose.m);
        shaderSimple_->setUniformValue("u_mvpMatrix", vp * pose);
        s.frustum.draw();
    }
    shaderSimple_->release();

    // Draw point clouds
    shaderPC_->bind();
    shaderPC_->setUniformValue("u_KeepColor", drawColors_);
    shaderPC_->setUniformValue("pointsize", ptSize_);

    for (auto& [id, s] : sensors_) {
        if (!s.canDraw)
            continue;
        QMatrix4x4 pose(s.pose.m);
        shaderPC_->setUniformValue("u_mvpMatrix", vp * pose);
        QVector3D clr(s.color.x, s.color.y, s.color.z);
        shaderPC_->setUniformValue("u_color", clr);
        shaderPC_->setUniformValue("u_isLidar", s.isLidar);
        s.pointCloud.draw();
    }
    shaderPC_->release();
}

void GlViewer::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    float hFov = rad2deg(2.f * atan(w / (2.f * 500.f)));
    float vFov = rad2deg(2.f * atan(h / (2.f * 500.f)));
    camera_->setProjection(hFov, vFov, 0.05f, 200.f);
}

void GlViewer::resetView() {
    camera_->setPosition(sl::Translation(0.f, 2.f, 3.f));
    camera_->setDirection(sl::Translation(0.f, -1.4f, -1.5f), sl::Translation(0, 1, 0));
    camera_->setOffsetFromPosition(sl::Translation(0, 0, 4));
}

// ─── Sensor management ──────────────────────────────────────────────────────

void GlViewer::addCamera(sl::CameraParameters& param, sl::Mat& pcRef, CUstream stream, int id, sl::float3 color) {
    auto& s = sensors_[id];
    s.color = color / 255.f;
    s.isLidar = false;

    // Build frustum
    const float Z_ = -0.2f;
    float fx_ = 1.f / param.fx;
    float fy_ = -1.f / param.fy;
    sl::float3 cam_0(0, 0, 0);
    sl::float3 cam_1((0 - param.cx) * Z_ * fx_, (0 - param.cy) * Z_ * fy_, Z_);
    sl::float3 cam_2((param.image_size.width - param.cx) * Z_ * fx_, (0 - param.cy) * Z_ * fy_, Z_);
    sl::float3 cam_3((param.image_size.width - param.cx) * Z_ * fx_, (param.image_size.height - param.cy) * Z_ * fy_, Z_);
    sl::float3 cam_4((0 - param.cx) * Z_ * fx_, (param.image_size.height - param.cy) * Z_ * fy_, Z_);

    sl::float3 clr0(0.7f, 0.7f, 0.75f);
    s.frustum.setDrawingType(GL_TRIANGLES);
    s.frustum.setStatic(true);
    s.frustum.addFace(cam_0, cam_1, cam_2, clr0, s.color);
    s.frustum.addFace(cam_0, cam_2, cam_3, clr0, s.color);
    s.frustum.addFace(cam_0, cam_3, cam_4, clr0, s.color);
    s.frustum.addFace(cam_0, cam_4, cam_1, clr0, s.color);

    s.pointCloud.initFromRef(pcRef, stream);
    s.canDraw = true;
}

void GlViewer::addLidar(int id, sl::float3 color) {
    auto& s = sensors_[id];
    s.color = color / 255.f;
    s.isLidar = true;

    // Build a small sphere / cube marker for the LiDAR
    sl::float3 clr0(0.7f, 0.7f, 0.75f);
    float sz = 0.05f;
    s.frustum.setDrawingType(GL_TRIANGLES);
    s.frustum.setStatic(true);
    // Simple diamond shape
    sl::float3 top(0, sz, 0), bot(0, -sz, 0);
    sl::float3 fr(0, 0, -sz), bk(0, 0, sz), lt(-sz, 0, 0), rt(sz, 0, 0);
    s.frustum.addFace(top, fr, rt, clr0, s.color);
    s.frustum.addFace(top, rt, bk, clr0, s.color);
    s.frustum.addFace(top, bk, lt, clr0, s.color);
    s.frustum.addFace(top, lt, fr, clr0, s.color);
    s.frustum.addFace(bot, rt, fr, clr0, s.color);
    s.frustum.addFace(bot, bk, rt, clr0, s.color);
    s.frustum.addFace(bot, lt, bk, clr0, s.color);
    s.frustum.addFace(bot, fr, lt, clr0, s.color);

    // Point cloud will be initialized later on first upload
    s.pointCloud.initSized(1024, 64); // reasonable default for Ouster
    s.canDraw = true;
}

void GlViewer::updateSensor(int id, sl::Transform pose, bool updatePc) {
    auto it = sensors_.find(id);
    if (it == sensors_.end())
        return;
    if (updatePc && !it->second.isLidar)
        it->second.pointCloud.ingest();
    it->second.pose = pose;
}

void GlViewer::updateLidarCloud(int id, sl::Mat& gpuPC) {
    auto it = sensors_.find(id);
    if (it == sensors_.end())
        return;
    it->second.pointCloud.upload(gpuPC);
}

void GlViewer::removeSensor(int id) {
    sensors_.erase(id);
}
void GlViewer::hideSensor(int id, bool hide) {
    auto it = sensors_.find(id);
    if (it != sensors_.end())
        it->second.canDraw = !hide;
}

// ─── Mouse / Keyboard ───────────────────────────────────────────────────────

void GlViewer::mousePressEvent(QMouseEvent* e) {
    if (e->buttons() & (Qt::LeftButton | Qt::RightButton)) {
        capturedMouse_ = true;
        lastMouse_ = sl::float2(e->pos().x(), e->pos().y());
    }
}

void GlViewer::mouseMoveEvent(QMouseEvent* e) {
    if (!capturedMouse_)
        return;
    currMouse_ = sl::float2(e->pos().x(), e->pos().y());
    if (e->buttons() & Qt::LeftButton) {
        tracking_ = (currMouse_ - lastMouse_) / 5.f;
        rotateCamera();
    }
    if (e->buttons() & Qt::RightButton) {
        tracking_ = (currMouse_ - lastMouse_) / 10.f;
        translateCamera();
    }
    lastMouse_ = currMouse_;
}

void GlViewer::mouseReleaseEvent(QMouseEvent*) {
    capturedMouse_ = false;
}

void GlViewer::wheelEvent(QWheelEvent* e) {
    float numDeg = e->angleDelta().y() / 8.f;
    camera_->translate(-1.f * camera_->getForward() * numDeg * MOUSE_WHEEL_SENSITIVITY);
}

void GlViewer::keyPressEvent(QKeyEvent* e) {
    switch (e->key()) {
        case Qt::Key_R:
            resetView();
            break;
        case Qt::Key_Right:
            tracking_ = {0.1f, 0};
            translateCamera();
            break;
        case Qt::Key_Left:
            tracking_ = {-0.1f, 0};
            translateCamera();
            break;
        case Qt::Key_Up:
            tracking_ = {0, -0.1f};
            translateCamera();
            break;
        case Qt::Key_Down:
            tracking_ = {0, 0.1f};
            translateCamera();
            break;
        case Qt::Key_C:
            drawColors_ = !drawColors_;
            break;
        default:
            break;
    }
}

void GlViewer::rotateCamera() {
    camera_->rotate(sl::Rotation(tracking_.x * MOUSE_R_SENSITIVITY, camera_->getVertical() * -1.f));
    camera_->rotate(sl::Rotation(tracking_.y * MOUSE_R_SENSITIVITY, camera_->getRight()));
}

void GlViewer::translateCamera() {
    camera_->translate(camera_->getRight() * tracking_.x * MOUSE_T_SENSITIVITY);
    camera_->translate(camera_->getUp() * tracking_.y * MOUSE_T_SENSITIVITY);
}
