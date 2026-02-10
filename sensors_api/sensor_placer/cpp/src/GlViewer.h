#ifndef GL_VIEWER_H
#define GL_VIEWER_H

#include <sl/Camera.hpp>

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>

#include <unordered_map>
#include <mutex>

// ─── Utility ─────────────────────────────────────────────────────────────────

inline float deg2rad(float d) {
    return d * (M_PI / 180.0f);
}
inline float rad2deg(float r) {
    return r * (180.0f / M_PI);
}

// ─── Simple3DObject (grid, frustums) ─────────────────────────────────────────

class Simple3DObject : protected QOpenGLFunctions {
public:
    Simple3DObject();
    ~Simple3DObject();

    void setStatic(bool s) {
        isStatic_ = s;
    }
    void setDrawingType(GLenum t) {
        drawingType_ = t;
    }

    void addPoint(float x, float y, float z, float r, float g, float b);
    void addLine(sl::float3 p1, sl::float3 p2, sl::float3 clr);
    void addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float3 clr0, sl::float3 clr);

    void pushToGPU();
    void clear();
    void draw();

private:
    std::vector<float> vertices_, colors_;
    std::vector<unsigned int> indices_;
    bool isStatic_ = false;
    bool needUpdate_ = false;
    bool canDraw_ = false;
    GLenum drawingType_ = GL_TRIANGLES;

    QOpenGLVertexArrayObject vao_;
    GLuint vbo_[3] = {};
};

// ─── PointCloud (CUDA ↔ GL interop) ─────────────────────────────────────────

class PointCloud : protected QOpenGLFunctions {
public:
    PointCloud();
    ~PointCloud();

    // Init from a GPU sl::Mat + CUDA stream  (ZED cameras)
    void initFromRef(sl::Mat& ref, CUstream stream);

    // Init with explicit dimensions (LiDAR — no shared ref)
    void initSized(int width, int height);

    // Copy new data from a GPU sl::Mat (LiDAR)
    void upload(sl::Mat& gpuMat);

    // Copy from shared ref mat  (ZED)
    void ingest();

    void draw();

    bool initialized() const {
        return initialized_;
    }

private:
    void ensureGLInit(); // lazily creates GL/CUDA resources

    CUstream strm_ = nullptr;
    int width_ = 0, height_ = 0;
    std::mutex mtx_;
    sl::Mat matGPU_; // shared ref for ZED
    bool initialized_ = false;
    bool glReady_ = false; // true once GL resources are created
    bool ownsRef_ = false; // true for LiDAR (initSized)
    size_t numBytes_ = 0;
    float* mappedBuf_ = nullptr;
    GLuint bufferGL_ = 0;
    cudaGraphicsResource* bufferCuda_ = nullptr;
};

// ─── CameraGL (3D viewport camera) ──────────────────────────────────────────

#define MOUSE_R_SENSITIVITY 0.03f
#define MOUSE_WHEEL_SENSITIVITY 0.01f
#define MOUSE_T_SENSITIVITY 0.2f

class CameraGL {
public:
    CameraGL(
        sl::Translation pos = sl::Translation(0, 0, 0),
        sl::Translation dir = sl::Translation(0, 0, -1),
        sl::Translation vert = sl::Translation(0, 1, 0)
    );
    ~CameraGL() = default;

    void update();
    void setProjection(float hFov, float vFov, float znear, float zfar);
    const sl::Transform& getViewProjectionMatrix() const;

    void setOffsetFromPosition(const sl::Translation& o);
    void setDirection(const sl::Translation& dir, const sl::Translation& vert);
    void translate(const sl::Translation& t);
    void setPosition(const sl::Translation& p);
    void rotate(const sl::Orientation& rot);
    void rotate(const sl::Rotation& m);

    const sl::Translation& getForward() const {
        return forward_;
    }
    const sl::Translation& getRight() const {
        return right_;
    }
    const sl::Translation& getUp() const {
        return up_;
    }
    const sl::Translation& getVertical() const {
        return vertical_;
    }

    static const sl::Translation ORIGINAL_FORWARD;
    static const sl::Translation ORIGINAL_UP;
    static const sl::Translation ORIGINAL_RIGHT;

    sl::Transform projection_;

private:
    void updateVectors();
    void updateView();
    void updateVP();

    sl::Translation offset_, position_, forward_, up_, right_, vertical_;
    sl::Orientation rotation_;
    sl::Transform view_, vp_;
    float hFov_ = 70, vFov_ = 70, znear_ = 0.05f, zfar_ = 100.f;
};

// ─── Per-sensor rendering data ──────────────────────────────────────────────

struct SensorRenderData {
    sl::Transform pose;
    PointCloud pointCloud;
    Simple3DObject frustum; // only for ZED
    sl::float3 color;
    bool canDraw = false;
    bool isLidar = false;
    int serial = 0;
};

// ─── GlViewer (QOpenGLWidget) ───────────────────────────────────────────────

class GlViewer : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT
public:
    explicit GlViewer(QWidget* parent = nullptr);
    ~GlViewer() override;

    void setAvailable(bool c) {
        canDraw_ = c;
    }

    // ZED cameras
    void addCamera(sl::CameraParameters& param, sl::Mat& pcRef, CUstream stream, int id, sl::float3 color);

    // LiDAR sensors
    void addLidar(int id, sl::float3 color);

    void updateSensor(int id, sl::Transform pose, bool updatePc);
    void updateLidarCloud(int id, sl::Mat& gpuPC);

    void removeSensor(int id);
    void hideSensor(int id, bool hide);

    void setPointSize(float s) {
        ptSize_ = s;
    }
    void drawColors(bool s) {
        drawColors_ = s;
    }

    void resetView();

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;

private:
    void createShader(const char* vs, const char* fs, QOpenGLShaderProgram** out);
    void buildGrid();
    void rotateCamera();
    void translateCamera();

    bool canDraw_ = false;
    float ptSize_ = 2.5f;
    bool drawColors_ = true;

    sl::float2 tracking_, lastMouse_, currMouse_;
    bool capturedMouse_ = false;

    CameraGL* camera_ = nullptr;
    QOpenGLShaderProgram* shaderSimple_ = nullptr;
    QOpenGLShaderProgram* shaderPC_ = nullptr;
    Simple3DObject* grid_ = nullptr;

    std::unordered_map<int, SensorRenderData> sensors_;
};

#endif // GL_VIEWER_H
