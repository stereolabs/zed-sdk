#ifndef GL_VIEWER_HPP
#define GL_VIEWER_HPP

#include "Viewer.hpp"

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>

#include <sl/Camera.hpp>
#include <sl/Lidar.hpp>

#include <mutex>
#include <iostream>
#include <unordered_map>
#include <functional>
#include <algorithm>

// A Simple Shader class for basic rendering
class Shader {
public:
    Shader()
        : verterxId_(0)
        , fragmentId_(0)
        , programId_(0) { }
    Shader(const GLchar* vs, const GLchar* fs);
    ~Shader();

    // Delete copy constructor and assignment
    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    // Move constructor and assignment
    Shader(Shader&& o) noexcept;
    Shader& operator=(Shader&& o) noexcept;

    GLuint getProgramId() const {
        return programId_;
    }

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_COLOR_POS = 1;
    static const GLint ATTRIB_TEXCOORD = 2;

private:
    bool compile(GLuint& shaderId, GLenum type, const GLchar* src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};

// Simple Camera class for the 3D viewer
class CameraGL {
public:
    CameraGL();
    ~CameraGL() = default;

    void update();
    void setProjection(float hfov, float ar, float zn, float zf);
    void setPosition(sl::float3 p);

    const sl::Transform& getViewProjectionMatrix() const {
        return vpMatrix_;
    }
    float getHorizontalFOV() const {
        return horizontalFOV_;
    }
    float getVerticalFOV() const {
        return verticalFOV_;
    }

    void setOffsetFromPosition(sl::float3 o);
    const sl::float3& getOffsetFromPosition() const {
        return offset_;
    }
    sl::float3 getPosition() const {
        return position_;
    }

    void setDirection(sl::float3 direction, sl::float3 vertical);
    void translate(sl::float3 t);
    void setRotation(sl::Rotation r);
    void rotate(const sl::Rotation& r);

private:
    void updateVectors();
    void updateView();
    void updateVPMatrix();

    sl::float3 offset_;
    sl::float3 up_;
    sl::float3 right_;
    sl::float3 forward_;
    sl::float3 vertical_;

    sl::Rotation rotation_;
    sl::float3 position_;

    sl::Transform view_;
    sl::Transform projection_;
    sl::Transform vpMatrix_;

    float horizontalFOV_;
    float verticalFOV_;

    bool usePerspective_;
};

// Point Cloud for GPU rendering with CUDA interop
class PointCloud {
public:
    PointCloud(sl::Resolution res);
    ~PointCloud();

    void setOffset(float x, float y, float z);
    void pushNewPC(sl::Mat& pc);
    void draw(const sl::Transform& vp, const Shader& shader);

    sl::float3 offset = {0, 0, 0};

private:
    GLuint bufferGLID_ = 0;
    cudaGraphicsResource* bufferCudaID_ = nullptr;
    size_t numPoints_;
    bool hasData_ = false;
};

// A GL Image for displaying camera images
class GLImage {
public:
    GLImage();
    ~GLImage();
    void init(int width, int height, bool bgra = true);
    void pushImage(sl::Mat& img);
    void draw(float x0, float y0, float width, float height, const Shader& shader);
    float getAspectRatio() const {
        return imgHeight_ > 0 ? (float)imgWidth_ / (float)imgHeight_ : 1.0f;
    }

private:
    GLuint texId_ = 0;
    GLuint vbo_ = 0;
    int imgWidth_ = 0;
    int imgHeight_ = 0;
    bool inited_ = false;
    bool isBGRA_ = true;
};

// Data structure for pending data updates
struct PendingData {
    sl::Mat pointCloud;
    sl::Mat image;
    sl::Resolution pointCloudRes;
    sl::Resolution imageRes;
    bool hasNewCloud = false;
    bool hasNewImage = false;
    sl::float3 cloudOffset {0, 0, 0};
    bool hasCloudOffset = false;
};

// Data structure for rendering in the GL viewer
struct CachedObjectData {
    std::unique_ptr<PointCloud> cloud;
    std::unique_ptr<GLImage> image;
};

// Layout modes for 2D image display
enum class ViewerLayout {
    SplitRows, // Lidar on top, cameras at bottom (default)
    Combined   // All images in single bottom row (legacy)
};

// The Main OpenGL Viewer class implementing our abstract Viewer interface
class GLViewer : public Viewer {
public:
    GLViewer();
    ~GLViewer() override;

    void init(int argc, char** argv) override;
    void setLayout(ViewerLayout layout) {
        layout_ = layout;
    }
    bool isAvailable() override;
    bool isDone() override;

    // Recording status callback
    void setRecordingCallback(std::function<std::string()> getStatusFn) {
        getRecordingStatusFn_ = std::move(getStatusFn);
    }

    // Recording toggle callback
    void setRecordingToggleCallback(std::function<void()> toggleFn) {
        recordingToggleFn_ = std::move(toggleFn);
    }

    // Update methods from parent interface
    void updateLidar(const std::string& id, sl::Mat& point_cloud, sl::Mat& intensity_img) override;
    void updateZed(const std::string& id, sl::Mat& image, sl::Mat* point_cloud = nullptr) override;
    void updateZedOne(const std::string& id, sl::Mat& image) override;
    void updateZedObjects(
        const std::string& id,
        const sl::Objects& objects,
        const sl::Matrix4f* transform = nullptr,
        sl::Timestamp timestamp = 0
    ) override;
    void
    updateZedBodies(const std::string& id, const sl::Bodies& bodies, const sl::Matrix4f* transform = nullptr, sl::Timestamp timestamp = 0)
        override;

    // Set lidar/camera offset for rendering
    void setLidarOffset(const std::string& id, float x, float y, float z);
    void setZedOffset(const std::string& id, float x, float y, float z);

private:
    // Internal GL callbacks (static with pointer to instance)
    static GLViewer* currentInstance_;
    static void drawCallback() {
        if (currentInstance_)
            currentInstance_->render();
    }
    static void reshapeCallback(int w, int h) {
        if (currentInstance_)
            currentInstance_->reshape(w, h);
    }
    static void keyboardCallback(unsigned char key, int x, int y) {
        if (currentInstance_)
            currentInstance_->keyboard(key, x, y);
    }
    static void mouseButtonCallback(int button, int state, int x, int y) {
        if (currentInstance_)
            currentInstance_->mouseButton(button, state, x, y);
    }
    static void mouseMotionCallback(int x, int y) {
        if (currentInstance_)
            currentInstance_->mouseMotion(x, y);
    }
    static void idleCallback() {
        glutPostRedisplay();
    }

    static void keyboardUpCallback(unsigned char key, int x, int y) {
        if (currentInstance_)
            currentInstance_->keyboardUp(key, x, y);
    }
    static void specialKeyDownCallback(int key, int x, int y) {
        if (currentInstance_)
            currentInstance_->specialKeyDown(key, x, y);
    }
    static void specialKeyUpCallback(int key, int x, int y) {
        if (currentInstance_)
            currentInstance_->specialKeyUp(key, x, y);
    }

    // Internal methods
    void render();
    void reshape(int w, int h);
    void keyboard(unsigned char key, int x, int y);
    void keyboardUp(unsigned char key, int x, int y);
    void specialKeyDown(int key, int x, int y);
    void specialKeyUp(int key, int x, int y);

    void mouseButton(int button, int state, int x, int y);
    void mouseMotion(int x, int y);

    void applyCameraMotion();
    void drawText(const std::string& text, int x, int y);

    // State
    std::mutex mtx_;
    bool initialized_ = false;
    bool available_ = true;
    int windowWidth_ = 1600;
    int windowHeight_ = 900;
    bool specialKeyState_[256] = {false};
    ViewerLayout layout_ = ViewerLayout::SplitRows;

    // Camera
    CameraGL camera_;
    sl::float3 cameraVelocity_ {0, 0, 0};
    bool keyState_[256] = {false};
    int lastMouseX_ = 0, lastMouseY_ = 0;
    bool mouseRotating_ = false;
    bool mousePanning_ = false;
    bool firstRotateMotion_ = true;

    // Shaders
    Shader pcShader_;
    Shader pcShader_zed_;
    Shader imgShader_;
    Shader imgLidarShader_;

    Shader lidarShader_;

    // GL objects
    std::unordered_map<std::string, CachedObjectData> lidars_;
    std::unordered_map<std::string, CachedObjectData> zeds_;

    // Pending data to be applied in render loop
    std::unordered_map<std::string, PendingData> pendingLidars_;
    std::unordered_map<std::string, PendingData> pendingZeds_;

    // Recording status callback
    std::function<std::string()> getRecordingStatusFn_;

    // Recording toggle callback
    std::function<void()> recordingToggleFn_;
};

#endif // GL_VIEWER_HPP
