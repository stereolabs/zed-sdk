#ifndef __VIEWER_INCLUDE__
#define __VIEWER_INCLUDE__

#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>   /* OpenGL Utility Toolkit header */

#include <list>

#include <cuda_gl_interop.h>

#ifndef M_PI
#define M_PI 3.141592653f
#endif

class CameraGL {
public:

    CameraGL() {
    }

    enum DIRECTION {
        UP, DOWN, LEFT, RIGHT, FORWARD, BACK
    };

    CameraGL(sl::Translation position, sl::Translation direction, sl::Translation vertical = sl::Translation(0, 1, 0));
    ~CameraGL();

    void update();
    void setProjection(float horizontalFOV, float verticalFOV, float znear, float zfar);
    const sl::Transform& getViewProjectionMatrix() const;

    float getHorizontalFOV() const;
    float getVerticalFOV() const;

    // Set an offset between the eye of the camera and its position
    // Note: Useful to use the camera as a trackball camera with z>0 and x = 0, y = 0
    // Note: coordinates are in local space
    void setOffsetFromPosition(const sl::Translation& offset);
    const sl::Translation& getOffsetFromPosition() const;

    void setDirection(const sl::Translation& direction, const sl::Translation &vertical);
    void translate(const sl::Translation& t);
    void setPosition(const sl::Translation& p);
    void rotate(const sl::Orientation& rot);
    void rotate(const sl::Rotation& m);
    void setRotation(const sl::Orientation& rot);
    void setRotation(const sl::Rotation& m);

    const sl::Translation& getPosition() const;
    const sl::Translation& getForward() const;
    const sl::Translation& getRight() const;
    const sl::Translation& getUp() const;
    const sl::Translation& getVertical() const;
    float getZNear() const;
    float getZFar() const;

    static const sl::Translation ORIGINAL_FORWARD;
    static const sl::Translation ORIGINAL_UP;
    static const sl::Translation ORIGINAL_RIGHT;

    sl::Transform projection_;
    //private:
    void updateVectors();
    void updateView();
    void updateVPMatrix();

    sl::Translation offset_;
    sl::Translation position_;
    sl::Translation forward_;
    sl::Translation up_;
    sl::Translation right_;
    sl::Translation vertical_;

    sl::Orientation rotation_;

    sl::Transform view_;
    sl::Transform vpMatrix_;
    float horizontalFieldOfView_;
    float verticalFieldOfView_;
    float znear_;
    float zfar_;
};

class Shader {
public:

    Shader() : verterxId_(0), fragmentId_(0), programId_(0) {}
    Shader(const GLchar* vs, const GLchar* fs);
    ~Shader();

    // Delete the move constructor and move assignment operator
    Shader(Shader&&) = delete;
    Shader& operator=(Shader&&) = delete;

    // Delete the copy constructor and copy assignment operator
    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    void set(const GLchar* vs, const GLchar* fs);
    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_COLOR_POS = 1;
    static const GLint ATTRIB_NORMAL_POS = 2;
    
private:
    bool compile(GLuint &shaderId, GLenum type, const GLchar* src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};

class Simple3DObject {
public:

    Simple3DObject();

    ~Simple3DObject();

    inline bool isInit() {
        return vaoID_ != 0;
    }

    void addPoint(sl::float3 pt, sl::float4 clr);
    void addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float3 clr);
    void pushToGPU();
    void clear();

    void setStatic(bool _static) {
        isStatic_ = _static;
    }

    void setDrawingType(GLenum type);

    void draw();

private:
    std::vector<sl::float3> vertices_;
    std::vector<sl::float4> colors_;
    std::vector<unsigned int> indices_;

    bool isStatic_ = false;
    bool need_update;
    GLenum drawingType_;
    GLuint vaoID_;
    GLuint vboID_[3];
};

class PointCloud {
public:
    PointCloud();
    ~PointCloud();

    // Initialize Opengl and Cuda buffers
    // Warning: must be called in the Opengl thread
    void initialize(sl::Mat&);
    // Push a new point cloud
    // Warning: can be called from any thread but the mutex "mutexData" must be locked
    void pushNewPC(CUstream);
    // Draw the point cloud
    // Warning: must be called in the Opengl thread
    void draw(const sl::Transform& vp);
    // Close (disable update)
    void close();

private:
    sl::Mat refMat;

    Shader shader_;
    GLuint shMVPMatrixLoc_;
    size_t numBytes_;
    float* xyzrgbaMappedBuf_;
    GLuint bufferGLID_;
    cudaGraphicsResource* bufferCudaID_;
};

struct ShaderObj {
    Shader it;
    GLuint MVPM;
};

class CameraViewer {
public:
    CameraViewer();
    ~CameraViewer();

    // Initialize Opengl and Cuda buffers
    bool initialize(sl::Mat& image);
    // Push a new Image + Z buffer and transform into a point cloud
    void pushNewImage(CUstream);
    // Draw the Image
    void draw2D();
    // Close (disable update)
    void close();

    Simple3DObject frustum;
    float aspect_ratio;
private:
    sl::Mat ref;
	cudaArray_t ArrIm;
    cudaGraphicsResource* cuda_gl_ressource;//cuda GL resource
    Shader shader;
    Shader shader_im;
    GLuint shMVPMatrixLocTex_;

    GLuint texture;
    GLuint vaoID_;
    GLuint vboID_[3];

    std::vector<sl::uint3> faces;
    std::vector<sl::float3> vert;
    std::vector<sl::float2> uv;
};

// This class manages input events, window and Opengl rendering pipeline

class GLViewer {
public:
    GLViewer();
    ~GLViewer();

    bool isAvailable();

    void exit();

    void init(int argc, char **argv, sl::Mat &, sl::Mat &, CUstream);
    void updateCameraPose(sl::Transform, sl::PositionalTrackingStatus);
    void pushTrackedLM(std::vector<sl::float3> &lm);
    void pushLM(std::map<uint64_t, sl::Landmark> &lm);
    
private:
    // Rendering loop method called each frame by glutDisplayFunc
    void render();
    // Everything that needs to be updated before rendering must be done in this method
    void update();
    // Once everything is updated, every renderable objects must be drawn in this method
    void draw();
    // Clear and refresh inputs' data
    void clearInputs();

    void printText();

    static GLViewer* currentInstance_;

    // Glut functions callbacks
    static void drawCallback();
    static void mouseButtonCallback(int button, int state, int x, int y);
    static void mouseMotionCallback(int x, int y);
    static void reshapeCallback(int width, int height);
    static void keyPressedCallback(unsigned char c, int x, int y);
    static void keyReleasedCallback(unsigned char c, int x, int y);
    static void specialKeyReleasedCallback(int c, int x, int y);
    static void idle();

    bool available;

    enum MOUSE_BUTTON {
        LEFT = 0,
        MIDDLE = 1,
        RIGHT = 2,
        WHEEL_UP = 3,
        WHEEL_DOWN = 4
    };

    enum KEY_STATE {
        UP = 'u',
        DOWN = 'd',
        FREE = 'f'
    };

    bool mouseButton_[3];
    int mouseWheelPosition_;
    int mouseCurrentPosition_[2];
    int mouseMotion_[2];
    int previousMouseMotion_[2];
    KEY_STATE keyStates_[256];

    // follow camera
    sl::Transform cam_pose;
    sl::PositionalTrackingStatus trackState;

    CameraGL camera_;

    ShaderObj shader;

    bool draw_live_point_cloud;
    bool draw_landmark;
    bool dark_background;
    bool follow_cam;

    Simple3DObject ZED_path;
    Simple3DObject origin_axis;

    PointCloud pc_render;
    CameraViewer camera_viewer;

    CUstream strm;

    bool SPLIT_DISPLAY = true;
    sl::Resolution wnd_size;

    Simple3DObject lms_tracked;
    Simple3DObject lms;

    float control_magnitude = 1.f;
};

#endif /* __VIEWER_INCLUDE__ */
