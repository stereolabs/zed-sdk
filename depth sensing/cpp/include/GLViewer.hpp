#ifndef __VIEWER_INCLUDE__
#define __VIEWER_INCLUDE__

#include <vector>
#include <mutex>

#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#ifndef M_PI
#define M_PI 3.141592653f
#endif

#define MOUSE_R_SENSITIVITY 0.03f
#define MOUSE_UZ_SENSITIVITY 0.5f
#define MOUSE_DZ_SENSITIVITY 1.25f
#define MOUSE_T_SENSITIVITY 0.05f
#define KEY_T_SENSITIVITY 0.1f


//// UTILS //////
using namespace std;
void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "") ;

/////////////////

class CameraGL {
public:

    CameraGL() {}
    enum DIRECTION {
        UP, DOWN, LEFT, RIGHT, FORWARD, BACK
    };
    CameraGL(sl::Translation position, sl::Translation direction, sl::Translation vertical = sl::Translation(0, 1, 0)); // vertical = Eigen::Vector3f(0, 1, 0)
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
private:
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

    Shader() {}
    Shader(GLchar* vs, GLchar* fs);
    ~Shader();
    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_COLOR_POS = 1;
private:
    bool compile(GLuint &shaderId, GLenum type, GLchar* src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};

class Simple3DObject {
public:

    Simple3DObject() {}
    Simple3DObject(sl::Translation position, bool isStatic);
    ~Simple3DObject();

    void addPoint(sl::float3 pt, sl::float3 clr);
    void addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float3 clr);
    void pushToGPU();
    void clear();

    void setDrawingType(GLenum type);

    void draw();

    void translate(const sl::Translation& t);
    void setPosition(const sl::Translation& p);

    void setRT(const sl::Transform& mRT);

    void rotate(const sl::Orientation& rot);
    void rotate(const sl::Rotation& m);
    void setRotation(const sl::Orientation& rot);
    void setRotation(const sl::Rotation& m);

    const sl::Translation& getPosition() const;

    sl::Transform getModelMatrix() const;
private:
    std::vector<float> vertices_;
    std::vector<float> colors_;
    std::vector<unsigned int> indices_;

    bool isStatic_;

    GLenum drawingType_;

    GLuint vaoID_;
    /*
            Vertex buffer IDs:
            - [0]: Vertices coordinates;
            - [1]: Vertices colors;
            - [2]: Indices;
     */
    GLuint vboID_[3];

    sl::Translation position_;
    sl::Orientation rotation_;

};

class PointCloud {
public:
    PointCloud();
    ~PointCloud();

    // Initialize Opengl and Cuda buffers
    // Warning: must be called in the Opengl thread
    void initialize(sl::Resolution res);
    // Push a new point cloud
    // Warning: can be called from any thread but the mutex "mutexData" must be locked
    void pushNewPC(sl::Mat &matXYZRGBA);
    // Update the Opengl buffer
    // Warning: must be called in the Opengl thread
    void update();
    // Draw the point cloud
    // Warning: must be called in the Opengl thread
    void draw(const sl::Transform& vp);
    // Close (disable update)
    void close();
    
    std::mutex mutexData;
private:
    sl::Mat matGPU_;
    bool hasNewPCL_ = false;
    Shader shader_;
    GLuint shMVPMatrixLoc_;
    size_t numBytes_;
    float* xyzrgbaMappedBuf_;
    GLuint bufferGLID_;
    cudaGraphicsResource* bufferCudaID_;
};

// This class manages input events, window and Opengl rendering pipeline
class GLViewer {
public:
    GLViewer();
    ~GLViewer();
    bool isAvailable();

    GLenum init(int argc, char **argv, sl::CameraParameters param);
    void updatePointCloud(sl::Mat &matXYZRGBA);

    void exit();
private:
    // Rendering loop method called each frame by glutDisplayFunc
    void render();
    // Everything that needs to be updated before rendering must be done in this method
    void update();
    // Once everything is updated, every renderable objects must be drawn in this method
    void draw();
    // Clear and refresh inputs' data
    void clearInputs();
    
    // Glut functions callbacks
    static void drawCallback();
    static void mouseButtonCallback(int button, int state, int x, int y);
    static void mouseMotionCallback(int x, int y);
    static void reshapeCallback(int width, int height);
    static void keyPressedCallback(unsigned char c, int x, int y);
    static void keyReleasedCallback(unsigned char c, int x, int y);
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
    sl::float3 bckgrnd_clr;

    Simple3DObject frustum;
    PointCloud pointCloud_;
    CameraGL camera_;
    Shader shader_;
    GLuint shMVPMatrixLoc_;
};

#endif /* __VIEWER_INCLUDE__ */
