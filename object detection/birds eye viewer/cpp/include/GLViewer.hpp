#ifndef __VIEWER_INCLUDE__
#define __VIEWER_INCLUDE__

#include <vector>
#include <mutex>

#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glut.h>   /* OpenGL Utility Toolkit header */

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <opencv2/opencv.hpp>

#include "utils.hpp"


#ifndef M_PI
#define M_PI 3.141592653f
#endif

#define MOUSE_R_SENSITIVITY 0.04f
#define MOUSE_UZ_SENSITIVITY 0.75f
#define MOUSE_DZ_SENSITIVITY 1.25f
#define MOUSE_T_SENSITIVITY 100.f
#define KEY_T_SENSITIVITY 0.1f

// Utils
std::vector<sl::float3> getBBoxOnFloorPlane(std::vector<sl::float3> &bbox, sl::Pose cam_pose);

// 3D

class CameraGL {
public:

    CameraGL() {
    }

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
    bool usePerspective_;
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

    Shader() {
    }
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

struct ShaderData {
    Shader it;
    GLuint MVP_Mat;
};

class Simple3DObject {
public:

    Simple3DObject() {
    }
    Simple3DObject(sl::Translation position, bool isStatic);
    ~Simple3DObject();

    void addPt(sl::float3 pt);
    void addClr(sl::float4 clr);

    void addBBox(std::vector<sl::float3> &pts, sl::float4 clr);
    void addPoint(sl::float3 pt, sl::float4 clr);
    void addTriangle(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float4 clr);
    void addLine(sl::float3 p1, sl::float3 p2, sl::float4 clr);

    // New 3D rendering
    void addFullEdges(std::vector<sl::float3> &pts, sl::float4 clr);
    void addVerticalEdges(std::vector<sl::float3> &pts, sl::float4 clr);
    void addTopFace(std::vector<sl::float3> &pts, sl::float4 clr);
    void addVerticalFaces(std::vector<sl::float3> &pts, sl::float4 clr);

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
    - [1]: Indices;
     */
    GLuint vboID_[2];

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

private:
    sl::Mat matGPU_;
    bool hasNewPCL_ = false;
    ShaderData shader;
    size_t numBytes_;
    float* xyzrgbaMappedBuf_;
    GLuint bufferGLID_;
    cudaGraphicsResource* bufferCudaID_;
};

struct ObjectClassName {
    sl::float3 position;
    std::string name;
    sl::float4 color;
};

// This class manages input events, window and Opengl rendering pipeline

class GLViewer {
public:
    GLViewer();
    ~GLViewer();
    bool isAvailable();

    void init(int argc, char **argv, sl::CameraParameters& param);
    void updateData(sl::Mat& matXYZRGBA, std::vector<sl::ObjectData>& objs, sl::Transform &cam_pose);

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

    void printText();
    void createBboxRendering(std::vector<sl::float3> &bbox, sl::float4 bbox_clr);
    void createIDRendering(sl::float3 &center, sl::float4 clr, unsigned int id);

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

    std::mutex mtx;

    Simple3DObject frustum;
    PointCloud pointCloud_;
    CameraGL camera_;
    ShaderData shaderLine;
    ShaderData shader;
    sl::float4 bckgrnd_clr;
    sl::Transform cam_pose;

    std::vector<ObjectClassName> objectsName;
    Simple3DObject BBox_edges;
    Simple3DObject BBox_faces;
    Simple3DObject skeletons;
    Simple3DObject floor_grid;
};

#endif /* __VIEWER_INCLUDE__ */
