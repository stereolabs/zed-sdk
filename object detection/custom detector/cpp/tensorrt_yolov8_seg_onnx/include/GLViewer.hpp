#ifndef __GLVIEWER_HPP__
#define __GLVIEWER_HPP__

#include <vector>
#include <mutex>

#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <opencv2/opencv.hpp>

#include "utils.h"


#ifndef M_PI
#define M_PI 3.141592653f
#endif

#define MOUSE_R_SENSITIVITY 0.03f
#define MOUSE_UZ_SENSITIVITY 0.5f
#define MOUSE_DZ_SENSITIVITY 1.25f
#define MOUSE_T_SENSITIVITY 0.05f
#define KEY_T_SENSITIVITY 0.1f

// Utils
std::vector<sl::float3> getBBoxOnFloorPlane(std::vector<sl::float3> const& bbox, sl::Pose const& cam_pose);

// 3D

class CameraGL {
public:

    CameraGL() {
    }

    enum DIRECTION {
        UP, DOWN, LEFT, RIGHT, FORWARD, BACK
    };
    CameraGL(sl::Translation const& position, sl::Translation const& direction, sl::Translation const& vertical = sl::Translation(0, 1, 0)); // vertical = Eigen::Vector3f(0, 1, 0)
    ~CameraGL();

    void update();
    void setProjection(float const horizontalFOV, float const verticalFOV, float const znear, float const zfar);
    const sl::Transform& getViewProjectionMatrix() const;

    float getHorizontalFOV() const;
    float getVerticalFOV() const;

    // Set an offset between the eye of the camera and its position
    // Note: Useful to use the camera as a trackball camera with z>0 and x = 0, y = 0
    // Note: coordinates are in local space
    void setOffsetFromPosition(sl::Translation const& offset);
    const sl::Translation& getOffsetFromPosition() const;

    void setDirection(sl::Translation const& direction, sl::Translation const &vertical);
    void translate(sl::Translation const& t);
    void setPosition(sl::Translation const& p);
    void rotate(sl::Orientation const& rot);
    void rotate(sl::Rotation const& m);
    void setRotation(sl::Orientation const& rot);
    void setRotation(sl::Rotation const& m);

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
private:
    bool compile(GLuint &shaderId, GLenum const type, GLchar const* src);
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
    Simple3DObject(sl::Translation const& position, bool const isStatic);
    ~Simple3DObject();

    void addPt(sl::float3 const& pt);
    void addClr(sl::float4 const& clr);

    void addBBox(std::vector<sl::float3> const& pts, sl::float4& clr);
    void addPoint(sl::float3 const& pt, sl::float4 const& clr);
    void addTriangle(sl::float3 const& p1, sl::float3 const& p2, sl::float3 const& p3, sl::float4 const& clr);
    void addLine(sl::float3 const& p1, sl::float3 const& p2, sl::float4 const& clr);

    // New 3D rendering
    void addFullEdges(std::vector<sl::float3> const& pts, sl::float4& clr);
    void addVerticalEdges(std::vector<sl::float3> const& pts, sl::float4& clr);
    void addTopFace(std::vector<sl::float3> const& pts, sl::float4& clr);
    void addVerticalFaces(std::vector<sl::float3> const& pts, sl::float4& clr);

    void pushToGPU();
    void clear();

    void setDrawingType(GLenum const type);

    void draw();

    void translate(sl::Translation const& t);
    void setPosition(sl::Translation const& p);

    void setRT(sl::Transform const& mRT);

    void rotate(sl::Orientation const& rot);
    void rotate(sl::Rotation const& m);
    void setRotation(sl::Orientation const& rot);
    void setRotation(sl::Rotation const& m);

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
    void initialize(sl::Resolution const& res);
    // Push a new point cloud
    // Warning: can be called from any thread but the mutex "mutexData" must be locked
    void pushNewPC(sl::Mat const& matXYZRGBA);
    // Update the Opengl buffer
    // Warning: must be called in the Opengl thread
    void update();
    // Draw the point cloud
    // Warning: must be called in the Opengl thread
    void draw(sl::Transform const& vp);
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
    bool isPlaying() const { return play; }
    void setPlaying(const bool p) { play = p; }

    void init(int argc, char **argv, sl::CameraParameters const& param, bool const isTrackingON);
    void updateData(sl::Mat const& matXYZRGBA, std::vector<sl::ObjectData> const& objs, sl::Transform const& cam_pose);

    int getKey() {
        int const key{last_key};
        last_key = -1;
        return key;
    }

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
    void createBboxRendering(std::vector<sl::float3> const&bbox, sl::float4& bbox_clr);
    void createIDRendering(sl::float3 const& center, sl::float4 const& clr, unsigned int const id);

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

    bool isTrackingON_ = false;

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

    bool play{true};
    int last_key{-1};
};

#endif // __GLVIEWER_HPP__
