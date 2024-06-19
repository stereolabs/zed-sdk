#ifndef __VIEWER_INCLUDE__
#define __VIEWER_INCLUDE__

#include <vector>
#include <mutex>
#include <map>
#include <deque>
#include <vector>
#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#ifndef M_PI
#define M_PI 3.141592653f
#endif

#define MOUSE_R_SENSITIVITY 0.03f
#define MOUSE_UZ_SENSITIVITY 0.75f
#define MOUSE_DZ_SENSITIVITY 1.25f
#define MOUSE_T_SENSITIVITY 50.f
#define KEY_T_SENSITIVITY 0.1f

using namespace sl;

const std::vector<std::pair<BODY_38_PARTS, BODY_38_PARTS>> BODY_BONES_FAST_RENDER
{
    {
        BODY_38_PARTS::PELVIS, BODY_38_PARTS::SPINE_1
    },
    {
        BODY_38_PARTS::SPINE_1, BODY_38_PARTS::SPINE_2
    },
    {
        BODY_38_PARTS::SPINE_2, BODY_38_PARTS::SPINE_3
    },
    {
        BODY_38_PARTS::SPINE_3, BODY_38_PARTS::NECK
    },
    // Face
    {
        BODY_38_PARTS::NECK, BODY_38_PARTS::NOSE
    },
    {
        BODY_38_PARTS::NOSE, BODY_38_PARTS::LEFT_EYE
    },
    {
        BODY_38_PARTS::LEFT_EYE, BODY_38_PARTS::LEFT_EAR
    },
    {
        BODY_38_PARTS::NOSE, BODY_38_PARTS::RIGHT_EYE
    },
    {
        BODY_38_PARTS::RIGHT_EYE, BODY_38_PARTS::RIGHT_EAR
    },
    // Left arm
    {
        BODY_38_PARTS::SPINE_3, BODY_38_PARTS::LEFT_CLAVICLE
    },
    {
        BODY_38_PARTS::LEFT_CLAVICLE, BODY_38_PARTS::LEFT_SHOULDER
    },
    {
        BODY_38_PARTS::LEFT_SHOULDER, BODY_38_PARTS::LEFT_ELBOW
    },
    {
        BODY_38_PARTS::LEFT_ELBOW, BODY_38_PARTS::LEFT_WRIST
    },
    // Right arm
    {
        BODY_38_PARTS::SPINE_3, BODY_38_PARTS::RIGHT_CLAVICLE
    },
    {
        BODY_38_PARTS::RIGHT_CLAVICLE, BODY_38_PARTS::RIGHT_SHOULDER
    },
    {
        BODY_38_PARTS::RIGHT_SHOULDER, BODY_38_PARTS::RIGHT_ELBOW
    },
    {
        BODY_38_PARTS::RIGHT_ELBOW, BODY_38_PARTS::RIGHT_WRIST
    },
    // Left leg
    {
        BODY_38_PARTS::PELVIS, BODY_38_PARTS::LEFT_HIP
    },
    {
        BODY_38_PARTS::LEFT_HIP, BODY_38_PARTS::LEFT_KNEE
    },
    {
        BODY_38_PARTS::LEFT_KNEE, BODY_38_PARTS::LEFT_ANKLE
    },
    {
        BODY_38_PARTS::LEFT_ANKLE, BODY_38_PARTS::LEFT_HEEL
    },
    {
        BODY_38_PARTS::LEFT_ANKLE, BODY_38_PARTS::LEFT_BIG_TOE
    },
    {
        BODY_38_PARTS::LEFT_ANKLE, BODY_38_PARTS::LEFT_SMALL_TOE
    },
    // Right leg
    {
        BODY_38_PARTS::PELVIS, BODY_38_PARTS::RIGHT_HIP
    },
    {
        BODY_38_PARTS::RIGHT_HIP, BODY_38_PARTS::RIGHT_KNEE
    },
    {
        BODY_38_PARTS::RIGHT_KNEE, BODY_38_PARTS::RIGHT_ANKLE
    },
    {
        BODY_38_PARTS::RIGHT_ANKLE, BODY_38_PARTS::RIGHT_HEEL
    },
    {
        BODY_38_PARTS::RIGHT_ANKLE, BODY_38_PARTS::RIGHT_BIG_TOE
    },
    {
        BODY_38_PARTS::RIGHT_ANKLE, BODY_38_PARTS::RIGHT_SMALL_TOE
    },
};

///////////////////////////////////////////////////////////////////////////////////////////////

class Shader {
public:

    Shader() {
    }
    Shader(const GLchar* vs, const GLchar* fs);
    ~Shader();
    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_COLOR_POS = 1;
    static const GLint ATTRIB_NORMAL = 2;
private:
    bool compile(GLuint &shaderId, GLenum type, const GLchar* src);
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

    Simple3DObject();
    Simple3DObject(sl::Translation position, bool isStatic);
    ~Simple3DObject();

    void addPt(sl::float3 pt);
    void addClr(sl::float4 clr);
    void addNormal(sl::float3 normal);
    void addPoints(std::vector<sl::float3> pts, sl::float4 base_clr);
    void addBoundingBox(std::vector<sl::float3> bbox, sl::float4 base_clr);
    void addPoint(sl::float3 pt, sl::float4 clr);
    void addLine(sl::float3 p1, sl::float3 p2, sl::float3 clr);
    void addCylinder(sl::float3 startPosition, sl::float3 endPosition, sl::float4 clr);
    void addSphere(sl::float3 position, sl::float4 clr, float radius = 0.01f * 1000.0f * 2);
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
    std::vector<float> normals_;

    bool isStatic_;

    GLenum drawingType_;

    GLuint vaoID_;
    /*
    Vertex buffer IDs:
    - [0]: Vertices coordinates;
    - [1]: Colors;
    - [2]: Indices;
    - [3]: Normals
     */
    GLuint vboID_[4];

    sl::Translation position_;
    sl::Orientation rotation_;
};

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

// This class manages input events, window and Opengl rendering pipeline

class GLViewer {
public:
    GLViewer();
    ~GLViewer();
    bool isAvailable();
    void init(int argc, char **argv);
    void updateData(Bodies &);
    void exit();

    void restart() {
        available = true;
        clearInputs();
    }

private:
    void render();
    void update();
    void draw();
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

    std::mutex mtx;

    ShaderData shaderLine;
    ShaderData shaderSK;

    sl::float3 bckgrnd_clr;

    CameraGL camera_;

    Simple3DObject skeletons;
    Simple3DObject floor_grid;
};

#endif /* __VIEWER_INCLUDE__ */
