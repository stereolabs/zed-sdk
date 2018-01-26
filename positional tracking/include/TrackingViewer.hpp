#ifndef __CLOUD_VIEWER_INCLUDE__
#define __CLOUD_VIEWER_INCLUDE__

#include <GL/glew.h>
#include <GL/glut.h>    /* OpenGL Utility Toolkit header */

#include <GL/freeglut.h>

#include <sl/Core.hpp>

#include <math.h>
#include <thread>         // std::thread
#include <mutex>          // std::mutex

#include "ZEDModel.hpp"    /* OpenGL Utility Toolkit header */
#include <sl/Camera.hpp>

#ifndef M_PI
#define M_PI 3.1416f
#endif


#define SAFE_DELETE( res ) if( res!=NULL )  { delete res; res = NULL; }

#define MOUSE_R_SENSITIVITY 0.015f
#define MOUSE_UZ_SENSITIVITY 0.75f
#define MOUSE_DZ_SENSITIVITY 1.25f
#define MOUSE_T_SENSITIVITY 0.1f
#define KEY_T_SENSITIVITY 0.1f

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

    Simple3DObject();
    Simple3DObject(sl::Translation position, bool isStatic);
    ~Simple3DObject();

    void addPoint(float x, float y, float z, float r, float g, float b);
    void addPoint(sl::float3 position, sl::float3 color);
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
    - [1]: RGB color values;
    - [2]: Indices;
    */
    GLuint vboID_[3];

    sl::Translation position_;
    sl::Orientation rotation_;

};

// This class manages input events, window and Opengl rendering pipeline
class GLViewer {
public:
    GLViewer();
    ~GLViewer();
    void exit();
    bool isEnded();
    void init(sl::MODEL camera_model);
    void updateZEDPosition(sl::Transform zed_rt);
    void updateText(std::string str_t, std::string str_r, sl::TRACKING_STATE state);

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
    static void idle();

    bool ended_;

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

    Simple3DObject grill;
    Simple3DObject zedModel;
    Simple3DObject zedPath;
    std::vector<sl::float3> vecPath;
    std::mutex mtx;
    bool updateZEDposition;

    std::string txtR;
    std::string txtT;
    sl::TRACKING_STATE trackState;
    const std::string str_tracking = "POSITIONAL TRACKING : ";

    CameraGL camera_;
    Shader shader_;
    GLuint shMVPMatrixLoc_;
};

#endif
