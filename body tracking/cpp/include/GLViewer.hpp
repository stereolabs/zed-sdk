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
#include <GL/gl.h>
#include <GL/glut.h>   /* OpenGL Utility Toolkit header */

#include <cuda.h>
#include <cuda_gl_interop.h>

#ifndef M_PI
#define M_PI 3.141592653f
#endif


///////////////////////////////////////////////////////////////////////////////////////////////

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

struct ShaderData {
    Shader it;
    GLuint MVP_Mat;
};

class Simple3DObject {
public:

    Simple3DObject();
    ~Simple3DObject();

    void init();
    bool isInit();
    void addPoints(std::vector<sl::float3> pts,sl::float4 base_clr);
    void addBoundingBox(std::vector<sl::float3> bbox,sl::float4 base_clr);
    void addLine(sl::float3 p1, sl::float3 p2, sl::float3 clr);
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
    bool is_init;

    GLenum drawingType_;

    GLuint vaoID_;
    /*
    Vertex buffer IDs:
    - [0]: Vertices coordinates;
    - [1]: Indices;
    */
    GLuint vboID_[2];

    ShaderData shader;

    sl::Translation position_;
    sl::Orientation rotation_;
};

class ImageHandler {
public:
    ImageHandler();
    ~ImageHandler();

    // Initialize Opengl and Cuda buffers
    bool initialize(sl::Resolution res);
    // Push a new Image + Z buffer and transform into a point cloud
    void pushNewImage(sl::Mat &image);
    // Draw the Image
    void draw();
    // Close (disable update)
    void close();

private:
    GLuint texID;
    GLuint imageTex;
    cudaGraphicsResource* cuda_gl_ressource;//cuda GL resource
    ShaderData shaderImage;
    GLuint quad_vb;
};

struct ObjectClassName {
    sl::float3 position;
    std::string name_lineA;
    std::string name_lineB;
    sl::float4 color;
};

struct ObjectExtPosition {
    sl::float3 position;
    sl::Timestamp timestamp;
};

// This class manages input events, window and Opengl rendering pipeline
class GLViewer {
public:
    GLViewer();
    ~GLViewer();
    bool isAvailable();
    void init(int argc, char **argv, sl::CameraParameters param);
    void updateView(sl::Mat image, sl::Objects &obj);
    void exit();
    void setFloorPlaneEquation(sl::float4 eq);

private:
    void render();
    void update();
    void draw();
    void clearInputs();
    void setRenderCameraProjection(sl::CameraParameters params,float znear, float zfar);

    void printText();

    // Glut functions callbacks
    static void drawCallback();
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


    KEY_STATE keyStates_[256];

    std::mutex mtx;

    sl::Transform projection_;

    ImageHandler image_handler;

    ShaderData shader;
    sl::float3 bckgrnd_clr;

    std::vector<ObjectClassName> objectsName;

    Simple3DObject BBox_obj;
    Simple3DObject skeletons_obj;

    sl::Resolution windowSize;
    sl::Resolution imageSize;
    sl::CameraParameters camera_parameters;
    bool g_showBox=true;
    bool g_showLabel=true;
    int f_count = 0;
    bool floor_plane_set = false;
    sl::float4 floor_plane_eq;
    sl::Timestamp last_send_message_ts = 0;
    bool message_ready_to_be_sent = true;

};

#endif /* __VIEWER_INCLUDE__ */
