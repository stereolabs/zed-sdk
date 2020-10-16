#ifndef __SIMPLE3DOBJECT_INCLUDE__
#define __SIMPLE3DOBJECT_INCLUDE__

#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <mutex>

#ifndef M_PI
#define M_PI 3.141592653f
#endif

//// UTILS //////
void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "") ;

/////////////////

struct UserAction {
    bool press_space;
    bool hit;
    sl::float2 hit_coord;

    void clear() {
        press_space = false;
        hit = false;
    }
};

class Shader {
public:
    Shader() {}
    Shader(GLchar* vs, GLchar* fs);
    ~Shader();

    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_VERTICES_DIST = 1;

private:
    bool compile(GLuint &shaderId, GLenum type, GLchar* src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};

struct ShaderData {
    Shader it;
    GLuint MVP_Mat;
    GLuint shColorLoc;
};

class MeshObject {
    GLuint vaoID_;
    GLuint vboID_[3];
    int current_fc;
    bool need_update;

    std::vector<sl::float3> vert;
    std::vector<float> edge_dist;
    std::vector<sl::uint3> tri;

public:
    MeshObject();
    ~MeshObject();
    void updateMesh(std::vector<sl::float3> &vertices, std::vector<sl::uint3> &triangles, std::vector<int> &border);
    void pushToGPU();
    void draw();
    void alloc();
    sl::PLANE_TYPE type;
    ShaderData shader;
};

class ImageHandler {
    public:
    ImageHandler();
    ~ImageHandler();

    // Initialize Opengl and Cuda buffers
    bool initialize(sl::Resolution res);
    // Push a new Image + Z buffer and transform into a point cloud
    void pushNewImage(sl::Mat& image);
    // Draw the Image
    void draw();
    // Close (disable update)
    void close();

    private:
    GLuint texID;
    GLuint imageTex;
    cudaGraphicsResource* cuda_gl_ressource;//cuda GL resource
    ShaderData shader;
    GLuint quad_vb;
};

class GLViewer {
public:
    GLViewer();
    ~GLViewer();
    bool isAvailable();
    bool init(int argc, char **argv, sl::CameraParameters &camLeft, bool has_imu);
    UserAction updateImageAndState(sl::Mat &image, sl::Transform &pose, sl::POSITIONAL_TRACKING_STATE track_state);
    void updateMesh(sl::Mesh &mesh, sl::PLANE_TYPE type);
    
    void exit();
private:
    // Rendering loop method called each frame by glutDisplayFunc
    void render();
    // Everything that needs to be updated before rendering must be done in this method
    void update();
    // Once everything is updated, every renderable objects must be drawn in this method
    void draw();

    void printText();

    void setRenderCameraProjection(sl::CameraParameters params, float znear, float zfar);

    static void drawCallback();
    static void reshapeCallback(int width, int height);
    static void keyReleasedCallback(unsigned char c, int x, int y);
    static void mouseButtonCallback(int button, int state, int x, int y);
    static void idle();

    std::mutex mtx;

    bool available;    
    sl::Transform pose;
    sl::POSITIONAL_TRACKING_STATE tracking_state;
    UserAction user_action;

    bool new_data;
    bool use_imu;
    int wnd_w, wnd_h;

    // Opengl object
    // OpenGL camera projection matrix
    sl::Transform camera_projection;
    ImageHandler image_handler;
    MeshObject mesh_object; // Opengl mesh container
};

#endif
