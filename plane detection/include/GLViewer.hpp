#ifndef __SIMPLE3DOBJECT_INCLUDE__
#define __SIMPLE3DOBJECT_INCLUDE__

#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <mutex>

//// UTILS //////
using namespace std;


//// UTILS //////
using namespace std;
void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "") ;

/////////////////

struct UserAction {
    bool press_space;
    bool hit;
    sl::uint2 hit_coord;

    void clear() {
        press_space = false;
        hit = false;
    }
};

class Shader {
public:
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
    sl::PLANE_TYPE type;
};

class GLViewer {
public:
    GLViewer();
    ~GLViewer();
    bool isAvailable();
    bool init(int argc, char **argv, sl::CameraInformation &cam_infos);
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

    static void drawCallback();
    static void keyReleasedCallback(unsigned char c, int x, int y);
    static void mouseButtonCallback(int button, int state, int x, int y);

    std::mutex mtx;

    bool available;
    bool change_state;

    // For CUDA-OpenGL interoperability
    cudaGraphicsResource* cuda_gl_ressource;//cuda GL resource
    MeshObject mesh_object;    // Opengl mesh container
    sl::Transform camera_projection; // OpenGL camera projection matrix
    
    sl::Mat image;
    sl::Transform pose;
    sl::POSITIONAL_TRACKING_STATE tracking_state;
    sl::MODEL cam_model;
    UserAction user_action;

    bool new_data;

    // Opengl object
    Shader *shader_mesh; //GLSL Shader for mesh
    Shader *shader_image;//GLSL Shader for image
    GLuint imageTex;            //OpenGL texture mapped with a cuda array (opengl gpu interop)
    GLuint shMVPMatrixLoc;      //Shader variable loc
    GLuint shColorLoc;          //Shader variable loc
    GLuint texID;               //Shader variable loc (sampler/texture)
    GLuint fbo = 0;             //FBO
    GLuint renderedTexture = 0; //Render Texture for FBO
    GLuint quad_vb;             //buffer for vertices/coords for image
};

#endif
