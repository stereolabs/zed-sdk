#pragma once

#ifndef __SIMPLE3DOBJECT_INCLUDE_H_
#define __SIMPLE3DOBJECT_INCLUDE_H_

#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <mutex>

#include <list>

#ifndef M_PI
#define M_PI 3.141592653f
#endif

//// UTILS //////
using namespace std;
void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "") ;

/////////////////
class Shader{
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
    GLuint shColorLoc;
};

class SubMapObj {
    GLuint vaoID_;
    GLuint vboID_[2];
    int current_fc;

    std::vector<sl::uint1> index;
public:
    SubMapObj();
    ~SubMapObj();
    template<typename T>
    void update(T &chunks);
    void draw();
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
    template<typename T>
    bool init(int argc, char **argv, sl::CameraParameters, T*ptr);
    bool updateImageAndState(sl::Mat &image,  sl::Transform &pose, sl::POSITIONAL_TRACKING_STATE track_state, sl::SPATIAL_MAPPING_STATE mapp_state);
    
    void updateChunks() {
        new_chunks = true;
        chunks_pushed = false;
    }

    bool chunksUpdated() {
        return chunks_pushed;
    }

    void clearCurrentMesh();

    void exit();
private:
    // Rendering loop method called each frame by glutDisplayFunc
    void render();
    // Everything that needs to be updated before rendering must be done in this method
    void update();
    // Once everything is updated, every renderable objects must be drawn in this method
    void draw();

    void setRenderCameraProjection(sl::CameraParameters params, float znear, float zfar);

    void printText();

    static void drawCallback();
    static void reshapeCallback(int width, int height);
    static void keyReleasedCallback(unsigned char c, int x, int y);
    static void idle();
    
    template<typename T>
    void initPtr(T* ptr);

    std::mutex mtx;

    bool available;
    bool change_state;

    std::list<SubMapObj> sub_maps;  // Opengl mesh container
    sl::float3 vertices_color;      // Defines the color of the mesh
    
    // OpenGL camera projection matrix
    sl::Transform camera_projection;

    sl::Transform pose;
    sl::POSITIONAL_TRACKING_STATE tracking_state;
    sl::SPATIAL_MAPPING_STATE mapping_state;

    bool new_chunks;
    bool chunks_pushed;
    bool ask_clear;
    bool draw_mesh;

    sl::Mesh* p_mesh;
    sl::FusedPointCloud* p_fpc;
    ImageHandler image_handler;
    ShaderData shader_obj;
};

/* Find MyDocuments directory for windows platforms.*/
std::string getDir();

#endif
