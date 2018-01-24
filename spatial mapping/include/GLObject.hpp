#ifndef __SIMPLE3DOBJECT_INCLUDE__
#define __SIMPLE3DOBJECT_INCLUDE__

#include <GL/glew.h>
#include <vector>
#include <sl/Camera.hpp>
#include "Shader.hpp"

class MeshObject {
    GLuint vaoID_;
    GLuint vboID_[2];
    int current_fc;

public:
    MeshObject();
    ~MeshObject();    

    void updateMesh(std::vector<sl::float3> &vertices, std::vector<sl::uint3> &triangles);
    void draw(GLenum drawing_mode);
};
#endif
