#ifndef __SIMPLE3DOBJECT_INCLUDE__
#define __SIMPLE3DOBJECT_INCLUDE__

#include <GL/glew.h>
#include <vector>
#include <sl_zed/Camera.hpp>
#include "Shader.hpp"

class MeshObject {
    GLuint vaoID_;
    GLuint vboID_[3];
    int current_fc;

public:
    MeshObject();
    ~MeshObject();
    sl::PLANE_TYPE type;
    void updateMesh(std::vector<sl::float3> &vertices, std::vector<sl::uint3> &triangles, std::vector<int> &border);
    void draw(GLuint type);

    std::vector<float> distance_border;
};
#endif
