#ifndef __SIMPLE3DOBJECT_INCLUDE__
#define __SIMPLE3DOBJECT_INCLUDE__

#include <GL/glew.h>
#include <vector>
#include <sl/Camera.hpp>
#include "Shader.hpp"

class GLObject {
public:
	GLObject(sl::Translation position, bool isStatic);
    ~GLObject();

    void addPoint(float x, float y, float z, float r, float g, float b);
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

    std::vector<float> m_vertices_;
    std::vector<float> m_colors_;
    std::vector<unsigned int> m_indices_;
 
    bool isStatic_;

    GLenum drawingType_;
    GLuint vaoID_;    
   /* Vertex buffer IDs:
    - [0]: vertices coordinates;
    - [1]: RGB color values;
    - [2]: indices;*/     
    GLuint vboID_[3];

    sl::Translation position_;
    sl::Orientation rotation_;
};


class MeshObject: public GLObject {

public:
    MeshObject(sl::Translation position,bool);
    ~MeshObject();

    std::vector<float> normal_;
    GLuint vboID_[4];
    int current_fc;

    void updateMesh(sl::Mesh &_mesh);
    void draw(GLenum drawing_mode);
};
#endif
