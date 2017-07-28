#include "GLObject.hpp"

MeshObject::MeshObject() {
    current_fc = 0;
    vaoID_ = 0;
}

MeshObject::~MeshObject() {
    current_fc = 0;
}

void MeshObject::updateMesh(std::vector<sl::float3> &vertices, std::vector<sl::uint3> &triangles) {
    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(2, vboID_);
    }

    glBindVertexArray(vaoID_);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(sl::float3), &vertices[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.size() * sizeof(sl::uint3), &triangles[0], GL_DYNAMIC_DRAW);

    current_fc = (int) triangles.size() * 3;

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MeshObject::draw(GLenum drawing_mode) {
    if ((current_fc > 0) && (vaoID_ > 0)) {
        glBindVertexArray(vaoID_);
        glDrawElements(drawing_mode, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}
