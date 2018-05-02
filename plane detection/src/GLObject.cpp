#include "GLObject.hpp"

MeshObject::MeshObject() {
    current_fc = 0;
    vaoID_ = 0;
}

MeshObject::~MeshObject() {
    distance_border.clear();
    current_fc = 0;
}

void MeshObject::updateMesh(std::vector<sl::float3> &vertices, std::vector<sl::uint3> &triangles, std::vector<int> &border) {
    if(vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(3, vboID_);
    }

    glBindVertexArray(vaoID_);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(sl::float3), &vertices[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    distance_border.resize(vertices.size());

    for(int i = 0; i < distance_border.size(); i++) {
        float d_min = std::numeric_limits<float>::max();
        sl::float3 v_current = vertices[i];
        for(int j = 0; j < border.size(); j++) {
            float dist_current = sl::float3::distance(v_current, vertices[border[j]]);
            if(dist_current < d_min) {
                d_min = dist_current;
            }
        }
        distance_border[i] = d_min >= 0.002 ? 0.4 : 0.0f;
    }


    glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ARRAY_BUFFER, distance_border.size() * sizeof(float), &distance_border[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_DIST, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_DIST);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangles.size() * sizeof(sl::uint3), &triangles[0], GL_DYNAMIC_DRAW);

    current_fc = (int) triangles.size() * 3;

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MeshObject::draw(GLuint type) {
    if((current_fc > 0) && (vaoID_ > 0)) {
        glBindVertexArray(vaoID_);
        glDrawElements(type, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}
