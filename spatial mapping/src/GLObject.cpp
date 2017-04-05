#include "GLObject.hpp"

using namespace sl;

GLObject::GLObject(Translation position, bool isStatic): isStatic_(isStatic){
    vaoID_ = 0;
    drawingType_ = GL_TRIANGLES;
    position_ = position;
    rotation_.setIdentity();
}

GLObject::~GLObject(){
    if(vaoID_ != 0){
        glDeleteBuffers(3, vboID_);
        glDeleteVertexArrays(1, &vaoID_);
    }
}

void GLObject::addPoint(float x, float y, float z, float r, float g, float b){
    m_vertices_.push_back(x);
    m_vertices_.push_back(y);
    m_vertices_.push_back(z);
    m_colors_.push_back(r);
    m_colors_.push_back(g);
    m_colors_.push_back(b);
    m_indices_.push_back(m_indices_.size());
}

void GLObject::pushToGPU(){
    if(!isStatic_ || vaoID_ == 0){
        if(vaoID_ == 0){
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(3, vboID_);
        }
        glBindVertexArray(vaoID_);
        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, m_vertices_.size() * sizeof(float), &m_vertices_[0], isStatic_?GL_STATIC_DRAW:GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, m_colors_.size() * sizeof(float), &m_colors_[0], isStatic_?GL_STATIC_DRAW:GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices_.size() * sizeof(unsigned int), &m_indices_[0], isStatic_?GL_STATIC_DRAW:GL_DYNAMIC_DRAW);

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void GLObject::clear(){
    m_vertices_.clear();
    m_colors_.clear();
    m_indices_.clear();
}

void GLObject::setDrawingType(GLenum type){
    drawingType_ = type;
}

void GLObject::draw(){
    glBindVertexArray(vaoID_);
    glDrawElements(drawingType_, (GLsizei)m_indices_.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void GLObject::translate(const Translation& t){
    position_ = position_ + t;
}

void GLObject::setPosition(const Translation& p){
    position_ = p;
}

void GLObject::setRT(const Transform& mRT){
    position_ = mRT.getTranslation();
    rotation_ = mRT.getOrientation();
}

void GLObject::rotate(const Orientation& rot){
    rotation_ = rot * rotation_;
}

void GLObject::rotate(const Rotation& m){
    this->rotate(sl::Orientation(m));
}

void GLObject::setRotation(const Orientation& rot){
    rotation_ = rot;
}

void GLObject::setRotation(const Rotation& m){
    this->setRotation(sl::Orientation(m));
}

const Translation& GLObject::getPosition() const{
    return position_;
}

Transform GLObject::getModelMatrix() const{
    Transform tmp = Transform::identity();
    tmp.setOrientation(rotation_);
    tmp.setTranslation(position_);
    return tmp;
}

MeshObject::MeshObject(sl::Translation position,bool is_static):GLObject(position, is_static){
    current_fc=0;
    vaoID_=0;
}

MeshObject::~MeshObject(){
}

void MeshObject::updateMesh(Mesh &_mesh){
    if(!isStatic_ || vaoID_ == 0){
        if(vaoID_ == 0){
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(4, vboID_);
        }

        glBindVertexArray(vaoID_);

        if (_mesh.vertices.size() >0) {
            glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
            glBufferData(GL_ARRAY_BUFFER, _mesh.vertices.size() * sizeof(sl::float3), &_mesh.vertices[0], isStatic_?GL_STATIC_DRAW:GL_DYNAMIC_DRAW);
            glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
        }

        if (_mesh.normals.size()>0) {
            glBindBuffer(GL_ARRAY_BUFFER, vboID_[2]);
            glBufferData(GL_ARRAY_BUFFER, _mesh.normals.size() * sizeof(sl::float3), &_mesh.normals[0], isStatic_?GL_STATIC_DRAW:GL_DYNAMIC_DRAW);
            glVertexAttribPointer(Shader::ATTRIB_NORM_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(Shader::ATTRIB_NORM_POS);
        }

        if (_mesh.triangles.size() > 0) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[3]);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, _mesh.triangles.size() * sizeof(sl::uint3), &_mesh.triangles[0], isStatic_?GL_STATIC_DRAW:GL_DYNAMIC_DRAW);
        }
        current_fc = _mesh.triangles.size() * 3;

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void MeshObject::draw(GLenum drawing_mode){
    if (current_fc>0 && vaoID_!=0){
        glBindVertexArray(vaoID_);
        glDrawElements(drawing_mode, (GLsizei)current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}
