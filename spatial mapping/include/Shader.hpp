#pragma once
#include <GL/glew.h>

class Shader{
public:
    Shader(GLchar* vs, GLchar* fs);
    ~Shader();

    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
private:
    bool compile(GLuint &shaderId, GLenum type, GLchar* src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};
