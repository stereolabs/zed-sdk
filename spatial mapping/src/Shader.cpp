#include "Shader.hpp"
#include <iostream>

Shader::Shader(GLchar* vs, GLchar* fs){
    if(!compile(verterxId_, GL_VERTEX_SHADER, vs)){
        std::cout << "ERROR: while compiling vertex shader" << std::endl;
    }
    if(!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)){
        std::cout << "ERROR: while compiling fragment shader" << std::endl;
    }

    programId_ = glCreateProgram();

    glAttachShader(programId_, verterxId_);
    glAttachShader(programId_, fragmentId_);

    glBindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_vertex");

    glLinkProgram(programId_);

    GLint errorlk(0);
    glGetProgramiv(programId_, GL_LINK_STATUS, &errorlk);
    if(errorlk != GL_TRUE){
        std::cout << "ERROR: while linking Shader :" << std::endl;
        GLint errorSize(0);
        glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &errorSize);

        char *error = new char[errorSize + 1];
        glGetShaderInfoLog(programId_, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteProgram(programId_);
    }
}

Shader::~Shader(){
    if(verterxId_ != 0)
        glDeleteShader(verterxId_);
    if(fragmentId_ != 0)
        glDeleteShader(fragmentId_);
    if(programId_ != 0)
        glDeleteShader(programId_);
}

GLuint Shader::getProgramId(){
    return programId_;
}

bool Shader::compile(GLuint &shaderId, GLenum type, GLchar* src){
    shaderId = glCreateShader(type);
    if(shaderId == 0){
        std::cout << "ERROR: shader type (" << type << ") does not exist" << std::endl;
        return false;
    }
    glShaderSource(shaderId, 1, (const char**)&src, 0);
    glCompileShader(shaderId);

    GLint errorCp(0);
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &errorCp);
    if(errorCp != GL_TRUE){
        std::cout << "ERROR: while compiling Shader :" << std::endl;
        GLint errorSize(0);
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

        char *error = new char[errorSize + 1];
        glGetShaderInfoLog(shaderId, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteShader(shaderId);
        return false;
    }
    return true;
}
