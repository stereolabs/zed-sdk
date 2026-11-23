///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <GL/gl.h>

//
// Shader
//
class Shader {
public:
    Shader()
        : verterxId_(0)
        , fragmentId_(0)
        , programId_(0) { }
    Shader(const GLchar* vs, const GLchar* fs);
    ~Shader();

    // Delete the move constructor and move assignment operator
    Shader(Shader&&) = delete;
    Shader& operator=(Shader&&) = delete;

    // Delete the copy constructor and copy assignment operator
    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    void set(const GLchar* vs, const GLchar* fs);
    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_COLOR_POS = 1;
    static const GLint ATTRIB_NORMAL_POS = 2;

private:
    bool compile(GLuint& shaderId, GLenum type, const GLchar* src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};

//
// ShaderObj
//
struct ShaderObj {
    Shader it;
    GLuint MVPM;
};
