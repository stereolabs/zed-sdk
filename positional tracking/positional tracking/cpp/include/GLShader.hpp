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

#include <GL/glew.h>

class GLShader {
public:
    GLShader();
    ~GLShader();

    GLuint id() const;

    bool load(const char* vertexShaderSource, const char* fragmentShaderSource);

    void use() const;

    //
    // Shader Library
    //
    static constexpr const char* MVP_VERTEX = R"(
        #version 330 core
        
        layout(location = 0) in vec3 inPosition;
        uniform mat4 uMVP;
        
        void main() {
            gl_Position = uMVP * vec4(inPosition, 1.0);
        }
    )";

    static constexpr const char* UNIFORM_COLOR_FRAGMENT = R"(
        #version 330 core

        uniform vec4 uColor;
        out vec4 fragColor;
        
        void main() {
            fragColor = uColor;
        }
    )";

    //
    // Deleted copy & move constructors
    //
    GLShader(const GLShader&) = delete;
    GLShader& operator=(const GLShader&) = delete;

    GLShader(GLShader&&) = delete;
    GLShader& operator=(GLShader&&) = delete;

private:
    GLuint _program = 0;
};
