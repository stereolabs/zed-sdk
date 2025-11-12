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

#include "sl/Camera.hpp"
#include "GLShader.hpp"
#include <vector>

class Simple3DPath {
public:
    Simple3DPath();
    ~Simple3DPath();

    void addPoint(const sl::float3& pt);
    void setColor(const sl::float4& color);
    void setMVP(const sl::Transform& mvp);

    void draw();
    void clear();

private:
    std::vector<sl::float3> _points;
    sl::float4 _color = {1.f, 1.f, 1.f, 1.f};
    sl::Transform _mvp = sl::Transform();

    GLuint _vaoID = 0;
    GLuint _vboID = 0;
    GLenum _drawingType = GL_LINE_STRIP;

    GLShader _shader;

    bool _needsUpdate = false;
    bool _isGLInitialized = false;

    void initGL();
    void pushToGPU();
};
