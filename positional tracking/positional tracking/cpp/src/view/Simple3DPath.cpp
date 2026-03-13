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

#include "Simple3DPath.hpp"
#include <iostream>

Simple3DPath::Simple3DPath() { }

Simple3DPath::~Simple3DPath() {
    glDeleteBuffers(1, &_vboID);
    glDeleteVertexArrays(1, &_vaoID);
}

void Simple3DPath::addPoint(const sl::float3& pt) {
    _points.push_back(pt);

    // Prevent unbounded memory growth by trimming old points
    if (_maxPoints > 0 && _points.size() > _maxPoints) {
        size_t excess = _points.size() - _maxPoints;
        _points.erase(_points.begin(), _points.begin() + excess);
    }

    _needsUpdate = true;
}

void Simple3DPath::setMaxPoints(size_t maxPoints) {
    _maxPoints = maxPoints;
}

void Simple3DPath::setColor(const sl::float4& color) {
    _color = color;
}

void Simple3DPath::setMVP(const sl::Transform& mvp) {
    _mvp = mvp;
}

void Simple3DPath::draw() {
    if (_points.empty()) {
        return;
    }

    if (!_isGLInitialized) {
        initGL();
    }

    pushToGPU();

    // Stash GL state
    GLint prevProgram = 0;
    GLint prevVAO = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVAO);

    _shader.use();

    GLint mvpLoc = glGetUniformLocation(_shader.id(), "uMVP");
    glUniformMatrix4fv(mvpLoc, 1, GL_TRUE, _mvp.m);

    GLint colorLoc = glGetUniformLocation(_shader.id(), "uColor");
    glUniform4f(colorLoc, _color.x, _color.y, _color.z, _color.w);

    glBindVertexArray(_vaoID);
    glDrawArrays(_drawingType, 0, static_cast<GLsizei>(_points.size()));

    // Pop GL state
    glBindVertexArray(prevVAO);
    glUseProgram(prevProgram);
}

void Simple3DPath::clear() {
    _points.clear();
    _needsUpdate = true;
}

void Simple3DPath::initGL() {
    glGenVertexArrays(1, &_vaoID);
    glGenBuffers(1, &_vboID);

    if (!_shader.load(GLShader::MVP_VERTEX, GLShader::UNIFORM_COLOR_FRAGMENT)) {
        std::cerr << "Failed to initialize Simple3DPath shader" << std::endl;
    }

    _isGLInitialized = true;
}

void Simple3DPath::pushToGPU() {
    if (!_needsUpdate) {
        return;
    }

    glBindVertexArray(_vaoID);
    glBindBuffer(GL_ARRAY_BUFFER, _vboID);
    glBufferData(GL_ARRAY_BUFFER, _points.size() * sizeof(sl::float3), _points.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    _needsUpdate = false;
}
