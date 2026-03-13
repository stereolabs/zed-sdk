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

#include <GL/glew.h>
#include "Simple3DObject.hpp"
#include "Shader.hpp"

Simple3DObject::Simple3DObject() {
    vaoID_ = 0;
    drawingType_ = GL_TRIANGLES;
    isStatic_ = need_update = false;
}

Simple3DObject::~Simple3DObject() {
    clear();
    if (vaoID_ != 0) {
        glDeleteBuffers(3, vboID_);
        glDeleteVertexArrays(1, &vaoID_);
    }
}

void Simple3DObject::addPoint(sl::float3 pt, sl::float4 clr) {
    vertices_.push_back(pt);
    colors_.push_back(clr);
    indices_.push_back((int)indices_.size());
    need_update = true;
}

void Simple3DObject::addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float3 clr) {
    addPoint(p1, clr);
    addPoint(p2, clr);
    addPoint(p3, clr);
}

void Simple3DObject::pushToGPU() {
    if (!need_update)
        return;

    if (!isStatic_ || vaoID_ == 0) {
        if (vaoID_ == 0) {
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(3, vboID_);
        }

        glBindVertexArray(vaoID_);
        glBindBuffer(GL_ARRAY_BUFFER, vboID_[Shader::ATTRIB_VERTICES_POS]);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(sl::float3), &vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[Shader::ATTRIB_COLOR_POS]);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(sl::float4), &colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            indices_.size() * sizeof(unsigned int),
            &indices_[0],
            isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW
        );

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        need_update = false;
    }
}

void Simple3DObject::clear() {
    vertices_.clear();
    colors_.clear();
    indices_.clear();

    // Release allocated memory since this object will be rebuilt from scratch
    vertices_.shrink_to_fit();
    colors_.shrink_to_fit();
    indices_.shrink_to_fit();
}

void Simple3DObject::setDrawingType(GLenum type) {
    drawingType_ = type;
}

void Simple3DObject::draw() {
    if (vaoID_) {
        glBindVertexArray(vaoID_);
        glDrawElements(drawingType_, (GLsizei)indices_.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}
