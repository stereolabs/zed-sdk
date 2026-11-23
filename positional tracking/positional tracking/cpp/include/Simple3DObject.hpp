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
#include <GL/glew.h>

class Simple3DObject {
public:
    Simple3DObject();

    ~Simple3DObject();

    inline bool isInit() {
        return vaoID_ != 0;
    }

    void addPoint(sl::float3 pt, sl::float4 clr);
    void addFace(sl::float3 p1, sl::float3 p2, sl::float3 p3, sl::float3 clr);
    void pushToGPU();
    void clear();

    void setStatic(bool _static) {
        isStatic_ = _static;
    }

    void setDrawingType(GLenum type);

    /// Recolor every vertex. Idempotent: re-applying the colour already in place does nothing, so
    /// a render loop may call this unconditionally every frame without forcing a GPU re-upload.
    void setUniformColor(sl::float4 color);

    void draw();

private:
    std::vector<sl::float3> vertices_;
    std::vector<sl::float4> colors_;
    std::vector<unsigned int> indices_;

    /// Colour currently applied to every vertex by setUniformColor(), when one is. Lets that call
    /// detect a no-op instead of dirtying the buffers on every frame.
    sl::float4 uniformColor_;
    bool hasUniformColor_ = false;

    bool isStatic_ = false;
    bool need_update;
    GLenum drawingType_;
    GLuint vaoID_;
    GLuint vboID_[3];
};
