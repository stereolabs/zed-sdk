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
#include <cuda_gl_interop.h>
#include "PointCloud.hpp"

#include <algorithm> // std::min, used to clamp an unorganized cloud to the allocated buffer

//
// Shaders
//
const GLchar* POINTCLOUD_VERTEX_SHADER
    = "#version 330 core\n"
      "layout(location = 0) in vec4 in_VertexRGBA;\n"
      "uniform mat4 u_mvpMatrix;\n"
      "out vec3 b_color;\n"
      "vec4 decomposeFloat(const in float value)\n"
      "{\n"
      "   uint rgbaInt = floatBitsToUint(value);\n"
      "	uint bIntValue = (rgbaInt / 256U / 256U) % 256U;\n"
      "	uint gIntValue = (rgbaInt / 256U) % 256U;\n"
      "	uint rIntValue = (rgbaInt) % 256U; \n"
      "	return vec4(bIntValue / 255.0f, gIntValue / 255.0f, rIntValue / 255.0f, 1.0); \n"
      "}\n"
      "void main() {\n"
      // Decompose the 4th channel of the XYZRGBA buffer to retrieve the color of the point (1float to 4uint)
      "   b_color = decomposeFloat(in_VertexRGBA.a).xyz;\n"
      "	gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n"
      "}";

const GLchar* POINTCLOUD_FRAGMENT_SHADER = "#version 330 core\n"
                                           "in vec3 b_color;\n"
                                           "layout(location = 0) out vec4 out_Color;\n"
                                           "void main() {\n"
                                           "   out_Color = vec4(b_color, 0.9);\n"
                                           "}";

//
// GLPointCloud
//
PointCloud::PointCloud() { }

PointCloud::~PointCloud() { }

void PointCloud::close() {
    // Gated on the GL buffer, not on the referenced Mat: the buffer is ours to release, and
    // keying cleanup off someone else's data would leak it whenever that Mat is freed first.
    if (bufferGLID_ == 0)
        return;

    if (_hasCudaInterop) {
        auto err = cudaGraphicsUnmapResources(1, &bufferCudaID_, 0);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA UnmapResources (" << err << ")" << std::endl;
    }

    glDeleteBuffers(1, &bufferGLID_);
    bufferGLID_ = 0;
    refMat = nullptr;
    _hasCudaInterop = false;
    _pendingCpuUpload = false;
    _lastValidCount = 0;
}

void PointCloud::initialize(sl::Mat& ref) {
    refMat = &ref;

    numBytes_ = refMat->getResolution().area() * 4 * sizeof(float);

    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, numBytes_, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register the buffer for CUDA-GL interop only when the Mat lives on GPU (organized
    // cloud). For CPU Mats (voxel cloud) pushNewPC() uploads via glBufferSubData instead.
    // The mapping is fixed for the lifetime of the buffer, hence the decision here; pushNewPC
    // still re-checks the Mat's memory type each frame in case it gets reallocated.
    _hasCudaInterop = (static_cast<int>(refMat->getMemoryType()) & static_cast<int>(sl::MEM::GPU)) != 0;

    if (_hasCudaInterop) {
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsNone);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA - OpenGL Interop failed (" << err << ")" << std::endl;

        err = cudaGraphicsMapResources(1, &bufferCudaID_, 0);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA MapResources (" << err << ")" << std::endl;

        err = cudaGraphicsResourceGetMappedPointer((void**)&xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA GetMappedPointer (" << err << ")" << std::endl;
    }

    shader_.set(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");
}

void PointCloud::pushNewPC(CUstream strm) {
    if (!refMat || !refMat->isInit())
        return;

    size_t N = refMat->getResolution().area();
    if (N == 0)
        return;

    // Clamp to the buffer allocated in initialize(): an unorganized cloud is reallocated by
    // the SDK on every retrieve, so N is not guaranteed to stay within the initial bound.
    size_t copyBytes = std::min(numBytes_, N * sizeof(sl::float4));
    size_t count = copyBytes / sizeof(sl::float4);

    // The Mat's memory type can change between frames (retrieveVoxelMeasure reallocates it),
    // so read it back here rather than trusting what initialize() saw.
    int memType = static_cast<int>(refMat->getMemoryType());

    if (_hasCudaInterop && (memType & static_cast<int>(sl::MEM::GPU))) {
        cudaMemcpyAsync(xyzrgbaMappedBuf_, refMat->getPtr<sl::float4>(sl::MEM::GPU), copyBytes, cudaMemcpyDeviceToDevice, strm);
        _lastValidCount = count;
    } else if (memType & static_cast<int>(sl::MEM::CPU)) {
        // Defer the GL upload to draw() where the context is guaranteed current.
        _pendingCpuUpload = true;
        _pendingCount = count;
    }
}

void PointCloud::draw(const sl::Transform& vp) {
    // CPU path: perform the deferred upload now that we are inside the GL callback.
    if (_pendingCpuUpload && refMat && refMat->isInit()) {
        size_t copyBytes = std::min(numBytes_, _pendingCount * sizeof(sl::float4));
        glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, copyBytes, refMat->getPtr<sl::float4>(sl::MEM::CPU));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // Draw only what was actually uploaded: glDrawArrays() on the unclamped count would
        // fetch vertices past the end of the buffer allocated in initialize().
        _lastValidCount = copyBytes / sizeof(sl::float4);
        _pendingCpuUpload = false;
    }

    if (_lastValidCount == 0)
        return;

    glUseProgram(shader_.getProgramId());
    glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_TRUE, vp.m);

    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(_lastValidCount));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
}
