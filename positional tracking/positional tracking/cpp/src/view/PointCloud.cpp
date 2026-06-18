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
    if (refMat.isInit()) {
        auto err = cudaGraphicsUnmapResources(1, &bufferCudaID_, 0);
        if (err != cudaSuccess)
            std::cerr << "Error: CUDA UnmapResources (" << err << ")" << std::endl;
        glDeleteBuffers(1, &bufferGLID_);
        refMat = sl::Mat();
    }
}

void PointCloud::initialize(sl::Mat& ref) {
    refMat = ref;

    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, refMat.getResolution().area() * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess)
        std::cerr << "Error: CUDA - OpenGL Interop failed (" << err << ")" << std::endl;

    err = cudaGraphicsMapResources(1, &bufferCudaID_, 0);
    if (err != cudaSuccess)
        std::cerr << "Error: CUDA MapResources (" << err << ")" << std::endl;

    err = cudaGraphicsResourceGetMappedPointer((void**)&xyzrgbaMappedBuf_, &numBytes_, bufferCudaID_);
    if (err != cudaSuccess)
        std::cerr << "Error: CUDA GetMappedPointer (" << err << ")" << std::endl;

    shader_.set(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");
}

void PointCloud::pushNewPC(CUstream strm) {
    if (refMat.isInit())
        cudaMemcpyAsync(xyzrgbaMappedBuf_, refMat.getPtr<sl::float4>(sl::MEM::GPU), numBytes_, cudaMemcpyDeviceToDevice, strm);
}

void PointCloud::draw(const sl::Transform& vp) {
    if (refMat.isInit()) {
        glUseProgram(shader_.getProgramId());
        glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_TRUE, vp.m);

        glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glDrawArrays(GL_POINTS, 0, refMat.getResolution().area());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
    }
}
