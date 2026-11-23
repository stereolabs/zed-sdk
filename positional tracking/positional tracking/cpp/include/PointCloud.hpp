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
#include "Shader.hpp"

class PointCloud {
public:
    PointCloud();
    ~PointCloud();

    // Initialize Opengl and Cuda buffers.
    // Warning: must be called in the Opengl thread.
    // The Mat is referenced, not copied: it must outlive this PointCloud, and close() must run
    // before it is freed. Referenced rather than copied on purpose: an unorganized (voxel) cloud
    // is reallocated by the SDK on every retrieve, so a copy would go stale after the first frame.
    void initialize(sl::Mat&);
    // Push a new point cloud
    // Warning: can be called from any thread but the mutex "mutexData" must be locked
    void pushNewPC(CUstream);
    // Draw the point cloud
    // Warning: must be called in the Opengl thread
    void draw(const sl::Transform& vp);
    // Close (disable update)
    void close();

private:
    sl::Mat* refMat = nullptr;
    bool _hasCudaInterop = false;

    // Last count used for glDrawArrays, stable across frames, only updates when N > 0.
    size_t _lastValidCount = 0;

    // CPU upload deferred to draw() so the GL context is guaranteed current.
    bool _pendingCpuUpload = false;
    size_t _pendingCount = 0;

    Shader shader_;
    GLuint shMVPMatrixLoc_;
    size_t numBytes_ = 0;
    float* xyzrgbaMappedBuf_ = nullptr;
    GLuint bufferGLID_ = 0;
    cudaGraphicsResource* bufferCudaID_ = nullptr;
};
