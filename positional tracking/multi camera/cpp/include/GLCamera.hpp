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

#include <sl/Camera.hpp>

class GLCamera {
public:
    GLCamera(
        std::array<float, 3> target = {0.f, 0.f, 0.f},
        float distance = 3.5,
        float yaw = -18,
        float pitch = 27,
        float fov = 60,
        float aspect = 16.0 / 9.0,
        float znear = 1,
        float zfar = 1000
    );

    void setProjection(float fovDegrees, float aspectRatio, float nearPlane, float farPlane);
    void resizeViewport(float aspectRatio);

    void zoom(float delta);
    void rotate(float yawDelta, float pitchDelta);
    void pan(float deltaX, float deltaY);
    void reset();

    std::array<float, 16> getViewMatrix() const;
    std::array<float, 16> getProjectionMatrix() const;

    sl::Transform getSLViewMatrix() const;
    sl::Transform getSLProjectionMatrix() const;

private:
    void updateViewMatrix();
    void updateProjectionMatrix();

    // Position and orientation
    std::array<float, 3> _target; // Point camera orbits around
    float _distance;              // Distance from target
    float _yaw;                   // Horizontal angle (degrees)
    float _pitch;                 // Vertical angle (degrees)

    // Projection parameters
    float _fov;
    float _aspect;
    float _znear;
    float _zfar;

    // Matrices (row-major)
    std::array<float, 16> _viewMatrix;
    std::array<float, 16> _projectionMatrix;

    // Reset state
    std::array<float, 3> _defaultTarget;
    float _defaultDistance;
    float _defaultYaw;
    float _defaultPitch;
};
