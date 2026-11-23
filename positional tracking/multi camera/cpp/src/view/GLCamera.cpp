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

#define _USE_MATH_DEFINES 1
#include <math.h>
#include "GLCamera.hpp"
#include <cmath>
#include <cstring>

static void normalize(std::array<float, 3>& v) {
    float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-6f) {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

static std::array<float, 3> cross(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

static float dot(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

GLCamera::GLCamera(std::array<float, 3> target, float distance, float yaw, float pitch, float fov, float aspect, float znear, float zfar)
    : _target(target)
    , _distance(distance)
    , _yaw(yaw)
    , _pitch(pitch)
    , _fov(fov)
    , _aspect(aspect)
    , _znear(znear)
    , _zfar(zfar)
    , _defaultDistance(distance)
    , _defaultYaw(yaw)
    , _defaultPitch(pitch)
    , _defaultTarget(target) {

    updateProjectionMatrix();
    updateViewMatrix();
}

void GLCamera::setProjection(float fovDegrees, float aspectRatio, float nearPlane, float farPlane) {
    _fov = fovDegrees;
    _aspect = aspectRatio;
    _znear = nearPlane;
    _zfar = farPlane;

    updateProjectionMatrix();
}

void GLCamera::resizeViewport(float aspectRatio) {
    _aspect = aspectRatio;
    updateProjectionMatrix();
}

void GLCamera::zoom(float delta) {
    _distance -= delta;

    if (_distance < 2) {
        _distance = 2;
    }

    updateViewMatrix();
}

void GLCamera::rotate(float yawDelta, float pitchDelta) {
    _yaw += yawDelta;
    _pitch += pitchDelta;

    if (_pitch > 89.f) {
        _pitch = 89.f;
    }

    if (_pitch < -89.f) {
        _pitch = -89.f;
    }

    updateViewMatrix();
}

void GLCamera::pan(float deltaX, float deltaY) {
    std::array<float, 3> forward;
    std::array<float, 3> right;
    std::array<float, 3> up = {0.f, 1.f, 0.f};
    std::array<float, 3> eye;

    float pitchRad = (_pitch)*M_PI / 180.0f;
    float yawRad = (_yaw)*M_PI / 180.0f;
    eye[0] = _target[0] + _distance * std::cos(pitchRad) * std::sin(yawRad);
    eye[1] = _target[1] + _distance * std::sin(pitchRad);
    eye[2] = _target[2] + _distance * std::cos(pitchRad) * std::cos(yawRad);

    forward[0] = _target[0] - eye[0];
    forward[1] = _target[1] - eye[1];
    forward[2] = _target[2] - eye[2];
    normalize(forward);

    right = cross(forward, up);
    normalize(right);

    std::array<float, 3> panUp = cross(right, forward);
    normalize(panUp);

    for (int i = 0; i < 3; ++i) {
        _target[i] += -right[i] * deltaX + panUp[i] * deltaY;
    }

    updateViewMatrix();
}

void GLCamera::reset() {
    _target = _defaultTarget;
    _distance = _defaultDistance;
    _yaw = _defaultYaw;
    _pitch = _defaultPitch;

    updateViewMatrix();
}

std::array<float, 16> GLCamera::getViewMatrix() const {
    return _viewMatrix;
}

std::array<float, 16> GLCamera::getProjectionMatrix() const {
    return _projectionMatrix;
}

sl::Transform GLCamera::getSLViewMatrix() const {
    sl::Matrix4f mat;

    // Transpose from row-major (OpenGL) to column-major (Stereolabs)
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            mat(col, row) = _viewMatrix[row * 4 + col];
        }
    }

    return sl::Transform(mat);
}

sl::Transform GLCamera::getSLProjectionMatrix() const {
    sl::Matrix4f mat;

    // Transpose from row-major (OpenGL) to column-major (Stereolabs)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            mat(j, i) = _projectionMatrix[i * 4 + j];
        }
    }

    return sl::Transform(mat);
}

void GLCamera::updateViewMatrix() {
    float pitchRad = _pitch * M_PI / 180.0f;
    float yawRad = _yaw * M_PI / 180.0f;

    std::array<float, 3> eye;
    eye[0] = _target[0] + _distance * std::cos(pitchRad) * std::sin(yawRad);
    eye[1] = _target[1] + _distance * std::sin(pitchRad);
    eye[2] = _target[2] + _distance * std::cos(pitchRad) * std::cos(yawRad);

    std::array<float, 3> forward = {_target[0] - eye[0], _target[1] - eye[1], _target[2] - eye[2]};
    normalize(forward);

    std::array<float, 3> up = {0.f, 1.f, 0.f};
    std::array<float, 3> right = cross(forward, up);
    normalize(right);
    up = cross(right, forward);
    normalize(up);

    _viewMatrix[0] = right[0];
    _viewMatrix[1] = up[0];
    _viewMatrix[2] = -forward[0];
    _viewMatrix[3] = 0.0f;

    _viewMatrix[4] = right[1];
    _viewMatrix[5] = up[1];
    _viewMatrix[6] = -forward[1];
    _viewMatrix[7] = 0.0f;

    _viewMatrix[8] = right[2];
    _viewMatrix[9] = up[2];
    _viewMatrix[10] = -forward[2];
    _viewMatrix[11] = 0.0f;

    _viewMatrix[12] = -dot(right, eye);
    _viewMatrix[13] = -dot(up, eye);
    _viewMatrix[14] = dot(forward, eye);
    _viewMatrix[15] = 1.0f;
}

void GLCamera::updateProjectionMatrix() {
    float f = 1.0f / tanf(_fov * M_PI / 180.0f * 0.5f);
    float nf = 1.0f / (_znear - _zfar);

    _projectionMatrix.fill(0.0f);

    _projectionMatrix[0] = f / _aspect;
    _projectionMatrix[5] = f;
    _projectionMatrix[10] = (_zfar + _znear) * nf;
    _projectionMatrix[11] = -1.0f;
    _projectionMatrix[14] = (2.0f * _zfar * _znear) * nf;
    _projectionMatrix[15] = 0.0f;
}
