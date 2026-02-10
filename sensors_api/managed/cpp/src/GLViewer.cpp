#include "GLViewer.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstring>
#include <sstream>

// Enable freeglut extension for glutMainLoopEvent
#define GL_GLEXT_PROTOTYPES
#include <GL/freeglut.h>

GLViewer* GLViewer::currentInstance_ = nullptr;

// =============================================
// Shader Vertex and Fragment Sources
// =============================================
static const char* POINTCLOUD_VERTEX_SHADER = R"GLSL(
#version 330 core
layout(location = 0) in vec4 in_vertex;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_vertex.xyz, 1.0);
    gl_PointSize = 2.0;
    // Color is packed as ABGR: (A << 24) | (B << 16) | (G << 8) | R (matching ZED SDK RGBA mode)
    uint color_uint = floatBitsToUint(in_vertex.w);
    float r = float((color_uint >> 0u) & 0xFFu) / 255.0;
    float g = float((color_uint >> 8u) & 0xFFu) / 255.0;
    float b = float((color_uint >> 16u) & 0xFFu) / 255.0;
    v_color = vec4(r, g, b, 1.0);
}
)GLSL";

static const char* POINTCLOUD_FRAGMENT_SHADER = R"GLSL(
#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
)GLSL";

static const char* IMAGE_VERTEX_SHADER = R"GLSL(
#version 330 core
layout(location = 0) in vec2 in_pos;
layout(location = 2) in vec2 in_tex;
out vec2 v_texcoord;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_texcoord = in_tex;
}
)GLSL";

static const char* IMAGE_FRAGMENT_SHADER = R"GLSL(
#version 330 core
in vec2 v_texcoord;
out vec4 frag_color;
uniform sampler2D u_texture;
void main() {
    frag_color = texture(u_texture, v_texcoord);
}
)GLSL";

// Simple line/color shader for 3D boxes and skeletons
static const char* SIMPLE3D_VERTEX_SHADER = R"GLSL(
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
)GLSL";

static const char* SIMPLE3D_FRAGMENT_SHADER = R"GLSL(
#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
)GLSL";

// Use the SDK's bone definitions directly: sl::BODY_18_BONES, sl::BODY_34_BONES, sl::BODY_38_BONES
// with sl::getIdx() to convert enum to int index

// Helper to draw skeleton using SDK bone definitions - simple thick lines
template <typename T>
void drawSkeletonBones(
    const sl::BodyData& body,
    const std::vector<std::pair<T, T>>& bones,
    Simple3DObject& skeletonObj,
    const sl::float4& color
) {
    for (const auto& bone : bones) {
        int idx1 = sl::getIdx(bone.first);
        int idx2 = sl::getIdx(bone.second);

        if (idx1 < static_cast<int>(body.keypoint.size()) && idx2 < static_cast<int>(body.keypoint.size())) {
            const sl::float3& kp1 = body.keypoint[idx1];
            const sl::float3& kp2 = body.keypoint[idx2];

            if (std::isfinite(kp1.x) && std::isfinite(kp1.y) && std::isfinite(kp1.z) && std::isfinite(kp2.x) && std::isfinite(kp2.y)
                && std::isfinite(kp2.z)) {
                skeletonObj.addLine(kp1, kp2, color);
            }
        }
    }
}

// Generate color from object ID
static sl::float4 generateColorFromID(int id) {
    // Use golden ratio for nice color distribution
    float hue = fmod(id * 0.618033988749895f, 1.0f);
    float s = 0.8f;
    float v = 0.9f;

    int h_i = static_cast<int>(hue * 6);
    float f = hue * 6 - h_i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);

    float r, g, b;
    switch (h_i % 6) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        default:
            r = v;
            g = p;
            b = q;
            break;
    }
    return sl::float4(r, g, b, 0.8f);
}

// =============================================
// Shader Class Implementation
// =============================================
Shader::Shader(const GLchar* vs, const GLchar* fs)
    : verterxId_(0)
    , fragmentId_(0)
    , programId_(0) {
    if (!compile(verterxId_, GL_VERTEX_SHADER, vs)) {
        std::cerr << "ERROR: Vertex Shader compilation failed!" << std::endl;
        return;
    }
    if (!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)) {
        std::cerr << "ERROR: Fragment Shader compilation failed!" << std::endl;
        return;
    }
    programId_ = glCreateProgram();
    glAttachShader(programId_, verterxId_);
    glAttachShader(programId_, fragmentId_);
    glBindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_vertex");
    glBindAttribLocation(programId_, ATTRIB_COLOR_POS, "in_color");
    glBindAttribLocation(programId_, ATTRIB_TEXCOORD, "in_texcoord");
    glLinkProgram(programId_);
    GLint linked;
    glGetProgramiv(programId_, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[1024];
        glGetProgramInfoLog(programId_, sizeof(log), nullptr, log);
        std::cerr << "Shader link error: " << log << std::endl;
    }
}

Shader::~Shader() {
    if (programId_) {
        glDeleteProgram(programId_);
    }
    if (verterxId_) {
        glDeleteShader(verterxId_);
    }
    if (fragmentId_) {
        glDeleteShader(fragmentId_);
    }
}

Shader::Shader(Shader&& o) noexcept
    : verterxId_(o.verterxId_)
    , fragmentId_(o.fragmentId_)
    , programId_(o.programId_) {
    o.verterxId_ = 0;
    o.fragmentId_ = 0;
    o.programId_ = 0;
}

Shader& Shader::operator=(Shader&& o) noexcept {
    if (this != &o) {
        if (programId_)
            glDeleteProgram(programId_);
        if (verterxId_)
            glDeleteShader(verterxId_);
        if (fragmentId_)
            glDeleteShader(fragmentId_);
        verterxId_ = o.verterxId_;
        fragmentId_ = o.fragmentId_;
        programId_ = o.programId_;
        o.verterxId_ = 0;
        o.fragmentId_ = 0;
        o.programId_ = 0;
    }
    return *this;
}

bool Shader::compile(GLuint& shaderId, GLenum type, const GLchar* src) {
    shaderId = glCreateShader(type);
    glShaderSource(shaderId, 1, &src, nullptr);
    glCompileShader(shaderId);
    GLint compiled;
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[1024];
        glGetShaderInfoLog(shaderId, sizeof(log), nullptr, log);
        std::cerr << "Shader compile error: " << log << std::endl;
        return false;
    }
    return true;
}

// =============================================
// CameraGL Class Implementation
// =============================================
CameraGL::CameraGL()
    : horizontalFOV_(70.f)
    , verticalFOV_(70.f)
    , usePerspective_(true) {
    setPosition(sl::float3(0.f, 0.f, 1000.f));
    setDirection(sl::float3(0.f, 0.f, -1.f), sl::float3(0.f, 1.f, 0.f));
    setProjection(70.f, 16.f / 9.f, 0.01f, 10000.f); // near=1cm, far=10km
}

void CameraGL::update() {
    updateView();
    updateVPMatrix();
}

void CameraGL::setProjection(float hfov, float ar, float zn, float zf) {
    horizontalFOV_ = hfov;
    float fovx_rad = hfov * M_PI / 180.f;
    float fovy_rad = 2.f * atan(tan(fovx_rad / 2.f) / ar);
    verticalFOV_ = fovy_rad * 180.f / M_PI;

    float f = 1.f / tan(fovy_rad / 2.f);
    projection_.setIdentity();
    projection_(0, 0) = f / ar;
    projection_(1, 1) = f;
    projection_(2, 2) = (zf + zn) / (zn - zf);
    projection_(3, 2) = -1.f;
    projection_(2, 3) = (2.f * zf * zn) / (zn - zf);
    projection_(3, 3) = 0.f;
}

void CameraGL::setPosition(sl::float3 p) {
    position_ = p;
}

void CameraGL::setOffsetFromPosition(sl::float3 o) {
    offset_ = o;
}

void CameraGL::setDirection(sl::float3 direction, sl::float3 vertical) {
    vertical_ = vertical;
    sl::float3 dir = direction;
    dir.x = -dir.x;
    dir.y = -dir.y;
    dir.z = -dir.z;
    float len = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (len > 1e-6) {
        forward_ = sl::float3(dir.x / len, dir.y / len, dir.z / len);
    }
    updateVectors();
}

void CameraGL::translate(sl::float3 t) {
    position_.x += t.x * right_.x + t.y * up_.x + t.z * forward_.x;
    position_.y += t.x * right_.y + t.y * up_.y + t.z * forward_.y;
    position_.z += t.x * right_.z + t.y * up_.z + t.z * forward_.z;
}

void CameraGL::setRotation(sl::Rotation r) {
    rotation_ = r;
    updateVectors();
}

void CameraGL::rotate(sl::Rotation r) {
    // Combine rotations (simplified)
    float rx = r.getRotationVector().x;
    float ry = r.getRotationVector().y;

    // Simple Euler angle rotation for mouse control
    sl::float3 dir = forward_;
    dir.x = -dir.x;
    dir.y = -dir.y;
    dir.z = -dir.z;

    // Rotate around up axis (yaw)
    float cosY = cos(ry), sinY = sin(ry);
    float newX = dir.x * cosY - dir.z * sinY;
    float newZ = dir.x * sinY + dir.z * cosY;
    dir.x = newX;
    dir.z = newZ;

    // Rotate around right axis (pitch) - limit pitch to avoid gimbal lock
    float cosP = cos(rx), sinP = sin(rx);
    float newY = dir.y * cosP - dir.z * sinP;
    newZ = dir.y * sinP + dir.z * cosP;
    dir.y = newY;
    dir.z = newZ;

    dir.x = -dir.x;
    dir.y = -dir.y;
    dir.z = -dir.z;
    float len = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (len > 1e-6) {
        forward_ = sl::float3(dir.x / len, dir.y / len, dir.z / len);
    }
    updateVectors();
}

void CameraGL::updateVectors() {
    // Compute right vector: right = vertical × forward
    // (forward_ is the negated direction, so this gives the correct right vector)
    right_.x = vertical_.y * forward_.z - vertical_.z * forward_.y;
    right_.y = vertical_.z * forward_.x - vertical_.x * forward_.z;
    right_.z = vertical_.x * forward_.y - vertical_.y * forward_.x;
    float len = sqrt(right_.x * right_.x + right_.y * right_.y + right_.z * right_.z);
    if (len > 1e-6) {
        right_.x /= len;
        right_.y /= len;
        right_.z /= len;
    }

    // Compute up vector: up = forward × right
    up_.x = forward_.y * right_.z - forward_.z * right_.y;
    up_.y = forward_.z * right_.x - forward_.x * right_.z;
    up_.z = forward_.x * right_.y - forward_.y * right_.x;
    len = sqrt(up_.x * up_.x + up_.y * up_.y + up_.z * up_.z);
    if (len > 1e-6) {
        up_.x /= len;
        up_.y /= len;
        up_.z /= len;
    }
}

void CameraGL::updateView() {
    sl::float3 lookAt;
    lookAt.x = position_.x - forward_.x;
    lookAt.y = position_.y - forward_.y;
    lookAt.z = position_.z - forward_.z;

    sl::float3 zaxis = forward_;
    sl::float3 xaxis = right_;
    sl::float3 yaxis = up_;

    view_.setIdentity();
    view_(0, 0) = xaxis.x;
    view_(0, 1) = xaxis.y;
    view_(0, 2) = xaxis.z;
    view_(1, 0) = yaxis.x;
    view_(1, 1) = yaxis.y;
    view_(1, 2) = yaxis.z;
    view_(2, 0) = zaxis.x;
    view_(2, 1) = zaxis.y;
    view_(2, 2) = zaxis.z;
    view_(0, 3) = -(xaxis.x * position_.x + xaxis.y * position_.y + xaxis.z * position_.z);
    view_(1, 3) = -(yaxis.x * position_.x + yaxis.y * position_.y + yaxis.z * position_.z);
    view_(2, 3) = -(zaxis.x * position_.x + zaxis.y * position_.y + zaxis.z * position_.z);
}

void CameraGL::updateVPMatrix() {
    // VP = Projection * View
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vpMatrix_(i, j) = 0;
            for (int k = 0; k < 4; k++) {
                vpMatrix_(i, j) += projection_(i, k) * view_(k, j);
            }
        }
    }
}

// =============================================
// Simple3DObject Class Implementation
// =============================================
Simple3DObject::Simple3DObject() { }

Simple3DObject::~Simple3DObject() {
    if (vboVertices_)
        glDeleteBuffers(1, &vboVertices_);
    if (vboColors_)
        glDeleteBuffers(1, &vboColors_);
    if (ebo_)
        glDeleteBuffers(1, &ebo_);
    if (vao_)
        glDeleteVertexArrays(1, &vao_);
}

void Simple3DObject::init() {
    if (inited_)
        return;

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vboVertices_);
    glGenBuffers(1, &vboColors_);
    glGenBuffers(1, &ebo_);

    shader_ = Shader(SIMPLE3D_VERTEX_SHADER, SIMPLE3D_FRAGMENT_SHADER);
    inited_ = true;
}

void Simple3DObject::clear() {
    vertices_.clear();
    colors_.clear();
    indices_.clear();
    numIndices_ = 0;
}

void Simple3DObject::addLine(const sl::float3& p1, const sl::float3& p2, const sl::float4& color) {
    unsigned int startIdx = vertices_.size() / 3;

    // Add vertices
    vertices_.push_back(p1.x);
    vertices_.push_back(p1.y);
    vertices_.push_back(p1.z);
    vertices_.push_back(p2.x);
    vertices_.push_back(p2.y);
    vertices_.push_back(p2.z);

    // Add colors
    colors_.push_back(color.x);
    colors_.push_back(color.y);
    colors_.push_back(color.z);
    colors_.push_back(color.w);
    colors_.push_back(color.x);
    colors_.push_back(color.y);
    colors_.push_back(color.z);
    colors_.push_back(color.w);

    // Add indices
    indices_.push_back(startIdx);
    indices_.push_back(startIdx + 1);
}

void Simple3DObject::addBox(const std::vector<sl::float3>& corners, const sl::float4& color) {
    if (corners.size() < 8)
        return;

    // Box has 8 corners: 0-3 top, 4-7 bottom (or vice versa)
    // Top edges: 0-1, 1-2, 2-3, 3-0
    addLine(corners[0], corners[1], color);
    addLine(corners[1], corners[2], color);
    addLine(corners[2], corners[3], color);
    addLine(corners[3], corners[0], color);

    // Bottom edges: 4-5, 5-6, 6-7, 7-4
    addLine(corners[4], corners[5], color);
    addLine(corners[5], corners[6], color);
    addLine(corners[6], corners[7], color);
    addLine(corners[7], corners[4], color);

    // Vertical edges: 0-4, 1-5, 2-6, 3-7
    addLine(corners[0], corners[4], color);
    addLine(corners[1], corners[5], color);
    addLine(corners[2], corners[6], color);
    addLine(corners[3], corners[7], color);
}

void Simple3DObject::addSphere(const sl::float3& center, float radius, const sl::float4& color, int segments) {
    // Simple sphere approximation using 3 circles
    for (int i = 0; i < segments; i++) {
        float angle1 = 2.0f * M_PI * i / segments;
        float angle2 = 2.0f * M_PI * (i + 1) / segments;

        // XY plane circle
        sl::float3 p1(center.x + radius * cos(angle1), center.y + radius * sin(angle1), center.z);
        sl::float3 p2(center.x + radius * cos(angle2), center.y + radius * sin(angle2), center.z);
        addLine(p1, p2, color);

        // XZ plane circle
        p1 = sl::float3(center.x + radius * cos(angle1), center.y, center.z + radius * sin(angle1));
        p2 = sl::float3(center.x + radius * cos(angle2), center.y, center.z + radius * sin(angle2));
        addLine(p1, p2, color);

        // YZ plane circle
        p1 = sl::float3(center.x, center.y + radius * cos(angle1), center.z + radius * sin(angle1));
        p2 = sl::float3(center.x, center.y + radius * cos(angle2), center.z + radius * sin(angle2));
        addLine(p1, p2, color);
    }
}

void Simple3DObject::addCylinder(const sl::float3& p1, const sl::float3& p2, float radius, const sl::float4& color, int segments) {
    // Compute direction and perpendicular vectors
    sl::float3 dir(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    float len = sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (len < 1e-6f)
        return;
    dir.x /= len;
    dir.y /= len;
    dir.z /= len;

    // Find perpendicular vectors
    sl::float3 up(0, 1, 0);
    if (fabs(dir.y) > 0.9f)
        up = sl::float3(1, 0, 0);

    sl::float3 right(dir.y * up.z - dir.z * up.y, dir.z * up.x - dir.x * up.z, dir.x * up.y - dir.y * up.x);
    float rightLen = sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
    if (rightLen > 1e-6f) {
        right.x /= rightLen;
        right.y /= rightLen;
        right.z /= rightLen;
    }

    sl::float3 forward(right.y * dir.z - right.z * dir.y, right.z * dir.x - right.x * dir.z, right.x * dir.y - right.y * dir.x);

    // Draw cylinder lines
    for (int i = 0; i < segments; i++) {
        float angle = 2.0f * M_PI * i / segments;
        float c = cos(angle), s = sin(angle);

        sl::float3 offset(
            radius * (c * right.x + s * forward.x),
            radius * (c * right.y + s * forward.y),
            radius * (c * right.z + s * forward.z)
        );

        sl::float3 cp1(p1.x + offset.x, p1.y + offset.y, p1.z + offset.z);
        sl::float3 cp2(p2.x + offset.x, p2.y + offset.y, p2.z + offset.z);
        addLine(cp1, cp2, color);
    }
}

void Simple3DObject::pushToGPU() {
    if (!inited_ || vertices_.empty())
        return;

    glBindVertexArray(vao_);

    glBindBuffer(GL_ARRAY_BUFFER, vboVertices_);
    glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), vertices_.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, vboColors_);
    glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(float), colors_.data(), GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int), indices_.data(), GL_DYNAMIC_DRAW);

    numIndices_ = indices_.size();

    glBindVertexArray(0);
}

void Simple3DObject::draw(const sl::Transform& vp) {
    if (!inited_ || numIndices_ == 0)
        return;

    glUseProgram(shader_.getProgramId());

    GLint mvpLoc = glGetUniformLocation(shader_.getProgramId(), "u_mvp");
    glUniformMatrix4fv(mvpLoc, 1, GL_TRUE, vp.m);

    glBindVertexArray(vao_);
    glLineWidth(4.0f); // Thicker lines for better visibility
    glDrawElements(GL_LINES, numIndices_, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    glUseProgram(0);
}

// =============================================
// PointCloud Class Implementation
// =============================================
PointCloud::PointCloud(sl::Resolution res)
    : numPoints_(res.width * res.height) {
    glGenBuffers(1, &bufferGLID_);
    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glBufferData(GL_ARRAY_BUFFER, numPoints_ * sizeof(sl::float4), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&bufferCudaID_, bufferGLID_, cudaGraphicsMapFlagsWriteDiscard);
}

PointCloud::~PointCloud() {
    if (bufferCudaID_) {
        cudaGraphicsUnregisterResource(bufferCudaID_);
    }
    if (bufferGLID_) {
        glDeleteBuffers(1, &bufferGLID_);
    }
}

void PointCloud::setOffset(float x, float y, float z) {
    offset = sl::float3(x, y, z);
}

void PointCloud::pushNewPC(sl::Mat& pc) {
    if (pc.getWidth() * pc.getHeight() != numPoints_) {
        std::cerr << "PointCloud::pushNewPC: size mismatch!" << std::endl;
        return;
    }
    void* d_ptr = nullptr;
    size_t size = 0;
    cudaGraphicsMapResources(1, &bufferCudaID_);
    cudaGraphicsResourceGetMappedPointer(&d_ptr, &size, bufferCudaID_);

    // Handle both CPU and GPU source memory
    if (pc.getMemoryType() == sl::MEM::GPU) {
        cudaMemcpy(d_ptr, pc.getPtr<sl::float4>(sl::MEM::GPU), numPoints_ * sizeof(sl::float4), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpy(d_ptr, pc.getPtr<sl::float4>(sl::MEM::CPU), numPoints_ * sizeof(sl::float4), cudaMemcpyHostToDevice);
    }

    cudaGraphicsUnmapResources(1, &bufferCudaID_);
    hasData_ = true;
}

void PointCloud::draw(const sl::Transform& vp, const Shader& shader) {
    if (!hasData_)
        return;

    glUseProgram(shader.getProgramId());

    // Create offset transform
    sl::Transform offsetMat;
    offsetMat.setIdentity();
    offsetMat(0, 3) = offset.x;
    offsetMat(1, 3) = offset.y;
    offsetMat(2, 3) = offset.z;

    // MVP = VP * offset
    sl::Transform mvp;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mvp(i, j) = 0;
            for (int k = 0; k < 4; k++) {
                mvp(i, j) += vp(i, k) * offsetMat(k, j);
            }
        }
    }

    GLint mvpLoc = glGetUniformLocation(shader.getProgramId(), "u_mvp");
    glUniformMatrix4fv(mvpLoc, 1, GL_TRUE, mvp.m);

    glBindBuffer(GL_ARRAY_BUFFER, bufferGLID_);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glDrawArrays(GL_POINTS, 0, numPoints_);
    glDisable(GL_PROGRAM_POINT_SIZE);

    glDisableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
}

// =============================================
// GLImage Class Implementation
// =============================================
GLImage::GLImage() { }

GLImage::~GLImage() {
    if (texId_)
        glDeleteTextures(1, &texId_);
    if (vbo_)
        glDeleteBuffers(1, &vbo_);
}

void GLImage::init(int width, int height, bool bgra) {
    imgWidth_ = width;
    imgHeight_ = height;
    isBGRA_ = bgra;

    glGenTextures(1, &texId_);
    glBindTexture(GL_TEXTURE_2D, texId_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, isBGRA_ ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenBuffers(1, &vbo_);

    inited_ = true;
}

void GLImage::pushImage(sl::Mat& img) {
    if (!inited_)
        return;
    glBindTexture(GL_TEXTURE_2D, texId_);

    // Handle both CPU and GPU source memory
    if (img.getMemoryType() == sl::MEM::GPU) {
        sl::Mat cpuMat;
        img.copyTo(cpuMat, sl::COPY_TYPE::GPU_CPU);
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            imgWidth_,
            imgHeight_,
            isBGRA_ ? GL_BGRA : GL_RGBA,
            GL_UNSIGNED_BYTE,
            cpuMat.getPtr<sl::uchar4>(sl::MEM::CPU)
        );
    } else {
        // Already on CPU
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            imgWidth_,
            imgHeight_,
            isBGRA_ ? GL_BGRA : GL_RGBA,
            GL_UNSIGNED_BYTE,
            img.getPtr<sl::uchar4>(sl::MEM::CPU)
        );
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLImage::draw(float x0, float y0, float width, float height, const Shader& shader) {
    if (!inited_)
        return;

    float verts[] = {x0, y0, 0.f, 1.f, x0 + width, y0, 1.f, 1.f, x0 + width, y0 + height, 1.f, 0.f, x0, y0 + height, 0.f, 0.f};

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);

    glUseProgram(shader.getProgramId());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texId_);
    GLint texLoc = glGetUniformLocation(shader.getProgramId(), "u_texture");
    glUniform1i(texLoc, 0);

    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(Shader::ATTRIB_TEXCOORD);
    glVertexAttribPointer(Shader::ATTRIB_TEXCOORD, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glDisableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);
    glDisableVertexAttribArray(Shader::ATTRIB_TEXCOORD);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

// =============================================
// GLViewer Class Implementation
// =============================================
GLViewer::GLViewer() { }

GLViewer::~GLViewer() {
    if (currentInstance_ == this) {
        currentInstance_ = nullptr;
    }
}

void GLViewer::init(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(windowWidth_, windowHeight_);
    glutCreateWindow("Sensors API Sample - Press 'R' to toggle recording, 'Q' to quit");

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW!" << std::endl;
        available_ = false;
        return;
    }

    currentInstance_ = this;
    glutDisplayFunc(drawCallback);
    glutReshapeFunc(reshapeCallback);
    glutKeyboardFunc(keyboardCallback);
    glutMouseFunc(mouseButtonCallback);
    glutMotionFunc(mouseMotionCallback);
    glutIdleFunc(idleCallback);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pcShader_ = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    imgShader_ = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);

    // Initialize 3D rendering objects for detections
    objectBoxes_.init();
    bodySkeletons_.init();

    camera_.setPosition(sl::float3(0.f, 2.f, 10.f)); // 10 meters back, 2 meters up
    camera_.setDirection(sl::float3(0.f, 0.f, -1.f), sl::float3(0.f, 1.f, 0.f));
    camera_.update();

    initialized_ = true;
    std::cout << "GLViewer initialized. Use WASD to move, mouse to look around." << std::endl;
}

bool GLViewer::isAvailable() {
    return available_;
}

bool GLViewer::isDone() {
    if (available_) {
        glutPostRedisplay();
        glutMainLoopEvent();
    }
    return !available_;
}

void GLViewer::updateLidar(const std::string& id, sl::Mat& point_cloud, sl::Mat& intensity_img) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto& pending = pendingLidars_[id];
    if (point_cloud.isInit()) {
        // Determine copy type based on source memory location
        sl::COPY_TYPE copyType = (point_cloud.getMemoryType() == sl::MEM::GPU) ? sl::COPY_TYPE::GPU_GPU : sl::COPY_TYPE::CPU_CPU;
        pending.pointCloud.setFrom(point_cloud, copyType);
        pending.pointCloudRes = point_cloud.getResolution();
        pending.hasNewCloud = true;
    }
    if (intensity_img.isInit()) {
        sl::COPY_TYPE copyType = (intensity_img.getMemoryType() == sl::MEM::GPU) ? sl::COPY_TYPE::GPU_GPU : sl::COPY_TYPE::CPU_CPU;
        pending.image.setFrom(intensity_img, copyType);
        pending.imageRes = intensity_img.getResolution();
        pending.hasNewImage = true;
    }
}

void GLViewer::updateZed(const std::string& id, sl::Mat& image, sl::Mat* point_cloud) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto& pending = pendingZeds_[id];
    if (image.isInit()) {
        // Determine copy type based on source memory location
        sl::COPY_TYPE copyType = (image.getMemoryType() == sl::MEM::GPU) ? sl::COPY_TYPE::GPU_GPU : sl::COPY_TYPE::CPU_CPU;
        pending.image.setFrom(image, copyType);
        pending.imageRes = image.getResolution();
        pending.hasNewImage = true;
    }
    if (point_cloud && point_cloud->isInit()) {
        sl::COPY_TYPE copyType = (point_cloud->getMemoryType() == sl::MEM::GPU) ? sl::COPY_TYPE::GPU_GPU : sl::COPY_TYPE::CPU_CPU;
        pending.pointCloud.setFrom(*point_cloud, copyType);
        pending.pointCloudRes = point_cloud->getResolution();
        pending.hasNewCloud = true;
    }
}

void GLViewer::updateZedOne(const std::string& id, sl::Mat& image) {
    updateZed(id, image, nullptr);
}

void GLViewer::updateZedObjects(const std::string& id, const sl::Objects& objects, const sl::Matrix4f* transform, sl::Timestamp timestamp) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto& pending = pendingObjects_[id];
    pending.objects = objects;
    pending.hasNew = true;
}

void GLViewer::updateZedBodies(const std::string& id, const sl::Bodies& bodies, const sl::Matrix4f* transform, sl::Timestamp timestamp) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto& pending = pendingBodies_[id];
    pending.bodies = bodies;
    pending.hasNew = true;
}

void GLViewer::setLidarOffset(const std::string& id, float x, float y, float z) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto& pending = pendingLidars_[id];
    pending.cloudOffset = sl::float3(x, y, z);
    pending.hasCloudOffset = true;
}

void GLViewer::setZedOffset(const std::string& id, float x, float y, float z) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto& pending = pendingZeds_[id];
    pending.cloudOffset = sl::float3(x, y, z);
    pending.hasCloudOffset = true;
}

void GLViewer::render() {
    applyCameraMotion();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.12f, 0.12f, 0.15f, 1.f);

    camera_.update();
    const sl::Transform& vp = camera_.getViewProjectionMatrix();

    // Apply pending data and draw point clouds
    {
        std::lock_guard<std::mutex> lock(mtx_);

        // Process LIDAR data
        for (auto& [id, pending] : pendingLidars_) {
            auto& cached = lidars_[id];

            if (pending.hasNewCloud) {
                if (!cached.cloud
                    || pending.pointCloudRes.width * pending.pointCloudRes.height
                        != pending.pointCloud.getWidth() * pending.pointCloud.getHeight()) {
                    cached.cloud = std::make_unique<PointCloud>(pending.pointCloudRes);
                }
                cached.cloud->pushNewPC(pending.pointCloud);
                pending.hasNewCloud = false;
            }
            if (pending.hasCloudOffset && cached.cloud) {
                cached.cloud->setOffset(pending.cloudOffset.x, pending.cloudOffset.y, pending.cloudOffset.z);
                pending.hasCloudOffset = false;
            }
        }

        // Process ZED data
        for (auto& [id, pending] : pendingZeds_) {
            auto& cached = zeds_[id];

            if (pending.hasNewCloud) {
                if (!cached.cloud
                    || pending.pointCloudRes.width * pending.pointCloudRes.height
                        != pending.pointCloud.getWidth() * pending.pointCloud.getHeight()) {
                    cached.cloud = std::make_unique<PointCloud>(pending.pointCloudRes);
                }
                cached.cloud->pushNewPC(pending.pointCloud);
                pending.hasNewCloud = false;
            }
            if (pending.hasCloudOffset && cached.cloud) {
                cached.cloud->setOffset(pending.cloudOffset.x, pending.cloudOffset.y, pending.cloudOffset.z);
                pending.hasCloudOffset = false;
            }
            if (pending.hasNewImage) {
                if (!cached.image) {
                    cached.image = std::make_unique<GLImage>();
                    cached.image->init(pending.imageRes.width, pending.imageRes.height, true);
                }
                cached.image->pushImage(pending.image);
                pending.hasNewImage = false;
            }
        }

        // Process detected objects - build 3D bounding boxes
        objectBoxes_.clear();
        for (auto& [id, pending] : pendingObjects_) {
            if (pending.hasNew) {
                for (const auto& obj : pending.objects.object_list) {
                    if (obj.tracking_state == sl::OBJECT_TRACKING_STATE::OK && obj.bounding_box.size() >= 8) {
                        sl::float4 color = generateColorFromID(obj.id);
                        std::vector<sl::float3> corners(obj.bounding_box.begin(), obj.bounding_box.end());
                        objectBoxes_.addBox(corners, color);
                    }
                }
                pending.hasNew = false;
            }
        }
        objectBoxes_.pushToGPU();

        // Process body tracking data - build skeleton lines using SDK bone definitions
        bodySkeletons_.clear();
        for (auto& [id, pending] : pendingBodies_) {
            if (pending.hasNew) {
                sl::BODY_FORMAT format = pending.bodies.body_format;
                for (const auto& body : pending.bodies.body_list) {
                    if (body.tracking_state == sl::OBJECT_TRACKING_STATE::OK && !body.keypoint.empty()) {
                        sl::float4 color = generateColorFromID(body.id);

                        // Use SDK bone definitions directly based on body format
                        if (format == sl::BODY_FORMAT::BODY_38) {
                            drawSkeletonBones(body, sl::BODY_38_BONES, bodySkeletons_, color);
                        } else if (format == sl::BODY_FORMAT::BODY_34) {
                            drawSkeletonBones(body, sl::BODY_34_BONES, bodySkeletons_, color);
                        } else {
                            drawSkeletonBones(body, sl::BODY_18_BONES, bodySkeletons_, color);
                        }
                    }
                }
                pending.hasNew = false;
            }
        }
        bodySkeletons_.pushToGPU();
    }

    // Draw point clouds (3D view)
    glViewport(0, windowHeight_ / 4, windowWidth_, windowHeight_ * 3 / 4);
    for (auto& [id, cached] : lidars_) {
        if (cached.cloud)
            cached.cloud->draw(vp, pcShader_);
    }
    for (auto& [id, cached] : zeds_) {
        if (cached.cloud)
            cached.cloud->draw(vp, pcShader_);
    }

    // Draw detected object bounding boxes
    objectBoxes_.draw(vp);

    // Draw body skeletons
    bodySkeletons_.draw(vp);

    // Draw images (2D thumbnails at bottom)
    glViewport(0, 0, windowWidth_, windowHeight_ / 4);
    glDisable(GL_DEPTH_TEST);

    int imgCount = 0;
    for (auto& [id, cached] : zeds_) {
        if (cached.image) {
            float w = 0.3f;
            float h = 1.0f;
            float x = -1.f + imgCount * w * 1.1f;
            cached.image->draw(x, -1.f, w, h, imgShader_);
            imgCount++;
        }
    }

    glEnable(GL_DEPTH_TEST);

    // Draw recording status text
    if (getRecordingStatusFn_) {
        std::string status = getRecordingStatusFn_();
        if (!status.empty()) {
            glViewport(0, 0, windowWidth_, windowHeight_);
            glDisable(GL_DEPTH_TEST);
            drawText(status, 10, windowHeight_ - 20);
            glEnable(GL_DEPTH_TEST);
        }
    }

    glutSwapBuffers();
}

void GLViewer::drawText(const std::string& text, int x, int y) {
    glColor3f(1.f, 1.f, 1.f);
    glRasterPos2f(2.f * x / windowWidth_ - 1.f, 2.f * y / windowHeight_ - 1.f);
    for (char c : text) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
    }
}

void GLViewer::reshape(int w, int h) {
    windowWidth_ = w;
    windowHeight_ = h;
    float ar = (float)w / (float)h;
    camera_.setProjection(70.f, ar, 0.01f, 10000.f); // near=1cm, far=10km
}

void GLViewer::keyboard(unsigned char key, int x, int y) {
    keyState_[key] = true;
    if (key == 'q' || key == 'Q' || key == 27) {
        available_ = false;
    }
    if (key == 'r' || key == 'R') {
        if (recordingToggleFn_) {
            recordingToggleFn_();
        }
    }
}

void GLViewer::mouseButton(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        mouseRotating_ = (state == GLUT_DOWN);
        if (mouseRotating_) {
            lastMouseX_ = x;
            lastMouseY_ = y;
        }
    }
    if (button == GLUT_RIGHT_BUTTON) {
        mousePanning_ = (state == GLUT_DOWN);
        if (mousePanning_) {
            lastMouseX_ = x;
            lastMouseY_ = y;
        }
    }
    // Scroll for zoom
    if (button == 3) {                                 // scroll up
        camera_.translate(sl::float3(0.f, 0.f, -2.f)); // 2 meters forward
    }
    if (button == 4) {                                // scroll down
        camera_.translate(sl::float3(0.f, 0.f, 2.f)); // 2 meters backward
    }
}

void GLViewer::mouseMotion(int x, int y) {
    if (mouseRotating_) {
        float dx = (float)(x - lastMouseX_) * 0.005f;
        float dy = (float)(y - lastMouseY_) * 0.005f;
        lastMouseX_ = x;
        lastMouseY_ = y;

        sl::Rotation r;
        r.setRotationVector(sl::float3(-dy, dx, 0.f));
        camera_.rotate(r);
    }
    if (mousePanning_) {
        float dx = (float)(x - lastMouseX_) * 0.01f; // Pan sensitivity
        float dy = (float)(y - lastMouseY_) * 0.01f;
        lastMouseX_ = x;
        lastMouseY_ = y;

        // Pan in camera-local X and Y directions
        camera_.translate(sl::float3(-dx, dy, 0.f));
    }
}

void GLViewer::applyCameraMotion() {
    float speed = 0.5f; // 0.5 meters per key press
    sl::float3 move(0, 0, 0);

    if (keyState_['w'] || keyState_['W'])
        move.z -= speed;
    if (keyState_['s'] || keyState_['S'])
        move.z += speed;
    if (keyState_['a'] || keyState_['A'])
        move.x -= speed;
    if (keyState_['d'] || keyState_['D'])
        move.x += speed;
    if (keyState_[' '])
        move.y += speed; // space for up
    if (keyState_['c'] || keyState_['C'])
        move.y -= speed; // c for down

    if (move.x != 0 || move.y != 0 || move.z != 0) {
        camera_.translate(move);
    }

    // Reset key state for WASD (non-sticky)
    keyState_['w'] = keyState_['W'] = false;
    keyState_['s'] = keyState_['S'] = false;
    keyState_['a'] = keyState_['A'] = false;
    keyState_['d'] = keyState_['D'] = false;
    keyState_[' '] = false;
    keyState_['c'] = keyState_['C'] = false;
}
