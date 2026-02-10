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
    // ZED stores color as RGBA in little-endian (R in lowest byte)
    uint color_uint = floatBitsToUint(in_vertex.w);
    float r = float((color_uint >> 0u) & 0xFFu) / 255.0;
    float g = float((color_uint >> 8u) & 0xFFu) / 255.0;
    float b = float((color_uint >> 16u) & 0xFFu) / 255.0;
    v_color = vec4(r, g, b, 1.0);
}
)GLSL";

static const char* POINTCLOUD_VERTEX_SHADER_ZED = R"GLSL(
#version 330 core
layout(location = 0) in vec4 in_vertex;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
    vec3 world_pos = in_vertex.xyz;
    gl_Position = u_mvp * vec4(in_vertex.xyz, 1.0);
    gl_PointSize = 1.5;
    // ZED stores color as RGBA in little-endian (R in lowest byte)
    uint color_uint = floatBitsToUint(in_vertex.w);
    float r = float((color_uint >> 0u) & 0xFFu) / 255.0;
    float g = float((color_uint >> 8u) & 0xFFu) / 255.0;
    float b = float((color_uint >> 16u) & 0xFFu) / 255.0;
  // ---- distance from camera ----
    //vec3 view_pos = (u_view * vec4(world_pos, 1.0)).xyz;
    float dist = length(world_pos);
    // ---- alpha fade ----
    float alpha = 1.0 - smoothstep(0.1,
                                    15,
                                    dist);
    v_color = vec4(r, g, b, alpha);
}
)GLSL";

static const char* LIDAR_VERTEX_SHADER = R"GLSL(
#version 330 core
layout(location = 0) in vec4 in_vertex;   // xyz = position, w = scalar (0..255)
uniform mat4 u_mvp;
out float v_field;
void main() {
    gl_Position = u_mvp * vec4(in_vertex.xyz, 1.0);
    gl_PointSize = 2.0;
    v_field = clamp(in_vertex.w, 0.0, 255.0) / 255.0;
}
)GLSL";

static const char* LIDAR_FRAGMENT_SHADER = R"GLSL(
#version 330 core
in float v_field;
out vec4 frag_color;
uniform sampler1D u_colormap;
void main() {
    vec3 rgb = texture(u_colormap, v_field).rgb;
    frag_color = vec4(rgb, 1.0);
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

static const char* IMAGE_LIDAR_VERTEX_SHADER = R"GLSL(
#version 330 core
layout(location = 0) in vec2 in_pos;
layout(location = 2) in vec2 in_tex;
out vec2 v_texcoord;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_texcoord = in_tex;
}
)GLSL";

static const char* IMAGE_LIDAR_FRAGMENT_SHADER = R"GLSL(
#version 330 core
in vec2 v_texcoord;
out vec4 frag_color;
uniform sampler2D u_texture;
void main() {
    vec4 tex = texture(u_texture, v_texcoord);
    // SIGNAL view returns grayscale in BGRA format
    float gray = tex.b;
    frag_color = vec4(gray, gray, gray, 1.0);
}
)GLSL";

// Utils function
static sl::float3 rotateAroundAxis(const sl::float3& v, const sl::float3& axis, float angle) {
    sl::float3 a = axis;
    float len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    if (len < 1e-6f)
        return v;
    a.x /= len;
    a.y /= len;
    a.z /= len;

    float c = cos(angle), s = sin(angle);
    // v*cos + (a×v)*sin + a*(a·v)*(1-cos)
    sl::float3 axv {a.y * v.z - a.z * v.y, a.z * v.x - a.x * v.z, a.x * v.y - a.y * v.x};
    float adv = a.x * v.x + a.y * v.y + a.z * v.z;

    return sl::float3 {
        v.x * c + axv.x * s + a.x * adv * (1 - c),
        v.y * c + axv.y * s + a.y * adv * (1 - c),
        v.z * c + axv.z * s + a.z * adv * (1 - c)
    };
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
static inline sl::float3 normalize3(const sl::float3& v) {
    float len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-6f)
        return sl::float3(0.f, 0.f, 0.f);
    return sl::float3(v.x / len, v.y / len, v.z / len);
}

static inline sl::float3 cross3(const sl::float3& a, const sl::float3& b) {
    return sl::float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static inline float dot3(const sl::float3& a, const sl::float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

CameraGL::CameraGL()
    : horizontalFOV_(70.f)
    , verticalFOV_(70.f)
    , usePerspective_(true) {
    setPosition(sl::float3(0.f, 0.f, 1000.f));
    setDirection(sl::float3(0.f, 0.f, 1.f), sl::float3(0.f, -1.f, 0.f));
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
    sl::float3 dir = normalize3(direction);
    if (fabs(dir.x) + fabs(dir.y) + fabs(dir.z) > 0.f)
        forward_ = dir;
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

void CameraGL::rotate(const sl::Rotation& r) {
    // Convert rotation to matrix (ZED SDK provides this)
    sl::Matrix3f R = r;
    // Apply to forward_ (and normalize)
    sl::float3 f = forward_;
    sl::float3 nf;
    nf.x = R(0, 0) * f.x + R(0, 1) * f.y + R(0, 2) * f.z;
    nf.y = R(1, 0) * f.x + R(1, 1) * f.y + R(1, 2) * f.z;
    nf.z = R(2, 0) * f.x + R(2, 1) * f.y + R(2, 2) * f.z;

    forward_ = normalize3(nf);

    // --- Pitch clamp without Euler state ---
    // Prevent forward_ from aligning with vertical_ (avoids flips)
    sl::float3 upW = normalize3(vertical_);
    float d = dot3(forward_, upW); // cos(angle)
    const float limit = 0.99f;     // ~8 deg from poles
    if (d > limit || d < -limit) {
        // Undo pitch component by projecting forward onto plane orthogonal to upW
        // (keeps yaw, removes the too-steep pitch)
        sl::float3 proj = sl::float3(forward_.x - d * upW.x, forward_.y - d * upW.y, forward_.z - d * upW.z);
        forward_ = normalize3(proj);
    }

    updateVectors();
}

void CameraGL::updateVectors() {
    sl::float3 f = normalize3(forward_);

    sl::float3 up = normalize3(vertical_);
    float parallel = fabs(dot3(f, up));
    if (parallel > 0.999f) {
        up = sl::float3(0.f, 0.f, 1.f);
        if (fabs(dot3(f, up)) > 0.999f)
            up = sl::float3(1.f, 0.f, 0.f);
        up = normalize3(up);
    }

    // right = normalize(f x up)  (or up x f depending on handedness)
    right_ = normalize3(cross3(f, up));
    up_ = cross3(right_, f); // orthogonal
}

void CameraGL::updateView() {
    // Convention:
    // forward_ points WHERE the camera looks (in world space).

    sl::float3 f = normalize3(forward_);

    // OpenGL view space looks down -Z, so camera "back" axis is -f
    sl::float3 zaxis(-f.x, -f.y, -f.z);

    // Use world up, but protect against forward parallel to up
    sl::float3 up = normalize3(vertical_);
    float parallel = fabs(dot3(f, up));
    if (parallel > 0.999f) {
        // Pick a different up if too close to parallel
        up = sl::float3(0.f, 0.f, 1.f);
        if (fabs(dot3(f, up)) > 0.999f) {
            up = sl::float3(1.f, 0.f, 0.f);
        }
        up = normalize3(up);
    }

    // Right-handed basis:
    // xaxis = normalize(up x zaxis)
    // yaxis = zaxis x xaxis
    sl::float3 xaxis = normalize3(cross3(up, zaxis));
    sl::float3 yaxis = cross3(zaxis, xaxis); // already orthogonal

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

    view_(0, 3) = -dot3(xaxis, position_);
    view_(1, 3) = -dot3(yaxis, position_);
    view_(2, 3) = -dot3(zaxis, position_);
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
    // glutCreateWindow("Sensors API Sample - Press 'R' to toggle recording, 'Q' to quit");
    glutCreateWindow("ZED Unified Viewer");

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
    glutKeyboardUpFunc(keyboardUpCallback);
    glutSpecialFunc(specialKeyDownCallback);
    glutSpecialUpFunc(specialKeyUpCallback);
    glutMouseFunc(mouseButtonCallback);
    glutMotionFunc(mouseMotionCallback);
    glutIdleFunc(idleCallback);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pcShader_ = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    pcShader_zed_ = Shader(POINTCLOUD_VERTEX_SHADER_ZED, POINTCLOUD_FRAGMENT_SHADER);

    imgShader_ = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER);
    lidarShader_ = Shader(POINTCLOUD_VERTEX_SHADER, POINTCLOUD_FRAGMENT_SHADER);
    imgLidarShader_ = Shader(IMAGE_LIDAR_VERTEX_SHADER, IMAGE_LIDAR_FRAGMENT_SHADER);

    camera_.setPosition(sl::float3(0.f, 2.f, 10.f)); // 10 meters back, 2 meters up
    camera_.setDirection(sl::float3(0.f, 0.f, -1.f), sl::float3(0.f, 2.f, 0.f));
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
    // For simplicity, this sample doesn't render 3D object boxes in GL
    // Use RerunViewer for that functionality
}

void GLViewer::updateZedBodies(const std::string& id, const sl::Bodies& bodies, const sl::Matrix4f* transform, sl::Timestamp timestamp) {
    // For simplicity, this sample doesn't render 3D body skeletons in GL
    // Use RerunViewer for that functionality
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
    glClearColor(0.1f, 0.1f, 0.1f, 1.f);

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

            if (pending.hasNewImage) {
                if (!cached.image) {
                    cached.image = std::make_unique<GLImage>();
                    cached.image->init(pending.imageRes.width, pending.imageRes.height, true);
                }
                cached.image->pushImage(pending.image);
                pending.hasNewImage = false;
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
    }

    if (layout_ == ViewerLayout::SplitRows) {
        // === SPLIT ROWS LAYOUT: Lidar on top, 3D in middle, cameras at bottom ===

        // Draw lidar 2D images (top row)
        int lidarRowHeight = windowHeight_ / 8;
        glViewport(0, windowHeight_ - lidarRowHeight, windowWidth_, lidarRowHeight);
        glDisable(GL_DEPTH_TEST);

        float lidarViewportAspect = (float)windowWidth_ / (float)lidarRowHeight;
        // Calculate total width needed for all lidar images to center them
        float lidarTotalWidth = 0.f;
        for (auto& [id, cached] : lidars_) {
            if (cached.image) {
                float imgAspect = cached.image->getAspectRatio();
                float w = 2.0f * imgAspect / lidarViewportAspect;
                lidarTotalWidth += w + 0.02f;
            }
        }
        lidarTotalWidth -= 0.02f; // Remove last gap
        float lidarX = -lidarTotalWidth / 2.f;
        for (auto& [id, cached] : lidars_) {
            if (cached.image) {
                float imgAspect = cached.image->getAspectRatio();
                float h = 2.0f; // Full height in normalized coords (-1 to 1)
                float w = h * imgAspect / lidarViewportAspect;
                cached.image->draw(lidarX, -1.f, w, h, imgLidarShader_);
                lidarX += w + 0.02f;
            }
        }
        glEnable(GL_DEPTH_TEST);

        // Draw point clouds (3D view - middle section)
        int cameraRowHeight = windowHeight_ / 8;
        glViewport(0, cameraRowHeight, windowWidth_, windowHeight_ - lidarRowHeight - cameraRowHeight);
        for (auto& [id, cached] : lidars_) {
            if (!cached.cloud)
                continue;
            cached.cloud->draw(vp, lidarShader_);
        }
        for (auto& [id, cached] : zeds_) {
            if (cached.cloud)
                cached.cloud->draw(vp, pcShader_zed_);
        }

        // Draw camera 2D images (bottom row)
        glViewport(0, 0, windowWidth_, cameraRowHeight);
        glDisable(GL_DEPTH_TEST);

        float cameraViewportAspect = (float)windowWidth_ / (float)cameraRowHeight;
        // Calculate total width needed for all camera images to center them
        float cameraTotalWidth = 0.f;
        float cameraGap = 0.05f;
        for (auto& [id, cached] : zeds_) {
            if (cached.image) {
                float imgAspect = cached.image->getAspectRatio();
                float w = 2.0f * imgAspect / cameraViewportAspect;
                cameraTotalWidth += w + cameraGap;
            }
        }
        cameraTotalWidth -= cameraGap; // Remove last gap
        float cameraX = -cameraTotalWidth / 2.f;
        for (auto& [id, cached] : zeds_) {
            if (cached.image) {
                float imgAspect = cached.image->getAspectRatio();
                float h = 2.0f; // Full height
                float w = h * imgAspect / cameraViewportAspect;
                cached.image->draw(cameraX, -1.f, w, h, imgShader_);
                cameraX += w + cameraGap;
            }
        }
    } else {
        // === COMBINED LAYOUT: All images in single bottom row, 3D view on top ===

        // Draw point clouds (3D view)
        glViewport(0, windowHeight_ / 4, windowWidth_, windowHeight_ * 3 / 4);
        for (auto& [id, cached] : lidars_) {
            if (!cached.cloud)
                continue;
            cached.cloud->draw(vp, lidarShader_);
        }
        for (auto& [id, cached] : zeds_) {
            if (cached.cloud)
                cached.cloud->draw(vp, pcShader_zed_);
        }

        // Draw all images (2D thumbnails at bottom)
        glViewport(0, 0, windowWidth_, windowHeight_ / 4);
        glDisable(GL_DEPTH_TEST);

        float viewportAspect = (float)windowWidth_ / ((float)windowHeight_ / 4.0f);
        float currentX = -1.f;

        for (auto& [id, cached] : zeds_) {
            if (cached.image) {
                float imgAspect = cached.image->getAspectRatio();
                float h = 2.0f;
                float w = h * imgAspect / viewportAspect;
                cached.image->draw(currentX, -1.f, w, h, imgShader_);
                currentX += w + 0.02f;
            }
        }

        for (auto& [id, cached] : lidars_) {
            if (cached.image) {
                float imgAspect = cached.image->getAspectRatio();
                float h = 2.0f;
                float w = h * imgAspect / viewportAspect;
                if (w > 1.9f) {
                    w = 1.9f;
                    h = w * viewportAspect / imgAspect;
                }
                cached.image->draw(currentX, -1.f, w, h, imgLidarShader_);
                currentX += w + 0.02f;
            }
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
    if (key == 'x' || key == 'X') {
        available_ = false;
    }
    if (key == 'r' || key == 'R') {
        if (recordingToggleFn_) {
            recordingToggleFn_();
        }
    }
}

void GLViewer::keyboardUp(unsigned char key, int /*x*/, int /*y*/) {
    keyState_[key] = false;
}

void GLViewer::specialKeyDown(int key, int /*x*/, int /*y*/) {
    if (key >= 0 && key < 256)
        specialKeyState_[key] = true;
}

void GLViewer::specialKeyUp(int key, int /*x*/, int /*y*/) {
    if (key >= 0 && key < 256)
        specialKeyState_[key] = false;
}

void GLViewer::mouseButton(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        mouseRotating_ = (state == GLUT_DOWN);
        if (mouseRotating_) {
            lastMouseX_ = x;
            lastMouseY_ = y;
            firstRotateMotion_ = true;
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
        if (firstRotateMotion_) {
            lastMouseX_ = x;
            lastMouseY_ = y;
            firstRotateMotion_ = false;
            return;
        }

        int dx_i = x - lastMouseX_;
        int dy_i = y - lastMouseY_;

        // clamp discontinuity spikes
        if (abs(dx_i) > 200 || abs(dy_i) > 200) {
            lastMouseX_ = x;
            lastMouseY_ = y;
            return;
        }

        float dx = (float)dx_i * 0.005f;
        float dy = (float)dy_i * 0.005f;

        lastMouseX_ = x;
        lastMouseY_ = y;

        sl::Rotation r;
        // mouse right => yaw right, mouse up => pitch up
        r.setRotationVector(sl::float3(-dy, -dx, 0.f));
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
    // ----------------------------
    // Translation (ZQSD)
    // ----------------------------
    float speed = 0.2f; // meters per frame
    sl::float3 move(0, 0, 0);

    if (keyState_['z'] || keyState_['Z'])
        move.z += speed;
    if (keyState_['s'] || keyState_['S'])
        move.z -= speed;
    if (keyState_['q'] || keyState_['Q'])
        move.x -= speed;
    if (keyState_['d'] || keyState_['D'])
        move.x += speed;

    if (move.x != 0 || move.y != 0 || move.z != 0) {
        camera_.translate(move);
    }

    // ----------------------------
    // Rotation (Arrow keys)
    // ----------------------------
    float rotSpeed = 0.025f; // radians per frame
    float yaw = 0.f;
    float pitch = 0.f;

    if (specialKeyState_[GLUT_KEY_LEFT])
        yaw += rotSpeed;
    if (specialKeyState_[GLUT_KEY_RIGHT])
        yaw -= rotSpeed;
    if (specialKeyState_[GLUT_KEY_UP])
        pitch += rotSpeed;
    if (specialKeyState_[GLUT_KEY_DOWN])
        pitch -= rotSpeed;

    if (yaw != 0.f || pitch != 0.f) {
        sl::Rotation r;
        r.setRotationVector(sl::float3(pitch, yaw, 0.f));
        camera_.rotate(r);
    }
}
