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
#include <GL/freeglut.h>
#include "SLAMView.hpp"

//
// Constants
//
constexpr int ESCAPE_KEY = 27;

//
// Style
//
const float to_f = 1.f / 255.f;
const sl::float4 colorLime(217 * to_f, 255 * to_f, 66 * to_f, 1.f);
const sl::float4 colorPearl(242 * to_f, 242 * to_f, 242 * to_f, 0.7f);
const sl::float4 colorIron(194 * to_f, 194 * to_f, 194 * to_f, 0.2f);
const sl::float4 colorSoil(25 * to_f, 25 * to_f, 25 * to_f, 1.f);
const sl::float4 colorGreen(67 * to_f, 255 * to_f, 63 * to_f, 1.f);
const sl::float4 colorRed(255 * to_f, 100 * to_f, 100 * to_f, 1.f);
const sl::float4 colorBlue(100 * to_f, 100 * to_f, 255 * to_f, 1.f);

void* font = GLUT_BITMAP_HELVETICA_18;

const GLchar* MESH_VERTEX_SHADER = "#version 330 core\n"
                                   "layout(location = 0) in vec3 in_Vertex;\n"
                                   "layout(location = 1) in vec4 in_Color;\n"
                                   "uniform mat4 u_mvpMatrix;\n"
                                   "out vec4 b_color;\n"
                                   "void main() {\n"
                                   "   b_color = in_Color;\n"
                                   "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
                                   "}";

const GLchar* MESH_FRAGMENT_SHADER = "#version 330 core\n"
                                     "in vec4 b_color;\n"
                                     "layout(location = 0) out vec4 color;\n"
                                     "void main() {\n"
                                     "   color = b_color;\n"
                                     "}";

//
// Interaction definitions
//
enum MOUSE_BUTTON {
    LEFT = 0,
    MIDDLE = 1,
    RIGHT = 2,
    WHEEL_UP = 3,
    WHEEL_DOWN = 4
};

//
// SLAMView
//
SLAMView::SLAMView(int argc, char** argv, sl::Mat* frame, sl::Mat* pointCloud, CUstream cudaStream, std::string title)
    : _title(title)
    , _frame(frame)
    , _darkMode(true)
    , _sideBySideMode(true)
    , _pointCloudMode(true)
    , _landmarkMode(false)
    , _followMode(false)
    , _cudaStream(cudaStream)
    , _frameTextureID(0) {

    _instance = this;

    glutInit(&argc, argv);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
    int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);

    _width = 0.6 * screenWidth;
    _height = 0.6 * screenHeight;

    glutInitWindowSize(_width, _height);
    glutInitWindowPosition(0.2 * screenWidth, 0.2 * screenHeight);
    glutCreateWindow(title.c_str());

    glutIdleFunc(SLAMView::onIdle);
    glutDisplayFunc(SLAMView::onDisplay);
    glutReshapeFunc(SLAMView::onReshape);
    glutKeyboardFunc(SLAMView::onKeyboard);
    glutMouseFunc(SLAMView::onMouseButtonPressed);
    glutMotionFunc(SLAMView::onMouseMotion);
    glutCloseFunc(SLAMView::onClose);

    glewInit();

    _shader.it.set((GLchar*)MESH_VERTEX_SHADER, (GLchar*)MESH_FRAGMENT_SHADER);
    _shader.MVPM = glGetUniformLocation(_shader.it.getProgramId(), "u_mvpMatrix");

    _camera = GLCamera(/*target=*/ {-1.4, 0.8, -0.9}, /*distance*/ 3.5, /* yaw */ -18, /*pitch*/ 27);

    _originAxes.setStatic(true);
    _originAxes.setDrawingType(GL_LINES);
    _originAxes.addPoint(sl::float3(0, 0, 0), sl::float4(1, 0, 0, 1));
    _originAxes.addPoint(sl::float3(1, 0, 0), sl::float4(1, 0, 0, 1));

    _originAxes.addPoint(sl::float3(0, 0, 0), sl::float4(0, 1, 0, 1));
    _originAxes.addPoint(sl::float3(0, 1, 0), sl::float4(0, 1, 0, 1));

    _originAxes.addPoint(sl::float3(0, 0, 0), sl::float4(0, 0, 1, 1));
    _originAxes.addPoint(sl::float3(0, 0, 1), sl::float4(0, 0, 1, 1));
    _originAxes.pushToGPU();

    float fx, fy, cx, cy;
    fx = fy = 1400;
    float width, height;
    width = 2208;
    height = 1242;
    cx = width / 2;
    cy = height / 2;

    float Z_ = .15f;
    sl::float3 toOGL(1, -1, -1);
    sl::float3 cam_0(0, 0, 0);
    sl::float3 cam_1, cam_2, cam_3, cam_4;

    float fx_ = 1.f / fx;
    float fy_ = 1.f / fy;

    cam_1.z = Z_;
    cam_1.x = (0 - cx) * Z_ * fx_;
    cam_1.y = (0 - cy) * Z_ * fy_;
    cam_1 *= toOGL;

    cam_2.z = Z_;
    cam_2.x = (width - cx) * Z_ * fx_;
    cam_2.y = (0 - cy) * Z_ * fy_;
    cam_2 *= toOGL;

    cam_3.z = Z_;
    cam_3.x = (width - cx) * Z_ * fx_;
    cam_3.y = (height - cy) * Z_ * fy_;
    cam_3 *= toOGL;

    cam_4.z = Z_;
    cam_4.x = (0 - cx) * Z_ * fx_;
    cam_4.y = (height - cy) * Z_ * fy_;
    cam_4 *= toOGL;

    _cameraFrustum.addPoint(cam_0, colorLime);
    _cameraFrustum.addPoint(cam_1, colorLime);

    _cameraFrustum.addPoint(cam_0, colorLime);
    _cameraFrustum.addPoint(cam_2, colorLime);

    _cameraFrustum.addPoint(cam_0, colorLime);
    _cameraFrustum.addPoint(cam_3, colorLime);

    _cameraFrustum.addPoint(cam_0, colorLime);
    _cameraFrustum.addPoint(cam_4, colorLime);

    _cameraFrustum.setDrawingType(GL_LINES);
    _cameraFrustum.pushToGPU();

    _pointCloud.initialize(*pointCloud);

    _landmarks.setDrawingType(GL_POINTS);
    _keyframes.setDrawingType(GL_LINES);
}

bool SLAMView::isLandmarkModeEnabled() const {
    return _landmarkMode;
}

void SLAMView::run(std::function<void()> callback) {
    _callback = callback;
    glutMainLoop();
}

void SLAMView::stop() {
    glutLeaveMainLoop();
}

void SLAMView::updatePoseTransform(sl::Transform poseTransform) {
    _poseTransform = poseTransform;

    _cameraPath.addPoint(poseTransform.getTranslation());

    if (_pointCloudMode) {
        _pointCloud.pushNewPC(_cudaStream);
    }
}

void SLAMView::updatePositionalTrackingStatus(sl::PositionalTrackingStatus positionalTrackingStatus) {
    _positionalTrackingStatus = positionalTrackingStatus;
}

void SLAMView::updateLandmarks(std::map<uint64_t, sl::Landmark>& landmarks) {
    _landmarks.clear();

    for (auto& landmark : landmarks) {
        _landmarks.addPoint(landmark.second.position, colorLime);
    }
}

sl::float3 applyRT(const sl::float3& point, const sl::Matrix4f& rt) {
    sl::float3 transformedPoint;
    transformedPoint.x = (rt.m[0] * point.x + rt.m[1] * point.y + rt.m[2] * point.z + rt.m[3]);
    transformedPoint.y = (rt.m[4] * point.x + rt.m[5] * point.y + rt.m[6] * point.z + rt.m[7]);
    transformedPoint.z = (rt.m[8] * point.x + rt.m[9] * point.y + rt.m[10] * point.z + rt.m[11]);
    return transformedPoint;
}

void SLAMView::updateKeyframes(const std::map<uint64_t, sl::KeyFrame>& keyframes) {
    _keyframes.clear();

    const float z_ = -.05f; // ogl convention
    const float width = 0.05f;
    const float height = width * 0.75f;

    std::vector<sl::float3> cameraCorners
        = {sl::float3(-width, -height, z_), sl::float3(width, -height, z_), sl::float3(width, height, z_), sl::float3(-width, height, z_)};

    for (const auto& kf_iter : keyframes) {
        sl::KeyFrame keyframe = kf_iter.second;
        sl::float3 origin(0, 0, 0);
        origin = applyRT(origin, keyframe.pose);
        auto clr = keyframe.is_loaded ? colorGreen : colorLime;
        for (auto& corner : cameraCorners) {
            _keyframes.addPoint(origin, clr);
            _keyframes.addPoint(applyRT(corner, keyframe.pose), clr);
        }

        if (keyframe.is_loaded) {
            _keyframes.addPoint(applyRT(cameraCorners.back(), keyframe.pose), clr);
            _keyframes.addPoint(applyRT(cameraCorners.front(), keyframe.pose), clr);
            for (int i = 1; i < cameraCorners.size(); i++) {
                _keyframes.addPoint(applyRT(cameraCorners[i - 1], keyframe.pose), clr);
                _keyframes.addPoint(applyRT(cameraCorners[i], keyframe.pose), clr);
            }
        }
    }
}

void SLAMView::idle() {
    _callback();

    if (_landmarkMode) {
        _landmarks.pushToGPU();
        _keyframes.pushToGPU();
    }

    glutPostRedisplay();
}

void SLAMView::display() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_DEPTH_TEST);

    sl::float4 color = _darkMode ? colorSoil : colorPearl;
    glClearColor(color[0], color[1], color[2], color[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    sl::Transform vpMatrix = _camera.getSLProjectionMatrix() * _camera.getSLViewMatrix();

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(3.f);

    glUseProgram(_shader.it.getProgramId());

    if (_followMode) {
        sl::Transform pose_w_t = _poseTransform;
        auto ea = pose_w_t.getEulerAngles();
        ea.x = 0;
        ea.z = 0;
        pose_w_t.setEulerAngles(ea);
        vpMatrix = vpMatrix * sl::Transform::inverse(pose_w_t);
    }

    glUniformMatrix4fv(_shader.MVPM, 1, GL_TRUE, vpMatrix.m);
    _originAxes.draw();

    _cameraPath.setColor(_darkMode ? colorPearl : colorSoil);
    _cameraPath.setMVP(vpMatrix);
    _cameraPath.draw();

    if (_landmarkMode) {
        glPointSize(3.f);
        _landmarks.draw();
        glLineWidth(2.f);
        _keyframes.draw();
    }

    glPointSize(2.f);

    sl::Transform transformedPose = vpMatrix * _poseTransform;
    glUniformMatrix4fv(_shader.MVPM, 1, GL_TRUE, transformedPose.m);
    _cameraFrustum.draw();
    glUseProgram(0);

    if (_pointCloudMode) {
        _pointCloud.draw(transformedPose);
    }

    glDisable(GL_DEPTH_TEST);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    renderPositionalTrackingStatus();

    if (_sideBySideMode) {
        renderImageFrame();
    }

    glutSwapBuffers();
}

void SLAMView::reshape(int width, int height) {
    _width = width;
    _height = height;

    glViewport(0, 0, width, height);

    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    _camera.resizeViewport(aspectRatio);
}

void SLAMView::keyboard(unsigned char key, int x, int y) {
    if (key == ' ') {
        _sideBySideMode = !_sideBySideMode;
    } else if (key == 'd') {
        _darkMode = !_darkMode;
    } else if (key == 'p') {
        _pointCloudMode = !_pointCloudMode;
    } else if (key == 'l') {
        _landmarkMode = !_landmarkMode;
    } else if (key == 'f') {
        _followMode = !_followMode;
        _camera.reset();
    } else if (key == 'z') {
        _camera.reset();
    } else if (key == ESCAPE_KEY) {
        stop();
    }
}

void SLAMView::mouseButtonPressed(int button, int state, int x, int y) {
    constexpr float zoomSensitivity = 0.75;

    if (button == MOUSE_BUTTON::LEFT) {
        _mouseButtonPressed = (state == GLUT_DOWN);

        if (_mouseButtonPressed) {
            _mousePosition[0] = x;
            _mousePosition[1] = y;

            _ctrlPressed = glutGetModifiers() & GLUT_ACTIVE_CTRL;
        }
    } else if (button == MOUSE_BUTTON::WHEEL_DOWN) {
        _camera.zoom(-zoomSensitivity);
    } else if (button == MOUSE_BUTTON::WHEEL_UP) {
        _camera.zoom(zoomSensitivity);
    }
}

void SLAMView::mouseMotion(int x, int y) {
    if (!_mouseButtonPressed) {
        return;
    }

    int deltaX = x - _mousePosition[0];
    int deltaY = y - _mousePosition[1];

    if (_ctrlPressed) {
        constexpr float rotateSensitivity = 0.05f;
        _camera.rotate(-deltaX * rotateSensitivity, deltaY * rotateSensitivity);
    } else {
        constexpr float translateSensitivity = 0.007f;
        _camera.pan(deltaX * translateSensitivity, deltaY * translateSensitivity);
    }

    _mousePosition[0] = x;
    _mousePosition[1] = y;
}

void SLAMView::close() {
    if (glIsTexture(_frameTextureID)) {
        glDeleteTextures(1, &_frameTextureID);
        _frameTextureID = 0;
    }

    _pointCloud.close();
}

//
// Rendering Content
//
void SLAMView::renderPositionalTrackingStatus() {
    // Save current projection and modelview matrices
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, _width, 0, _height);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    static const int startX = 50;
    const int startY = _height - 50;
    static const int verticalSpacing = 25;
    static const int statusValueX = startX + 150;

    sl::float4 textColor = _darkMode ? colorPearl : colorSoil;

    // Fusion status
    renderText(startX, startY, font, textColor, "Fusion:");

    sl::float4 statusColor
        = _positionalTrackingStatus.tracking_fusion_status != sl::POSITIONAL_TRACKING_FUSION_STATUS::UNAVAILABLE ? colorGreen : colorRed;

    renderText(statusValueX, startY, font, statusColor, sl::toString(_positionalTrackingStatus.tracking_fusion_status).c_str());

    // Spatial memory status
    renderText(startX, startY - verticalSpacing, font, textColor, "Area Memory:");

    if (_positionalTrackingStatus.spatial_memory_status == sl::SPATIAL_MEMORY_STATUS::OFF
        || _positionalTrackingStatus.spatial_memory_status == sl::SPATIAL_MEMORY_STATUS::LOST) {
        statusColor = colorRed;
    } else if (_positionalTrackingStatus.spatial_memory_status == sl::SPATIAL_MEMORY_STATUS::INITIALIZING
               || _positionalTrackingStatus.spatial_memory_status == sl::SPATIAL_MEMORY_STATUS::SEARCHING) {
        statusColor = colorLime;
    } else if (_positionalTrackingStatus.spatial_memory_status == sl::SPATIAL_MEMORY_STATUS::LOOP_CLOSED) {
        statusColor = colorPearl;
    } else {
        statusColor = colorGreen;
    }

    std::string spatialMemoryStatusText = sl::toString(_positionalTrackingStatus.spatial_memory_status).c_str();

    renderText(statusValueX, startY - verticalSpacing, font, statusColor, spatialMemoryStatusText);

    // Odometry status
    renderText(startX, startY - 2 * verticalSpacing, font, textColor, "Odometry:");

    statusColor = _positionalTrackingStatus.odometry_status != sl::ODOMETRY_STATUS::UNAVAILABLE ? colorGreen : colorRed;

    std::string odometryStatusText = sl::toString(_positionalTrackingStatus.odometry_status).c_str();

    renderText(statusValueX, startY - 2 * verticalSpacing, font, statusColor, odometryStatusText);

    // Pose transform
    renderText(startX, startY - 4 * verticalSpacing, font, textColor, "Translation (m):");
    renderText(statusValueX, startY - 4 * verticalSpacing, font, textColor, formatNumericText(_poseTransform.getTranslation()));

    renderText(startX, startY - 5 * verticalSpacing, font, textColor, "Rotation   (rad):");
    renderText(statusValueX, startY - 5 * verticalSpacing, font, textColor, formatNumericText(_poseTransform.getRotationVector()));

    // Restore matrices
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void SLAMView::renderImageFrame() {
    if (_frame->getWidth() == 0 || _frame->getHeight() == 0) {
        return;
    }

    uploadImageToTexture(*_frame, _frameTextureID);

    const static float maxWidthFraction = 0.3f;
    const static int minWidth = 512;

    int maxWidth = static_cast<int>(_width * maxWidthFraction);
    int displayWidth = std::max(minWidth, maxWidth);

    float scale = static_cast<float>(displayWidth) / _frame->getWidth();
    int scaledWidth = displayWidth;
    int scaledHeight = static_cast<int>(_frame->getHeight() * scale);

    renderTexture(_frameTextureID, 50, 50, scaledWidth, scaledHeight);
}

//
// Rendering Utilities
//
std::string SLAMView::formatNumericText(sl::float3 value) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

void SLAMView::renderText(int x, int y, void* font, const sl::float4& color, const std::string& str) {
    glColor3f(color[0], color[1], color[2]);
    glRasterPos2i(x, y);
    glutBitmapString(font, reinterpret_cast<const unsigned char*>(str.c_str()));
}

void SLAMView::uploadImageToTexture(const sl::Mat& frame, GLuint& textureID) {
    if (textureID == 0) {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            frame.getWidth(),
            frame.getHeight(),
            0,
            GL_BGRA,
            GL_UNSIGNED_BYTE,
            frame.getPtr<sl::uchar1>()
        );
    } else {
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.getWidth(), frame.getHeight(), GL_BGRA, GL_UNSIGNED_BYTE, frame.getPtr<sl::uchar1>());
    }
}

void SLAMView::renderTexture(GLuint textureID, int x, int y, int width, int height) {
    // Save current projection and model view
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, _width, 0, _height);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // Bind to texture
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glColor3f(1.0f, 1.0f, 1.0f);

    // Render in rect
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2i(x, y);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2i(x + width, y);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2i(x + width, y + height);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2i(x, y + height);
    glEnd();

    glDisable(GL_TEXTURE_2D);

    // Restore projection and model view
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

//
// Static Forwarding
//
void SLAMView::onIdle() {
    _instance->idle();
}

void SLAMView::onDisplay() {
    _instance->display();
}

void SLAMView::onReshape(int w, int h) {
    _instance->reshape(w, h);
}

void SLAMView::onKeyboard(unsigned char key, int x, int y) {
    _instance->keyboard(key, x, y);
}

void SLAMView::onMouseButtonPressed(int button, int state, int x, int y) {
    _instance->mouseButtonPressed(button, state, x, y);
}

void SLAMView::onMouseMotion(int x, int y) {
    _instance->mouseMotion(x, y);
}

void SLAMView::onClose() {
    _instance->close();
}
