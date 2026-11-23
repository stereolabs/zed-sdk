///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2026, STEREOLABS.
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

#include <cassert>
#include <iomanip>
#include <sstream>

#include "RigView.hpp"
#include "Overlay2D.hpp"

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
const sl::float4 colorSoil(25 * to_f, 25 * to_f, 25 * to_f, 1.f);
const sl::float4 colorGreen(67 * to_f, 255 * to_f, 63 * to_f, 1.f);
const sl::float4 colorRed(255 * to_f, 100 * to_f, 100 * to_f, 1.f);

// One color per camera of the rig, the first one for the camera that drives the tracking.
const sl::float4 cameraPalette[] = {
    colorLime,
    sl::float4(80 * to_f, 200 * to_f, 255 * to_f, 1.f),
    sl::float4(255 * to_f, 150 * to_f, 60 * to_f, 1.f),
    sl::float4(230 * to_f, 100 * to_f, 255 * to_f, 1.f),
    sl::float4(255 * to_f, 220 * to_f, 90 * to_f, 1.f),
    sl::float4(90 * to_f, 255 * to_f, 200 * to_f, 1.f),
};
constexpr size_t cameraPaletteSize = sizeof(cameraPalette) / sizeof(cameraPalette[0]);

// Layout of the overlays in the top left corner, in pixels.
constexpr float overlayMargin = 30.f;
constexpr float cardWidth = 330.f;
constexpr float cardSpacing = 12.f;

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

namespace {

    /**
     * @brief Applies a rigid transform to a point.
     * @param point The point.
     * @param rt The transform.
     * @return The transformed point.
     */
    sl::float3 applyRT(const sl::float3& point, const sl::Matrix4f& rt) {
        sl::float3 transformedPoint;
        transformedPoint.x = (rt.m[0] * point.x + rt.m[1] * point.y + rt.m[2] * point.z + rt.m[3]);
        transformedPoint.y = (rt.m[4] * point.x + rt.m[5] * point.y + rt.m[6] * point.z + rt.m[7]);
        transformedPoint.z = (rt.m[8] * point.x + rt.m[9] * point.y + rt.m[10] * point.z + rt.m[11]);
        return transformedPoint;
    }

    /**
     * @brief Colors a tracking state.
     * @param state The state.
     * @return Green when OK, red otherwise.
     */
    sl::float4 trackingStateColor(sl::POSITIONAL_TRACKING_STATE state) {
        return state == sl::POSITIONAL_TRACKING_STATE::OK ? colorGreen : colorRed;
    }

    /**
     * @brief Colors a spatial memory status.
     * @param status The status.
     * @return Red when lost or off, lime while searching, pearl on a loop closure, green otherwise.
     */
    sl::float4 spatialMemoryColor(sl::SPATIAL_MEMORY_STATUS status) {
        switch (status) {
            case sl::SPATIAL_MEMORY_STATUS::OFF:
            case sl::SPATIAL_MEMORY_STATUS::LOST:
            case sl::SPATIAL_MEMORY_STATUS::NOT_ENOUGH_MEMORY_FOR_TRACKING:
                return colorRed;
            case sl::SPATIAL_MEMORY_STATUS::INITIALIZING:
            case sl::SPATIAL_MEMORY_STATUS::SEARCHING:
                return colorLime;
            case sl::SPATIAL_MEMORY_STATUS::LOOP_CLOSED:
                return colorPearl;
            default:
                return colorGreen;
        }
    }

    /**
     * @brief Formats the time elapsed since a start time.
     * @param start The start time.
     * @return The elapsed time as hh:mm:ss.
     */
    std::string formatElapsed(std::chrono::steady_clock::time_point start) {
        const auto totalSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();

        std::stringstream stream;
        stream << std::setfill('0') << std::setw(2) << totalSeconds / 3600 << ":" << std::setw(2) << (totalSeconds % 3600) / 60 << ":"
               << std::setw(2) << totalSeconds % 60;
        return stream.str();
    }

} // namespace

//
// RigView
//
RigView::RigView(
    int argc,
    char** argv,
    sl::Mat* frame,
    const std::vector<std::string>& cameraLabels,
    size_t trackingCamera,
    std::string title
)
    : _title(title)
    , _isOpen(true)
    , _frame(frame)
    , _trackingCamera(trackingCamera)
    , _displayedCamera(trackingCamera)
    , _darkMode(true)
    , _sideBySideMode(true)
    , _landmarkMode(false)
    , _followMode(false)
    , _pathMode(true)
    , _startTime(std::chrono::steady_clock::now())
    , _frameTextureID(0) {
    assert(frame != nullptr);
    assert(trackingCamera < cameraLabels.size());

    _instance = this;

    size_t nextColor = 1;
    for (size_t i = 0; i < cameraLabels.size(); i++) {
        CameraState camera;
        camera.label = cameraLabels[i];
        camera.color = i == trackingCamera ? cameraPalette[0] : cameraPalette[nextColor++ % cameraPaletteSize];
        _cameras.push_back(camera);
    }

    createWindow(argc, argv);
    createScene();
}

RigView::~RigView() {
    _instance = nullptr;
}

void RigView::createWindow(int argc, char** argv) {
    glutInit(&argc, argv);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
    int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);

    _width = 0.6 * screenWidth;
    _height = 0.6 * screenHeight;

    glutInitWindowSize(_width, _height);
    glutInitWindowPosition(0.2 * screenWidth, 0.2 * screenHeight);
    glutCreateWindow(_title.c_str());

    glutDisplayFunc(RigView::onDisplay);
    glutReshapeFunc(RigView::onReshape);
    glutKeyboardFunc(RigView::onKeyboard);
    glutMouseFunc(RigView::onMouseButtonPressed);
    glutMotionFunc(RigView::onMouseMotion);
    glutCloseFunc(RigView::onClose);

    glewInit();
}

void RigView::createScene() {
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

    // Corners of the image plane of a generic camera, 15 cm in front of it. A camera looks down its -Z axis with Y up in
    // RIGHT_HANDED_Y_UP, hence the flip of the image axes.
    const float fx = 1400, fy = 1400;
    const float width = 2208, height = 1242;
    const float cx = width / 2, cy = height / 2;
    const float Z_ = .15f;
    const sl::float3 toOGL(1, -1, -1);

    for (const sl::float2& pixel : {sl::float2(0, 0), sl::float2(width, 0), sl::float2(width, height), sl::float2(0, height)}) {
        sl::float3 corner((pixel.x - cx) * Z_ / fx, (pixel.y - cy) * Z_ / fy, Z_);
        _frustumCorners.push_back(corner * toOGL);
    }

    _rig.setDrawingType(GL_LINES);
    _landmarks.setDrawingType(GL_POINTS);
    _keyframes.setDrawingType(GL_LINES);
}

bool RigView::isLandmarkModeEnabled() const {
    return _landmarkMode;
}

size_t RigView::displayedCamera() const {
    return _displayedCamera;
}

bool RigView::isOpen() const {
    return _isOpen;
}

void RigView::render() {
    if (!_isOpen) {
        return;
    }

    if (_landmarkMode) {
        _landmarks.pushToGPU();
        _keyframes.pushToGPU();
    }

    glutPostRedisplay();
    glutMainLoopEvent();
}

void RigView::updateRigPose(const sl::Transform& rigPose) {
    _rigPose = rigPose;
    _rigPath.addPoint(rigPose.getTranslation());
}

void RigView::updateCameraPose(size_t camera, const sl::Transform& cameraPose, sl::POSITIONAL_TRACKING_STATE state) {
    assert(camera < _cameras.size());
    _cameras[camera].pose = cameraPose;
    _cameras[camera].state = state;
}

void RigView::updatePositionalTrackingStatus(const sl::PositionalTrackingStatus& positionalTrackingStatus) {
    _positionalTrackingStatus = positionalTrackingStatus;
}

void RigView::enablePauseButton() {
    _pauseButtonEnabled = true;
}

bool RigView::isPaused() const {
    return _paused;
}

void RigView::updateLandmarks(const std::map<uint64_t, sl::Landmark>& landmarks) {
    _landmarks.clear();

    for (const auto& landmark : landmarks) {
        _landmarks.addPoint(landmark.second.position, colorPearl);
    }
}

void RigView::updateKeyframes(const std::vector<std::map<uint64_t, sl::KeyFrame>>& keyframes) {
    assert(keyframes.size() == _cameras.size());

    _keyframes.clear();

    for (size_t camera = 0; camera < keyframes.size(); camera++) {
        for (const auto& keyframe : keyframes[camera]) {
            addKeyframe(keyframe.second, _cameras[camera].color);
        }
    }
}

void RigView::addKeyframe(const sl::KeyFrame& keyframe, const sl::float4& color) {
    const float z_ = -.05f; // ogl convention
    const float width = 0.05f;
    const float height = width * 0.75f;
    const sl::float3 corners[4]
        = {sl::float3(-width, -height, z_), sl::float3(width, -height, z_), sl::float3(width, height, z_), sl::float3(-width, height, z_)};

    const sl::float3 origin = applyRT(sl::float3(0, 0, 0), keyframe.pose);
    for (const sl::float3& corner : corners) {
        _keyframes.addPoint(origin, color);
        _keyframes.addPoint(applyRT(corner, keyframe.pose), color);
    }

    if (keyframe.is_loaded) {
        for (size_t i = 0; i < 4; i++) {
            _keyframes.addPoint(applyRT(corners[i], keyframe.pose), color);
            _keyframes.addPoint(applyRT(corners[(i + 1) % 4], keyframe.pose), color);
        }
    }
}

void RigView::buildRig() {
    _rig.clear();

    const sl::float3 rigOrigin = _rigPose.getTranslation();
    const sl::float4 armColor = textColor();

    for (const CameraState& camera : _cameras) {
        const sl::float3 apex = applyRT(sl::float3(0, 0, 0), camera.pose);

        _rig.addPoint(rigOrigin, armColor);
        _rig.addPoint(apex, armColor);

        for (size_t i = 0; i < _frustumCorners.size(); i++) {
            const sl::float3 corner = applyRT(_frustumCorners[i], camera.pose);
            const sl::float3 nextCorner = applyRT(_frustumCorners[(i + 1) % _frustumCorners.size()], camera.pose);
            _rig.addPoint(apex, camera.color);
            _rig.addPoint(corner, camera.color);
            _rig.addPoint(corner, camera.color);
            _rig.addPoint(nextCorner, camera.color);
        }
    }

    _rig.pushToGPU();
}

void RigView::display() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_DEPTH_TEST);

    sl::float4 color = _darkMode ? colorSoil : colorPearl;
    glClearColor(color[0], color[1], color[2], color[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    sl::Transform vpMatrix = _camera.getSLProjectionMatrix() * _camera.getSLViewMatrix();

    if (_followMode) {
        sl::Transform pose_w_t = _rigPose;
        auto ea = pose_w_t.getEulerAngles();
        ea.x = 0;
        ea.z = 0;
        pose_w_t.setEulerAngles(ea);
        vpMatrix = vpMatrix * sl::Transform::inverse(pose_w_t);
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(3.f);

    glUseProgram(_shader.it.getProgramId());

    glUniformMatrix4fv(_shader.MVPM, 1, GL_TRUE, vpMatrix.m);
    _originAxes.draw();

    if (_pathMode) {
        _rigPath.setColor(textColor());
        _rigPath.setMVP(vpMatrix);
        _rigPath.draw();
    }

    if (_landmarkMode) {
        glPointSize(1.5f);
        _landmarks.draw();
        glLineWidth(2.f);
        _keyframes.draw();
    }

    glLineWidth(3.f);
    buildRig();
    _rig.draw();
    glUseProgram(0);

    glDisable(GL_DEPTH_TEST);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    renderOverlays();

    if (_sideBySideMode) {
        renderImageFrame();
    }

    // Last, so the camera image never covers it.
    renderPauseButton();

    glutSwapBuffers();
}

void RigView::reshape(int width, int height) {
    _width = width;
    _height = height;

    glViewport(0, 0, width, height);

    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    _camera.resizeViewport(aspectRatio);
}

void RigView::keyboard(unsigned char key, int x, int y) {
    if (key == ' ') {
        _sideBySideMode = !_sideBySideMode;
    } else if (key == 'n') {
        _displayedCamera = (_displayedCamera + 1) % _cameras.size();
    } else if (key == 'd') {
        _darkMode = !_darkMode;
    } else if (key == 'l') {
        _landmarkMode = !_landmarkMode;
    } else if (key == 'f') {
        _followMode = !_followMode;
        _camera.reset();
    } else if (key == 'z') {
        _camera.reset();
    } else if (key == 'c') {
        _pathMode = !_pathMode;
    } else if (key == 'k' && _pauseButtonEnabled) {
        _paused = !_paused;
    } else if (key == ESCAPE_KEY) {
        _isOpen = false;
    }
}

void RigView::mouseButtonPressed(int button, int state, int x, int y) {
    constexpr float zoomSensitivity = 0.75;

    if (button == MOUSE_BUTTON::LEFT) {
        _mouseButtonPressed = (state == GLUT_DOWN);

        if (_mouseButtonPressed) {
            _mousePosition[0] = x;
            _mousePosition[1] = y;
            _mouseDownPosition[0] = x;
            _mouseDownPosition[1] = y;

            _ctrlPressed = glutGetModifiers() & GLUT_ACTIVE_CTRL;
        } else if (std::abs(x - _mouseDownPosition[0]) <= 3 && std::abs(y - _mouseDownPosition[1]) <= 3) {
            // A release where the press happened is a click: toggle the pause button if it is under it.
            const sl::float4 button = pauseButtonRect();
            const float py = static_cast<float>(_height - y); // GLUT counts from the top
            if (_pauseButtonEnabled && x >= button.x && x <= button.x + button.z && py >= button.y && py <= button.y + button.w) {
                _paused = !_paused;
            }
        }
    } else if (button == MOUSE_BUTTON::WHEEL_DOWN) {
        _camera.zoom(-zoomSensitivity);
    } else if (button == MOUSE_BUTTON::WHEEL_UP) {
        _camera.zoom(zoomSensitivity);
    }
}

void RigView::mouseMotion(int x, int y) {
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

void RigView::close() {
    if (glIsTexture(_frameTextureID)) {
        glDeleteTextures(1, &_frameTextureID);
        _frameTextureID = 0;
    }

    _isOpen = false;
}

//
// Rendering Content
//
sl::float4 RigView::textColor() const {
    return _darkMode ? colorPearl : colorSoil;
}

sl::float4 RigView::panelColor() const {
    return _darkMode ? sl::float4(14 * to_f, 16 * to_f, 20 * to_f, 0.85f) : sl::float4(1.f, 1.f, 1.f, 0.85f);
}

void RigView::renderOverlays() {
    overlay::begin(_width, _height);

    const float badgesBottom = renderStatusBadges(static_cast<float>(_height) - overlayMargin);
    const float cardBottom = renderPoseCard(badgesBottom - cardSpacing);
    renderCameraList(cardBottom - cardSpacing);

    overlay::end();
}

float RigView::renderStatusBadges(float top) {
    const sl::float4 mutedColor(textColor().x, textColor().y, textColor().z, 0.6f);
    void* labelFont = GLUT_BITMAP_HELVETICA_12;

    struct Badge {
        std::string label;
        std::string value;
        sl::float4 color;
    };
    const Badge badges[] = {
        {"Fusion",
         sl::toString(_positionalTrackingStatus.tracking_fusion_status).c_str(),
         _positionalTrackingStatus.tracking_fusion_status != sl::POSITIONAL_TRACKING_FUSION_STATUS::UNAVAILABLE ? colorGreen : colorRed},
        {"Area memory",
         sl::toString(_positionalTrackingStatus.spatial_memory_status).c_str(),
         spatialMemoryColor(_positionalTrackingStatus.spatial_memory_status)                                                           },
        {"Odometry",
         sl::toString(_positionalTrackingStatus.odometry_status).c_str(),
         _positionalTrackingStatus.odometry_status != sl::ODOMETRY_STATUS::UNAVAILABLE ? colorGreen : colorRed                         },
    };

    const float badgeHeight = 34.f;
    float badgeX = overlayMargin;

    for (const Badge& badge : badges) {
        const float width
            = static_cast<float>(std::max(overlay::textWidth(labelFont, badge.label), overlay::textWidth(labelFont, badge.value)) + 24);
        overlay::fillRect(badgeX, top - badgeHeight, width, badgeHeight, panelColor());
        overlay::strokeRect(badgeX, top - badgeHeight, width, badgeHeight, sl::float4(badge.color.x, badge.color.y, badge.color.z, 0.5f));
        overlay::text(badgeX + 12.f, top - 14.f, labelFont, mutedColor, badge.label);
        overlay::text(badgeX + 12.f, top - 28.f, labelFont, badge.color, badge.value);
        badgeX += width + 10.f;
    }

    return top - badgeHeight;
}

float RigView::renderPoseCard(float top) {
    const sl::float4 mutedColor(textColor().x, textColor().y, textColor().z, 0.6f);
    void* labelFont = GLUT_BITMAP_HELVETICA_12;

    const float cardHeight = 108.f;
    overlay::fillRect(overlayMargin, top - cardHeight, cardWidth, cardHeight, panelColor());

    overlay::text(42.f, top - 22.f, labelFont, mutedColor, "Up time");
    overlay::text(150.f, top - 22.f, labelFont, textColor(), formatElapsed(_startTime));

    const float columns[3] = {150.f, 210.f, 270.f};
    const char* const axes[3] = {"x", "y", "z"};
    for (int i = 0; i < 3; i++) {
        overlay::text(columns[i], top - 48.f, labelFont, mutedColor, axes[i]);
    }

    const auto poseRow = [&](float y, const std::string& label, const sl::float3& value) {
        overlay::text(42.f, y, labelFont, mutedColor, label);
        for (int i = 0; i < 3; i++) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << value[i];
            overlay::text(columns[i], y, labelFont, textColor(), stream.str());
        }
    };
    poseRow(top - 72.f, "Rig position (m)", _rigPose.getTranslation());
    poseRow(top - 94.f, "Rig rotation (rad)", _rigPose.getRotationVector());

    return top - cardHeight;
}

void RigView::renderCameraList(float top) {
    const sl::float4 mutedColor(textColor().x, textColor().y, textColor().z, 0.6f);
    void* labelFont = GLUT_BITMAP_HELVETICA_12;

    const float rowHeight = 22.f;
    const float listHeight = 12.f + rowHeight * _cameras.size();
    overlay::fillRect(overlayMargin, top - listHeight, cardWidth, listHeight, panelColor());

    for (size_t i = 0; i < _cameras.size(); i++) {
        const CameraState& camera = _cameras[i];
        const float y = top - 6.f - rowHeight * (i + 1);

        if (i == _displayedCamera && _sideBySideMode) {
            overlay::triangle(36.f, y + 4.f, 36.f, y + 14.f, 44.f, y + 9.f, textColor());
        }
        overlay::fillRect(50.f, y + 4.f, 10.f, 10.f, camera.color);

        const std::string label = camera.label + (i == _trackingCamera ? "  (tracking)" : "");
        overlay::text(70.f, y + 5.f, labelFont, textColor(), label);
        overlay::text(250.f, y + 5.f, labelFont, trackingStateColor(camera.state), sl::toString(camera.state).c_str());
    }

    overlay::text(overlayMargin, top - listHeight - 18.f, labelFont, mutedColor, "'n' shows the next camera");
}

sl::float4 RigView::pauseButtonRect() const {
    // Level with the top of the pose card: 76 px below the top of the window.
    const float height = 30.f;
    return sl::float4(overlayMargin + cardWidth + 10.f, static_cast<float>(_height) - 76.f - height, 124.f, height);
}

void RigView::renderPauseButton() {
    if (!_pauseButtonEnabled) {
        return;
    }

    const sl::float4 r = pauseButtonRect();

    overlay::begin(_width, _height);
    overlay::fillRect(r.x, r.y, r.z, r.w, panelColor());
    const sl::float4 outlineColor = _paused ? colorLime : sl::float4(textColor().x, textColor().y, textColor().z, 0.4f);
    overlay::strokeRect(r.x, r.y, r.z, r.w, outlineColor, _paused ? 2.f : 1.f);

    // The icon shows what a click does: play while paused, pause while playing.
    const float iconX = r.x + 12.f;
    const float iconY = r.y + 9.f;
    const float iconSize = 12.f;
    if (_paused) {
        overlay::triangle(iconX, iconY, iconX, iconY + iconSize, iconX + iconSize, iconY + iconSize / 2.f, textColor());
    } else {
        overlay::fillRect(iconX, iconY, 4.f, iconSize, textColor());
        overlay::fillRect(iconX + 8.f, iconY, 4.f, iconSize, textColor());
    }
    overlay::text(iconX + iconSize + 10.f, r.y + 11.f, GLUT_BITMAP_HELVETICA_12, textColor(), _paused ? "Resume (k)" : "Pause (k)");
    overlay::end();
}

void RigView::renderImageFrame() {
    if (_frame->getWidth() == 0 || _frame->getHeight() == 0) {
        return;
    }

    uploadImageToTexture(*_frame);

    const static float maxWidthFraction = 0.3f;
    const static int minWidth = 512;

    int maxWidth = static_cast<int>(_width * maxWidthFraction);
    int displayWidth = std::max(minWidth, maxWidth);

    float scale = static_cast<float>(displayWidth) / _frame->getWidth();
    int scaledWidth = displayWidth;
    int scaledHeight = static_cast<int>(_frame->getHeight() * scale);

    renderTexture(_frameTextureID, 50, 50, scaledWidth, scaledHeight);

    overlay::begin(_width, _height);
    const CameraState& camera = _cameras[_displayedCamera];
    overlay::fillRect(50.f, 50.f + scaledHeight, 10.f, 10.f, camera.color);
    overlay::text(66.f, 51.f + scaledHeight, GLUT_BITMAP_HELVETICA_12, textColor(), camera.label);
    overlay::end();
}

//
// Rendering Utilities
//
void RigView::uploadImageToTexture(const sl::Mat& frame) {
    const sl::Resolution size(frame.getWidth(), frame.getHeight());

    if (_frameTextureID == 0 || size.width != _frameTextureSize.width || size.height != _frameTextureSize.height) {
        if (_frameTextureID == 0) {
            glGenTextures(1, &_frameTextureID);
        }
        glBindTexture(GL_TEXTURE_2D, _frameTextureID);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size.width, size.height, 0, GL_BGRA, GL_UNSIGNED_BYTE, frame.getPtr<sl::uchar1>());
        _frameTextureSize = size;
    } else {
        glBindTexture(GL_TEXTURE_2D, _frameTextureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, size.width, size.height, GL_BGRA, GL_UNSIGNED_BYTE, frame.getPtr<sl::uchar1>());
    }
}

void RigView::renderTexture(GLuint textureID, int x, int y, int width, int height) {
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
void RigView::onDisplay() {
    _instance->display();
}

void RigView::onReshape(int w, int h) {
    _instance->reshape(w, h);
}

void RigView::onKeyboard(unsigned char key, int x, int y) {
    _instance->keyboard(key, x, y);
}

void RigView::onMouseButtonPressed(int button, int state, int x, int y) {
    _instance->mouseButtonPressed(button, state, x, y);
}

void RigView::onMouseMotion(int x, int y) {
    _instance->mouseMotion(x, y);
}

void RigView::onClose() {
    if (_instance)
        _instance->close();
}
