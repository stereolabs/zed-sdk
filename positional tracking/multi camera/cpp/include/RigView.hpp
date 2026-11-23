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

#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include <sl/Sensors.hpp>

#include "GLCamera.hpp"
#include "Simple3DObject.hpp"
#include "Simple3DPath.hpp"
#include "Shader.hpp"

/**
 * @brief OpenGL viewer of a tracked rig: every camera at its pose, the path of the rig, the map (landmarks, and keyframes in
 * the color of the camera that took them), the tracking status, and the image of one camera of the rig.
 */
class RigView {
public:
    /**
     * @brief Opens the window.
     * @param argc Number of command line arguments, for GLUT.
     * @param argv The command line arguments, for GLUT.
     * @param frame Image shown next to the scene, filled by the application; must outlive the viewer.
     * @param cameraLabels Label of each camera of the rig, in rig order; must not be empty.
     * @param trackingCamera Index of the camera that drives the tracking.
     * @param title Title of the window.
     */
    RigView(
        int argc,
        char** argv,
        sl::Mat* frame,
        const std::vector<std::string>& cameraLabels,
        size_t trackingCamera,
        std::string title = "ZED Positional Tracking - Multi Camera"
    );

    /**
     * @brief Releases the instance the GLUT callbacks forward to.
     */
    ~RigView();

    /**
     * @brief Tells whether landmarks are displayed (toggled with l).
     * @return True when they are, so the application retrieves them.
     */
    bool isLandmarkModeEnabled() const;

    /**
     * @brief Gives the camera whose image is shown (changed with n).
     * @return Its index in the rig.
     */
    size_t displayedCamera() const;

    /**
     * @brief Tells whether the window is still open.
     * @return False once the window is closed or esc is pressed.
     */
    bool isOpen() const;

    /**
     * @brief Draws one frame and handles the pending window events.
     */
    void render();

    /**
     * @brief Releases the OpenGL resources and marks the viewer as closed.
     */
    void close();

    /**
     * @brief Sets the pose of the rig and extends its path.
     * @param rigPose Pose of the rig (BASELINK) in the world frame.
     */
    void updateRigPose(const sl::Transform& rigPose);

    /**
     * @brief Sets the pose of one camera.
     * @param camera Index of the camera in the rig.
     * @param cameraPose Pose of the camera in the world frame.
     * @param state Tracking state of the camera.
     */
    void updateCameraPose(size_t camera, const sl::Transform& cameraPose, sl::POSITIONAL_TRACKING_STATE state);

    /**
     * @brief Sets the tracking status shown in the badges.
     * @param positionalTrackingStatus Status of the tracking of the rig.
     */
    void updatePositionalTrackingStatus(const sl::PositionalTrackingStatus& positionalTrackingStatus);

    /**
     * @brief Replaces the landmarks drawn.
     * @param landmarks The map of the rig, in the world frame.
     */
    void updateLandmarks(const std::map<uint64_t, sl::Landmark>& landmarks);

    /**
     * @brief Replaces the keyframes drawn.
     * @param keyframes The keyframes of each camera, in rig order, in the world frame.
     */
    void updateKeyframes(const std::vector<std::map<uint64_t, sl::KeyFrame>>& keyframes);

    /**
     * @brief Shows the Pause / Resume button, also toggled with k.
     */
    void enablePauseButton();

    /**
     * @brief Tells whether the replay is paused.
     * @return True while paused.
     */
    bool isPaused() const;

private:
    /**
     * @brief What the viewer draws of one camera.
     */
    struct CameraState {
        std::string label;
        sl::Transform pose;
        sl::POSITIONAL_TRACKING_STATE state = sl::POSITIONAL_TRACKING_STATE::OFF;
        sl::float4 color;
    };

    int _width;
    int _height;
    std::string _title;
    bool _isOpen;

    sl::Mat* _frame;
    sl::Transform _rigPose;
    std::vector<CameraState> _cameras;
    size_t _trackingCamera;
    size_t _displayedCamera;
    sl::PositionalTrackingStatus _positionalTrackingStatus;

    bool _darkMode;
    bool _sideBySideMode;
    bool _landmarkMode;
    bool _followMode;
    bool _pathMode;
    bool _pauseButtonEnabled = false;
    bool _paused = false;

    std::chrono::steady_clock::time_point _startTime;

    // Interaction
    bool _ctrlPressed = false;
    bool _mouseButtonPressed = false;
    int _mousePosition[2] = {0, 0};
    int _mouseDownPosition[2] = {0, 0};

    // Rendering
    GLCamera _camera;
    ShaderObj _shader;
    Simple3DObject _originAxes;
    Simple3DObject _rig;
    Simple3DPath _rigPath;
    Simple3DObject _landmarks;
    Simple3DObject _keyframes;
    std::vector<sl::float3> _frustumCorners;

    GLuint _frameTextureID;
    sl::Resolution _frameTextureSize;

    /**
     * @brief Creates the GLUT window and registers its callbacks.
     * @param argc Number of command line arguments.
     * @param argv The command line arguments.
     */
    void createWindow(int argc, char** argv);

    /**
     * @brief Creates the shader and the static geometry: origin axes and camera frustum.
     */
    void createScene();

    /**
     * @brief Draws the scene, then the overlays.
     */
    void display();

    /**
     * @brief Follows a window resize.
     * @param width New width in pixels.
     * @param height New height in pixels.
     */
    void reshape(int width, int height);

    /**
     * @brief Handles a key press.
     * @param key The key.
     * @param x Mouse position, unused.
     * @param y Mouse position, unused.
     */
    void keyboard(unsigned char key, int x, int y);

    /**
     * @brief Handles a mouse button: drag start, click on the pause button, zoom.
     * @param button The button.
     * @param state GLUT_DOWN or GLUT_UP.
     * @param x Mouse position in pixels, from the left.
     * @param y Mouse position in pixels, from the top.
     */
    void mouseButtonPressed(int button, int state, int x, int y);

    /**
     * @brief Pans, or rotates with ctrl, the view while dragging.
     * @param x Mouse position in pixels, from the left.
     * @param y Mouse position in pixels, from the top.
     */
    void mouseMotion(int x, int y);

    /**
     * @brief Rebuilds the rig geometry from the current poses: one frustum per camera and the arms joining them to the rig origin.
     */
    void buildRig();

    /**
     * @brief Adds one keyframe to the keyframe geometry, closed when it was loaded from an area file.
     * @param keyframe The keyframe.
     * @param color Color of the camera that took it.
     */
    void addKeyframe(const sl::KeyFrame& keyframe, const sl::float4& color);

    /**
     * @brief Gives the text color of the current theme.
     * @return The color.
     */
    sl::float4 textColor() const;

    /**
     * @brief Gives the panel background color of the current theme.
     * @return The color.
     */
    sl::float4 panelColor() const;

    /**
     * @brief Draws the overlays in the top left corner: status badges, pose card and camera list.
     */
    void renderOverlays();

    /**
     * @brief Draws one badge per tracking status.
     * @param top Top of the badges, in pixels from the bottom.
     * @return Bottom of the badges.
     */
    float renderStatusBadges(float top);

    /**
     * @brief Draws the up time and the pose of the rig.
     * @param top Top of the card, in pixels from the bottom.
     * @return Bottom of the card.
     */
    float renderPoseCard(float top);

    /**
     * @brief Draws one row per camera: its color, its label and its tracking state.
     * @param top Top of the list, in pixels from the bottom.
     */
    void renderCameraList(float top);

    /**
     * @brief Draws the Pause / Resume button.
     */
    void renderPauseButton();

    /**
     * @brief Gives the rectangle of the pause button, right of the pose card.
     * @return x, y, width, height in pixels, origin at the bottom left.
     */
    sl::float4 pauseButtonRect() const;

    /**
     * @brief Draws the image of the displayed camera with its label.
     */
    void renderImageFrame();

    /**
     * @brief Uploads an image to the frame texture, recreating the texture when the image size changes.
     * @param frame A BGRA image in CPU memory.
     */
    void uploadImageToTexture(const sl::Mat& frame);

    /**
     * @brief Draws a texture in a window rectangle.
     * @param textureID The texture.
     * @param x Left of the rectangle, in pixels.
     * @param y Bottom of the rectangle, in pixels.
     * @param width Width in pixels.
     * @param height Height in pixels.
     */
    void renderTexture(GLuint textureID, int x, int y, int width, int height);

    static inline RigView* _instance = nullptr;

    /// GLUT callbacks, forwarded to _instance.
    static void onDisplay();
    static void onReshape(int w, int h);
    static void onKeyboard(unsigned char key, int x, int y);
    static void onMouseButtonPressed(int button, int state, int x, int y);
    static void onMouseMotion(int x, int y);
    static void onClose();
};
