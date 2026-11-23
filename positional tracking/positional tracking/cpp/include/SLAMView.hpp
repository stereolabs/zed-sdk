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

#include <chrono>
#include <functional>
#include "sl/Camera.hpp"
#include "GLCamera.hpp"
#include "Simple3DObject.hpp"
#include "Simple3DPath.hpp"
#include "Shader.hpp"
#include "PointCloud.hpp"
#include "MotionPriorView.hpp"

//
// SLAMView
//
class SLAMView {
public:
    SLAMView(
        int argc,
        char** argv,
        sl::Mat* frame,
        sl::Mat* pointCloud,
        CUstream cudaStream,
        std::string title = "ZED Positional Tracking"
    );
    ~SLAMView();

    bool isLandmarkModeEnabled() const;

    bool isOpen() const;
    void render();
    void close();

    void updatePoseTransform(sl::Transform transform);
    // Call after updatePoseTransform() for the same image: the motion prior view pairs the two.
    void updatePositionalTrackingStatus(sl::PositionalTrackingStatus positionalTrackingStatus, sl::Timestamp imageTimestamp);
    void updateLandmarks(std::map<uint64_t, sl::Landmark>& landmarks);
    void updateKeyframes(const std::map<uint64_t, sl::KeyFrame>& keyframes);

    // Motion prior: the samples given to the tracking and the keyframe links it reports (see MotionPriorView).
    void enableMotionPrior(const std::string& source, const sl::Transform& sensorTransform);
    void clearMotionPrior();
    void addMotionPriorSample(const sl::RelativePoseData& sample, uint64_t arrivalTimestamp);
    void updateMotionPriorLinks(const std::map<uint64_t, sl::MotionPriorLink>& links);

    // SVO input only: a Pause / Resume button next to the pose card, also toggled with the k key. While
    // paused the application stops reading images and keeps calling render(), so the view stays live.
    void enablePauseButton();
    bool isPaused() const;

private:
    int _width;
    int _height;
    std::string _title;
    bool _isOpen;

    sl::Mat* _frame;
    sl::Transform _poseTransform;
    sl::PositionalTrackingStatus _positionalTrackingStatus;

    bool _darkMode;
    bool _sideBySideMode;
    bool _pointCloudMode;
    bool _landmarkMode;
    bool _followMode;
    bool _pathMode; // hidden while the motion prior is drawn, so the arrows on it stay readable
    bool _pauseButtonEnabled = false;
    bool _paused = false;

    std::chrono::steady_clock::time_point _startTime;

    void display();
    void reshape(int width, int height);
    void keyboard(unsigned char key, int x, int y);
    void mouseButtonPressed(int button, int state, int x, int y);
    void mouseMotion(int x, int y);

    // Interaction
    bool _ctrlPressed = false;
    bool _mouseButtonPressed = false;
    int _mousePosition[2] = {0, 0};
    int _mouseDownPosition[2] = {0, 0}; // a release this close to it is a click, not a drag

    // Rendering
    GLCamera _camera;
    ShaderObj _shader;
    Simple3DObject _originAxes;
    Simple3DObject _cameraFrustum;
    Simple3DPath _cameraPath;
    PointCloud _pointCloud;
    Simple3DObject _landmarks;
    Simple3DObject _keyframes;
    MotionPriorView _motionPrior;

    CUstream _cudaStream;

    GLuint _frameTextureID;

    void renderPositionalTrackingStatus();
    void renderPauseButton();
    // x, y, width, height in window pixels, origin at the bottom left. Right of the pose card.
    sl::float4 pauseButtonRect() const;
    void renderImageFrame();

    void uploadImageToTexture(const sl::Mat& frame, GLuint& textureID);
    void renderTexture(GLuint textureID, int x, int y, int width, int height);

    //
    // Static Forwarding
    //
    static inline SLAMView* _instance = nullptr;

    static void onDisplay();
    static void onReshape(int w, int h);
    static void onKeyboard(unsigned char key, int x, int y);
    static void onMouseButtonPressed(int button, int state, int x, int y);
    static void onMouseMotion(int x, int y);
    static void onClose();
};
