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

#include <functional>
#include "sl/Camera.hpp"
#include "GLCamera.hpp"
#include "Simple3DObject.hpp"
#include "Simple3DPath.hpp"
#include "Shader.hpp"
#include "PointCloud.hpp"

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

    bool isLandmarkModeEnabled() const;

    void run(std::function<void()> callback);
    void stop();

    void updatePoseTransform(sl::Transform transform);
    void updatePositionalTrackingStatus(sl::PositionalTrackingStatus positionalTrackingStatus);
    void updateLandmarks(std::map<uint64_t, sl::Landmark>& landmarks);
    void updateKeyframes(const std::map<uint64_t, sl::KeyFrame>& keyframes);

private:
    int _width;
    int _height;
    std::string _title;
    std::function<void()> _callback;

    sl::Mat* _frame;
    sl::Transform _poseTransform;
    sl::PositionalTrackingStatus _positionalTrackingStatus;

    bool _darkMode;
    bool _sideBySideMode;
    bool _pointCloudMode;
    bool _landmarkMode;
    bool _followMode;

    void idle();
    void display();
    void reshape(int width, int height);
    void keyboard(unsigned char key, int x, int y);
    void mouseButtonPressed(int button, int state, int x, int y);
    void mouseMotion(int x, int y);
    void close();

    // Interaction
    bool _ctrlPressed = false;
    bool _mouseButtonPressed = false;
    int _mousePosition[2] = {0, 0};

    // Rendering
    GLCamera _camera;
    ShaderObj _shader;
    Simple3DObject _originAxes;
    Simple3DObject _cameraFrustum;
    Simple3DPath _cameraPath;
    PointCloud _pointCloud;
    Simple3DObject _landmarks;
    Simple3DObject _keyframes;

    CUstream _cudaStream;

    GLuint _frameTextureID;

    void renderPositionalTrackingStatus();
    void renderImageFrame();

    std::string formatNumericText(sl::float3 value);
    void renderText(int x, int y, void* font, const sl::float4& color, const std::string& str);
    void uploadImageToTexture(const sl::Mat& frame, GLuint& textureID);
    void renderTexture(GLuint textureID, int x, int y, int width, int height);

    //
    // Static Forwarding
    //
    static inline SLAMView* _instance = nullptr;

    static void onIdle();
    static void onDisplay();
    static void onReshape(int w, int h);
    static void onKeyboard(unsigned char key, int x, int y);
    static void onMouseButtonPressed(int button, int state, int x, int y);
    static void onMouseMotion(int x, int y);
    static void onClose();
};
