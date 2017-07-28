///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
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


/********************************************************************************
 ** This sample demonstrates how to grab images and change the camera settings **
 ** with the ZED SDK                                                           **
 ********************************************************************************/


 // Standard includes
#include <stdio.h>
#include <string.h>

// ZED include
#include <sl/Camera.hpp>

// OpenCV include (for display)
#include "opencv2/opencv.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

// Sample functions
void updateCameraSettings(char key, sl::Camera &zed);
void switchCameraSettings();
void printHelp();

// Sample variables
CAMERA_SETTINGS camera_settings_ = CAMERA_SETTINGS_BRIGHTNESS;
string str_camera_settings = "BRIGHTNESS";
int step_camera_setting = 1;


int main(int argc, char **argv) {

    // Create a ZED Camera object
    Camera zed;

    // Open the camera
    ERROR_CODE err = zed.open();
    if (err != SUCCESS) {
        cout << errorCode2str(err) << endl;
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Print help in console
    printHelp();

    // Print camera information
    printf("ZED Serial Number         : %d\n", zed.getCameraInformation().serial_number);
    printf("ZED Firmware              : %d\n", zed.getCameraInformation().firmware_version);
    printf("ZED Camera Resolution     : %dx%d\n", (int) zed.getResolution().width, (int) zed.getResolution().height);
    printf("ZED Camera FPS            : %d\n", (int) zed.getCameraFPS());

    // Create a Mat to store images
    Mat zed_image;

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {

        // Check that grab() is successful
        if (zed.grab() == SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(zed_image, VIEW_LEFT);

            // Display image with OpenCV
            cv::imshow("VIEW", cv::Mat((int) zed_image.getHeight(), (int) zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM_CPU)));
            key = cv::waitKey(5);

            // Change camera settings with keyboard
            updateCameraSettings(key, zed);
        } else
            key = cv::waitKey(5);
    }

    // Exit
    zed.close();
    return EXIT_SUCCESS;
}

/**
    This function updates camera settings
 **/
void updateCameraSettings(char key, sl::Camera &zed) {
    int current_value;

    // Keyboard shortcuts
    switch (key) {

        // Switch to the next camera parameter
        case 's':
        switchCameraSettings();
        break;

        // Increase camera settings value ('+' key)
        case '+':
        current_value = zed.getCameraSettings(camera_settings_);
        zed.setCameraSettings(camera_settings_, current_value + step_camera_setting);
        std::cout << str_camera_settings << ": " << current_value + step_camera_setting << std::endl;
        break;

        // Decrease camera settings value ('-' key)
        case '-':
        current_value = zed.getCameraSettings(camera_settings_);
        if (current_value >= 1) {
            zed.setCameraSettings(camera_settings_, current_value - step_camera_setting);
            std::cout << str_camera_settings << ": " << current_value - step_camera_setting << std::endl;
        }
        break;

        // Reset to default parameters
        case 'r':
        std::cout << "Reset all settings to default" << std::endl;
        zed.setCameraSettings(sl::CAMERA_SETTINGS_BRIGHTNESS, -1, true);
        zed.setCameraSettings(sl::CAMERA_SETTINGS_CONTRAST, -1, true);
        zed.setCameraSettings(sl::CAMERA_SETTINGS_HUE, -1, true);
        zed.setCameraSettings(sl::CAMERA_SETTINGS_SATURATION, -1, true);
        zed.setCameraSettings(sl::CAMERA_SETTINGS_GAIN, -1, true);
        zed.setCameraSettings(sl::CAMERA_SETTINGS_EXPOSURE, -1, true);
        zed.setCameraSettings(sl::CAMERA_SETTINGS_WHITEBALANCE, -1, true);
        break;
    }
}

/**
    This function toggles between camera settings
 **/
void switchCameraSettings() {
    step_camera_setting = 1;
    switch (camera_settings_) {
        case CAMERA_SETTINGS_BRIGHTNESS:
        camera_settings_ = CAMERA_SETTINGS_CONTRAST;
        str_camera_settings = "Contrast";
        std::cout << "Camera Settings: CONTRAST" << std::endl;
        break;

        case CAMERA_SETTINGS_CONTRAST:
        camera_settings_ = CAMERA_SETTINGS_HUE;
        str_camera_settings = "Hue";
        std::cout << "Camera Settings: HUE" << std::endl;
        break;

        case CAMERA_SETTINGS_HUE:
        camera_settings_ = CAMERA_SETTINGS_SATURATION;
        str_camera_settings = "Saturation";
        std::cout << "Camera Settings: SATURATION" << std::endl;
        break;

        case CAMERA_SETTINGS_SATURATION:
        camera_settings_ = CAMERA_SETTINGS_GAIN;
        str_camera_settings = "Gain";
        std::cout << "Camera Settings: GAIN" << std::endl;
        break;

        case CAMERA_SETTINGS_GAIN:
        camera_settings_ = CAMERA_SETTINGS_EXPOSURE;
        str_camera_settings = "Exposure";
        std::cout << "Camera Settings: EXPOSURE" << std::endl;
        break;

        case CAMERA_SETTINGS_EXPOSURE:
        camera_settings_ = CAMERA_SETTINGS_WHITEBALANCE;
        str_camera_settings = "White Balance";
        step_camera_setting = 100;
        std::cout << "Camera Settings: WHITE BALANCE" << std::endl;
        break;

        case CAMERA_SETTINGS_WHITEBALANCE:
        camera_settings_ = CAMERA_SETTINGS_BRIGHTNESS;
        str_camera_settings = "Brightness";
        std::cout << "Camera Settings: BRIGHTNESS" << std::endl;
        break;
    }
}

/**
    This function displays help
 **/
void printHelp() {
    cout << endl;
    cout << endl;
    cout << "Camera controls hotkeys: " << endl;
    cout << "  Increase camera settings value:            '+'" << endl;
    cout << "  Decrease camera settings value:            '-'" << endl;
    cout << "  Toggle camera settings:                    's'" << endl;
    cout << "  Reset all parameters:                      'r'" << endl;
    cout << endl;
    cout << "Exit : 'q'" << endl;
    cout << endl;
    cout << endl;
    cout << endl;
}
