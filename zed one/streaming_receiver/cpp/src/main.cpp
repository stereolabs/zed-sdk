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

/**
 * @file main.cpp
 * @brief ZED One Streaming Receiver Sample
 *
 * This sample demonstrates how to receive and display a video stream
 * from a remote ZED One camera with camera settings control.
 */

// ZED include
#include <sl/CameraOne.hpp>

// Sample includes
#include "utils.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

// Camera settings variables
VIDEO_SETTINGS camera_settings_ = VIDEO_SETTINGS::BRIGHTNESS;
string str_camera_settings = "BRIGHTNESS";
int step_camera_setting = 1;

// ROI selection variables
bool selectInProgress = false;
sl::Rect selection_rect;
cv::Point origin_rect;

// Function declarations
void updateCameraSettings(char key, sl::CameraOne& zed);
void switchCameraSettings();
void printHelp();

static void onMouse(int event, int x, int y, int, void*) {
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            origin_rect = cv::Point(x, y);
            selectInProgress = true;
            break;
        case cv::EVENT_LBUTTONUP:
            selectInProgress = false;
            break;
        case cv::EVENT_RBUTTONDOWN:
            selectInProgress = false;
            selection_rect = sl::Rect(0, 0, 0, 0);
            break;
    }

    if (selectInProgress) {
        selection_rect.x = MIN(x, origin_rect.x);
        selection_rect.y = MIN(y, origin_rect.y);
        selection_rect.width = abs(x - origin_rect.x) + 1;
        selection_rect.height = abs(y - origin_rect.y) + 1;
    }
}

vector<string> split(const string& s, char separator) {
    vector<string> output;
    string::size_type prev_pos = 0, pos = 0;
    while ((pos = s.find(separator, pos)) != string::npos) {
        output.push_back(s.substr(prev_pos, pos - prev_pos));
        prev_pos = ++pos;
    }
    output.push_back(s.substr(prev_pos, pos - prev_pos));
    return output;
}

void setStreamParameter(InitParametersOne& init_p, string& argument) {
    vector<string> configStream = split(argument, ':');
    String ip(configStream.at(0).c_str());
    if (configStream.size() == 2) {
        init_p.input.setFromStream(ip, atoi(configStream.at(1).c_str()));
    } else {
        init_p.input.setFromStream(ip);
    }
}

int main(int argc, char** argv) {
    CameraOne zed;
    InitParametersOne init_parameters;
    init_parameters.sdk_verbose = true;

    // Get stream address
    string stream_params;
    if (argc > 1) {
        stream_params = string(argv[1]);
    } else {
        cout << "\nOpening the stream requires the IP of the sender\n";
        cout << "Usage: ./ZED_One_Streaming_Receiver IP:[port]\n";
        cout << "Enter IP:[port]: ";
        cin >> stream_params;
    }

    setStreamParameter(init_parameters, stream_params);

    // Setup OpenCV window
    cv::String win_name = "ZED One - Remote Control";
    cv::namedWindow(win_name);
    cv::setMouseCallback(win_name, onMouse);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Print camera information
    auto camera_info = zed.getCameraInformation();
    cout << "\n=== Stream Connected ===" << endl;
    cout << "Model:      " << camera_info.camera_model << endl;
    cout << "Serial:     " << camera_info.serial_number << endl;
    cout << "Resolution: " << camera_info.camera_configuration.resolution.width << "x" << camera_info.camera_configuration.resolution.height
         << endl;
    cout << "FPS:        " << zed.getInitParameters().camera_fps << endl;
    cout << "========================\n" << endl;

    printHelp();

    // Create image mat
    Mat image;
    switchCameraSettings();

    // Capture loop
    char key = ' ';
    while (key != 'q') {
        returned_state = zed.grab();
        if (returned_state <= ERROR_CODE::SUCCESS) {
            zed.retrieveImage(image);
            cv::Mat cvImage = slMat2cvMat(image);

            // Draw selection rectangle if valid
            if (!selection_rect.isEmpty() && selection_rect.isContained(sl::Resolution(cvImage.cols, cvImage.rows))) {
                cv::rectangle(
                    cvImage,
                    cv::Rect(selection_rect.x, selection_rect.y, selection_rect.width, selection_rect.height),
                    cv::Scalar(0, 255, 0),
                    2
                );
            }

            cv::imshow(win_name, cvImage);
        } else {
            print("Error during capture", returned_state);
            break;
        }

        key = cv::waitKey(5);
        updateCameraSettings(key, zed);
    }

    zed.close();
    return EXIT_SUCCESS;
}

void updateCameraSettings(char key, sl::CameraOne& zed) {
    int current_value;

    switch (key) {
        case 's':
            switchCameraSettings();
            zed.getCameraSettings(camera_settings_, current_value);
            break;

        case '+':
            zed.getCameraSettings(camera_settings_, current_value);
            zed.setCameraSettings(camera_settings_, current_value + step_camera_setting);
            zed.getCameraSettings(camera_settings_, current_value);
            print(str_camera_settings + ": " + std::to_string(current_value));
            break;

        case '-':
            zed.getCameraSettings(camera_settings_, current_value);
            current_value = current_value > 0 ? current_value - step_camera_setting : 0;
            zed.setCameraSettings(camera_settings_, current_value);
            zed.getCameraSettings(camera_settings_, current_value);
            print(str_camera_settings + ": " + std::to_string(current_value));
            break;

        case 'r':
            print("Reset all settings to default");
            for (int s = (int)VIDEO_SETTINGS::BRIGHTNESS; s < (int)VIDEO_SETTINGS::LED_STATUS; s++) {
                zed.setCameraSettings(static_cast<VIDEO_SETTINGS>(s), sl::VIDEO_SETTINGS_VALUE_AUTO);
            }
            break;

        case 'a':
            if (!selection_rect.isEmpty()) {
                print("Set AEC_AGC_ROI");
                zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI, selection_rect);
            }
            break;

        case 'f':
            print("Reset AEC_AGC_ROI to full image");
            zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI, selection_rect, true);
            break;

        default:
            break;
    }
}

void switchCameraSettings() {
    camera_settings_ = static_cast<VIDEO_SETTINGS>((int)camera_settings_ + 1);

    if (camera_settings_ == VIDEO_SETTINGS::LED_STATUS)
        camera_settings_ = VIDEO_SETTINGS::BRIGHTNESS;

    step_camera_setting = (camera_settings_ == VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE) ? 100 : 1;
    str_camera_settings = string(sl::toString(camera_settings_).c_str());

    print("Switch to camera settings: ", ERROR_CODE::SUCCESS, str_camera_settings);
}

void printHelp() {
    cout << "Camera controls:\n";
    cout << "  +  Increase setting value\n";
    cout << "  -  Decrease setting value\n";
    cout << "  s  Switch camera setting\n";
    cout << "  r  Reset all settings\n";
    cout << "  a  Apply exposure ROI\n";
    cout << "  f  Reset exposure ROI\n";
    cout << "  q  Quit\n\n";
}
