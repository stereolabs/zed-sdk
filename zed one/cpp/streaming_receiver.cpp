///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
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

/*********************************************************************************
 ** This sample demonstrates how to capture and process the streaming video feed **
 ** provided by an application that uses the ZED SDK with streaming enabled.     **
 **********************************************************************************/

// ZED include
#include <sl/CameraOne.hpp>

// Sample includes
#include "utils.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

// Sample functions
void updateCameraSettings(char key, sl::CameraOne &zed);
void switchCameraSettings();
void printHelp();

// Sample variables
VIDEO_SETTINGS camera_settings_ = VIDEO_SETTINGS::BRIGHTNESS;
string str_camera_settings = "BRIGHTNESS";
int step_camera_setting = 1;

bool selectInProgress = false;
sl::Rect selection_rect;
cv::Point origin_rect;
static void onMouse(int event, int x, int y, int, void*)
{
    switch (event)
    {
        case cv::EVENT_LBUTTONDOWN:
        {
            origin_rect = cv::Point(x, y);
            selectInProgress = true;
            break;
        }

    case cv::EVENT_LBUTTONUP:
        {
            selectInProgress = false;
            break;
        }

    case cv::EVENT_RBUTTONDOWN:
        {
            //Reset selection
            selectInProgress = false;
            selection_rect = sl::Rect(0,0,0,0);
            break;
        }
    }

    if (selectInProgress)
    {
        selection_rect.x = MIN(x, origin_rect.x);
        selection_rect.y = MIN(y, origin_rect.y);
        selection_rect.width = abs(x - origin_rect.x) + 1;
        selection_rect.height = abs(y - origin_rect.y) + 1;
    }
}

vector< string> split(const string& s, char seperator) {
    vector< string> output;
    string::size_type prev_pos = 0, pos = 0;

    while ((pos = s.find(seperator, pos)) != string::npos) {
        string substring(s.substr(prev_pos, pos - prev_pos));
        output.push_back(substring);
        prev_pos = ++pos;
    }

    output.push_back(s.substr(prev_pos, pos - prev_pos));
    return output;
}

void setStreamParameter(InitParametersOne& init_p, string& argument) {
    vector< string> configStream = split(argument, ':');
    String ip(configStream.at(0).c_str());
    if (configStream.size() == 2) {
        init_p.input.setFromStream(ip, atoi(configStream.at(1).c_str()));
    } else init_p.input.setFromStream(ip);
}

int main(int argc, char **argv) {
    CameraOne zed;
    // Set configuration parameters for the ZED
    InitParametersOne init_parameters;
    init_parameters.sdk_verbose = true;

    string stream_params;
    if (argc > 1) {
        stream_params = string(argv[1]);
    } else {
        cout << "\nOpening the stream requires the IP of the sender\n";
        cout << "Usage : ./ZED_Streaming_Receiver IP:[port]\n";
        cout << "You can specify it now, then press ENTER, 'IP:[port]': ";
        cin >> stream_params;
    }

    setStreamParameter(init_parameters, stream_params);

    cv::String win_name = "Camera Remote Control";
    cv::namedWindow(win_name);
    cv::setMouseCallback(win_name, onMouse);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Print camera information
    auto camera_info = zed.getCameraInformation();
    cout << endl;
    cout << "ZED Model                 : " << camera_info.camera_model << endl;
    cout << "ZED Serial Number         : " << camera_info.serial_number << endl;
    cout << "ZED Camera Firmware       : " << camera_info.camera_configuration.firmware_version << "/" << camera_info.sensors_configuration.firmware_version << endl;
    cout << "ZED Camera Resolution     : " << camera_info.camera_configuration.resolution.width << "x" << camera_info.camera_configuration.resolution.height << endl;
    cout << "ZED Camera FPS            : " << zed.getInitParameters().camera_fps << endl;

    // Print help in console
    printHelp();

    // Create a Mat to store images
    Mat image;

    // Initialise camera setting
    switchCameraSettings();

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(image);

            // Convert sl::Mat to cv::Mat (share buffer)
            cv::Mat cvImage = slMat2cvMat(image);
            
            //Check that selection rectangle is valid and draw it on the image
            if (!selection_rect.isEmpty() && selection_rect.isContained(sl::Resolution(cvImage.cols, cvImage.rows)))
                cv::rectangle(cvImage, cv::Rect(selection_rect.x,selection_rect.y,selection_rect.width,selection_rect.height),cv::Scalar(0, 255, 0), 2);

            // Display image with OpenCV
            cv::imshow(win_name, cvImage);

        } else {
            print("Error during capture : ", returned_state);
            break;
        }

        key = cv::waitKey(5);
        // Change camera settings with keyboard
        updateCameraSettings(key, zed);
    }

    // Exit
    zed.close();
    return EXIT_SUCCESS;
}

/**
    This function updates camera settings
 **/
void updateCameraSettings(char key, sl::CameraOne &zed) {
    int current_value;

    // Keyboard shortcuts
    switch (key) {
            // Switch to the next camera parameter
            case 's':
            switchCameraSettings();
            zed.getCameraSettings(camera_settings_,current_value);
            break;

            // Increase camera settings value ('+' key)
            case '+':
            zed.getCameraSettings(camera_settings_,current_value);
            zed.setCameraSettings(camera_settings_, current_value + step_camera_setting);
            zed.getCameraSettings(camera_settings_,current_value);
            print(str_camera_settings+": "+std::to_string(current_value));
            break;

            // Decrease camera settings value ('-' key)
            case '-':
            zed.getCameraSettings(camera_settings_,current_value);
            current_value = current_value > 0 ? current_value - step_camera_setting : 0; // take care of the 'default' value parameter:  VIDEO_SETTINGS_VALUE_AUTO
            zed.setCameraSettings(camera_settings_, current_value);
            zed.getCameraSettings(camera_settings_,current_value);
            print(str_camera_settings+": "+std::to_string(current_value));
            break;

            // Reset to default parameters
        case 'r':
            print("Reset all settings to default");
            for (int s = (int) VIDEO_SETTINGS::BRIGHTNESS; s <= (int) VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE; s++)
                zed.setCameraSettings(static_cast<VIDEO_SETTINGS> (s), sl::VIDEO_SETTINGS_VALUE_AUTO);
            break;

        default :
        break;
        }
}

/**
    This function toggles between camera settings
 **/
void switchCameraSettings() {
    camera_settings_ = static_cast<VIDEO_SETTINGS> ((int) camera_settings_ + 1);

    // reset to 1st setting
    if (camera_settings_ == VIDEO_SETTINGS::LED_STATUS)
        camera_settings_ = VIDEO_SETTINGS::BRIGHTNESS;

    // select the right step
    step_camera_setting = (camera_settings_ == VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE) ? 100 : 1;

    // get the name of the selected SETTING
    str_camera_settings = string(sl::toString(camera_settings_).c_str());

    print("Switch to camera settings: ", ERROR_CODE::SUCCESS, str_camera_settings);
}

/**
    This function displays help
 **/
void printHelp() {
    cout << "\n\nCamera controls hotkeys:\n";
    cout << "* Increase camera settings value:  '+'\n";
    cout << "* Decrease camera settings value:  '-'\n";
    cout << "* Toggle camera settings:          's'\n";
    cout << "* Reset all parameters:            'r'\n";
    cout << "* Exit :                           'q'\n\n";
}
