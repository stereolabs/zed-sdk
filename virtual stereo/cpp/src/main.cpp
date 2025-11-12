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

/********************************************************************************
 ** This sample demonstrates how to grab images and change the camera settings **
 ** with the ZED SDK                                                           **
 ********************************************************************************/

// Standard includes
#include <stdio.h>
#include <string.h>

// ZED include
#include <sl/Camera.hpp>
#include <sl/CameraOne.hpp>

// OpenCV include (for display)
#include <opencv2/opencv.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

// Sample functions
void updateCameraSettings(char key, sl::Camera& zed);
void switchCameraSettings();
void printHelp();
void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
bool parseArgs(int argc, char** argv, sl::InitParameters& param);
void printDevices();

// Sample variables
VIDEO_SETTINGS camera_settings_ = VIDEO_SETTINGS::BRIGHTNESS;
string str_camera_settings = "BRIGHTNESS";
int step_camera_setting = 1;
bool led_on = true;

bool selectInProgress = false;
sl::Rect selection_rect;
cv::Point origin_rect;

static void onMouse(int event, int x, int y, int, void*) {
    switch (event) {
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
                // Reset selection
                selectInProgress = false;
                selection_rect = sl::Rect(0, 0, 0, 0);
                break;
            }
    }

    if (selectInProgress) {
        selection_rect.x = MIN(x, origin_rect.x);
        selection_rect.y = MIN(y, origin_rect.y);
        selection_rect.width = abs(x - origin_rect.x) + 1;
        selection_rect.height = abs(y - origin_rect.y) + 1;
    }
}

int main(int argc, char** argv) {

    // Create a ZED Camera object
    Camera zed;

    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.camera_resolution = sl::RESOLUTION::AUTO;
    init_parameters.depth_mode = sl::DEPTH_MODE::NONE; // no depth computation required here

    printDevices();

#if 0
    // Example with a virtual stereo rig from 2 ZED X One cameras
    // Serial numbers of the 2 ZED X One cameras
    unsigned int serial_left = 306179455;  // Left camera serial number
    unsigned int serial_right = 300967311; // Right camera serial number
    // use hash function to create a virtual serial number that follows the same rules as physical camera serial numbers
    // the virtual serial number must be unique to avoid conflicts with real camera serial numbers
    unsigned int virt_sn = generateVirtualStereoSerialNumber(serial_left, serial_right);
    #if 0
    // From the 2 ZED X One cameras serial numbers, defining the Left and Right cameras of the virtual stereo rig
    init_parameters.input.setVirtualStereoFromSerialNumbers(serial_left, serial_right, virt_sn);
    #else
    // Or from the GMSL IDs
    init_parameters.input.setVirtualStereoFromCameraIDs(1, 0, virt_sn);
    #endif
#else
    if (!parseArgs(argc, argv, init_parameters)) {
        std::cout << "Usage examples:" << std::endl;
        std::cout << "  - IP address: " << argv[0] << " 192.168.1.2" << std::endl;
        std::cout << "  - IP with port: " << argv[0] << " 192.168.1.2:30000" << std::endl;
        std::cout << "  - Camera IDs (and virtual SN): " << argv[0] << " 0 1 9999" << std::endl;
        std::cout << "  - Serial Numbers (and virtual SN): " << argv[0] << " 10012 10015 9999" << std::endl;
        return EXIT_FAILURE;
    }
#endif

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state == ERROR_CODE::INVALID_CALIBRATION_FILE) {
        std::cout << "WARNING: Virtual Stereo rig not calibrated, only unrectified images and recording are available" << std::endl;
    } else if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    cv::String win_name = "Camera Control LEFT";
    cv::String win_name2 = "Camera Control RIGHT";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::namedWindow(win_name2, cv::WINDOW_NORMAL);
    cv::setMouseCallback(win_name, onMouse);
    cv::setMouseCallback(win_name2, onMouse);

    // Print camera information
    auto camera_info = zed.getCameraInformation();
    auto camera_conf = camera_info.camera_configuration;
    cout << endl;
    cout << "ZED Model                 : " << camera_info.camera_model << endl;
    cout << "ZED Serial Number         : " << camera_info.serial_number << endl;
    cout << "ZED Camera Firmware       : " << camera_conf.firmware_version << "/" << camera_info.sensors_configuration.firmware_version
         << endl;
    cout << "ZED Camera Resolution     : " << camera_conf.resolution.width << "x" << camera_conf.resolution.height << endl;
    cout << "ZED Camera FPS            : " << zed.getInitParameters().camera_fps << endl;

    // Print help in console
    printHelp();

    // Create a Mat to store images
    Mat zed_image, zed_image2;

    // Initialise camera setting
    switchCameraSettings();

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();

        if (returned_state == ERROR_CODE::CORRUPTED_FRAME) {
            // Get the detailed health status
            auto health = zed.getHealthStatus();
            std::cout << "Health status: ";
            if (health.low_image_quality)
                std::cout << "Low image quality - ";
            if (health.low_lighting)
                std::cout << "Low lighting - ";
            if (health.low_depth_reliability)
                std::cout << "Low depth reliability - ";
            if (health.low_motion_sensors_reliability)
                std::cout << "Low motion sensors reliability - ";
            std::cout << std::endl;
        } else if (returned_state != ERROR_CODE::SUCCESS)
            std::cout << "returned_state " << returned_state << std::endl;
        int current_value = 10;
        zed.getCameraSettings(VIDEO_SETTINGS::EXPOSURE, current_value);
        if (returned_state <= ERROR_CODE::SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(zed_image, VIEW::LEFT_UNRECTIFIED);
            zed.retrieveImage(zed_image2, VIEW::RIGHT_UNRECTIFIED);

            // Convert sl::Mat to cv::Mat (share buffer)
            cv::Mat cvImage
                = cv::Mat((int)zed_image.getHeight(), (int)zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));
            cv::Mat cvImage2
                = cv::Mat((int)zed_image2.getHeight(), (int)zed_image2.getWidth(), CV_8UC4, zed_image2.getPtr<sl::uchar1>(sl::MEM::CPU));

            // Check that selection rectangle is valid and draw it on the image
            if (!selection_rect.isEmpty() && selection_rect.isContained(sl::Resolution(cvImage.cols, cvImage.rows)))
                cv::rectangle(
                    cvImage,
                    cv::Rect(selection_rect.x, selection_rect.y, selection_rect.width, selection_rect.height),
                    cv::Scalar(220, 180, 20),
                    2
                );

            // Display the image
            cv::imshow(win_name, cvImage);
            cv::imshow(win_name2, cvImage2);
        } else {
            print("Error during capture : ", returned_state);
            if (returned_state != sl::ERROR_CODE::CAMERA_REBOOTING)
                break;
        }

        key = cv::waitKey(10);
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
void updateCameraSettings(char key, sl::Camera& zed) {
    int current_value;

    // Keyboard shortcuts
    switch (key) {

            // Switch to the next camera parameter
        case 's':
            switchCameraSettings();
            zed.getCameraSettings(camera_settings_, current_value);
            std::cout << " Current Value : " << current_value << std::endl;
            break;

            // Increase camera settings value ('+' key)
        case '+':
            zed.getCameraSettings(camera_settings_, current_value);
            zed.setCameraSettings(camera_settings_, current_value + step_camera_setting);
            zed.getCameraSettings(camera_settings_, current_value);
            print(str_camera_settings + ": " + std::to_string(current_value));
            break;

            // Decrease camera settings value ('-' key)
        case '-':
            zed.getCameraSettings(camera_settings_, current_value);
            current_value = current_value > 0 ? current_value - step_camera_setting
                                              : 0; // take care of the 'default' value parameter:  VIDEO_SETTINGS_VALUE_AUTO
            zed.setCameraSettings(camera_settings_, current_value);
            zed.getCameraSettings(camera_settings_, current_value);
            print(str_camera_settings + ": " + std::to_string(current_value));
            break;

            // switch LED On :
        case 'l':
            led_on = !led_on;
            zed.setCameraSettings(sl::VIDEO_SETTINGS::LED_STATUS, led_on);
            break;

            // Reset to default parameters
        case 'r':
            print("Reset all settings to default\n");
            for (int s = (int)VIDEO_SETTINGS::BRIGHTNESS; s <= (int)VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE; s++)
                zed.setCameraSettings(static_cast<VIDEO_SETTINGS>(s), sl::VIDEO_SETTINGS_VALUE_AUTO);
            break;

        case 'a':
            {
                cout << "[Sample] set AEC_AGC_ROI on target [" << selection_rect.x << "," << selection_rect.y << "," << selection_rect.width
                     << "," << selection_rect.height << "]\n";
                zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI, selection_rect, sl::SIDE::BOTH);
            }
            break;

        case 'f':
            print("reset AEC_AGC_ROI to full res");
            zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI, selection_rect, sl::SIDE::BOTH, true);
            break;
    }
}

/**
    This function toggles between camera settings
 **/
void switchCameraSettings() {
    camera_settings_ = static_cast<VIDEO_SETTINGS>((int)camera_settings_ + 1);

    // reset to 1st setting
    if (camera_settings_ > VIDEO_SETTINGS::SCENE_ILLUMINANCE)
        camera_settings_ = VIDEO_SETTINGS::BRIGHTNESS;

    // increment if AEC_AGC_ROI since it using the overloaded function
    if (camera_settings_ == VIDEO_SETTINGS::AEC_AGC_ROI)
        camera_settings_ = static_cast<VIDEO_SETTINGS>((int)camera_settings_ + 1);

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
    cout << "* Toggle camera LED:               'l' (lower L)\n";
    cout << "* Reset all parameters:            'r'\n";
    cout << "* Reset exposure ROI to full image 'f'\n";
    cout << "* Use mouse to select an image area to apply exposure (press 'a')\n";
    cout << "* Exit :                           'q'\n\n";
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code > ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else if (err_code < ERROR_CODE::SUCCESS)
        cout << "[Warning] ";
    else
        cout << " ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

void printDevices() {
    auto devices = sl::CameraOne::getDeviceList();

    if (devices.empty()) {
        std::cout << "No ZED cameras found." << std::endl;
        return;
    }
    for (auto& it : devices) {
        if (it.camera_state == sl::CAMERA_STATE::AVAILABLE && isCameraOne(it.camera_model)) {
            std::cout << "- ZED One camera: " << it.camera_model << " (ID : " << it.id << ", Serial: " << it.serial_number << ")"
                      << std::endl;
        }
    }
}

// Helper function to check if a string contains only digits.
bool isNumber(const std::string& s) {
    return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
}

bool parseArgs(int argc, char** argv, sl::InitParameters& param) {
    // A constant to differentiate camera IDs from serial numbers.
    const unsigned int MIN_SERIAL_NUMBER = 20;

    if (argc <= 1) {
        std::cout << "Error: Insufficient arguments." << std::endl;
        return false;
    }

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    const std::string& first_arg = args[0];

    // Disallow SVO file input for this specific sample.
    if (first_arg.find(".svo") != std::string::npos) {
        std::cout << "Error: SVO input mode is not available for this sample." << std::endl;
        return false;
    }

    // --- 1. Attempt to parse as an IP address ---
    unsigned int a, b, c, d, port;
    if (sscanf(first_arg.c_str(), "%u.%u.%u.%u:%u", &a, &b, &c, &d, &port) == 5) {
        std::string ip_str = std::to_string(a) + "." + std::to_string(b) + "." + std::to_string(c) + "." + std::to_string(d);
        param.input.setFromStream(sl::String(ip_str.c_str()), port);
        std::cout << "[Sample] Using Stream input, IP: " << ip_str << ", Port: " << port << std::endl;
        return true;
    }

    if (sscanf(first_arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
        param.input.setFromStream(sl::String(first_arg.c_str()));
        std::cout << "[Sample] Using Stream input, IP: " << first_arg << std::endl;
        return true;
    }

    // --- 2. Attempt to parse as camera identifiers (ID or Serial Number) ---
    if (args.size() >= 2) {
        // Check if all arguments are valid numbers before converting.
        if (!isNumber(args[0]) || !isNumber(args[1]) || (args.size() > 2 && !isNumber(args[2]))) {
            std::cout << "Error: Non-numeric argument detected for camera/serial number." << std::endl;
            return false;
        }

        unsigned long num1 = std::stoul(args[0]);
        unsigned long num2 = std::stoul(args[1]);
        unsigned long num3 = (args.size() > 2) ? std::stoul(args[2]) : 0;
        if (num3 == 0) {
            std::cout << "Invalid Virtual Serial Number" << std::endl;
            return false;
        }

        const bool are_camera_ids = num1 < MIN_SERIAL_NUMBER && num2 < MIN_SERIAL_NUMBER;
        const bool are_serial_numbers = num1 >= MIN_SERIAL_NUMBER && num2 >= MIN_SERIAL_NUMBER;

        if (are_camera_ids) {
            std::cout << "[Sample] Using Virtual Stereo from Camera IDs: " << num1 << ", " << num2;
            if (num3 > 0)
                std::cout << " with virtual SN: " << num3;
            std::cout << std::endl;
            return !param.input.setVirtualStereoFromCameraIDs(num1, num2, num3);
        }

        if (are_serial_numbers) {
            std::cout << "[Sample] Using Virtual Stereo from Serial Numbers: " << num1 << ", " << num2;
            if (num3 > 0)
                std::cout << " with virtual SN: " << num3;
            std::cout << std::endl;
            return !param.input.setVirtualStereoFromSerialNumbers(num1, num2, num3);
        }

        // If numbers are a mix of ID and SN, it's an error.
        std::cout << "Error: Both numbers must be either camera IDs (<" << MIN_SERIAL_NUMBER
                  << ") or serial numbers (>=" << MIN_SERIAL_NUMBER << ")." << std::endl;
        return false;
    }
    return false;
}
