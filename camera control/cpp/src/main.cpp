///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
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
#include <opencv2/opencv.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

// Sample functions
void updateCameraSettings(char key, sl::Camera &zed);
void switchCameraSettings();
void printHelp();
void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv,sl::InitParameters& param);

// Sample variables
VIDEO_SETTINGS camera_settings_ = VIDEO_SETTINGS::BRIGHTNESS;
string str_camera_settings = "BRIGHTNESS";
int step_camera_setting = 1;
bool led_on = true;


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

int main(int argc, char **argv) {
    // Create a ZED Camera object
    Camera zed;
    
    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.camera_resolution= sl::RESOLUTION::HD720;
    init_parameters.depth_mode = sl::DEPTH_MODE::NONE; // no depth computation required here
    parseArgs(argc,argv, init_parameters);


    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }
    
    cv::String win_name = "Camera Control";
    cv::namedWindow(win_name);
    cv::setMouseCallback(win_name, onMouse);

    // Print camera information
    auto camera_info = zed.getCameraInformation();
    cout << endl;
    cout <<"ZED Model                 : "<<camera_info.camera_model << endl;
    cout <<"ZED Serial Number         : "<< camera_info.serial_number << endl;
    cout <<"ZED Camera Firmware       : "<< camera_info.camera_configuration.firmware_version <<"/"<<camera_info.sensors_configuration.firmware_version<< endl;
    cout <<"ZED Camera Resolution     : "<< camera_info.camera_configuration.resolution.width << "x" << camera_info.camera_configuration.resolution.height << endl;
    cout <<"ZED Camera FPS            : "<< zed.getInitParameters().camera_fps << endl;
    
    // Print help in console
    printHelp();

    // Create a Mat to store images
    Mat zed_image;

    // Initialise camera setting
    switchCameraSettings();

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();
        if (returned_state == ERROR_CODE::SUCCESS) {
            // Retrieve left image
            zed.retrieveImage(zed_image, VIEW::LEFT);

            // Convert sl::Mat to cv::Mat (share buffer)
            cv::Mat cvImage = cv::Mat((int) zed_image.getHeight(), (int) zed_image.getWidth(), CV_8UC4, zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));

            //Check that selection rectangle is valid and draw it on the image
            if (!selection_rect.isEmpty() && selection_rect.isContained(sl::Resolution(cvImage.cols, cvImage.rows)))
                cv::rectangle(cvImage, cv::Rect(selection_rect.x,selection_rect.y,selection_rect.width,selection_rect.height),cv::Scalar(220, 180, 20), 2);

            //Display the image
            cv::imshow(win_name, cvImage);
        }else {
            print("Error during capture : ", returned_state);
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
void updateCameraSettings(char key, sl::Camera &zed) {
    int current_value;

    // Keyboard shortcuts
    switch (key) {

        // Switch to the next camera parameter
        case 's':
        switchCameraSettings();
        current_value = zed.getCameraSettings(camera_settings_);
        break;

        // Increase camera settings value ('+' key)
        case '+':
		current_value = zed.getCameraSettings(camera_settings_);
		zed.setCameraSettings(camera_settings_, current_value + step_camera_setting);
        print(str_camera_settings+": "+to_string(zed.getCameraSettings(camera_settings_)));
        break;

        // Decrease camera settings value ('-' key)
        case '-':
		current_value = zed.getCameraSettings(camera_settings_);
        current_value = current_value > 0 ? current_value - step_camera_setting : 0; // take care of the 'default' value parameter:  VIDEO_SETTINGS_VALUE_AUTO
        zed.setCameraSettings(camera_settings_, current_value);
        print(str_camera_settings+": "+to_string(zed.getCameraSettings(camera_settings_)));
        break;

        //switch LED On :
        case 'l':
        led_on = !led_on;
        zed.setCameraSettings(sl::VIDEO_SETTINGS::LED_STATUS, led_on);
        break;

        // Reset to default parameters
        case 'r':
        print("Reset all settings to default\n");
        for(int s = (int)VIDEO_SETTINGS::BRIGHTNESS; s <= (int)VIDEO_SETTINGS::WHITEBALANCE_TEMPERATURE; s++)
            zed.setCameraSettings(static_cast<VIDEO_SETTINGS>(s), sl::VIDEO_SETTINGS_VALUE_AUTO);
        break;

        case 'a':
        {
        cout<<"[Sample] set AEC_AGC_ROI on target ["<<selection_rect.x<<","<<selection_rect.y<<","<<selection_rect.width<<","<<selection_rect.height<<"]\n";
        zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI, selection_rect, sl::SIDE::BOTH);
        }
        break;

        case 'f':
        print("reset AEC_AGC_ROI to full res");
        zed.setCameraSettings(VIDEO_SETTINGS::AEC_AGC_ROI,selection_rect,sl::SIDE::BOTH,true);
        break;

    }
}

/**
    This function toggles between camera settings
 **/
void switchCameraSettings() {
    camera_settings_ = static_cast<VIDEO_SETTINGS>((int)camera_settings_ + 1);

    // reset to 1st setting
    if (camera_settings_ == VIDEO_SETTINGS::LED_STATUS)
        camera_settings_ = VIDEO_SETTINGS::BRIGHTNESS;

    // increment if AEC_AGC_ROI since it using the overloaded function
    if (camera_settings_==VIDEO_SETTINGS::AEC_AGC_ROI)
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
    cout <<"[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

void parseArgs(int argc, char **argv,sl::InitParameters& param)
{
    if (argc > 1 && string(argv[1]).find(".svo")!=string::npos) {
        // SVO input mode not available in camera control
	cout<<"SVO Input mode is not available for camera control sample"<<endl;
    } else if (argc > 1 && string(argv[1]).find(".svo")==string::npos) {
        string arg = string(argv[1]);
        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()),port);
            cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<<endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout<<"[Sample] Using Stream input, IP : "<<argv[1]<<endl;
        }
        else if (arg.find("HD2K")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            cout<<"[Sample] Using Camera in resolution HD2K"<<endl;
        } else if (arg.find("HD1080")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            cout<<"[Sample] Using Camera in resolution HD1080"<<endl;
        } else if (arg.find("HD720")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            cout<<"[Sample] Using Camera in resolution HD720"<<endl;
        } else if (arg.find("VGA")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            cout<<"[Sample] Using Camera in resolution VGA"<<endl;
        }
    } else {
        // Default
    }
}
