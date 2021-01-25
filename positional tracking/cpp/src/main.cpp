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

/*************************************************************************
 ** This sample demonstrates how to use the ZED for positional tracking  **
 ** and display camera motion in an OpenGL window. 		                **
 **************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std namespace
using namespace std;
using namespace sl;

#define IMU_ONLY 0
const int MAX_CHAR = 128;

inline void setTxt(sl::float3 value, char* ptr_txt) {
    snprintf(ptr_txt, MAX_CHAR, "%3.2f; %3.2f; %3.2f", value.x, value.y, value.z);
}

void parseArgs(int argc, char **argv, sl::InitParameters& param);

int main(int argc, char **argv) {

    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters init_parameters;
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_parameters.sdk_verbose = true;
    parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    auto camera_model = zed.getCameraInformation().camera_model;
    GLViewer viewer;
    // Initialize OpenGL viewer
    viewer.init(argc, argv, camera_model);

    // Create text for GUI
    char text_rotation[MAX_CHAR];
    char text_translation[MAX_CHAR];

    // Set parameters for Positional Tracking
    PositionalTrackingParameters positional_tracking_param;
    positional_tracking_param.enable_area_memory = true;
    // enable Positional Tracking
    returned_state = zed.enablePositionalTracking(positional_tracking_param);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Enabling positionnal tracking failed: ", returned_state);
        zed.close();
        return EXIT_FAILURE;
    }

    Pose camera_path;
    POSITIONAL_TRACKING_STATE tracking_state;
#if IMU_ONLY
    SensorsData sensors_data;
#endif
    
    while (viewer.isAvailable()) {
        if (zed.grab() == ERROR_CODE::SUCCESS) {
            // Get the position of the camera in a fixed reference frame (the World Frame)
            tracking_state = zed.getPosition(camera_path, REFERENCE_FRAME::WORLD);

#if IMU_ONLY
            if (zed.getSensorsData(sensors_data, TIME_REFERENCE::IMAGE) == sl::ERROR_CODE::SUCCESS) {
                setTxt(sensors_data.imu.pose.getEulerAngles(), text_rotation); //only rotation is computed for IMU
                viewer.updateData(sensors_data.imu.pose, string(text_translation), string(text_rotation), sl::POSITIONAL_TRACKING_STATE::OK);
            }
#else
            if (tracking_state == POSITIONAL_TRACKING_STATE::OK) {
                // Get rotation and translation and displays it
                setTxt(camera_path.getEulerAngles(), text_rotation);
                setTxt(camera_path.getTranslation(), text_translation);
            }

            // Update rotation, translation and tracking state values in the OpenGL window
            viewer.updateData(camera_path.pose_data, string(text_translation), string(text_rotation), tracking_state);
#endif

        } else
            sleep_ms(1);
    }

    zed.disablePositionalTracking();

    //zed.disableRecording();
    zed.close();
    return EXIT_SUCCESS;
}

void parseArgs(int argc, char **argv, sl::InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        } else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        } else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    } else {
        // Default
    }
}

