///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
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
 ** This sample demonstrates how to capture 3D point cloud and detected objects  **
 **      with the ZED SDK and display the result in an OpenGL window. 	        **
 **********************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
bool checkIsJetson();
void parseArgs(int argc, char **argv, sl::InitParameters& param);

int main(int argc, char **argv) {
    // Create ZED objects
    Camera zed;

    // Configure init parameters
    InitParameters initParameters;
    bool isJetson = checkIsJetson();
    // On Jetson (Nano, TX1/2) the object detection combined with an heavy depth mode could reduce the frame rate too much
    initParameters.depth_mode = isJetson ? DEPTH_MODE::PERFORMANCE : DEPTH_MODE::ULTRA;
    initParameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    initParameters.coordinate_units = sl::UNIT::METER;
    initParameters.depth_maximum_distance = 15.f;
#if (ZED_SDK_MAJOR_VERSION*10+ZED_SDK_MINOR_VERSION > 31)
    initParameters.camera_image_flip = sl::FLIP_MODE::AUTO; // 3.2 and ZED2 --> detect automatically the flip mode
#else
    initParameters.camera_image_flip = false;
#endif
    parseArgs(argc, argv, initParameters);

    // Open the camera
    ERROR_CODE zed_error = zed.open(initParameters);
    if (zed_error != ERROR_CODE::SUCCESS) {
        std::cout << sl::toVerbose(zed_error) << "\nExit program." << std::endl;
        zed.close();
        return 1; // Quit if an error occurred
    }

    auto camera_info = zed.getCameraInformation();

    // Security : Only ZED2 has object detection
    if (camera_info.camera_model != sl::MODEL::ZED2) {
        std::cout << " ERROR : Use ZED2 Camera only" << std::endl;
        exit(0);
    }

    // Enable Positional tracking (mandatory for object detection)
    sl::PositionalTrackingParameters trc_params;
    //If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    //trc_params.set_as_static = true;
    zed_error = zed.enablePositionalTracking(trc_params);
    if (zed_error != ERROR_CODE::SUCCESS) {
        std::cout << sl::toVerbose(zed_error) << "\nExit program." << std::endl;
        zed.close();
        return 1;
    }

    // Enable the Objects detection module
    sl::ObjectDetectionParameters obj_det_params;
    obj_det_params.image_sync = true;
    obj_det_params.enable_tracking = true;
    obj_det_params.enable_mask_output = false;

#if (ZED_SDK_MAJOR_VERSION*10+ZED_SDK_MINOR_VERSION > 31)
    obj_det_params.detection_model = isJetson ? DETECTION_MODEL::HUMAN_BODY_FAST : DETECTION_MODEL::HUMAN_BODY_ACCURATE;
#endif

    zed_error = zed.enableObjectDetection(obj_det_params);
    if (zed_error != ERROR_CODE::SUCCESS) {
        std::cout << sl::toVerbose(zed_error) << "\nExit program." << std::endl;
        zed.close();
        return 1;
    }

    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv, camera_info.calibration_parameters.left_cam);

    // Configure object detection runtime parameters
    ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    objectTracker_parameters_rt.detection_confidence_threshold = 50;
    objectTracker_parameters_rt.object_class_filter.clear();
    objectTracker_parameters_rt.object_class_filter.push_back(sl::OBJECT_CLASS::PERSON);

    // Create ZED Objects filled in the main loop
    Objects objects;
    Mat pImage;

    sl::Plane floor_plane; // floor plane handle
    sl::Transform reset_from_floor_plane; // camera transform once floor plane is detected

    // Main Loop
    bool need_floor_plane = zed.getPositionalTrackingParameters().set_as_static;
    while (viewer.isAvailable()) {
        // Grab images
        sl::ERROR_CODE zed_error = zed.grab();
        if (zed_error == sl::ERROR_CODE::SUCCESS) {

            // Once the camera has started, get the floor plane to stick the bounding box to the floor plane.
            // Only called if camera is static (see sl::PositionalTrackingParameters)
            if (need_floor_plane) {
                if (zed.findFloorPlane(floor_plane, reset_from_floor_plane) == sl::ERROR_CODE::SUCCESS) {
                    need_floor_plane = false;
                    viewer.setFloorPlaneEquation(floor_plane.getPlaneEquation());
                }
            }

            // Retrieve left image
            zed.retrieveImage(pImage, VIEW::LEFT, MEM::GPU);

            // Retrieve Objects
            zed.retrieveObjects(objects, objectTracker_parameters_rt);

            //Update GL View
            viewer.updateView(pImage, objects);
        } else
            sleep_us(100);

    }


    // Release objects
    pImage.free();
    floor_plane.clear();
    objects.object_list.clear();

    // Disable modules
    zed.disablePositionalTracking();
    zed.disableObjectDetection();
    zed.close();
    return 0;
}

bool checkIsJetson() {
    bool isJetson = false;
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int CC = (prop.major * 10 + prop.minor);
    isJetson = ((CC == 62)/* TX2 */ || (CC == 35)/*  TX1 */ || (CC == 53) /* Nano */);
    return isJetson;
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
        //
    }
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
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
