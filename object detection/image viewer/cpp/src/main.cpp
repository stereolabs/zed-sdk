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

/*****************************************************************************************
 ** This sample demonstrates how to detect objects and retrieve their 3D position       **
 **         with the ZED SDK and display the result in an OpenGL window.                **
 *****************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
void parseArgs(int argc, char **argv, InitParameters& param);

int main(int argc, char **argv) {

#ifdef _SL_JETSON_
    const bool isJetson = true;
#else
    const bool isJetson = false;
#endif

    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    init_parameters.coordinate_units = UNIT::METER;
    parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable Positional tracking (mandatory for object detection)
    PositionalTrackingParameters positional_tracking_parameters;
    //If the camera is static, uncomment the following line to have better performances.
    //positional_tracking_parameters.set_as_static = true;
    returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enable Positional Tracking", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable the Objects detection module
    ObjectDetectionParameters obj_det_params;
    obj_det_params.enable_tracking = true;
    obj_det_params.detection_model = isJetson ? OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_FAST : OBJECT_DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE;

    returned_state = zed.enableObjectDetection(obj_det_params);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enable Object Detection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    auto camera_info = zed.getCameraInformation().camera_configuration;
    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv, camera_info.calibration_parameters.left_cam, obj_det_params.enable_tracking = true);

    // Configure object detection runtime parameters
    ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    int detection_confidence = 60;
    objectTracker_parameters_rt.detection_confidence_threshold = detection_confidence;
    // To select a set of specific object classes, like persons, vehicles and animals for instance:
    objectTracker_parameters_rt.object_class_filter = {OBJECT_CLASS::PERSON, OBJECT_CLASS::VEHICLE, OBJECT_CLASS::ANIMAL};
    // To set a specific threshold
    objectTracker_parameters_rt.object_class_detection_confidence_threshold[OBJECT_CLASS::PERSON] = detection_confidence;
    //detection_parameters_rt.object_class_detection_confidence_threshold[OBJECT_CLASS::CAR] = detection_confidence;

    // Create ZED Objects filled in the main loop
    Objects objects;
    Mat image;

    // Main Loop
    while (viewer.isAvailable()) {
        // Grab images
        if (zed.grab() == ERROR_CODE::SUCCESS) {

            // Retrieve left image for display
            zed.retrieveImage(image, VIEW::LEFT, MEM::GPU);

            // Retrieve Detected Objects
            zed.retrieveObjects(objects, objectTracker_parameters_rt);

            //Update rendering, image with detected object overlay
            viewer.updateView(image, objects);
        }
    }

    // Release objects
    image.free();
    objects.object_list.clear();

    // Disable modules
    zed.disableObjectDetection();
    zed.disablePositionalTracking();
    zed.close();
    return EXIT_SUCCESS;
}

void parseArgs(int argc, char **argv, InitParameters& param) {
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
            param.input.setFromStream(String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        } else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        } else if (arg.find("HD1200") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1200;
            cout << "[Sample] Using Camera in resolution HD1200" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        } else if (arg.find("SVGA") != string::npos) {
            param.camera_resolution = RESOLUTION::SVGA;
            cout << "[Sample] Using Camera in resolution SVGA" << endl;
        } else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    }
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error]";
    cout << " " << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
