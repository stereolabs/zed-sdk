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
 ** This sample demonstrates how to capture 3D point cloud and detected objects **
 **      with the ZED SDK and display the result in an OpenGL window. 	        **
 *********************************************************************************/

// Standard includes
#include <iostream>
#include <fstream>

// Flag to disable the GUI to increase detection performances
// On low-end hardware such as Jetson Nano, the GUI significantly slows
// down the detection and increase the memory consumption
#define ENABLE_GUI 1

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#if ENABLE_GUI
#include "GLViewer.hpp"
#include "TrackingViewer.hpp"
#endif

// Using std and sl namespaces
using namespace std;
using namespace sl;
bool is_playback = false;
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
    init_parameters.camera_resolution = RESOLUTION::HD2K;
	init_parameters.sdk_verbose = true;
    // On Jetson (Nano, TX1/2) the object detection combined with an heavy depth mode could reduce the frame rate too much
    init_parameters.depth_mode = isJetson ? DEPTH_MODE::PERFORMANCE : DEPTH_MODE::ULTRA;
    init_parameters.depth_maximum_distance = 50.0f * 1000.0f;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    parseArgs(argc, argv, init_parameters);
    
    // Open the camera
    auto returned_state  = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Define the Objects detection module parameters
    ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_mask_output = false; // designed to give person pixel mask
    detection_parameters.detection_model = isJetson ? DETECTION_MODEL::MULTI_CLASS_BOX : DETECTION_MODEL::MULTI_CLASS_BOX_ACCURATE;

    auto camera_config = zed.getCameraInformation().camera_configuration;

    PositionalTrackingParameters positional_tracking_parameters;
    // If the camera is static in space, enabling this settings below provides better depth quality and faster computation
    // positional_tracking_parameters.set_as_static = true;
    zed.enablePositionalTracking(positional_tracking_parameters);

    print("Object Detection: Loading Module...");
    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Detection runtime parameters
    // default detection threshold, apply to all object class
    int detection_confidence = 35;
    ObjectDetectionRuntimeParameters detection_parameters_rt(detection_confidence);
    // To select a set of specific object classes:
    detection_parameters_rt.object_class_filter = {OBJECT_CLASS::VEHICLE, OBJECT_CLASS::PERSON};
    // To set a specific threshold
    detection_parameters_rt.object_class_detection_confidence_threshold[OBJECT_CLASS::PERSON] = 35;
    detection_parameters_rt.object_class_detection_confidence_threshold[OBJECT_CLASS::VEHICLE] = 35;

    // Detection output
    Objects objects;
    bool quit = false;

#if ENABLE_GUI
    Resolution display_resolution(min((int)camera_config.resolution.width, 1280) , min((int)camera_config.resolution.height, 720));
    Mat image_left(display_resolution, MAT_TYPE::U8_C4);
    sl::float2 img_scale(display_resolution.width / (float)camera_config.resolution.width, display_resolution.height / (float)camera_config.resolution.height);

    // 2D tracks
    TrackingViewer track_view_generator;
    // With OpenGL coordinate system, Y is the vertical axis, and negative z values correspond to objects in front of the camera
    track_view_generator.setZMin(-1.0f * zed.getInitParameters().depth_maximum_distance);
    track_view_generator.setFPS(camera_config.fps);
    track_view_generator.configureFromFPS();
    track_view_generator.setCameraCalibration(camera_config.calibration_parameters);
    cv::Mat track_view(track_view_generator.getWindowHeight(), track_view_generator.getWindowWidth(), CV_8UC3, cv::Scalar::all(0));

    string window_left_name = "Left";
    string window_birdview_name = "Bird view";
    if (detection_parameters.enable_tracking) window_birdview_name = "Tracks";
    cv::namedWindow(window_left_name, cv::WINDOW_AUTOSIZE); // Create Window
    cv::createTrackbar("Confidence", window_left_name, &detection_confidence, 100);

    char key = ' ';
    Resolution pc_resolution(min((int)camera_config.resolution.width, 720) , min((int)camera_config.resolution.height, 404));
    auto camera_parameters = zed.getCameraInformation(pc_resolution).camera_configuration.calibration_parameters.left_cam;
    Mat point_cloud(pc_resolution, MAT_TYPE::F32_C4, MEM::GPU);
    GLViewer viewer;
    viewer.init(argc, argv, camera_parameters);
#endif

    RuntimeParameters runtime_parameters;
    runtime_parameters.confidence_threshold = 50;
        
    Pose cam_pose;
    cam_pose.pose_data.setIdentity();
    bool gl_viewer_available=true;
    while (
#if ENABLE_GUI
            gl_viewer_available &&
#endif
            !quit && zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {
        detection_parameters_rt.detection_confidence_threshold = detection_confidence;
        returned_state = zed.retrieveObjects(objects, detection_parameters_rt);

        if ((returned_state == ERROR_CODE::SUCCESS) && objects.is_new) {
#if ENABLE_GUI
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, pc_resolution);
            zed.getPosition(cam_pose, REFERENCE_FRAME::WORLD);
            viewer.updateData(point_cloud, objects.object_list, cam_pose.pose_data);
            gl_viewer_available = viewer.isAvailable();

            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
            render_2D(image_left, img_scale, objects.object_list, true);
            zed.getPosition(cam_pose, REFERENCE_FRAME::CAMERA);
            track_view_generator.generate_view(objects, cam_pose, track_view, objects.is_tracked);
#else
            cout << "Detected " << objects.object_list.size() << " Object(s)" << endl;
#endif
        }

        if (is_playback && zed.getSVOPosition() == zed.getSVONumberOfFrames()) {
            quit = true;
        }

#if ENABLE_GUI
        cv::Mat left_display = slMat2cvMat(image_left);
        cv::imshow(window_left_name, left_display);
        cv::imshow(window_birdview_name, track_view);
        key = cv::waitKey(33);
        if (key == 'i') {
            track_view_generator.zoomIn();
        } else if (key == 'o') {
            track_view_generator.zoomOut();
        } else if (key == 'q') {
            quit = true;
        } else if (key == 'a') {
            detection_parameters_rt.object_class_filter.clear();
            detection_parameters_rt.object_class_filter.push_back(OBJECT_CLASS::PERSON);
            cout << "Person only" << endl;
        } else if (key == 'z') {
            detection_parameters_rt.object_class_filter.clear();
            detection_parameters_rt.object_class_filter.push_back(OBJECT_CLASS::VEHICLE);
            cout << "Vehicle only" << endl;
        } else if (key == 'e') {
            detection_parameters_rt.object_class_filter.clear();
            cout << "No filter" << endl;
        }
        
#endif
    }
#if ENABLE_GUI
    viewer.exit();
    point_cloud.free();
    image_left.free();
#endif
    zed.disableObjectDetection();
    zed.close();
    return EXIT_SUCCESS;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample] ";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}

void parseArgs(int argc, char **argv, InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        is_playback = true;
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
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        } else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    }
}
