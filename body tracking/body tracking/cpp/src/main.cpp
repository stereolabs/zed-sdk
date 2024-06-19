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
 ** This sample demonstrates how to detect human bodies and retrieves their 3D position **
 **         with the ZED SDK and display the result in an OpenGL window.                **
 *****************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"
#include "TrackingViewer.hpp"


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

    // Create ZED Bodies
    Camera zed;
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::AUTO;
    init_parameters.depth_mode = isJetson ? DEPTH_MODE::PERFORMANCE : DEPTH_MODE::ULTRA;
    //init_parameters.depth_mode = DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;

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
    //If the camera is static, uncomment the following line to have better performances
    //positional_tracking_parameters.set_as_static = true;

    returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enable Positional Tracking", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable the Body tracking module
    BodyTrackingParameters body_tracker_params;
    body_tracker_params.enable_tracking = true; // track people across images flow
    body_tracker_params.enable_body_fitting = false; // smooth skeletons moves
    body_tracker_params.body_format = sl::BODY_FORMAT::BODY_34;
    body_tracker_params.detection_model = isJetson ? BODY_TRACKING_MODEL::HUMAN_BODY_FAST : BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
    //body_tracker_params.allow_reduced_precision_inference = true;

    returned_state = zed.enableBodyTracking(body_tracker_params);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("enable Object Detection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    auto camera_config = zed.getCameraInformation().camera_configuration;

    // For 2D GUI
    float image_aspect_ratio = camera_config.resolution.width / (1.f * camera_config.resolution.height);
    int requested_low_res_w = min(1280, (int)camera_config.resolution.width);
    sl::Resolution display_resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio);

    cv::Mat image_left_ocv(display_resolution.height, display_resolution.width, CV_8UC4, 1);
    Mat image_left(display_resolution, MAT_TYPE::U8_C4, image_left_ocv.data, image_left_ocv.step);
    sl::float2 img_scale(display_resolution.width / (float) camera_config.resolution.width, display_resolution.height / (float) camera_config.resolution.height);


    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv);

    Pose cam_pose;
    cam_pose.pose_data.setIdentity();

    // Configure object detection runtime parameters
    BodyTrackingRuntimeParameters body_tracker_parameters_rt;
    body_tracker_parameters_rt.detection_confidence_threshold = 40;
    body_tracker_parameters_rt.skeleton_smoothing = 0.7;
    
    // Create ZED Bodies filled in the main loop
    Bodies bodies;

    // Main Loop
    bool quit = false;
    string window_name = "ZED| 2D View";
    int key_wait = 10;
    char key = ' ';
    while (!quit) {
        // Grab images
        auto err = zed.grab();
        if (err == ERROR_CODE::SUCCESS) {
            // Retrieve Detected Human Bodies
            zed.retrieveBodies(bodies, body_tracker_parameters_rt);

            //OCV View
            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
            zed.getPosition(cam_pose, REFERENCE_FRAME::WORLD);

            //Update GL View
            viewer.updateData(bodies, cam_pose.pose_data);

            //printf("bodies is tracked %d \n", bodies.is_tracked);
            render_2D(image_left_ocv, img_scale, bodies.body_list, bodies.is_tracked);
            cv::imshow(window_name, image_left_ocv);

            key = cv::waitKey(key_wait);

            if (key == 'q') quit = true;
            if (key == 'm') {
                if (key_wait > 0) key_wait = 0;
                else key_wait = 10;
            }
            if (!viewer.isAvailable()) quit = true;
        } 
        else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            zed.setSVOPosition(0);
        }
        else
            quit = true;
    }

    // Release Bodies
    viewer.exit();
    image_left.free();
    bodies.body_list.clear();

    // Disable modules
    zed.disableBodyTracking();
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
        }else if (arg.find("HD1200") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1200;
            cout << "[Sample] Using Camera in resolution HD1200" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        }else if (arg.find("SVGA") != string::npos) {
            param.camera_resolution = RESOLUTION::SVGA;
            cout << "[Sample] Using Camera in resolution SVGA" << endl;
        }else if (arg.find("VGA") != string::npos) {
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