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

/*****************************************************************************************
 ** This sample demonstrates how to detect human bodies and retrieves their 3D position **
 **         with the ZED SDK and display the result in an OpenGL window.                **
 *****************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

#include <cstdlib>
#include <utility>
#include <vector>

// Sample includes
#include "GLViewer.hpp"
#include "TrackingViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");

/// Everything the command line can change, pre-filled with the sample defaults.
struct SampleOptions {
    BODY_TRACKING_MODEL detection_model;
    BODY_TRACKING_MODEL_GEN model_gen = BODY_TRACKING_MODEL_GEN::GEN_2;
    BODY_FORMAT body_format = BODY_FORMAT::BODY_34;
    bool enable_body_fitting = false;
    bool enable_segmentation = true;
    float detection_confidence = 40.f;
    float skeleton_smoothing = 0.7f;
};

bool parseArgs(int argc, char** argv, InitParameters& param, SampleOptions& opts);
void printUsage(const char* exe);

int main(int argc, char** argv) {

#ifdef _SL_JETSON_
    const bool isJetson = true;
#else
    const bool isJetson = false;
#endif

    // Create ZED Bodies
    Camera zed;
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::AUTO;
    init_parameters.depth_mode = isJetson ? DEPTH_MODE::NEURAL_LIGHT : DEPTH_MODE::NEURAL;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;

    SampleOptions opts;
    opts.detection_model = isJetson ? BODY_TRACKING_MODEL::HUMAN_BODY_FAST : BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;

    if (!parseArgs(argc, argv, init_parameters, opts))
        return EXIT_FAILURE;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable Positional tracking (mandatory for object detection)
    PositionalTrackingParameters positional_tracking_parameters;
    // If the camera is static, uncomment the following line to have better performances
    // positional_tracking_parameters.set_as_static = true;

    returned_state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("enable Positional Tracking", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Enable the Body tracking module
    BodyTrackingParameters body_tracker_params;
    body_tracker_params.enable_tracking = true;                         // track people across images flow
    body_tracker_params.enable_body_fitting = opts.enable_body_fitting; // smooth skeletons moves
    body_tracker_params.body_format = opts.body_format;
    body_tracker_params.enable_segmentation = opts.enable_segmentation;
    body_tracker_params.detection_model = opts.detection_model;
    body_tracker_params.model_gen = opts.model_gen;
    // body_tracker_params.allow_reduced_precision_inference = true;

    cout << "[Sample] Body tracking: " << toString(body_tracker_params.detection_model) << " | " << toString(body_tracker_params.model_gen)
         << " | " << toString(body_tracker_params.body_format) << " | fitting " << (body_tracker_params.enable_body_fitting ? "on" : "off")
         << " | segmentation " << (body_tracker_params.enable_segmentation ? "on" : "off") << endl;

    returned_state = zed.enableBodyTracking(body_tracker_params);
    if (returned_state > ERROR_CODE::SUCCESS) {
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
    sl::float2 img_scale(
        display_resolution.width / (float)camera_config.resolution.width,
        display_resolution.height / (float)camera_config.resolution.height
    );

    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv);

    Pose cam_pose;
    cam_pose.pose_data.setIdentity();

    // Configure object detection runtime parameters
    BodyTrackingRuntimeParameters body_tracker_parameters_rt;
    body_tracker_parameters_rt.detection_confidence_threshold = opts.detection_confidence;
    body_tracker_parameters_rt.skeleton_smoothing = opts.skeleton_smoothing;

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
        if (err <= ERROR_CODE::SUCCESS) {
            // Retrieve Detected Human Bodies
            zed.retrieveBodies(bodies, body_tracker_parameters_rt);

            // OCV View
            zed.retrieveImage(image_left, VIEW::LEFT, MEM::CPU, display_resolution);
            zed.getPosition(cam_pose, REFERENCE_FRAME::WORLD);

            // Update GL View
            viewer.updateData(bodies, cam_pose.pose_data);

            // printf("bodies is tracked %d \n", bodies.is_tracked);
            render_2D(image_left_ocv, img_scale, bodies.body_list, bodies.is_tracked);
            cv::imshow(window_name, image_left_ocv);

            key = cv::waitKey(key_wait);

            if (key == 'q')
                quit = true;
            if (key == 'm') {
                if (key_wait > 0)
                    key_wait = 0;
                else
                    key_wait = 10;
            }
            if (!viewer.isAvailable())
                quit = true;
        } else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            zed.setSVOPosition(0);
        } else
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

void printUsage(const char* exe) {
    cout << "\nUsage: " << exe
         << " [input] [options]\n\n"
            "Input (optional, positional or with --input; defaults to a live camera):\n"
            "  <file.svo2>              Play back an SVO file\n"
            "  <a.b.c.d[:port]>         Receive a network stream\n\n"
            "Options:\n"
            "  --input <src>            Same as the positional input above\n"
            "  --resolution <res>       HD2K | HD1200 | HD1080 | HD720 | SVGA | XVGA | TXGA | VGA | AUTO\n"
            "  --depth <mode>           neural_light | neural | neural_plus\n"
            "  --model <m>              Body tracking model: fast | medium | accurate\n"
            "  --generation <g>         Body tracking network generation: 1 | 2\n"
            "  --format <f>             Keypoint format: 18 | 34 | 38\n"
            "  --confidence <0-100>     Detection confidence threshold\n"
            "  --smoothing <0-1>        Skeleton smoothing\n"
            "  --fitting                Enable body fitting (required by the 34-keypoint format)\n"
            "  --no-segmentation        Disable the body segmentation mask\n"
            "  --help, -h               Show this message\n\n"
            "Controls:\n"
            "  q                        Quit\n"
            "  m                        Pause / resume\n\n"
            "Examples:\n"
            "  "
         << exe
         << " --model fast --generation 1\n"
            "  "
         << exe
         << " recording.svo2 --format 18 --confidence 50\n"
            "  "
         << exe << " 192.168.1.42:30000 --resolution HD1080\n"
         << endl;
}

/// Set the input from an SVO path or an "ip[:port]" stream address.
/// Returns false if the argument is neither.
static bool setInput(const string& arg, InitParameters& param) {
    if (arg.find(".svo") != string::npos) {
        param.input.setFromSVOFile(arg.c_str());
        cout << "[Sample] Using SVO File input: " << arg << endl;
        return true;
    }
    unsigned int a, b, c, d, port;
    if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
        string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
        param.input.setFromStream(String(ip_adress.c_str()), port);
        cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        return true;
    }
    if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
        param.input.setFromStream(String(arg.c_str()));
        cout << "[Sample] Using Stream input, IP : " << arg << endl;
        return true;
    }
    return false;
}

/// Returns false if the resolution name is unknown.
static bool setResolution(const string& arg, InitParameters& param) {
    // The SDK exports this conversion and the positional tracking sample already
    // uses it, so this sample no longer carries its own name table, which had to
    // stay ordered longest-first or "HD1200" would never be reached through
    // "HD120". Note it returns an ERROR flag, not success.
    RESOLUTION resolution;
    if (fromString(arg.c_str(), resolution))
        return false;
    param.camera_resolution = resolution;
    cout << "[Sample] Using Camera in resolution " << toString(resolution) << endl;
    return true;
}

/// Parse a float option, rejecting junk instead of silently yielding 0.
static float parseFloatOption(const char* text, const char* name, float lo, float hi, bool& ok) {
    try {
        size_t used = 0;
        const float v = std::stof(text, &used);
        if (used == std::string(text).size() && v >= lo && v <= hi) {
            ok = true;
            return v;
        }
    } catch (...) { }
    cout << name << " expects a number between " << lo << " and " << hi << ", got '" << text << "'" << endl;
    ok = false;
    return lo;
}

bool parseArgs(int argc, char** argv, InitParameters& param, SampleOptions& opts) {
    bool ok = false;
    // A value-taking option needs its argument; report instead of reading past the end.
    auto needValue = [&](int i, const string& name) {
        if (i + 1 >= argc) {
            cout << "[Sample][Error] " << name << " expects a value." << endl;
            return false;
        }
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        } else if (arg == "--input") {
            if (!needValue(i, arg))
                return false;
            if (!setInput(argv[++i], param)) {
                cout << "[Sample][Error] Unrecognized input '" << argv[i] << "' (expected an .svo2 file or an ip[:port] address)." << endl;
                return false;
            }
        } else if (arg == "--resolution") {
            if (!needValue(i, arg))
                return false;
            if (!setResolution(argv[++i], param)) {
                cout << "[Sample][Error] Unknown resolution '" << argv[i] << "'." << endl;
                return false;
            }
        } else if (arg == "--depth") {
            if (!needValue(i, arg))
                return false;
            string v = argv[++i];
            if (v == "neural_light")
                param.depth_mode = DEPTH_MODE::NEURAL_LIGHT;
            else if (v == "neural")
                param.depth_mode = DEPTH_MODE::NEURAL;
            else if (v == "neural_plus")
                param.depth_mode = DEPTH_MODE::NEURAL_PLUS;
            else {
                cout << "[Sample][Error] Unknown depth mode '" << v << "'." << endl;
                return false;
            }
        } else if (arg == "--model") {
            if (!needValue(i, arg))
                return false;
            string v = argv[++i];
            if (v == "fast")
                opts.detection_model = BODY_TRACKING_MODEL::HUMAN_BODY_FAST;
            else if (v == "medium")
                opts.detection_model = BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
            else if (v == "accurate")
                opts.detection_model = BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
            else {
                cout << "[Sample][Error] Unknown model '" << v << "' (fast|medium|accurate)." << endl;
                return false;
            }
        } else if (arg == "--generation") {
            if (!needValue(i, arg))
                return false;
            string v = argv[++i];
            if (v == "1")
                opts.model_gen = BODY_TRACKING_MODEL_GEN::GEN_1;
            else if (v == "2")
                opts.model_gen = BODY_TRACKING_MODEL_GEN::GEN_2;
            else {
                cout << "[Sample][Error] Unknown generation '" << v << "' (1|2)." << endl;
                return false;
            }
        } else if (arg == "--format") {
            if (!needValue(i, arg))
                return false;
            string v = argv[++i];
            if (v == "18")
                opts.body_format = BODY_FORMAT::BODY_18;
            else if (v == "34")
                opts.body_format = BODY_FORMAT::BODY_34;
            else if (v == "38")
                opts.body_format = BODY_FORMAT::BODY_38;
            else {
                cout << "[Sample][Error] Unknown body format '" << v << "' (18|34|38)." << endl;
                return false;
            }
        } else if (arg == "--confidence") {
            if (!needValue(i, arg))
                return false;
            opts.detection_confidence = parseFloatOption(argv[++i], "--confidence", 0.f, 100.f, ok);
            if (!ok)
                return false;
        } else if (arg == "--smoothing") {
            if (!needValue(i, arg))
                return false;
            opts.skeleton_smoothing = parseFloatOption(argv[++i], "--smoothing", 0.f, 1.f, ok);
            if (!ok)
                return false;
        } else if (arg == "--fitting") {
            opts.enable_body_fitting = true;
        } else if (arg == "--no-segmentation") {
            opts.enable_segmentation = false;
        } else if (arg.rfind("--", 0) == 0 || arg.rfind("-", 0) == 0) {
            cout << "[Sample][Error] Unknown option '" << arg << "'." << endl;
            printUsage(argv[0]);
            return false;
        } else {
            // Bare argument: an input, or a resolution name (kept for backward compatibility).
            if (!setInput(arg, param) && !setResolution(arg, param)) {
                cout << "[Sample][Error] Unrecognized argument '" << arg << "'." << endl;
                printUsage(argv[0]);
                return false;
            }
        }
    }

    // The SDK force-enables fitting for BODY_34 anyway (and warns); doing it here
    // keeps the configuration we print below honest.
    if (opts.body_format == BODY_FORMAT::BODY_34 && !opts.enable_body_fitting) {
        cout << "[Sample] The 34-keypoint format requires body fitting; enabling it." << endl;
        opts.enable_body_fitting = true;
    }
    return true;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code > ERROR_CODE::SUCCESS)
        cout << "[Error]";
    else if (err_code < ERROR_CODE::SUCCESS)
        cout << "[Warning]";
    cout << " " << msg_prefix << " ";
    if (err_code > ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
