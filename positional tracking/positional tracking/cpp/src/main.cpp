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

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include <opencv2/opencv.hpp>
#include <optional>
#include "SLAMView.hpp"

// Using the sl namespace
using namespace sl;

//
// Arguments
//
struct Arguments {
    std::optional<RESOLUTION> resolution = std::nullopt;
    std::optional<std::string> svoFile = std::nullopt;
    std::optional<std::string> streamIP = std::nullopt;
    std::optional<int> streamPort = std::nullopt;
    bool map = false;
    std::optional<std::string> inputAreaFile = std::nullopt;
    std::optional<std::string> outputAreaFile = std::nullopt;
    std::optional<std::string> roiFile = std::nullopt;
    bool customInitialPose = false;
    bool enable2dGroundMode = false;
    bool exportTUMFile = false;
};

//
// Utility function declarations
//
void printHeader(const std::string& text);
void printUsage(const std::string& programName);
bool parseArgs(int argc, char* argv[], Arguments& args);
void printArgs(const Arguments& args);
void printTrackingParameters(PositionalTrackingParameters trackingParameters);
void print(std::string message, std::optional<ERROR_CODE> errorCode = std::nullopt, bool showErrorDetail = true);

cv::Mat slMat2cvMat(Mat& input);
cv::Scalar interpolate_color(const cv::Scalar& color1, const cv::Scalar& color2, float dynamic_confidence);

//
// Main
//
int main(int argc, char** argv) {
    //
    // Parse arguments
    //
    printHeader("ZED Positional Tracking");

    if (argc > 1 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return 0;
    }

    Arguments args;
    if (!parseArgs(argc, argv, args)) {
        std::cout << "\n";
        printUsage(argv[0]);
        return 1;
    }

    printArgs(args);

    //
    // Configure a camera
    //
    Camera zed;

    InitParameters initParameters;
    initParameters.depth_mode = DEPTH_MODE::NEURAL;
    initParameters.coordinate_units = UNIT::METER;
    initParameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;

    if (args.resolution) {
        initParameters.camera_resolution = args.resolution.value();
    }

    if (args.svoFile) {
        initParameters.input.setFromSVOFile(args.svoFile.value().c_str());
    } else if (args.streamIP && args.streamPort) {
        initParameters.input.setFromStream(args.streamIP.value().c_str(), args.streamPort.value());
    } else if (args.streamIP) {
        initParameters.input.setFromStream(args.streamIP.value().c_str());
    }

    //
    // Open the camera
    //
    ERROR_CODE status = zed.open(initParameters);

    if (status > ERROR_CODE::SUCCESS) {
        print("Failed to open the camera", status);
        zed.close();
        return 1;
    }

    //
    // Configure positional tracking
    //
    PositionalTrackingParameters trackingParameters;

    if (args.inputAreaFile) {
        trackingParameters.area_file_path = args.inputAreaFile.value().c_str();
    }

    trackingParameters.set_floor_as_origin = false;
    trackingParameters.set_gravity_as_origin = true;
    trackingParameters.enable_area_memory = true;
    trackingParameters.enable_pose_smoothing = false;
    trackingParameters.enable_imu_fusion = true;
    trackingParameters.set_as_static = false;
    trackingParameters.depth_min_range = -1;
    trackingParameters.enable_2d_ground_mode = args.enable2dGroundMode;
    trackingParameters.enable_localization_only = false;
    trackingParameters.mode = POSITIONAL_TRACKING_MODE::GEN_3;

    if (args.roiFile) {
        sl::Mat roi;
        roi.read(args.roiFile->c_str());
        zed.setRegionOfInterest(roi);
    }

    //
    // Optionally define an initial pose for the tracking to start from.
    // This is useful if you know the initial pose of the camera in a previously mapped area.
    //
    if (args.customInitialPose) {
        // clang-format off

        // Homogeneous transformation matrix for the initial pose:
        // For example, we have a translation to (2.0, 4.0, 1.0) in the right-most column,
        // and a rotation of 45° about the Z-axis in the upper-left 3x3 portion of the matrix.
        float initialPose[16] = {
            0.7071f, -0.7071f, 0.0f, 2.0f, 
            0.7071f,  0.7071f, 0.0f, 4.0f,
            0.0f,     0.0f,    1.0f, 1.0f,
            0.0f,     0.0f,    0.0f, 1.0f
        };
        // clang-format on

        trackingParameters.initial_world_transform = Transform(initialPose);
    }

    //
    // Enable positional tracking
    //
    status = zed.enablePositionalTracking(trackingParameters);

    if (status > ERROR_CODE::SUCCESS) {
        print("Failed to enable positional tracking", status);
        zed.close();
        return 1;
    }

    printTrackingParameters(trackingParameters);

    //
    // Set runtime parameters: for example, a low depth confidence to avoid introducing noise
    //
    RuntimeParameters runtime_parameters;
    runtime_parameters.confidence_threshold = 30;

    //
    // Configure display parameters
    //
    std::map<uint64_t, Landmark> landmarkMap;
    std::vector<Landmark2D> landmarks2D;
    std::map<uint64_t, KeyFrame> keyframes;

    uint64_t lastLandmarkUpdate = getCurrentTimeStamp().getSeconds();
    Resolution displayResolution = zed.getRetrieveMeasureResolution();

    //
    // Prepare out file for TUM trajectory
    //
    std::ofstream out_tum;
    if (args.exportTUMFile)
        out_tum.open("out.tum");

    //
    // Main loop
    //
    Mat leftImage = sl::Mat();
    Mat pointCloud(displayResolution, MAT_TYPE::F32_C4, MEM::GPU);

    Pose pose;
    SLAMView view = SLAMView(argc, argv, &leftImage, &pointCloud, zed.getCUDAStream());
    view.run([&]() {
        //
        // Grab the next image frame
        //
        ERROR_CODE status = zed.grab(runtime_parameters);

        if (status == ERROR_CODE::END_OF_SVOFILE_REACHED) {
            view.stop();
            return;
        } else if (status > ERROR_CODE::SUCCESS) {
            print("Failed to grab image frame", status);
            view.stop();
            return;
        }

        // Retrieve the left image
        if (trackingParameters.mode == POSITIONAL_TRACKING_MODE::GEN_3)
            zed.retrieveImage(leftImage, VIEW::LEFT_UNRECTIFIED, MEM::CPU, displayResolution);
        else
            zed.retrieveImage(leftImage, VIEW::LEFT, MEM::CPU, displayResolution);

        // Retrieve the calculated point cloud
        zed.retrieveMeasure(pointCloud, MEASURE::XYZBGRA, MEM::GPU, displayResolution);

        // Retrieve the calculated camera pose
        zed.getPosition(pose);

        // Export the pose in TUM format
        if (args.exportTUMFile) {
            out_tum << std::fixed << std::setprecision(9) << pose.timestamp.getMilliseconds() << " " << pose.getTranslation().tx << " "
                    << pose.getTranslation().ty << " " << pose.getTranslation().tz << " " << pose.getOrientation().ox << " "
                    << pose.getOrientation().oy << " " << pose.getOrientation().oz << " " << pose.getOrientation().ow << std::endl;
            out_tum.flush();
        }

        // Update display
        //
        view.updatePoseTransform(pose.pose_data);
        view.updatePositionalTrackingStatus(zed.getPositionalTrackingStatus());

        if (zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getSeconds() - lastLandmarkUpdate > 1) {
            zed.getPositionalTrackingLandmarks(landmarkMap);
            view.updateLandmarks(landmarkMap);
            lastLandmarkUpdate = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getSeconds();

            zed.getPositionalTrackingKeyframes(keyframes);
            view.updateKeyframes(keyframes);
        }

        if (view.isLandmarkModeEnabled()) {
            zed.getPositionalTrackingLandmarks2D(landmarks2D);

            static const cv::Scalar inlierLandmarkColor(63, 255, 67, 255);
            static const cv::Scalar outlierLandmarkColor(100, 100, 255, 255);

            Resolution resolution = zed.getCameraInformation().camera_configuration.resolution;
            float widthRatio = displayResolution.width / (float)resolution.width;
            float heightRatio = displayResolution.height / (float)resolution.height;

            cv::Mat leftImageCVMat = slMat2cvMat(leftImage);
            for (auto& landmark2D : landmarks2D) {
                cv::Scalar color = interpolate_color(inlierLandmarkColor, outlierLandmarkColor, landmark2D.dynamic_confidence);

                cv::circle(
                    leftImageCVMat,
                    cv::Point2f(landmark2D.image_position[0] * widthRatio, landmark2D.image_position[1] * heightRatio),
                    2,
                    color,
                    -1
                );
            }
        }
    });

    if (args.exportTUMFile)
        out_tum.close();

    //
    // OpenGL cleanup
    //
    pointCloud.free();

    //
    // Saving area map
    //
    if (args.map) {
        print("Saving area map to " + args.outputAreaFile.value() + " ...");

        ERROR_CODE status = zed.saveAreaMap(args.outputAreaFile.value().c_str());

        if (status == ERROR_CODE::SUCCESS) {
            AREA_EXPORTING_STATE exportState = zed.getAreaExportState();

            while (exportState == sl::AREA_EXPORTING_STATE::RUNNING) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                exportState = zed.getAreaExportState();
            }

            if (exportState == AREA_EXPORTING_STATE::SUCCESS) {
                print("Successfully saved area map to " + args.outputAreaFile.value());
            } else {
                print("Failed to save area map: " + std::string(toString(exportState).c_str()), ERROR_CODE::FAILURE, false);
            }

        } else {
            print("Failed to save area map", status);
        }
    }

    // Close the camera
    zed.close();

    return 0;
}

//
// Utility function definitions
//
auto colorBool = [](bool val) -> std::string {
    return val ? "\033[32mtrue\033[0m" : "\033[31mfalse\033[0m";
};

auto colorValue = [](const auto& val) -> std::string {
    std::ostringstream oss;
    oss << "\033[36m" << val << "\033[0m";
    return oss.str();
};

void printHeader(const std::string& text) {
    const int width = 80;
    const std::string border(width, '=');
    int padding = (width - text.length()) / 2;

    std::cout << "\n" << border << "\n";
    std::cout << std::string(padding, ' ') << text << "\n";
    std::cout << border << "\n\n";
}

void printUsage(const std::string& programName) {
    std::cout << "Usage: " << programName << " [options]\n\n"
              << "Options:\n"
              << "  --help                      Shows usage information.\n"
              << "  --resolution <mode>         Optional. Resolution options: (HD2K | HD1200 | HD1080 | HD720 | SVGA | VGA)\n"
              << "  --svo <filename.svo>        Optional. Use SVO file input. Mutually exclusive with --stream\n"
              << "  --stream <ip[:port]>        Optional. Use network streaming input. Mutually exclusive with --svo\n"
              << "  -i <input_area_file>        Optional. Input area file used in explore mode (default) or --map mode\n"
              << "  --map -o <output_area_file> Optional. Map mode creates or updates an .area file.\n"
              << "                                        Requires -o <output_area_filename> for generated map\n"
              << "  --roi <roi_filepath>        Optional. Region of interest image mask to ignore a static area\n"
              << "  --custom-initial-pose       Optional. Use custom initial pose (see code comments for more detail)\n"
              << "  --2d-ground-mode            Optional. Enable 2D ground mode\n"
              << "  --export-tum                Optional. Export camera trajectory to out.tum file in TUM format\n"
              << "\nExamples:\n"
              << "  " << programName << " --map -o new_map.area\n"
              << "  " << programName << " --svo recording.svo2 -i map.area\n";
}

bool parseArgs(int argc, char* argv[], Arguments& args) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--resolution" && i + 1 < argc) {
            RESOLUTION resolution;
            bool error = fromString(argv[++i], resolution);

            if (error) {
                print("Invalid resolution: " + std::string(argv[i]), ERROR_CODE::FAILURE, false);
                return false;
            }
            args.resolution = resolution;
        } else if (arg == "--svo" && i + 1 < argc) {
            if (args.streamIP) {
                print("Options --svo and --stream are mutually exclusive", ERROR_CODE::FAILURE, false);
                return false;
            }

            std::string filename = argv[++i];
            if (filename.find(".svo") == std::string::npos) {
                print("Option --svo requires a filename that contains '.svo'", ERROR_CODE::FAILURE, false);
                return false;
            }

            args.svoFile = filename;
        } else if (arg == "--stream" && i + 1 < argc) {
            if (args.svoFile) {
                print("Options --stream and --svo are mutually exclusive", ERROR_CODE::FAILURE, false);
                return false;
            }

            std::string ip;
            std::optional<int> port;

            const std::string& address = argv[++i];
            auto colonPos = address.find(':');

            if (colonPos != std::string::npos) {
                ip = address.substr(0, colonPos);
                std::string portStr = address.substr(colonPos + 1);

                try {
                    port = std::stoi(portStr);
                } catch (const std::exception& e) {
                    print("Invalid port in --stream: " + portStr, ERROR_CODE::FAILURE, false);
                    return false;
                }
            } else {
                ip = address;
            }

            args.streamIP = ip;
            args.streamPort = port;
        } else if (arg == "--map") {
            args.map = true;
        } else if (arg == "-i" && i + 1 < argc) {
            args.inputAreaFile = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            args.outputAreaFile = argv[++i];
        } else if (arg == "--roi" && i + 1 < argc) {
            args.roiFile = argv[++i];
        } else if (arg == "--custom-initial-pose") {
            args.customInitialPose = true;
        } else if (arg == "--2d-ground-mode") {
            args.enable2dGroundMode = true;
        } else if (arg == "--export-tum") {
            args.exportTUMFile = true;
        } else {
            print("Unrecognized or incomplete argument: " + arg, ERROR_CODE::FAILURE, false);
            return false;
        }
    }

    if (args.map && !args.outputAreaFile) {
        print("Option --map requires -o <output_area_file>", ERROR_CODE::FAILURE, false);
        return false;
    }

    return true;
}

void printArgs(const Arguments& args) {
    if (args.resolution) {
        std::string message = "Preferred Resolution: ";

        switch (args.resolution.value()) {
            case RESOLUTION::HD2K:
                message += "HD2K";
                break;
            case RESOLUTION::HD1200:
                message += "HD1200";
                break;
            case RESOLUTION::HD1080:
                message += "HD1080";
                break;
            case RESOLUTION::HD720:
                message += "HD720";
                break;
            case RESOLUTION::SVGA:
                message += "SVGA";
                break;
            case RESOLUTION::VGA:
                message += "VGA";
                break;
            default:
                break;
        }

        print(message);
    }

    if (args.svoFile) {
        print("Using SVO file input: " + args.svoFile.value());
    }

    if (args.streamIP) {
        std::string suffix = args.streamPort ? ":" + std::to_string(args.streamPort.value()) : "";
        print("Using stream input " + args.streamIP.value() + suffix);
    }

    if (args.map) {
        print("Positional tracking mode: Map");
    } else {
        print("Positional tracking mode: Explore (default)");
    }

    if (args.inputAreaFile) {
        print("Using input area file: " + args.inputAreaFile.value());
    }

    if (args.outputAreaFile) {
        print("Output file: " + args.outputAreaFile.value());
    }

    if (args.roiFile) {
        print("Using ROI file: " + args.roiFile.value());
    }

    if (args.customInitialPose) {
        print("Enabled custom initial pose");
    }

    if (args.enable2dGroundMode) {
        print("Enabled 2D ground mode");
    }

    if (args.exportTUMFile) {
        print("Enabled TUM trajectory export to out.tum");
    }

    std::cout << "\n";
}

void printTrackingParameters(PositionalTrackingParameters trackingParameters) {
    printHeader("Positional Tracking Parameters");
    print("set_floor_as_origin   " + colorBool(trackingParameters.set_floor_as_origin));
    print("set_gravity_as_origin " + colorBool(trackingParameters.set_gravity_as_origin));
    print("enable_area_memory    " + colorBool(trackingParameters.enable_area_memory));
    print("enable_pose_smoothing " + colorBool(trackingParameters.enable_pose_smoothing));
    print("enable_imu_fusion     " + colorBool(trackingParameters.enable_imu_fusion));
    print("set_as_static         " + colorBool(trackingParameters.set_as_static));
    print("depth_min_range       " + colorValue(trackingParameters.depth_min_range));
    print("enable_2d_ground_mode " + colorBool(trackingParameters.enable_2d_ground_mode));
    print("mode                  " + colorValue(toString(trackingParameters.mode)));

    printHeader("Interaction");
    print("'space' to toggle camera view visibility");
    print("'d' to switch background color from dark to light");
    print("'p' to enable / disable current live point cloud display");
    print("'l' to enable / disable landmark display");
    print("'f' to follow the camera");
    print("'z' to reset the view");
    print("'ctrl' + drag to rotate");
    print("'esc' to exit");
    std::cout << "\n";
}

void print(std::string message, std::optional<ERROR_CODE> errorCode, bool showErrorDetail) {
    std::cout << "\033[36m[Sample]\033[0m";

    if (errorCode && errorCode > ERROR_CODE::SUCCESS) {
        std::cout << " \033[31m[Error]\033[0m ";
    } else if (errorCode && errorCode < ERROR_CODE::SUCCESS) {
        std::cout << " \033[33m[Warning]\033[0m ";
    } else {
        std::cout << " ";
    }

    std::cout << message;

    if (errorCode && errorCode != ERROR_CODE::SUCCESS && showErrorDetail) {
        std::cout << " | " << toString(errorCode.value()) << ": " << toVerbose(errorCode.value());
    }

    std::cout << "\n";
}

cv::Mat slMat2cvMat(Mat& input) {
    int cv_type = -1;

    switch (input.getDataType()) {
        case MAT_TYPE::F32_C1:
            cv_type = CV_32FC1;
            break;
        case MAT_TYPE::F32_C2:
            cv_type = CV_32FC2;
            break;
        case MAT_TYPE::F32_C3:
            cv_type = CV_32FC3;
            break;
        case MAT_TYPE::F32_C4:
            cv_type = CV_32FC4;
            break;
        case MAT_TYPE::U8_C1:
            cv_type = CV_8UC1;
            break;
        case MAT_TYPE::U8_C2:
            cv_type = CV_8UC2;
            break;
        case MAT_TYPE::U8_C3:
            cv_type = CV_8UC3;
            break;
        case MAT_TYPE::U8_C4:
            cv_type = CV_8UC4;
            break;
        default:
            break;
    }

    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}

cv::Scalar interpolate_color(const cv::Scalar& color1, const cv::Scalar& color2, float dynamic_confidence) {

    if (dynamic_confidence == -1.f) {
        return cv::Scalar(240, 25, 25, 255);
    }

    if (dynamic_confidence < 0.0f) {
        dynamic_confidence = 0.0f;
    }

    if (dynamic_confidence > 1.0f) {
        dynamic_confidence = 1.0f;
    }

    return cv::Scalar(
        color1[0] * dynamic_confidence + color2[0] * (1 - dynamic_confidence),
        color1[1] * dynamic_confidence + color2[1] * (1 - dynamic_confidence),
        color1[2] * dynamic_confidence + color2[2] * (1 - dynamic_confidence),
        color1[3] * dynamic_confidence + color2[3] * (1 - dynamic_confidence)
    );
}
