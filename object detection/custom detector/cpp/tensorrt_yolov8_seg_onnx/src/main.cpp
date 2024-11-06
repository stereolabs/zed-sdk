#include <iostream>
#include <chrono>
#include <cmath>

#include <sl/Camera.hpp>
#include <NvInfer.h>

#include "GLViewer.hpp"
#include "utils.h"
#include "yolov8-seg.hpp"
#include "yolov8-seg_optim.hpp"

using namespace nvinfer1;

void print(std::string const& msg_prefix, sl::ERROR_CODE const err_code, std::string const& msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << err_code << ": ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

static void draw_objects(cv::Mat const& image,
                         cv::Mat &res,
                         sl::Objects const& objs,
                         std::vector<std::vector<int>> const& colors)
{
    res = image.clone();
    cv::Mat mask{image.clone()};
    for (sl::ObjectData const& obj : objs.object_list) {
        size_t const idx_color{obj.id % colors.size()};
        cv::Scalar const color{cv::Scalar(colors[idx_color][0U], colors[idx_color][1U], colors[idx_color][2U])};

        cv::Rect const rect{static_cast<int>(obj.bounding_box_2d[0U].x),
                            static_cast<int>(obj.bounding_box_2d[0U].y),
                            static_cast<int>(obj.bounding_box_2d[1U].x - obj.bounding_box_2d[0U].x),
                            static_cast<int>(obj.bounding_box_2d[2U].y - obj.bounding_box_2d[0U].y)};
        cv::rectangle(res, rect, color, 2);

        char text[256U];
        sprintf(text, "Class %d - %.1f%%", obj.raw_label, obj.confidence);
        if (obj.mask.isInit() && obj.mask.getWidth() > 0U && obj.mask.getHeight() > 0U) {
            const cv::Mat obj_mask = slMat2cvMat(obj.mask);
            mask(rect).setTo(color, obj_mask);
        }

        int baseLine{0};
        cv::Size const label_size{cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine)};

        int const x{rect.x};
        int const y{std::min(rect.y + 1, res.rows)};

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
}

static std::vector<sl::uint2> convertCvRect2SdkBbox(cv::Rect_<float> const& bbox_in) {
    std::vector<sl::uint2> bbox_out;
    bbox_out.push_back(sl::uint2(static_cast<unsigned int>(bbox_in.x),
                                 static_cast<unsigned int>(bbox_in.y)));
    bbox_out.push_back(sl::uint2(static_cast<unsigned int>(bbox_in.x + bbox_in.width),
                                 static_cast<unsigned int>(bbox_in.y)));
    bbox_out.push_back(sl::uint2(static_cast<unsigned int>(bbox_in.x + bbox_in.width),
                                 static_cast<unsigned int>(bbox_in.y + bbox_in.height)));
    bbox_out.push_back(sl::uint2(static_cast<unsigned int>(bbox_in.x),
                                 static_cast<unsigned int>(bbox_in.y + bbox_in.height)));
    return bbox_out;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Usage:" << std::endl
                  << "  1. ./yolov8_seg_onnx_zed -s yolov8s-seg.onnx yolov8s-seg.engine" << std::endl
                  << "  2. ./yolov8_seg_onnx_zed -s yolov8s-seg.onnx yolov8s-seg.engine images:1x3x640x640" << std::endl
                  << "  3. ./yolov8_seg_onnx_zed yolov8s-seg.engine <SVO path>" << std::endl
                  << "  4. ./yolov8_seg_onnx_zed yolov8s-seg.engine" << std::endl;
        return 0;
    }
    
    // Check Optim engine first
    if (std::string(argv[1U]) == "-s" && (argc >= 4)) {
        std::string const onnx_path{std::string(argv[2U])};
        std::string const engine_path{std::string(argv[3U])};

        OptimDim dyn_dim_profile;
        if (argc == 5) {
            std::string const optim_profile{std::string(argv[4U])};
            if (dyn_dim_profile.setFromString(optim_profile) != 0) {
                std::cerr << "Invalid dynamic dimension argument '" << optim_profile << "',"
                          << " expecting something like 'images:1x3x512x512'" << std::endl;
                return -1;
            }
        }

        build_engine(onnx_path, engine_path, dyn_dim_profile);
        return 0;
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed

    if (argc > 2) {
        std::string const zed_opt{argv[2U]};
        if (zed_opt.find(".svo") != std::string::npos)
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
    }

    // Open the camera
    sl::Camera zed;
    sl::ERROR_CODE state{zed.open(init_parameters)};
    if (state != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "[ERROR] Opening camera: " << state << std::endl;
        return EXIT_FAILURE;
    }

    // Enable Positional Tracking
    zed.enablePositionalTracking();

    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = true;
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    state = zed.enableObjectDetection(detection_parameters);
    if (state != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "[ERROR] Enabling Object Detection: " << state << std::endl;
        zed.close();
        return EXIT_FAILURE;
    }

    // Get camera configuration
    sl::CameraConfiguration const camera_config{zed.getCameraInformation().camera_configuration};
    sl::Resolution const pc_resolution{std::min(camera_config.resolution.width, 720UL),
                                       std::min(camera_config.resolution.height, 404UL)};
    sl::CameraConfiguration const camera_info{zed.getCameraInformation(pc_resolution).camera_configuration};

    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv, camera_info.calibration_parameters.left_cam, true);

    // Creating the inference engine class
    std::string const engine_name{argv[1U]};
    YOLOv8_seg detector{engine_name};
    detector.make_pipe(false);

    // Prepare detector input/output
    cv::Mat left_cv, pred_raw;
    int const topk{100};
    int const seg_channels{32};
    float const score_thres{0.5F};
    float const iou_thres{0.65F};
    std::vector<seg::Object> objs;

    // Prepare SDK input/output
    sl::Mat left_sl, point_cloud;
    cv::Mat pred_sdk;
    sl::Objects objects;
    sl::Pose cam_w_pose;
    cam_w_pose.pose_data.setIdentity();
    int key{0};

    sl::CustomObjectDetectionRuntimeParameters cod_rt_param;

    while (key != 'q' && key != 27) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            // Get image for inference
            zed.retrieveImage(left_sl, sl::VIEW::LEFT);
            left_cv = slMat2cvMat(left_sl);
            cv::cvtColor(left_cv, left_cv, cv::COLOR_BGRA2BGR);

            // Running inference
            detector.copy_from_Mat(left_cv);
            detector.infer();

            // Post process output
            objs.clear();
            detector.postprocess(objs, score_thres, iou_thres, topk, seg_channels);

            // Preparing for ZED SDK ingesting
            std::vector<sl::CustomMaskObjectData> objects_in;
            objects_in.reserve(objs.size());
            for (seg::Object& obj : objs) {
                objects_in.emplace_back();
                sl::CustomMaskObjectData &tmp{objects_in.back()};
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = obj.prob;
                tmp.label       = obj.label;
                tmp.bounding_box_2d = convertCvRect2SdkBbox(obj.rect);
                tmp.is_grounded = (obj.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                                                    // others are tracked in full 3D space
                cvMat2slMat(obj.boxMask).copyTo(tmp.box_mask, sl::COPY_TYPE::CPU_CPU);
            }

            // Send the custom detected boxes with masks to the ZED
            zed.ingestCustomMaskObjects(objects_in);

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, cod_rt_param);

            // Draw raw prediction
            draw_objects(left_cv, pred_sdk, objects, CLASS_COLORS);
            cv::imshow("ZED SDK Objects", pred_sdk);

            // GL Viewer
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
            zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
            viewer.updateData(point_cloud, objects.object_list, cam_w_pose.pose_data);

            int const cv_key{cv::waitKey(10)};
            int const gl_key{viewer.getKey()};
            key = (gl_key == -1) ? cv_key : gl_key;
            if (key == 'p' || key == 32) {
                viewer.setPlaying(!viewer.isPlaying());
            }
            while ((key == -1) && !viewer.isPlaying() && viewer.isAvailable()) {
                int const cv_key{cv::waitKey(10)};
                int const gl_key{viewer.getKey()};
                key = (gl_key == -1) ? cv_key : gl_key;
                if (key == 'p' || key == 32) {
                    viewer.setPlaying(!viewer.isPlaying());
                }
            }
        }
        if (!viewer.isAvailable())
            break;
    }

    cv::destroyAllWindows();
    viewer.exit();

    return 0;
}
