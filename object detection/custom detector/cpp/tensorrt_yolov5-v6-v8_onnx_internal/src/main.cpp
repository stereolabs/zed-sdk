#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.h"

#include "GLViewer.hpp"

#include <sl/Camera.hpp>


static void draw_objects(const cv::Mat& image,
                         cv::Mat &res,
                         const sl::Objects& objs,
                         const std::vector<std::vector<int>>& colors,
                         const bool isTrackingON)
{
    res = image.clone();
    cv::Mat mask{image.clone()};
    for (sl::ObjectData const& obj : objs.object_list) {
        if (!renderObject(obj, isTrackingON))
            continue;
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
        const cv::Size label_size{cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine)};

        const int x{rect.x};
        const int y{std::min(rect.y + 1, res.rows)};

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
}

void print(const std::string& msg_prefix, const sl::ERROR_CODE err_code, const std::string& msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Usage: ./yolo_onnx_zed yolov8s.onnx <Optional: SVO path or Cam ID>" << std::endl;
        return 0;
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed

    if (argc > 2) {
        const std::string zed_opt = argv[2];
        if (zed_opt.find(".svo") != std::string::npos)
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
    }

    // Open the camera
    const sl::ERROR_CODE open_ret = zed.open(init_parameters);
    if (open_ret != sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", open_ret, "Exit program.");
        return EXIT_FAILURE;
    }

    // Enable Positional tracking
    zed.enablePositionalTracking();

    // Enable Custom OD
    constexpr bool enable_tracking = true;
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = enable_tracking;
    detection_parameters.enable_segmentation = false;
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_YOLOLIKE_BOX_OBJECTS;
    detection_parameters.custom_onnx_file.set(argv[1U]);
    detection_parameters.custom_onnx_dynamic_input_shape = sl::Resolution(320, 320); // Provide resolution for dynamic shape model
    const sl::ERROR_CODE od_ret = zed.enableObjectDetection(detection_parameters);
    if (od_ret != sl::ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", od_ret, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Camera config
    const sl::CameraConfiguration camera_config = zed.getCameraInformation().camera_configuration;
    const sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    const sl::CameraConfiguration camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;

    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv, camera_info.calibration_parameters.left_cam, enable_tracking);

    // Prepare SDK's output retrieval
    const sl::Resolution display_resolution = zed.getCameraInformation().camera_configuration.resolution;
    sl::Mat left_sl, point_cloud;
    cv::Mat left_cv;
    sl::CustomObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    // All classes parameters
    objectTracker_parameters_rt.object_detection_properties.detection_confidence_threshold = 20.f;
    // objectTracker_parameters_rt.object_detection_properties.is_static = true;
    // objectTracker_parameters_rt.object_detection_properties.tracking_timeout = 100.f;
    // // Per classes paramters override
    // objectTracker_parameters_rt.object_class_detection_properties[0U].detection_confidence_threshold = 80.f;
    // objectTracker_parameters_rt.object_class_detection_properties[1U].min_box_width_normalized = 0.01f;
    // objectTracker_parameters_rt.object_class_detection_properties[1U].max_box_width_normalized = 0.5f;
    // objectTracker_parameters_rt.object_class_detection_properties[1U].min_box_height_normalized = 0.01f;
    // objectTracker_parameters_rt.object_class_detection_properties[1U].max_box_height_normalized = 0.5f;

    sl::Objects objects;
    sl::Pose cam_w_pose;
    cam_w_pose.pose_data.setIdentity();

    // Main loop
    while (viewer.isAvailable()) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {

            // Get image for display
            zed.retrieveImage(left_sl, sl::VIEW::LEFT);
            left_cv = slMat2cvMat(left_sl);

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, objectTracker_parameters_rt);

            // GL Viewer
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
            zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
            viewer.updateData(point_cloud, objects.object_list, cam_w_pose.pose_data, objectTracker_parameters_rt);

            // Displaying the SDK objects
            draw_objects(left_cv, left_cv, objects, CLASS_COLORS, enable_tracking);
            cv::imshow("ZED retrieved Objects", left_cv);
            int const key{cv::waitKey(10)};
            if (key == 'q' || key == 'Q' || key == 27)
                break;
        }
    }

    viewer.exit();

    return 0;
}
