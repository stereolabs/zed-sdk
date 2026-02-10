#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"

#include "yolo.hpp"

#include <sl/CameraOne.hpp>
#include <NvInfer.h>

using namespace nvinfer1;
#define NMS_THRESH 0.4
#define CONF_THRESH 0.3

static void draw_objects(cv::Mat& image, std::vector<sl::CustomBoxObjectData> const& objs, std::vector<std::vector<int>> const& colors) {
    cv::Mat mask {image.clone()};
    for (size_t i = 0; i < objs.size(); i++) {
        sl::CustomBoxObjectData const& obj = objs[i];
        size_t const idx_color {i % colors.size()};
        cv::Scalar const color {cv::Scalar(colors[idx_color][0U], colors[idx_color][1U], colors[idx_color][2U])};

        cv::Rect const rect {
            static_cast<int>(obj.bounding_box_2d[0U].x),
            static_cast<int>(obj.bounding_box_2d[0U].y),
            static_cast<int>(obj.bounding_box_2d[1U].x - obj.bounding_box_2d[0U].x),
            static_cast<int>(obj.bounding_box_2d[2U].y - obj.bounding_box_2d[0U].y)
        };
        cv::rectangle(mask, rect, color, -1); // Filled rectangle on mask
        cv::rectangle(image, rect, color, 2); // Outline on image

        char text[256U];
        sprintf(text, "Class %d - %.1f%%", obj.label, obj.probability * 100.f);

        int baseLine {0};
        cv::Size const label_size {cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine)};

        int const x {rect.x};
        int const y {std::min(rect.y + 1, image.rows)};

        cv::rectangle(image, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
}

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample]";
    if (err_code > sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    else if (err_code < sl::ERROR_CODE::SUCCESS)
        std::cout << "[Warning] ";
    else
        std::cout << " ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

inline std::vector<sl::uint2> cvt(const BBox& bbox_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
    bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
    bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
    bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
    return bbox_out;
}

std::mutex detector_mtx, tensor_mtx;
bool exit_detector = false;
std::vector<sl::CustomBoxObjectData> objects_in;
sl::Tensor input_tensor;
sl::Timestamp prev_ts = 0, custom_data_ts = 0;
sl::Resolution display_resolution;
Yolo detector;

void run_detector() {
    while (!exit_detector) {

        if (prev_ts != input_tensor.timestamp) {

            // Running inference - tensor is already preprocessed by retrieveTensor
            auto detections = detector.run(input_tensor, display_resolution.height, display_resolution.width, CONF_THRESH);

            // Preparing for ZED SDK ingesting
            std::vector<sl::CustomBoxObjectData> objects_tmp;
            for (auto& it : detections) {
                sl::CustomBoxObjectData tmp;
                // Fill the detections into the correct format
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = it.prob;
                tmp.label = (int)it.label;
                tmp.bounding_box_2d = cvt(it.box);
                tmp.velocity_smoothing_factor = 0.5f;
                tmp.is_grounded = ((int)it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                // others are tracked in full 3D space
                objects_tmp.push_back(tmp);
            }

            detector_mtx.lock();
            objects_tmp.swap(objects_in);
            custom_data_ts = input_tensor.timestamp;
            detector_mtx.unlock();

            prev_ts = input_tensor.timestamp;
        }

        sl::sleep_ms(1);
    }
}

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Usage: \n 1. ./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine\n 2. ./yolo_onnx_zed -s yolov8s.onnx yolov8s.engine "
                     "images:1x3x512x512\n 3. ./yolo_onnx_zed yolov8s.engine <SVO path>"
                  << std::endl;
        return 0;
    }

    // Check Optim engine first
    if (std::string(argv[1]) == "-s" && (argc >= 4)) {
        std::string onnx_path = std::string(argv[2]);
        std::string engine_path = std::string(argv[3]);
        OptimDim dyn_dim_profile;

        if (argc == 5) {
            std::string optim_profile = std::string(argv[4]);
            bool error = dyn_dim_profile.setFromString(optim_profile);
            if (error) {
                std::cerr << "Invalid dynamic dimension argument, expecting something like 'images:1x3x512x512'" << std::endl;
                return EXIT_FAILURE;
            }
        }

        Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
        return 0;
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::CameraOne zed;
    sl::InitParametersOne init_parameters;
    init_parameters.sdk_verbose = true;

    if (argc > 2) {
        std::string zed_opt = argv[2];
        if (zed_opt.find(".svo") != std::string::npos)
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
    }

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Creating the inference engine class
    std::string engine_name = "";
    if (argc > 0)
        engine_name = argv[1];
    else {
        std::cout << "Error: missing engine name as argument" << std::endl;
        return EXIT_FAILURE;
    }
    if (detector.init(engine_name)) {
        std::cerr << "Detector init failed!" << std::endl;
        return EXIT_FAILURE;
    }

    display_resolution = zed.getCameraInformation().camera_configuration.resolution;
    sl::Mat left_sl; // For display
    cv::Mat left_cv;

    // Get tensor parameters from detector (configured based on engine input size)
    sl::TensorParameters tensor_params = detector.getTensorParameters();

    std::thread detection_thread(run_detector);

    while (1) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            // Retrieve pre-processed tensor for inference (replaces blobFromImage)
            zed.retrieveTensor(input_tensor, tensor_params, detector.stream);

            // wait for the detections
            while (input_tensor.timestamp != custom_data_ts)
                sl::sleep_ms(1);

            // Get image for display
            zed.retrieveImage(left_sl, sl::VIEW::LEFT, sl::MEM::CPU);

            // Get image for display
            left_cv = slMat2cvMat(left_sl);
            // Displaying the SDK objects
            draw_objects(left_cv, objects_in, CLASS_COLORS);
            cv::imshow("ZED retrieved Objects", left_cv);
            int const key {cv::waitKey(1)};
            if (key == 'q' || key == 'Q' || key == 27)
                break;
        }
    }

    exit_detector = true;
    detection_thread.join();
    return 0;
}
