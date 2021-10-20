#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <sl/Camera.hpp>

#include "GLViewer.hpp"

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;
constexpr int INFERENCE_SIZE = 416;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof (colors) / sizeof (colors[0]);

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
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

std::vector<sl::uint2> cvt(const cv::Rect &bbox_in){
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}

int main(int argc, char** argv) {
    std::vector<std::string> class_names;
    {
        std::ifstream class_file("coco.names.txt");
        if (!class_file) {
            for (int i = 0; i < NUM_CLASSES; i++)
                class_names.push_back(std::to_string(i));
        } else {
            std::string line;
            while (std::getline(class_file, line))
                class_names.push_back(line);
        }
    }

    /// Opening the ZED camera before the model deserialization to avoid cuda context issue
    sl::Camera zed;
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD1080;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed

    if (argc >= 2) {
        std::string zed_opt = argv[1];
        if (zed_opt.find(".svo") != std::string::npos)
            init_parameters.input.setFromSVOFile(zed_opt.c_str());
    }
    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }
    zed.enablePositionalTracking();
    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    // Let's define the model as custom box object to specify that the inference is done externally
    detection_parameters.detection_model = sl::DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != sl::ERROR_CODE::SUCCESS) {
        print("enableObjectDetection", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }
    auto camera_config = zed.getCameraInformation().camera_configuration;
    sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
    auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;
    // Create OpenGL Viewer
    GLViewer viewer;
    viewer.init(argc, argv, camera_info.calibration_parameters.left_cam, true);
    sl::Mat left_sl, point_cloud;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;
    sl::Pose cam_w_pose;
    cam_w_pose.pose_data.setIdentity();
    // ---------

    // Weight can be downloaded from https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    auto net = cv::dnn::readNetFromDarknet("yolov4.cfg", "yolov4.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();

    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;
    while (viewer.isAvailable()) {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS) {

            zed.retrieveImage(left_sl, sl::VIEW::LEFT);

            // Preparing inference
            cv::Mat left_cv_rgba = slMat2cvMat(left_sl);
            cv::cvtColor(left_cv_rgba, frame, cv::COLOR_BGRA2BGR);

            cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(INFERENCE_SIZE, INFERENCE_SIZE), cv::Scalar(), true, false, CV_32F);
            net.setInput(blob);
            net.forward(detections, output_names);

            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];

            for (auto& output : detections) {
                const auto num_boxes = output.rows;
                for (int i = 0; i < num_boxes; i++) {
                    auto x = output.at<float>(i, 0) * frame.cols;
                    auto y = output.at<float>(i, 1) * frame.rows;
                    auto width = output.at<float>(i, 2) * frame.cols;
                    auto height = output.at<float>(i, 3) * frame.rows;
                    cv::Rect rect(x - width / 2, y - height / 2, width, height);

                    for (int c = 0; c < NUM_CLASSES; c++) {
                        auto confidence = *output.ptr<float>(i, 5 + c);
                        if (confidence >= CONFIDENCE_THRESHOLD) {
                            boxes[c].push_back(rect);
                            scores[c].push_back(confidence);
                        }
                    }
                }
            }

            for (int c = 0; c < NUM_CLASSES; c++)
                cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

            std::vector<sl::CustomBoxObjectData> objects_in;
            for (int c = 0; c < NUM_CLASSES; c++) {
                for (size_t i = 0; i < indices[c].size(); ++i) {
                    const auto color = colors[c % NUM_COLORS];

                    auto idx = indices[c][i];
                    const auto& rect = boxes[c][idx];
                    auto& rect_score = scores[c][idx];

                    // Fill the detections into the correct format
                    sl::CustomBoxObjectData tmp;
                    tmp.unique_object_id = sl::generate_unique_id();
                    tmp.probability = rect_score;
                    tmp.label = c;
                    tmp.bounding_box_2d = cvt(rect);
                    tmp.is_grounded = (c == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                    // others are tracked in full 3D space
                    objects_in.push_back(tmp);
                    //--

                    cv::rectangle(frame, rect, color, 3);

                    std::ostringstream label_ss;
                    label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                    auto label = label_ss.str();

                    int baseline;
                    auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                    cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                }
            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);

            cv::imshow("Objects", frame);
            cv::waitKey(10);

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, objectTracker_parameters_rt);
            // GL Viewer
            zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, pc_resolution);
            zed.getPosition(cam_w_pose, sl::REFERENCE_FRAME::WORLD);
            viewer.updateData(point_cloud, objects.object_list, cam_w_pose.pose_data);
        }
    }
    return 0;
}