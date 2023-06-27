#ifndef YOLO_HPP
#define YOLO_HPP

#include <NvInfer.h>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"

enum class YOLO_MODEL_VERSION_OUTPUT_STYLE {
    YOLOV6,
    YOLOV8_V5
};

struct BBox {
    float x1, y1, x2, y2;
};

struct BBoxInfo {
    BBox box;
    int label;
    float prob;
};

inline std::vector<std::string> split_str(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}


struct OptimDim {
    nvinfer1::Dims4 size;
    std::string tensor_name;

    bool setFromString(std::string &arg) {
        // "images:1x3x512x512"
        std::vector<std::string> v_ = split_str(arg, ":");
        if (v_.size() != 2) return true;

        std::string dims_str = v_.back();
        std::vector<std::string> v = split_str(dims_str, "x");

        size.nbDims = 4;
        // assuming batch is 1 and channel is 3
        size.d[0] = 1;
        size.d[1] = 3;

        if (v.size() == 2) {
            size.d[2] = stoi(v[0]);
            size.d[3] = stoi(v[1]);
        } else if (v.size() == 3) {
            size.d[2] = stoi(v[1]);
            size.d[3] = stoi(v[2]);
        } else if (v.size() == 4) {
            size.d[2] = stoi(v[2]);
            size.d[3] = stoi(v[3]);
        } else return true;

        if (size.d[2] != size.d[3]) std::cerr << "Warning only squared input are currently supported" << std::endl;

        tensor_name = v_.front();
        return false;
    }
};

class Yolo {
public:
    Yolo();
    ~Yolo();

    static int build_engine(std::string onnx_path, std::string engine_path, OptimDim dyn_dim_profile);

    int init(std::string engine_path);
    std::vector<BBoxInfo> run(sl::Mat left_sl, int orig_image_h, int orig_image_w, float thres);

    sl::Resolution getInferenceSize() {
        return sl::Resolution(input_width, input_height);
    }

private:

    cv::Mat left_cv_rgb;

    float nms = 0.4;

    // Get input dimension size
    std::string input_binding_name = "images"; // input
    std::string output_name = "classes";
    int inputIndex, outputIndex;
    size_t input_width = 0, input_height = 0, batch_size = 1;
    // Yolov6 1x8400x85 //  85=5+80=cxcy+cwch+obj_conf+cls_conf //https://github.com/DefTruth/lite.ai.toolkit/blob/1267584d5dae6269978e17ffd5ec29da496e503e/lite/ort/cv/yolov6.cpp#L97
    // Yolov8/yolov5 1x84x8400
    size_t out_dim = 8400, out_class_number = 80 /*for COCO*/, out_box_struct_number = 4; // https://github.com/ultralytics/yolov3/issues/750#issuecomment-569783354
    size_t output_size = 0;

    YOLO_MODEL_VERSION_OUTPUT_STYLE yolo_model_version;


    float *h_input, *h_output;
    float *d_input, *d_output;

    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;

    bool is_init = false;


};

#endif /* YOLO_HPP */

