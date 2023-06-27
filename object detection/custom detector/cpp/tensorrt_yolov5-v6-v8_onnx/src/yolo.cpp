#include "yolo.hpp"
#include "NvOnnxParser.h"

using namespace nvinfer1;

static Logger gLogger;

inline int clamp(int val, int min, int max) {
    if (val <= min) return min;
    if (val >= max) return max;
    return val;
}


#define WEIGHTED_NMS

std::vector<BBoxInfo> nonMaximumSuppression(const float nmsThresh, std::vector<BBoxInfo> binfo) {

    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min) {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto computeIoU = [&overlap1D](BBox& bbox1, BBox & bbox2) -> float {
        float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
        float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
        float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(), [](const BBoxInfo& b1, const BBoxInfo & b2) {
        return b1.prob > b2.prob; });

    std::vector<BBoxInfo> out;

#if defined(WEIGHTED_NMS)
    std::vector<std::vector < BBoxInfo> > weigthed_nms_candidates;
#endif
    for (auto& i : binfo) {
        bool keep = true;

#if defined(WEIGHTED_NMS)
        int j_index = 0;
#endif

        for (auto& j : out) {
            if (keep) {
                float overlap = computeIoU(i.box, j.box);
                keep = overlap <= nmsThresh;
#if defined(WEIGHTED_NMS)
                if (!keep && fabs(j.prob - i.prob) < 0.52f) // add label similarity check
                    weigthed_nms_candidates[j_index].push_back(i);
#endif
            } else
                break;

#if defined(WEIGHTED_NMS)  
            j_index++;
#endif

        }
        if (keep) {
            out.push_back(i);
#if defined(WEIGHTED_NMS)
            weigthed_nms_candidates.emplace_back();
            weigthed_nms_candidates.back().clear();
#endif
        }
    }

#if defined(WEIGHTED_NMS)

    for (int i = 0; i < out.size(); i++) {
        // the best confidence
        BBoxInfo& best = out[i];
        float sum_tl_x = best.box.x1 * best.prob;
        float sum_tl_y = best.box.y1 * best.prob;
        float sum_br_x = best.box.x2 * best.prob;
        float sum_br_y = best.box.y2 * best.prob;

        float weight = best.prob;
        for (auto& it : weigthed_nms_candidates[i]) {
            sum_tl_x += it.box.x1 * it.prob;
            sum_tl_y += it.box.y1 * it.prob;
            sum_br_x += it.box.x2 * it.prob;
            sum_br_y += it.box.y2 * it.prob;
            weight += it.prob;
        }

        weight = 1.f / weight;
        best.box.x1 = sum_tl_x * weight;
        best.box.y1 = sum_tl_y * weight;
        best.box.x2 = sum_br_x * weight;
        best.box.y2 = sum_br_y * weight;
    }

#endif

    return out;
}

Yolo::Yolo() {
}

Yolo::~Yolo() {
    if (is_init) {

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();

        delete[] h_input;
        delete[] h_output;
    }
    is_init = false;
}

int Yolo::build_engine(std::string onnx_path, std::string engine_path, OptimDim dyn_dim_profile) {


    std::vector<uint8_t> onnx_file_content;
    if (readFile(onnx_path, onnx_file_content)) return 1;

    if ((!onnx_file_content.empty())) {

        ICudaEngine * engine;
        // Create engine (onnx)
        std::cout << "Creating engine from onnx model" << std::endl;

        gLogger.setReportableSeverity(Severity::kINFO);
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            std::cerr << "createInferBuilder failed" << std::endl;
            return 1;
        }

        auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicitBatch);

        if (!network) {
            std::cerr << "createNetwork failed" << std::endl;
            return 1;
        }

        auto config = builder->createBuilderConfig();
        if (!config) {
            std::cerr << "createBuilderConfig failed" << std::endl;
            return 1;
        }

        ////////// Dynamic dimensions handling : support only 1 size at a time
        if (!dyn_dim_profile.tensor_name.empty()) {

            IOptimizationProfile* profile = builder->createOptimizationProfile();

            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kMIN, dyn_dim_profile.size);
            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kOPT, dyn_dim_profile.size);
            profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), OptProfileSelector::kMAX, dyn_dim_profile.size);

            config->addOptimizationProfile(profile);
            builder->setMaxBatchSize(1);
        }

        auto parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser) {
            std::cerr << "nvonnxparser::createParser failed" << std::endl;
            return 1;
        }

        bool parsed = false;
        unsigned char *onnx_model_buffer = onnx_file_content.data();
        size_t onnx_model_buffer_size = onnx_file_content.size() * sizeof (char);
        parsed = parser->parse(onnx_model_buffer, onnx_model_buffer_size);

        if (!parsed) {
            std::cerr << "onnx file parsing failed" << std::endl;
            return 1;
        }

        if (builder->platformHasFastFp16()) {
            std::cout << "FP16 enabled!" << std::endl;
            config->setFlag(BuilderFlag::kFP16);
        }

        //////////////// Actual engine building

        engine = builder->buildEngineWithConfig(*network, *config);

        onnx_file_content.clear();

        // write plan file if it is specified        
        if (engine == nullptr) return 1;
        IHostMemory* ptr = engine->serialize();
        assert(ptr);
        if (ptr == nullptr) return 1;

        FILE *fp = fopen(engine_path.c_str(), "wb");
        fwrite(reinterpret_cast<const char*> (ptr->data()), ptr->size() * sizeof (char), 1, fp);
        fclose(fp);

        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();

        engine->destroy();

        return 0;
    } else return 1;


}

int Yolo::init(std::string engine_name) {


    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    if (!trtModelStream) return 1;
    file.read(trtModelStream, size);
    file.close();

    // prepare input data ---------------------------
    runtime = createInferRuntime(gLogger);
    if (runtime == nullptr) return 1;
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    if (engine == nullptr) return 1;
    context = engine->createExecutionContext();
    if (context == nullptr) return 1;

    delete[] trtModelStream;
    if (engine->getNbBindings() != 2) return 1;


    const int bindings = engine->getNbBindings();
    for (int i = 0; i < bindings; i++) {
        if (engine->bindingIsInput(i)) {
            input_binding_name = engine->getBindingName(i);
            Dims bind_dim = engine->getBindingDimensions(i);
            input_width = bind_dim.d[3];
            input_height = bind_dim.d[2];
            inputIndex = i;
            std::cout << "Inference size : " << input_height << "x" << input_width << std::endl;
        }//if (engine->getTensorIOMode(engine->getBindingName(i)) == TensorIOMode::kOUTPUT) 
        else {
            output_name = engine->getBindingName(i);
            // fill size, allocation must be done externally
            outputIndex = i;
            Dims bind_dim = engine->getBindingDimensions(i);
            size_t batch = bind_dim.d[0];
            if (batch > batch_size) {
                std::cout << "batch > 1 not supported" << std::endl;
                return 1;
            }
            size_t dim1 = bind_dim.d[1];
            size_t dim2 = bind_dim.d[2];

            if (dim1 > dim2) {
                // Yolov6 1x8400x85 //  85=5+80=cxcy+cwch+obj_conf+cls_conf
                out_dim = dim1;
                out_box_struct_number = 5;
                out_class_number = dim2 - out_box_struct_number;
                yolo_model_version = YOLO_MODEL_VERSION_OUTPUT_STYLE::YOLOV6;
                std::cout << "YOLOV6 format" << std::endl;
            } else {
                // Yolov8 1x84x8400
                out_dim = dim2;
                out_box_struct_number = 4;
                out_class_number = dim1 - out_box_struct_number;
                yolo_model_version = YOLO_MODEL_VERSION_OUTPUT_STYLE::YOLOV8_V5;
                std::cout << "YOLOV8/YOLOV5 format" << std::endl;
            }
        }
    }
    output_size = out_dim * (out_class_number + out_box_struct_number);
    h_input = new float[batch_size * 3 * input_height * input_width];
    h_output = new float[batch_size * output_size];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * 3 * input_height * input_width * sizeof (float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * output_size * sizeof (float)));
    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (batch_size != 1) return 1; // This sample only support batch 1 for now

    is_init = true;
    return 0;
}

std::vector<BBoxInfo> Yolo::run(sl::Mat left_sl, int orig_image_h, int orig_image_w, float thres) {
    std::vector<BBoxInfo> binfo;

    size_t frame_s = input_height * input_width;

    /////// Preparing inference
    cv::Mat left_cv_rgba = slMat2cvMat(left_sl);
    cv::cvtColor(left_cv_rgba, left_cv_rgb, cv::COLOR_BGRA2BGR);
    if (left_cv_rgb.empty()) return binfo;
    cv::Mat pr_img = preprocess_img(left_cv_rgb, input_width, input_height); // letterbox BGR to RGB
    int i = 0;
    int batch = 0;
    for (int row = 0; row < input_height; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < input_width; ++col) {
            h_input[batch * 3 * frame_s + i] = (float) uc_pixel[2] / 255.0;
            h_input[batch * 3 * frame_s + i + frame_s] = (float) uc_pixel[1] / 255.0;
            h_input[batch * 3 * frame_s + i + 2 * frame_s] = (float) uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    /////// INFERENCE
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, batch_size * 3 * frame_s * sizeof (float), cudaMemcpyHostToDevice, stream));

    std::vector<void*> d_buffers_nvinfer(2);
    d_buffers_nvinfer[inputIndex] = d_input;
    d_buffers_nvinfer[outputIndex] = d_output;
    context->enqueueV2(&d_buffers_nvinfer[0], stream, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, batch_size * output_size * sizeof (float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);


    /////// Extraction

    float scalingFactor = std::min(static_cast<float> (input_width) / orig_image_w, static_cast<float> (input_height) / orig_image_h);
    float xOffset = (input_width - scalingFactor * orig_image_w) * 0.5f;
    float yOffset = (input_height - scalingFactor * orig_image_h) * 0.5f;
    scalingFactor = 1.f / scalingFactor;
    float scalingFactor_x = scalingFactor;
    float scalingFactor_y = scalingFactor;


    switch (yolo_model_version) {
        default:
        case YOLO_MODEL_VERSION_OUTPUT_STYLE::YOLOV8_V5:
        {
            // https://github.com/triple-Mu/YOLOv8-TensorRT/blob/df11cec3abaab7fefb28fb760f1cebbddd5ec826/csrc/detect/normal/include/yolov8.hpp#L343
            auto num_channels = out_class_number + out_box_struct_number;
            auto num_anchors = out_dim;
            auto num_labels = out_class_number;

            auto& dw = xOffset;
            auto& dh = yOffset;

            auto& width = orig_image_w;
            auto& height = orig_image_h;

            cv::Mat output = cv::Mat(
                    num_channels,
                    num_anchors,
                    CV_32F,
                    static_cast<float*> (h_output)
                    );
            output = output.t();
            for (int i = 0; i < num_anchors; i++) {
                auto row_ptr = output.row(i).ptr<float>();
                auto bboxes_ptr = row_ptr;
                auto scores_ptr = row_ptr + out_box_struct_number;
                auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_labels);
                float score = *max_s_ptr;
                if (score > thres) {
                    int label = max_s_ptr - scores_ptr;

                    BBoxInfo bbi;

                    float x = *bboxes_ptr++ - dw;
                    float y = *bboxes_ptr++ - dh;
                    float w = *bboxes_ptr++;
                    float h = *bboxes_ptr;

                    float x0 = clamp((x - 0.5f * w) * scalingFactor_x, 0.f, width);
                    float y0 = clamp((y - 0.5f * h) * scalingFactor_y, 0.f, height);
                    float x1 = clamp((x + 0.5f * w) * scalingFactor_x, 0.f, width);
                    float y1 = clamp((y + 0.5f * h) * scalingFactor_y, 0.f, height);

                    cv::Rect_<float> bbox;
                    bbox.x = x0;
                    bbox.y = y0;
                    bbox.width = x1 - x0;
                    bbox.height = y1 - y0;

                    bbi.box.x1 = x0;
                    bbi.box.y1 = y0;
                    bbi.box.x2 = x1;
                    bbi.box.y2 = y1;

                    if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2)) break;

                    bbi.label = label;
                    bbi.prob = score;

                    binfo.push_back(bbi);
                }
            }
            break;
        }
        case YOLO_MODEL_VERSION_OUTPUT_STYLE::YOLOV6:
        {
            // https://github.com/DefTruth/lite.ai.toolkit/blob/1267584d5dae6269978e17ffd5ec29da496e503e/lite/ort/cv/yolov6.cpp#L97

            auto& dw_ = xOffset;
            auto& dh_ = yOffset;

            auto& width = orig_image_w;
            auto& height = orig_image_h;

            const unsigned int num_anchors = out_dim; // n = ?
            const unsigned int num_classes = out_class_number; // - out_box_struct_number; // 80

            for (unsigned int i = 0; i < num_anchors; ++i) {
                const float *offset_obj_cls_ptr = h_output + (i * (num_classes + 5)); // row ptr
                float obj_conf = offset_obj_cls_ptr[4]; /*always == 1 for some reasons*/
                float cls_conf = offset_obj_cls_ptr[5];

                // The confidence is remapped because it output basically garbage with conf < ~0.1 and we don't want to clamp it either
                const float conf_offset = 0.1;
                const float input_start = 0;
                const float output_start = input_start;
                const float output_end = 1;
                const float input_end = output_end - conf_offset;

                float conf = (obj_conf * cls_conf) - conf_offset;
                if (conf < 0) conf = 0;
                conf = (conf - input_start) / (input_end - input_start) * (output_end - output_start) + output_start;

                if (conf > thres) {

                    unsigned int label = 0;
                    for (unsigned int j = 0; j < num_classes; ++j) {
                        float tmp_conf = offset_obj_cls_ptr[j + 5];
                        if (tmp_conf > cls_conf) {
                            cls_conf = tmp_conf;
                            label = j;
                        }
                    } // argmax

                    BBoxInfo bbi;

                    float cx = offset_obj_cls_ptr[0];
                    float cy = offset_obj_cls_ptr[1];
                    float w = offset_obj_cls_ptr[2];
                    float h = offset_obj_cls_ptr[3];
                    float x1 = ((cx - w / 2.f) - (float) dw_) * scalingFactor_x;
                    float y1 = ((cy - h / 2.f) - (float) dh_) * scalingFactor_y;
                    float x2 = ((cx + w / 2.f) - (float) dw_) * scalingFactor_x;
                    float y2 = ((cy + h / 2.f) - (float) dh_) * scalingFactor_y;

                    bbi.box.x1 = std::max(0.f, x1);
                    bbi.box.y1 = std::max(0.f, y1);
                    bbi.box.x2 = std::min(x2, (float) width - 1.f);
                    bbi.box.y2 = std::min(y2, (float) height - 1.f);

                    if ((bbi.box.x1 > bbi.box.x2) || (bbi.box.y1 > bbi.box.y2)) break;

                    bbi.label = label;
                    bbi.prob = conf;

                    binfo.push_back(bbi);
                }
            }
            break;
        }
    };

    /// NMS
    binfo = nonMaximumSuppression(nms, binfo);

    return binfo;
}
