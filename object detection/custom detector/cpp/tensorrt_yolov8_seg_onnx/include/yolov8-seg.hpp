// https://github.com/triple-Mu/YOLOv8-TensorRT/blob/main/csrc/segment/normal/include/yolov8-seg.hpp

#ifndef SEGMENT_NORMAL_YOLOV8_SEG_HPP
#define SEGMENT_NORMAL_YOLOV8_SEG_HPP
#include "NvInferPlugin.h"
#include "yolov8-seg_common.hpp"
#include <fstream>
#include <sl/Camera.hpp>


#if NV_TENSORRT_MAJOR >= 10
#define trt_name_engine_get_binding_name getIOTensorName
#define trt_name_engine_get_nb_binding getNbIOTensors
#else
#define trt_name_engine_get_binding_name getBindingName
#define trt_name_engine_get_nb_binding getNbBindings
#endif


using namespace seg;

class YOLOv8_seg {
public:
    explicit YOLOv8_seg(const std::string& engine_file_path);
    ~YOLOv8_seg();

    void make_pipe();
    void copy_from_Mat(sl::Mat& image);
    void setParams(sl::Resolution orig_res, sl::Resolution net_res);
    void infer();
    void postprocess(std::vector<Object>& objs,
            float score_thres = 0.25f,
            float iou_thres = 0.65f,
            int topk = 100,
            int seg_channels = 32);

    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*> host_ptrs;
    std::vector<void*> device_ptrs_in; // ref, not allocated
    std::vector<void*> device_ptrs_out; // allocated

    sl::Mat blob; // input tensor

    PreParam pparam;
    cudaStream_t stream = nullptr;

private:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

YOLOv8_seg::YOLOv8_seg(const std::string& engine_file_path) {
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->trt_name_engine_get_nb_binding();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding binding;
        nvinfer1::Dims dims;
#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR > 4)
        nvinfer1::DataType dtype = this->engine->getTensorDataType(this->engine->getIOTensorName(i));
#else
        nvinfer1::DataType dtype = this->engine->getBindingDataType(i);
#endif
        std::string name = this->engine->trt_name_engine_get_binding_name(i);
        binding.name = name;
        binding.dsize = type_to_size(dtype);

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR >= 8 && NV_TENSORRT_MINOR > 4)
        if (engine->getTensorIOMode(engine->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT)
#else
        if (engine->bindingIsInput(i))
#endif
        {
            this->num_inputs += 1;
#if NV_TENSORRT_MAJOR >= 10
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
#else
            dims = this->engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
#if NV_TENSORRT_MAJOR >= 10
            this->context->setInputShape(name.c_str(), dims);
#else
            this->context->setBindingDimensions(i, dims);
#endif
        } else {
#if NV_TENSORRT_MAJOR >= 10
            dims = this->engine->getTensorShape(this->engine->getIOTensorName(i));
#else
            dims = this->context->getBindingDimensions(i);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8_seg::~YOLOv8_seg() {
#if NV_TENSORRT_MAJOR >= 10
    delete this->context;
    delete this->engine;
    delete this->runtime;
#else
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
#endif
    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs_out) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8_seg::make_pipe() {
    this->device_ptrs_in.resize(1);

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs_out.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }
}

void YOLOv8_seg::setParams(sl::Resolution orig_res, sl::Resolution net_res) {
    const float inp_h = net_res.height;
    const float inp_w = net_res.width;
    float height = orig_res.height;
    float width = orig_res.width;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;

    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
}

void YOLOv8_seg::copy_from_Mat(sl::Mat& image) {
    auto& in_binding = this->input_bindings[0];
    auto width = in_binding.dims.d[3];
    auto height = in_binding.dims.d[2];
    this->setParams(image.getResolution(), sl::Resolution(width, height));

#if NV_TENSORRT_MAJOR >= 10
            this->context->setInputShape(in_binding.name.c_str(), nvinfer1::Dims{4,        {1, 3, height, width}});
#else
            this->context->setBindingDimensions(0, nvinfer1::Dims{4,        {1, 3, height, width}});
#endif

    sl::blobFromImage(image, blob, sl::Resolution(width, height), 1 / 255.f, sl::float3(0, 0, 0), sl::float3(1, 1, 1), true, true, stream);

    this->device_ptrs_in[0] = blob.getPtr<float>(sl::MEM::GPU);
}

void YOLOv8_seg::infer() {
#if (NV_TENSORRT_MAJOR < 8) || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
    this->context->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
#else
    for (unsigned int i = 0; i < this->num_inputs; ++i) {
        this->context->setTensorAddress(this->input_bindings[i].name.c_str(), this->device_ptrs_in[i]);
    }
    for (unsigned int i = 0; i < this->num_outputs; ++i) {
        this->context->setTensorAddress(this->output_bindings[i].name.c_str(), this->device_ptrs_out[i]);
    }
    this->context->enqueueV3(this->stream);
#endif
    for (int i = 0; i < this->num_outputs; ++i) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(this->host_ptrs[i], this->device_ptrs_out[i], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8_seg::postprocess(
        std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int seg_channels) {
    objs.clear();
    auto input_h = this->input_bindings[0].dims.d[2];
    auto input_w = this->input_bindings[0].dims.d[3];
    int seg_h = input_h / 4;
    int seg_w = input_w / 4;
    int num_channels, num_anchors, num_classes;
    bool flag = false;
    int bid;
    int bcnt = -1;
    for (auto& o : this->output_bindings) {
        bcnt += 1;
        if (o.dims.nbDims == 3) {
            num_channels = o.dims.d[1];
            num_anchors = o.dims.d[2];
            flag = true;
            bid = bcnt;
        }
    }
    assert(flag);
    num_classes = num_channels - seg_channels - 4;

    auto& dw = this->pparam.dw;
    auto& dh = this->pparam.dh;
    auto& width = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio = this->pparam.ratio;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*> (this->host_ptrs[bid]));
    output = output.t();

    cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F, static_cast<float*> (this->host_ptrs[1 - bid]));

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> mask_confs;
    std::vector<int> indices;

    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto mask_confs_ptr = row_ptr + 4 + num_classes;
        auto max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_classes);
        float score = *max_s_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            cv::Mat mask_conf = cv::Mat(1, seg_channels, CV_32F, mask_confs_ptr);

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            mask_confs.push_back(mask_conf);
        }
    }

#if defined(BATCHED_NMS)
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    cv::Mat masks;
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    if (masks.empty()) {
        // masks is empty
    } else {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(),{seg_h, seg_w});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size((int) width, (int) height), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > 0.5f;
        }
    }
}

#endif  // SEGMENT_NORMAL_YOLOV8_SEG_HPP
