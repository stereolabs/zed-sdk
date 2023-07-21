
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "model.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "cuda_utils.h"
#include "logging.h"


Logger gLogger;
// using namespace sl;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void serialize_engine(std::string &wts_name, std::string &engine_name, std::string &sub_type) {
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    if (sub_type == "n") {
        serialized_engine = buildEngineYolov8n(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "s") {
        serialized_engine = buildEngineYolov8s(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "m") {
        serialized_engine = buildEngineYolov8m(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "l") {
        serialized_engine = buildEngineYolov8l(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "x") {
        serialized_engine = buildEngineYolov8x(builder, config, DataType::kFLOAT, wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}


void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                    float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device, std::string cuda_post_process) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **) input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    if (cuda_post_process == "c") {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void **)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process) {
    // infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    context.enqueue(batchsize, buffers, stream, nullptr);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float *)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);//cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference and gpu postprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}


bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, std::string &img_dir, std::string &sub_type, std::string &cuda_post_process) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && argc == 5) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        sub_type = std::string(argv[4]);
    } else if (std::string(argv[1]) == "-d" && argc == 5) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
        cuda_post_process = std::string(argv[4]);
    } else {
        return false;
    }
    return true;
}

// int getOCVtype(sl::MAT_TYPE type);

// cv::Mat slMat2cvMat(Mat& input);
// int getOCVtype(sl::MAT_TYPE type);

// cv::Mat slMat2cvMat(Mat& input) {
//     // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
//     // cv::Mat and sl::Mat will share a single memory structure
//     return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
// }

// // Mapping between MAT_TYPE and CV_TYPE
// int getOCVtype(sl::MAT_TYPE type) {
//     int cv_type = -1;
//     switch (type) {
//         case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
//         case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
//         case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
//         case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
//         case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
//         case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
//         case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
//         case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
//         default: break;
//     }
//     return cv_type;
// }

