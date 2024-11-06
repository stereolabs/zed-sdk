#include "yolov8-seg_optim.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "utils.h"
#include "yolov8-seg_common.hpp"

static Logger gLogger{nvinfer1::ILogger::Severity::kINFO};

int OptimDim::setFromString(std::string const& arg) {
    // "images:1x3x512x512"
    std::vector<std::string> const v_{split_str(arg, ":")};
    if (v_.size() != 2U)
        return -1;

    std::string const dims_str{v_.back()};
    std::vector<std::string> const v{split_str(dims_str, "x")};

    size.nbDims = 4;
    // assuming batch is 1 and channel is 3
    size.d[0U] = 1;
    size.d[1U] = 3;

    if (v.size() == 2U) {
        size.d[2U] = stoi(v[0U]);
        size.d[3U] = stoi(v[1U]);
    }
    else if (v.size() == 3U) {
        size.d[2U] = stoi(v[1U]);
        size.d[3U] = stoi(v[2U]);
    }
    else if (v.size() == 4U) {
        size.d[2U] = stoi(v[2U]);
        size.d[3U] = stoi(v[3U]);
    }
    else
        return -1;

    if (size.d[2U] != size.d[3U])
        std::cerr << "Warning only squared input are currently supported" << std::endl;

    tensor_name = v_.front();
    return 0;
}

int build_engine(std::string const& onnx_path, std::string const& engine_path, OptimDim const& dyn_dim_profile) {
    std::vector<uint8_t> onnx_file_content;
    if (readFile(onnx_path, onnx_file_content) != 0)
        return -1;

    if (onnx_file_content.empty())
        return -1;

    nvinfer1::ICudaEngine * engine;
    // Create engine (onnx)
    std::cout << "Creating engine from onnx model" << std::endl;

    auto builder = nvinfer1::createInferBuilder(gLogger);
    if (!builder) {
        std::cerr << "createInferBuilder failed" << std::endl;
        return 1;
    }

    auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    if (!network) {
        std::cerr << "createNetwork failed" << std::endl;
        return -1;
    }

    auto config = builder->createBuilderConfig();
    if (!config) {
        std::cerr << "createBuilderConfig failed" << std::endl;
        return -1;
    }

    ////////// Dynamic dimensions handling : support only 1 size at a time
    if (!dyn_dim_profile.tensor_name.empty()) {

        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

        profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dyn_dim_profile.size);
        profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dyn_dim_profile.size);
        profile->setDimensions(dyn_dim_profile.tensor_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dyn_dim_profile.size);

        config->addOptimizationProfile(profile);
#if NV_TENSORRT_MAJOR < 10
        builder->setMaxBatchSize(1);
#endif
    }

    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser) {
        std::cerr << "nvonnxparser::createParser failed" << std::endl;
        return -1;
    }

    bool parsed = false;
    unsigned char *onnx_model_buffer = onnx_file_content.data();
    size_t onnx_model_buffer_size = onnx_file_content.size() * sizeof (char);
    parsed = parser->parse(onnx_model_buffer, onnx_model_buffer_size);

    if (!parsed) {
        std::cerr << "onnx file parsing failed" << std::endl;
        return -1;
    }

    if (builder->platformHasFastFp16()) {
        std::cout << "FP16 enabled!" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    //////////////// Actual engine building

#if NV_TENSORRT_MAJOR >= 10
    engine = nullptr;
    if (builder->isNetworkSupported(*network, *config)) {
        std::unique_ptr<nvinfer1::IHostMemory> serializedEngine{builder->buildSerializedNetwork(*network, *config)};
        if (serializedEngine != nullptr && serializedEngine.get() && serializedEngine->size() > 0) {
            nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);
            engine = infer->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
        }
    }
#else
    engine = builder->buildEngineWithConfig(*network, *config);
#endif

    onnx_file_content.clear();

    // write plan file if it is specified
    if (engine == nullptr) return -1;
    nvinfer1::IHostMemory* ptr = engine->serialize();
    assert(ptr);
    if (ptr == nullptr) return -1;

    FILE *fp = fopen(engine_path.c_str(), "wb");
    fwrite(reinterpret_cast<const char*> (ptr->data()), ptr->size() * sizeof (char), 1, fp);
    fclose(fp);

#if NV_TENSORRT_MAJOR >= 10
    delete parser;
    delete network;
    delete config;
    delete builder;
    delete engine;
#else
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    engine->destroy();
#endif

    return 0;
}
