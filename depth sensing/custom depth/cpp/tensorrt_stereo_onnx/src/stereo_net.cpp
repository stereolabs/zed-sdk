#include "stereo_net.hpp"

#include <NvOnnxParser.h>
#include <fstream>
#include <iostream>
#include <memory>

using namespace nvinfer1;

namespace {
    class Logger : public ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cout << "[TRT] " << msg << std::endl;
        }
    } gLogger;

    inline size_t volume(const Dims& d) {
        size_t v = 1;
        for (int i = 0; i < d.nbDims; i++)
            v *= static_cast<size_t>(d.d[i] > 0 ? d.d[i] : 1);
        return v;
    }
} // namespace

StereoNet::~StereoNet() {
    if (d_output)
        cudaFree(d_output);
#if NV_TENSORRT_MAJOR < 10
    if (context)
        context->destroy();
    if (engine)
        engine->destroy();
    if (runtime)
        runtime->destroy();
#else
    delete context;
    delete engine;
    delete runtime;
#endif
}

bool StereoNet::buildEngineFromOnnx(const std::string& onnx_path, const std::string& engine_path) {
    std::cout << "Building TensorRT engine from " << onnx_path << " (cached at " << engine_path << ", this can take a few minutes)..."
              << std::endl;
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
    if (!builder)
        return false;

#if NV_TENSORRT_MAJOR >= 10
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));
#else
    auto network = std::unique_ptr<INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))
    );
#endif
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse " << onnx_path << std::endl;
        return false;
    }

    auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (builder->platformHasFastFp16())
        config->setFlag(BuilderFlag::kFP16);

    auto serialized = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serialized)
        return false;

    std::ofstream f(engine_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    return loadEngine(engine_path);
}

bool StereoNet::loadEngine(const std::string& engine_path) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f.good())
        return false;
    f.seekg(0, std::ios::end);
    const size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    f.read(blob.data(), size);

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(blob.data(), size);
    if (!engine)
        return false;
    context = engine->createExecutionContext();
    if (!context)
        return false;

    // Identify IO: two inputs (left first, in declaration order), first output = disparity
    std::vector<std::string> inputs, outputs;
    std::vector<Dims> input_dims, output_dims;
#if NV_TENSORRT_MAJOR >= 10 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5)
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            inputs.push_back(name);
            input_dims.push_back(engine->getTensorShape(name));
        } else {
            outputs.push_back(name);
            output_dims.push_back(engine->getTensorShape(name));
        }
    }
#else
    for (int i = 0; i < engine->getNbBindings(); i++) {
        if (engine->bindingIsInput(i)) {
            inputs.push_back(engine->getBindingName(i));
            input_dims.push_back(engine->getBindingDimensions(i));
        } else {
            outputs.push_back(engine->getBindingName(i));
            output_dims.push_back(engine->getBindingDimensions(i));
        }
    }
#endif

    if (inputs.size() != 2 || outputs.empty()) {
        std::cerr << "Expected a network with 2 image inputs and at least 1 output, found " << inputs.size() << " input(s) / "
                  << outputs.size() << " output(s)" << std::endl;
        return false;
    }

    input_left_name = inputs[0];
    input_right_name = inputs[1];
    output_name = outputs[0];

    const Dims in_d = input_dims[0]; // expect (1,C,H,W)
    if (in_d.nbDims != 4) {
        std::cerr << "Expected NCHW inputs, got " << in_d.nbDims << " dims" << std::endl;
        return false;
    }
    net_c = in_d.d[1];
    net_h = in_d.d[2];
    net_w = in_d.d[3];
    input_size_elems = volume(in_d);
    output_size_elems = volume(output_dims[0]);

    if (output_size_elems < static_cast<size_t>(net_w * net_h)) {
        std::cerr << "Disparity output is smaller than the input resolution, unsupported layout" << std::endl;
        return false;
    }

    cudaMalloc(reinterpret_cast<void**>(&d_output), output_size_elems * sizeof(float));

    std::cout << "Stereo network ready: inputs '" << input_left_name << "' / '" << input_right_name << "' (" << net_c << "x" << net_h << "x"
              << net_w << "), output '" << output_name << "'" << std::endl;
    return true;
}

bool StereoNet::init(const std::string& model_path) {
    if (model_path.size() > 5 && model_path.substr(model_path.size() - 5) == ".onnx") {
        const std::string engine_path = model_path.substr(0, model_path.size() - 5) + ".engine";
        std::ifstream cached(engine_path, std::ios::binary);
        if (cached.good())
            return loadEngine(engine_path);
        return buildEngineFromOnnx(model_path, engine_path);
    }
    return loadEngine(model_path);
}

bool StereoNet::infer(const float* d_left, const float* d_right, cudaStream_t stream) {
    // The inputs are already on the GPU (sl::Camera::retrieveTensor output): bind them directly.
#if NV_TENSORRT_MAJOR >= 10 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 5)
    context->setTensorAddress(input_left_name.c_str(), const_cast<float*>(d_left));
    context->setTensorAddress(input_right_name.c_str(), const_cast<float*>(d_right));
    context->setTensorAddress(output_name.c_str(), d_output);
    return context->enqueueV3(stream);
#else
    void* bindings[3] = {const_cast<float*>(d_left), const_cast<float*>(d_right), d_output};
    return context->enqueueV2(bindings, stream, nullptr);
#endif
}
