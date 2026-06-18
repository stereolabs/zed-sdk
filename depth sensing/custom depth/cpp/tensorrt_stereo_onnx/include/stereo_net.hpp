#ifndef STEREO_NET_HPP
#define STEREO_NET_HPP

#include <string>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime.h>

// Minimal generic TensorRT runner for stereo disparity networks exported to ONNX with:
//  - two image inputs (left then right), NCHW, float32, 1 or 3 channels, fixed size
//  - one disparity output, (1[,1],H,W) float32, in pixels at the network resolution
// The first run builds and caches a .engine file next to the .onnx (like the custom OD samples).
class StereoNet {
public:
    StereoNet() = default;
    ~StereoNet();

    // model_path: .onnx (engine built + cached on first run) or .engine
    bool init(const std::string& model_path);

    // d_left / d_right: DEVICE pointers to densely packed CHW float32 blobs of size
    // netChannels*netH*netW, already pre-processed (e.g. from sl::Camera::retrieveTensor).
    // Bound directly as network inputs, no copy. Asynchronous on the given stream.
    bool infer(const float* d_left, const float* d_right, cudaStream_t stream);

    // device pointer to the disparity output (netH x netW float32, contiguous)
    float* getOutputDevicePtr() const {
        return d_output;
    }

    int netWidth() const {
        return net_w;
    }
    int netHeight() const {
        return net_h;
    }
    int netChannels() const {
        return net_c;
    }

private:
    bool buildEngineFromOnnx(const std::string& onnx_path, const std::string& engine_path);
    bool loadEngine(const std::string& engine_path);

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    std::string input_left_name, input_right_name, output_name;
    int net_w = 0, net_h = 0, net_c = 3;
    size_t input_size_elems = 0, output_size_elems = 0;

    float* d_output = nullptr;
};

#endif // STEREO_NET_HPP
