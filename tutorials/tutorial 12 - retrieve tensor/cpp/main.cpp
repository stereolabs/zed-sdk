///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

#include <sl/Camera.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// Helper to save a NCHW BGR float tensor to a PNG image
// This function reverses the normalization formula: normalized = (pixel * scale - mean) / std
void saveNCHWBGRTensorAsPNG(sl::Tensor& tensor, const std::string& filename, const sl::TensorParameters& params) {
    // 1. Download data from GPU to CPU
    sl::ERROR_CODE err = tensor.updateCPUfromGPU();
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to download tensor from GPU: " << err << std::endl;
        return;
    }

    // 2. Get tensor dimensions
    // For NCHW: [Batch, Channels, Height, Width]
    std::vector<size_t> dims = tensor.getDims();
    size_t C = dims[1]; // Number of channels (3 for RGB)
    size_t H = dims[2]; // Height of one plane
    size_t W = dims[3]; // Width of one plane

    // Pointer to the float data
    float* data = tensor.getPtr<float>(sl::MEM::CPU);
    if (data == nullptr) {
        std::cerr << "CPU pointer is null!" << std::endl;
        return;
    }

    // Get the step (stride) in float elements
    size_t step = tensor.getStep<float>(sl::MEM::CPU);

    // Denormalization: original_pixel = (normalized * std + mean) / scale
    sl::float3 scale_inv = {1.0f / params.scale.x, 1.0f / params.scale.y, 1.0f / params.scale.z};

    // Create output Mat (BGR for PNG compatibility)
    sl::Mat output(sl::Resolution(W, H), sl::MAT_TYPE::U8_C3, sl::MEM::CPU);

    // Plane size for NCHW indexing
    size_t plane_size = H * step;

    for (size_t y = 0; y < H; ++y) {
        auto row_ptr = output.getPtr<sl::uchar3>(sl::MEM::CPU) + y * output.getStep<sl::uchar3>(sl::MEM::CPU);
        for (size_t x = 0; x < W; ++x) {
            size_t idx = y * step + x;

            // Get normalized values from each channel plane
            float b_norm = data[idx];
            float g_norm = data[idx + plane_size];
            float r_norm = data[idx + 2 * plane_size];

            // Denormalize
            float b_val = (b_norm * params.std.z + params.mean.z) * scale_inv.z;
            float r_val = (r_norm * params.std.x + params.mean.x) * scale_inv.x;
            float g_val = (g_norm * params.std.y + params.mean.y) * scale_inv.y;

            // Clamp to [0, 255] and convert to uint8
            sl::uchar3 pixel;
            pixel.x = static_cast<unsigned char>(std::clamp(std::round(b_val), 0.0f, 255.0f)); // B
            pixel.y = static_cast<unsigned char>(std::clamp(std::round(g_val), 0.0f, 255.0f)); // G
            pixel.z = static_cast<unsigned char>(std::clamp(std::round(r_val), 0.0f, 255.0f)); // R

            row_ptr[x] = pixel;
        }
    }

    // Save as PNG
    err = output.write(filename.c_str());
    if (err == sl::ERROR_CODE::SUCCESS) {
        std::cout << "  Saved: " << filename << std::endl;
    } else {
        std::cerr << "  Failed to save " << filename << ": " << err << std::endl;
    }
}

int main(int argc, char** argv) {
    // Create a ZED camera object
    sl::Camera zed;

    // Set configuration parameters
    sl::InitParameters init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.camera_fps = 30;
    init_parameters.depth_mode = sl::DEPTH_MODE::NONE;

    // Open the camera
    sl::ERROR_CODE err = zed.open(init_parameters);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Error " << err << ", exit program." << std::endl;
        return EXIT_FAILURE;
    }

    // Prepare Tensor buffer
    sl::Tensor tensor;

    // Tensor parameters - Configure the output format for deep learning inference
    sl::TensorParameters tensor_params;
    tensor_params.target_size = sl::Resolution(501, 501);
    tensor_params.batch_size = 1;
    tensor_params.layout = sl::TensorParameters::LAYOUT::NCHW;
    tensor_params.pixel_type = sl::TensorParameters::PIXEL_TYPE::FLOAT;
    tensor_params.color_format = sl::TensorParameters::COLOR_FORMAT::RGB;
    tensor_params.memory_type = sl::MEM::GPU;
    tensor_params.scale = sl::float3(1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f);
    tensor_params.stretch = true;
    // Default normalization: ImageNet mean={0.485, 0.456, 0.406}, std={0.229, 0.224, 0.225}
    // no normalization
    tensor_params.mean = sl::float3(0.0f, 0.0f, 0.0f);
    tensor_params.std = sl::float3(1.0f, 1.0f, 1.0f);

    sl::Mat depth_map;

    // Main loop
    int frame_count = 0;
    while (frame_count < 50) {
        if (zed.grab() <= sl::ERROR_CODE::SUCCESS) {
            // Retrieve pre-processed tensor for inference
            sl::ERROR_CODE res = zed.retrieveTensor(tensor, tensor_params);

            if (res == sl::ERROR_CODE::SUCCESS) {
                if (frame_count % 10 == 0) {
                    std::cout << "Frame " << frame_count << ":" << std::endl;

                    std::vector<size_t> dims = tensor.getDims();
                    std::cout << "  Dims: [" << dims[0] << ", " << dims[1] << ", " << dims[2] << ", " << dims[3] << "]" << std::endl;

                    std::string filename = "tensor_" + std::to_string(frame_count) + ".png";
                    saveNCHWBGRTensorAsPNG(tensor, filename, tensor_params);

                    // Optionally retrieve and display depth map info
                    zed.retrieveImage(depth_map, sl::VIEW::DEPTH);
                    filename = "depth_" + std::to_string(frame_count) + ".png";
                    depth_map.write(filename.c_str());
                    std::cout << "  Saved depth map: " << filename << std::endl;
                }
            } else {
                std::cout << "Error retrieving tensor: " << res << std::endl;
            }
            frame_count++;
        }
    }

    zed.close();
    return EXIT_SUCCESS;
}
