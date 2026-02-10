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

/**
 * @file main.cpp
 * @brief ZED One Streaming Sender Sample
 *
 * This sample demonstrates how to stream live video from a ZED One camera
 * over the network using H.265 codec.
 */

// ZED includes
#include <sl/CameraOne.hpp>

// Sample includes
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

void printUsage() {
    cout << "Usage: ZED_One_Streaming_Sender [port]" << endl;
    cout << endl;
    cout << "Arguments:" << endl;
    cout << "  port  - Network port for streaming (default: 30000)" << endl;
    cout << endl;
    cout << "Press Ctrl+C to stop streaming." << endl;
}

int main(int argc, char** argv) {
    // Create a ZED camera
    CameraOne zed;

    // Set configuration parameters
    InitParametersOne init_parameters;
    init_parameters.camera_resolution = sl::RESOLUTION::AUTO;
    init_parameters.sdk_verbose = 1;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Configure streaming parameters
    StreamingParameters stream_params;
    if (argc >= 2) {
        stream_params.port = atoi(argv[1]);
    }

    // Enable streaming
    returned_state = zed.enableStreaming(stream_params);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Streaming initialization error", returned_state);
        zed.close();
        return EXIT_FAILURE;
    }

    // Print streaming info
    auto cam_info = zed.getCameraInformation();
    std::cout << "\n=== Streaming Started ===" << std::endl;
    std::cout << "Port:       " << stream_params.port << std::endl;
    std::cout << "Resolution: " << cam_info.camera_configuration.resolution.width << "x" << cam_info.camera_configuration.resolution.height
              << std::endl;
    std::cout << "FPS:        " << cam_info.camera_configuration.fps << std::endl;
    std::cout << "=========================\n" << std::endl;

    std::cout << "Streaming on port " << stream_params.port << "..." << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;

    SetCtrlHandler();

    while (!exit_app) {
        if (zed.grab() > ERROR_CODE::SUCCESS) {
            sl::sleep_ms(5);
        }
    }

    std::cout << "\nStopping stream..." << std::endl;

    // Disable streaming
    zed.disableStreaming();
    zed.close();

    return EXIT_SUCCESS;
}
