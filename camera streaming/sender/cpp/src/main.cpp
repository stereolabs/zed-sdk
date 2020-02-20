///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
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

/************************************************************************************************
 ** This sample shows how to stream remotely the video of a ZED camera. Any application using   **
 ** the ZED SDK can receive and process this stream. See Camera Streaming/Receiver example.     **
 *************************************************************************************************/

// Standard includes
#include <stdio.h>
#include <string.h>

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

void print(std::string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, std::string msg_suffix = "");
int parseArgs(int argc, char **argv, sl::InitParameters& param);

int main(int argc, char **argv) {
    // Create a ZED camera
    Camera zed;

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.camera_resolution = sl::RESOLUTION::HD720;
    initParameters.depth_mode = DEPTH_MODE::NONE;
    initParameters.sdk_verbose = true;
    int res_arg = parseArgs(argc, argv, initParameters);

    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != ERROR_CODE::SUCCESS) {
        print("Opening camera error: ", err);
        zed.close();
        return -1; // Quit if an error occurred
    }

    StreamingParameters stream_params;
    stream_params.codec = STREAMING_CODEC::H264;
    stream_params.bitrate = 10000;
    if (argc == 2 && res_arg == 1) stream_params.port = atoi(argv[1]);
    if (argc > 2) stream_params.port = atoi(argv[2]);

    err = zed.enableStreaming(stream_params);
    if (err != ERROR_CODE::SUCCESS) {
        print("Streaming initialization error: ", err);
        return -2; // Quit if an error occurred
    }

    print("Streaming on port " + std::to_string(stream_params.port));

    SetCtrlHandler();

    while (!exit_app) {
        if (zed.grab() != ERROR_CODE::SUCCESS)
            sleep_ms(1);
    }

    // disable Streaming
    zed.disableStreaming();

    // close the Camera
    zed.close();
    return 0;
}

void print(std::string msg_prefix, ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    else
        std::cout << " ";
    std::cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

int parseArgs(int argc, char **argv, sl::InitParameters& param) {
    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        } else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        } else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        } else
            return 1;
    } else {
        // Default
        return 1;
    }
    return 0;
}

