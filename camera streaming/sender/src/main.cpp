///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2019, STEREOLABS.
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

/****************************************************************************************
 ** This sample shows how to record video in Stereolabs SVO format.					   **
 ** SVO video files can be played with the ZED API and used with its different modules  **
 *****************************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "utils.hpp"

// Using namespace
using namespace sl;

int main(int argc, char **argv) {
    // Create a ZED camera
    Camera zed;

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.camera_resolution = RESOLUTION_HD2K;
    initParameters.depth_mode = DEPTH_MODE_NONE;
	initParameters.sdk_verbose = true;

    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != SUCCESS) {
        std::cout << toString(err) << std::endl;
        zed.close();
        return -1; // Quit if an error occurred
    }

    sl::StreamingParameters stream_params;
    stream_params.codec = sl::STREAMING_CODEC_AVCHD;
    stream_params.bitrate = 8000;
    if(argc > 1) stream_params.port = atoi(argv[1]);

    err = zed.enableStreaming(stream_params);
    if (err != SUCCESS) {
        std::cout << "Streaming initialization error. " << toString(err) << std::endl;
        zed.close();
        return -2;
    }

    std::cout << "Streaming on port " << stream_params.port << std::endl;

    SetCtrlHandler();

    int fc = 0;
    while (!exit_app) {
        if (zed.grab() == SUCCESS) {
            sl::sleep_ms(1);
            fc++;
        }
    }

    // Stop recording
    zed.disableStreaming();
    zed.close();
    return 0;
}
