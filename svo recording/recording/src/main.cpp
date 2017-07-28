///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
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

    if (argc != 2) {
        std::cout << "Only the path of the output SVO file should be passed as argument.\n";
        return 1;
    }

    // Create a ZED camera
    Camera zed;

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.camera_resolution = RESOLUTION_HD720;
    initParameters.depth_mode = DEPTH_MODE_NONE;

    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != SUCCESS) {
        std::cout << errorCode2str(err) << std::endl;
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Enable recording with the filename specified in argument
    String path_output(argv[1]);
    err = zed.enableRecording(path_output, SVO_COMPRESSION_MODE_LOSSLESS);

    if (err != SUCCESS) {
        std::cout << "Recording initialization error. " << errorCode2str(err) << std::endl;
        if (err == ERROR_CODE_SVO_RECORDING_ERROR) std::cout << " Note : This error mostly comes from a wrong path or missing writing permissions." << std::endl;
        zed.close();
        return 1;
    }

    // Start recording SVO, stop with Ctrl-C command
    std::cout << "SVO is Recording, use Ctrl-C to stop." << std::endl;
    SetCtrlHandler();
    int frames_recorded = 0;

    while (!exit_app) {
        if (zed.grab() == SUCCESS) {
            // Each new frame is added to the SVO file
            zed.record();
            frames_recorded++;
            std::cout << "Frame count: " << frames_recorded << "\r";
        }
    }

    // Stop recording
    zed.disableRecording();
    zed.close();
    return 0;
}