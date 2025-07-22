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

/****************************************************************************************
** This sample shows how to record video in Stereolabs SVO format.					   **
** SVO video files can be played with the ZED API and used with its different modules  **
*****************************************************************************************/

// ZED includes
#include <sl/CameraOne.hpp>

// Sample includes
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

int main(int argc, char **argv) {

    if (argc < 2) {
        cout << "No arguments provided, an output SVO name is expected.\n";
        return EXIT_FAILURE;
    }

    // Create a ZED camera
    CameraOne zed;
    
    // Set configuration parameters for the ZED
    InitParametersOne init_parameters;

    // Open the camera
    auto returned_state  = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Enable recording with the filename specified in argument
    RecordingParameters recording_parameters;
    recording_parameters.video_filename.set(argv[1]);
    recording_parameters.compression_mode = SVO_COMPRESSION_MODE::H265;
    returned_state = zed.enableRecording(recording_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera enable Recording", returned_state, "Exit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    // Start recording SVO, stop with Ctrl-C command
    std::cout << "SVO is Recording, use Ctrl-C to stop." << std::endl;
    SetCtrlHandler();
    sl::RecordingStatus rec_status;
    auto start = std::chrono::high_resolution_clock::now();
    while (!exit_app) {
        if (zed.grab() <= ERROR_CODE::SUCCESS) {

            // Each new frame is automatically added to the SVO file

            if(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() > 1000){
                rec_status = zed.getRecordingStatus();
                std::cout<<"Recording: "<<rec_status.number_frames_ingested<<"frames, compression ratio: "<<rec_status.average_compression_ratio<<"\r"<<std::flush;
                start = std::chrono::high_resolution_clock::now();
            }
        }
        else
            break;
    }

    std::cout<<std::endl;

    // Stop recording
    zed.disableRecording();
    zed.close();
    return EXIT_SUCCESS;
}