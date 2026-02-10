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
 * @file recording.cpp
 * @brief ZED One SVO External Data Recording Sample
 *
 * This sample demonstrates how to record video while embedding
 * custom data into the SVO file.
 */

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

void printUsage() {
    cout << "Usage: ZED_One_SVO_External_Data_Recording <output_svo_file>" << endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return EXIT_FAILURE;
    }

    // Create a ZED camera
    CameraOne zed;

    // Set configuration parameters
    InitParametersOne init_parameters;
    init_parameters.sdk_verbose = 1;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Enable recording with the filename specified in argument
    RecordingParameters recording_parameters;
    recording_parameters.video_filename.set(argv[1]);
    recording_parameters.compression_mode = SVO_COMPRESSION_MODE::H265;

    returned_state = zed.enableRecording(recording_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Enable Recording", returned_state);
        zed.close();
        return EXIT_FAILURE;
    }

    std::cout << "\n=== SVO External Data Recording ===" << std::endl;
    std::cout << "Output file: " << argv[1] << std::endl;
    std::cout << "Recording 100 frames with embedded data..." << std::endl;
    std::cout << "====================================\n" << std::endl;

    SetCtrlHandler();
    int frames_recorded = 0;
    sl::RecordingStatus rec_status;

    while (frames_recorded < 100 && !exit_app) {
        if (zed.grab() <= ERROR_CODE::SUCCESS) {
            rec_status = zed.getRecordingStatus();
            if (rec_status.status) {
                // Get current timestamp
                unsigned long long timestamp_ns = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);

                // Create custom data
                sl::SVOData data;
                data.key = "TEST";
                data.setContent("Hello, SVO World >> " + std::to_string(timestamp_ns));
                data.timestamp_ns = timestamp_ns;

                // Ingest data into SVO
                auto err = zed.ingestDataIntoSVO(data);
                if (err != ERROR_CODE::SUCCESS) {
                    std::cout << "Ingest error: " << err << std::endl;
                }

                frames_recorded++;
                std::cout << "Frame: " << frames_recorded << "/100" << "\r" << std::flush;
            }
        } else {
            break;
        }
    }

    std::cout << std::endl;
    std::cout << "\nRecording complete. " << frames_recorded << " frames with external data." << std::endl;

    // Stop recording
    zed.disableRecording();
    zed.close();
    return EXIT_SUCCESS;
}
