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
 * @file playback.cpp
 * @brief ZED One SVO External Data Playback Sample
 *
 * This sample demonstrates how to read an SVO file and retrieve
 * the embedded custom data.
 */

// ZED include
#include <sl/Camera.hpp>

// Sample includes
#include <opencv2/opencv.hpp>
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

void printUsage() {
    cout << "Usage: ZED_One_SVO_External_Data_Playback <input_svo_file>" << endl;
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        printUsage();
        return EXIT_FAILURE;
    }

    // Create ZED objects
    CameraOne zed;
    InitParametersOne init_parameters;
    init_parameters.input.setFromSVOFile(argv[1]);
    init_parameters.sdk_verbose = true;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // List available data channels
    std::string channels;
    for (const auto& key : zed.getSVODataKeys()) {
        channels += key + "; ";
    }

    std::cout << "\n=== SVO External Data Playback ===" << std::endl;
    std::cout << "File: " << argv[1] << std::endl;
    std::cout << "Data channels: " << (channels.empty() ? "None" : channels) << std::endl;
    std::cout << "==================================\n" << std::endl;

    // Read all data at once
    std::map<sl::Timestamp, sl::SVOData> data_map;
    std::cout << "Reading all external data..." << std::endl;
    auto err = zed.retrieveSVOData("TEST", data_map);

    if (err == ERROR_CODE::SUCCESS && !data_map.empty()) {
        std::cout << "Found " << data_map.size() << " data entries:\n" << std::endl;

        int count = 0;
        for (const auto& d : data_map) {
            std::string content;
            d.second.getContent(content);
            std::cout << "  [" << d.first << "] " << content << std::endl;

            // Limit output for large files
            if (++count >= 10) {
                std::cout << "  ... and " << (data_map.size() - 10) << " more entries" << std::endl;
                break;
            }
        }
    } else {
        std::cout << "No external data found or error: " << err << std::endl;
    }

    std::cout << "\n--- Frame-by-frame retrieval ---\n" << std::endl;
    std::cout << "Press 'q' to quit..." << std::endl;

    unsigned long long last_timestamp_ns = 0;
    char key = ' ';

    while (key != 'q') {
        returned_state = zed.grab();
        if (returned_state <= ERROR_CODE::SUCCESS) {
            // Retrieve data for this frame
            std::map<sl::Timestamp, sl::SVOData> frame_data;
            auto current_ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);

            zed.retrieveSVOData("TEST", frame_data, last_timestamp_ns, current_ts);

            for (const auto& d : frame_data) {
                std::string content;
                d.second.getContent(content);
                std::cout << "Frame data: " << content << std::endl;
            }

            last_timestamp_ns = current_ts;
            key = cv::waitKey(10);

        } else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            print("SVO end reached.");
            break;
        } else {
            print("Grab error", returned_state);
            break;
        }
    }

    zed.close();
    return EXIT_SUCCESS;
}
