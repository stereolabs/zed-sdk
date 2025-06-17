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

/************************************************************
 ** This sample demonstrates how to read a SVO video file. **
 ** We use OpenCV to display the video.					   **
 *************************************************************/

// ZED include
#include <sl/Camera.hpp>

// Sample includes
#include <opencv2/opencv.hpp>
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

int main(int argc, char **argv) {

    if (argc <= 1) {
        cout << "Usage: \n";
        cout << "$ ZED_SVO_Playback <SVO_file> \n";
        cout << "  ** SVO file is mandatory in the application ** \n\n";
        return EXIT_FAILURE;
    }

    // Create ZED objects
    CameraOne zed;
    InitParametersOne init_parameters;
    init_parameters.input.setFromSVOFile(argv[1]);
    init_parameters.sdk_verbose = true;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    std::string s;
    for (const auto &piece : zed.getSVODataKeys()) s += piece + "; ";
    std::cout << "Channels that are in the SVO: " << s << std::endl;

    unsigned long long last_timestamp_ns;

    std::map<sl::Timestamp, sl::SVOData> data_map;
    std::cout << "Reading everything all at once." << std::endl;
    auto ing = zed.retrieveSVOData("TEST", data_map);

    for (const auto& d : data_map) {
        std::string s;
        d.second.getContent(s);
        std::cout << d.first << " (//) " << s << std::endl;
    }

    std::cout << "#########\n";

    // Setup key, images, times
    char key = ' ';
    while (key != 'q') {
        returned_state = zed.grab();
        if (returned_state <= ERROR_CODE::SUCCESS) {
            std::map<sl::Timestamp, sl::SVOData> data_map;
            std::cout << "Reading between " << last_timestamp_ns << " and " << zed.getTimestamp(sl::TIME_REFERENCE::IMAGE) << std::endl;
            auto ing = zed.retrieveSVOData("TEST", data_map, last_timestamp_ns, zed.getTimestamp(sl::TIME_REFERENCE::IMAGE));
            for (const auto& d : data_map) {
                std::string s;
                d.second.getContent(s);
                std::cout << d.first << " // " << s << std::endl;
            }

            // Display the frame
            key = cv::waitKey(10);
        } else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            print("SVO end has been reached. Looping back to 0\n");
            zed.setSVOPosition(0);
            break;
        } else {
            print("Grab ZED : ", returned_state);
            break;
        }
        last_timestamp_ns = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE);
    }
    zed.close();
    return EXIT_SUCCESS;
}