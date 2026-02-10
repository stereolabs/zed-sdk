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
 * @brief ZED One SVO Playback Sample
 *
 * This sample demonstrates how to read and play back SVO files
 * recorded with a ZED One camera.
 */

// ZED include
#include <sl/CameraOne.hpp>

// Sample includes
#include <utils.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

void printUsage() {
    cout << "Usage: ZED_One_SVO_Playback <svo_file>" << endl;
    cout << endl;
    cout << "Controls:" << endl;
    cout << "  q - Quit" << endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return EXIT_FAILURE;
    }

    // Create a ZED Camera object
    CameraOne zed;

    InitParametersOne init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.input.setFromSVOFile(argv[1]);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    int total_frames = zed.getSVONumberOfFrames();
    std::cout << "\n=== SVO Information ===" << std::endl;
    std::cout << "File:         " << argv[1] << std::endl;
    std::cout << "Total frames: " << total_frames << std::endl;
    std::cout << "=======================\n" << std::endl;
    std::cout << "Press 'q' to quit..." << std::endl;

    // Create a Mat to store images
    Mat zed_image;

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();
        if (returned_state <= ERROR_CODE::SUCCESS) {
            // Retrieve image
            zed.retrieveImage(zed_image);

            // Get current position
            int current_frame = zed.getSVOPosition();

            // Add frame info overlay
            cv::Mat cv_image = slMat2cvMat(zed_image);
            std::string info = "Frame: " + std::to_string(current_frame) + "/" + std::to_string(total_frames);
            cv::putText(cv_image, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            // Display the image
            cv::imshow("ZED One - SVO Playback", cv_image);
        } else if (returned_state == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) {
            print("SVO end reached, looping back to start.");
            zed.setSVOPosition(0);
        } else {
            print("Grab", returned_state);
            break;
        }

        key = cv::waitKey(10);
    }

    zed.close();
    return EXIT_SUCCESS;
}
