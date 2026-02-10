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
 * @brief ZED One Live Capture Sample
 *
 * This sample demonstrates how to grab images from a ZED One camera
 * and display them using OpenCV.
 */

// ZED include
#include <sl/CameraOne.hpp>

// Sample includes
#include <utils.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

int main(int argc, char** argv) {
    // Create a ZED Camera object
    CameraOne zed;

    InitParametersOne init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.camera_resolution = sl::RESOLUTION::AUTO;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state > ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Create a Mat to store images
    Mat zed_image;

    // Get and display camera information
    auto cam_info = zed.getCameraInformation();
    std::cout << "\n=== Camera Information ===" << std::endl;
    std::cout << "Model:      " << cam_info.camera_model << std::endl;
    std::cout << "Input:      " << cam_info.input_type << std::endl;
    std::cout << "Serial:     " << cam_info.serial_number << std::endl;
    std::cout << "Resolution: " << cam_info.camera_configuration.resolution.width << "x" << cam_info.camera_configuration.resolution.height
              << std::endl;
    std::cout << "FPS:        " << cam_info.camera_configuration.fps << std::endl;
    std::cout << "==========================\n" << std::endl;

    std::cout << "Press 'q' to quit..." << std::endl;

    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q') {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();
        if (returned_state <= ERROR_CODE::SUCCESS) {
            // Retrieve image
            zed.retrieveImage(zed_image);
            // Display the image
            cv::imshow("ZED One - Live", slMat2cvMat(zed_image));
        } else {
            print("Grab", returned_state);
            if (returned_state != sl::ERROR_CODE::CAMERA_REBOOTING)
                break;
        }

        key = cv::waitKey(10);
    }

    zed.close();
    return EXIT_SUCCESS;
}
