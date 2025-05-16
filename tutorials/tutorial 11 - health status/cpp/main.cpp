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


#include <sl/Camera.hpp>

using namespace std;
using namespace sl;

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::AUTO; // Use HD720 opr HD1200 video mode, depending on camera type.
    init_parameters.camera_fps = 30; // Set fps at 30
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
    // '1' enables the health check, higher number enables more advanced verification (see documentation) but also increase the computation time
    // '2' enables processing regarding the image quality (absolute quality, and difference between left and right)
    // '3' enables advanced blur and quality check
    init_parameters.enable_image_validity_check = 1;
    if(argc > 1) init_parameters.input.setFromSVOFile(argv[1]);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program." << endl;
        return EXIT_FAILURE;
    }

    while (1) {
        std::cout << "\033[2J\033[H"; // Clear screen and move cursor to top-left
        // Grab an image
        returned_state = zed.grab();
        // A new image is available if grab() returns ERROR_CODE::SUCCESS
        if (returned_state == ERROR_CODE::SUCCESS) {

       } else if(returned_state == ERROR_CODE::CORRUPTED_FRAME) {
            // If the health check detect a corrupted frame, the grab will return a warning error code as ERROR_CODE::CORRUPTED_FRAME
            cout << "**** Corrupted frame detected ! *****" << endl;
        } else {
            cout << "Error " << returned_state << endl;
        }

        // Get the detailed health status
        auto health = zed.getHealthStatus();
        std::cout << "Health check enabled " << health.enabled << std::endl;
        std::cout << "low_image_quality " << health.low_image_quality << std::endl;
        std::cout << "low_lighting " << health.low_lighting << std::endl;
        std::cout << "low_depth_reliability " << health.low_depth_reliability << std::endl;
        std::cout << "low_motion_sensors_reliability " << health.low_motion_sensors_reliability << std::endl;
    }

    // Close the camera
    zed.close();
    return EXIT_SUCCESS;
}
