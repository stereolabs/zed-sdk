///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
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

/********************************************************************************
 ** This sample demonstrates how to grab images and change the camera settings **
 ** with the ZED SDK                                                           **
 ********************************************************************************/

// ZED include
#include <sl/CameraOne.hpp>

// Sample includes
#include <utils.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

int main(int argc, char **argv)
{
    // Create a ZED Camera object
    CameraOne zed;

    InitParametersOne init_parameters;
    init_parameters.sdk_verbose = true;
    init_parameters.camera_resolution = sl::RESOLUTION::AUTO;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS)
    {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }
    // Create a Mat to store images
    Mat zed_image;

    auto _cam_infos = zed.getCameraInformation();

    std::cout<<"CAM INFOS\n "<<
        "MODEL "<<_cam_infos.camera_model<<"\n"<<
        "INPUT "<<_cam_infos.input_type<<"\n"<<
        "SERIAL "<<_cam_infos.serial_number<<"\n"<<
        "Resolution "<<_cam_infos.camera_configuration.resolution.width <<"x"<<_cam_infos.camera_configuration.resolution.height<<"\n"<<
        "FPS "<<_cam_infos.camera_configuration.fps<<std::endl;
        
    // Capture new images until 'q' is pressed
    char key = ' ';
    while (key != 'q')
    {
        // Check that a new image is successfully acquired
        returned_state = zed.grab();
        if (returned_state == ERROR_CODE::SUCCESS)
        {
            // Retrieve  image
            zed.retrieveImage(zed_image);
            // Display the image
            cv::imshow("ZED-One", slMat2cvMat(zed_image));
        }
        else
        {
            print("Grab", returned_state);
            if (returned_state != sl::ERROR_CODE::CAMERA_REBOOTING)
                break;
        }

        key = cv::waitKey(10);
        // Change camera settings with keyboard
    }

    zed.close();
    return EXIT_SUCCESS;
}
