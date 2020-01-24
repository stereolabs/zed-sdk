///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
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

/*********************************************************************************
 ** This sample demonstrates how to use the objects detection module            **
 **      with the ZED SDK and display the result                                **
 *********************************************************************************/

// Standard includes
#include <iostream>
#include <fstream>

// ZED includes
#include <sl/Camera.hpp>


// Using std and sl namespaces
using namespace std;
using namespace sl;

int main(int argc, char **argv) {
    // Create ZED objects
    Camera zed;
    InitParameters initParameters;
    initParameters.camera_resolution = RESOLUTION::HD720;
    initParameters.depth_mode = DEPTH_MODE::PERFORMANCE;
    initParameters.coordinate_units = UNIT::METER;
    initParameters.sdk_verbose = true;

    // Open the camera
    ERROR_CODE zed_error = zed.open(initParameters);
    if (zed_error != ERROR_CODE::SUCCESS) {
        zed.close();
        return 1; // Quit if an error occurred
    }

    // Define the Objects detection module parameters
    ObjectDetectionParameters detection_parameters;
    detection_parameters.image_sync = true;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_mask_output = true;
    
    auto camera_infos = zed.getCameraInformation();

    // If you want to have object tracking you need to enable positional tracking first
    if (detection_parameters.enable_tracking) {
        PositionalTrackingParameters positional_tracking_parameters;
        //positional_tracking_parameters.set_as_static = true;
        positional_tracking_parameters.set_floor_as_origin = true;
        zed.enablePositionalTracking(positional_tracking_parameters);
    }

    std::cout << "Object Detection: Loading Module..." << std::endl;
    zed_error = zed.enableObjectDetection(detection_parameters);
    if (zed_error != ERROR_CODE::SUCCESS) {
        zed.close();
        return 1;
    }
    // detection runtime parameters
    ObjectDetectionRuntimeParameters detection_parameters_rt;
    // detection output
    Objects objects;
    std::cout << std::setprecision(2);

    while (zed.grab() == ERROR_CODE::SUCCESS) {
        zed_error = zed.retrieveObjects(objects, detection_parameters_rt);
        printf("\033c");

        if (objects.is_new) {

            std::cout << objects.object_list.size() << " Object(s) detected\n\n";

            if (!objects.object_list.empty()) {

                auto first_object = objects.object_list.front();

                std::cout << "First object attributes :\n";
                std::cout << " Label '" << first_object.label << "' (conf. "
                        << first_object.confidence << "/100)\n";

                if (detection_parameters.enable_tracking)
                    std::cout << " Tracking ID: " << first_object.id << " tracking state: " <<
                        first_object.tracking_state << " / " << first_object.action_state << std::endl;

                std::cout << " 3D position: " << first_object.position <<
                        " Velocity: " << first_object.velocity;

                std::cout << " 3D dimensions: " << first_object.dimensions;

                if (first_object.mask.isInit())
                    std::cout << " 2D mask available\n";

                std::cout << " Bounding Box 2D \n";
                for (auto it : first_object.bounding_box_2d)
                    std::cout << "    " << it;

                std::cout << " Bounding Box 3D \n";
                for (auto it : first_object.bounding_box)
                    std::cout << "    " << it;

                std::cout << "\nPress 'Enter' to continue...\n";
                cin.ignore();
            }

        }

    }
    zed.close();
    return 0;
}