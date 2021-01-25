///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, STEREOLABS.
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

int main(int argc, char** argv) {
    // Create ZED objects
    Camera zed;
    InitParameters init_parameters;
    init_parameters.camera_resolution = RESOLUTION::HD720;
    init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.sdk_verbose = true;

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program.\n";
        return EXIT_FAILURE;
    }

    // Define the Objects detection module parameters
    ObjectDetectionParameters detection_parameters;
    // run detection for every Camera grab
    detection_parameters.image_sync = true;
    // track detects object accross time and space
    detection_parameters.enable_tracking = true;
    // compute a binary mask for each object aligned on the left image
    detection_parameters.enable_mask_output = true; // designed to give person pixel mask

    // If you want to have object tracking you need to enable positional tracking first
    if (detection_parameters.enable_tracking)
        zed.enablePositionalTracking();    

    cout << "Object Detection: Loading Module..." << endl;
    returned_state = zed.enableObjectDetection(detection_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        cout << "Error " << returned_state << ", exit program.\n";
        zed.close();
        return EXIT_FAILURE;
    }
    // detection runtime parameters
    ObjectDetectionRuntimeParameters detection_parameters_rt;
    // detection output
    Objects objects;
    cout << setprecision(3);

    int nb_detection = 0;
    while (nb_detection < 100) {

        if(zed.grab() == ERROR_CODE::SUCCESS){
           zed.retrieveObjects(objects, detection_parameters_rt);

            if (objects.is_new) {
                cout << objects.object_list.size() << " Object(s) detected\n\n";
                if (!objects.object_list.empty()) {

                    auto first_object = objects.object_list.front();

                    cout << "First object attributes :\n";
                    cout << " Label '" << first_object.label << "' (conf. "
                        << first_object.confidence << "/100)\n";

                    if (detection_parameters.enable_tracking)
                        cout << " Tracking ID: " << first_object.id << " tracking state: " <<
                        first_object.tracking_state << " / " << first_object.action_state << "\n";

                    cout << " 3D position: " << first_object.position <<
                        " Velocity: " << first_object.velocity << "\n";

                    cout << " 3D dimensions: " << first_object.dimensions << "\n";

                    if (first_object.mask.isInit())
                        cout << " 2D mask available\n";

                    cout << " Bounding Box 2D \n";
                    for (auto it : first_object.bounding_box_2d)
                        cout << "    " << it<<"\n";

                    cout << " Bounding Box 3D \n";
                    for (auto it : first_object.bounding_box)
                        cout << "    " << it << "\n";

                    cout << "\nPress 'Enter' to continue...\n";
                    cin.ignore();
                }
                nb_detection++;
            }
        }
    }
    zed.close();
    return EXIT_SUCCESS;
}