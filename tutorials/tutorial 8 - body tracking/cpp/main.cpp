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
 ** This sample demonstrates how to use the body tracking module                **
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

std::string printBodyParts(BODY_PARTS part) {
    std::string out;
    switch (part) {
        case BODY_PARTS::NOSE:
            out = "Nose";
            break;
        case BODY_PARTS::NECK:
            out = "Neck";
            break;
        case BODY_PARTS::RIGHT_SHOULDER:
            out = "R Shoulder";
            break;
        case BODY_PARTS::RIGHT_ELBOW:
            out = "R Elbow";
            break;
        case BODY_PARTS::RIGHT_WRIST:
            out = "R Wrist";
            break;
        case BODY_PARTS::LEFT_SHOULDER:
            out = "L Shoulder";
            break;
        case BODY_PARTS::LEFT_ELBOW:
            out = "L Elbow";
            break;
        case BODY_PARTS::LEFT_WRIST:
            out = "L Wrist";
            break;
        case BODY_PARTS::RIGHT_HIP:
            out = "R Hip";
            break;
        case BODY_PARTS::RIGHT_KNEE:
            out = "R Knee";
            break;
        case BODY_PARTS::RIGHT_ANKLE:
            out = "R Ankle";
            break;
        case BODY_PARTS::LEFT_HIP:
            out = "L Hip";
            break;
        case BODY_PARTS::LEFT_KNEE:
            out = "L Knee";
            break;
        case BODY_PARTS::LEFT_ANKLE:
            out = "L Ankle";
            break;
        case BODY_PARTS::RIGHT_EYE:
            out = "R Eye";
            break;
        case BODY_PARTS::LEFT_EYE:
            out = "L Eye";
            break;
        case BODY_PARTS::RIGHT_EAR:
            out = "R Ear";
            break;
        case BODY_PARTS::LEFT_EAR:
            out = "L Ear";
            break;
    }
    return out;
}

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
    // Different model can be chosen, optimizing the runtime or the accuracy
    detection_parameters.detection_model = DETECTION_MODEL::HUMAN_BODY_FAST;
    // run detection for every Camera grab
    detection_parameters.image_sync = true;
    // track detects object across time and space
    detection_parameters.enable_tracking = true;
    // Optimize the person joints position, requires more computations
    detection_parameters.enable_body_fitting = true;

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
    // For outdoor scene or long range, the confidence should be lowered to avoid missing detections (~20-30)
    // For indoor scene or closer range, a higher confidence limits the risk of false positives and increase the precision (~50+)
    detection_parameters_rt.detection_confidence_threshold = 40;
    // detection output
    Objects objects;
    cout << setprecision(3);

    int nb_detection = 0;
    while (nb_detection < 100) {

        if (zed.grab() == ERROR_CODE::SUCCESS) {
            zed.retrieveObjects(objects, detection_parameters_rt);

            if (objects.is_new) {
                cout << objects.object_list.size() << " Person(s) detected\n\n";
                if (!objects.object_list.empty()) {

                    auto first_object = objects.object_list.front();

                    cout << "First Person attributes :\n";
                    cout << " Confidence (" << first_object.confidence << "/100)\n";

                    if (detection_parameters.enable_tracking)
                        cout << " Tracking ID: " << first_object.id << " tracking state: " <<
                            first_object.tracking_state << " / " << first_object.action_state << "\n";

                    cout << " 3D position: " << first_object.position <<
                            " Velocity: " << first_object.velocity << "\n";

                    cout << " 3D dimensions: " << first_object.dimensions << "\n";

                    cout << " Keypoints 2D \n";
                    // The body part meaning can be obtained by casting the index into a BODY_PARTS
                    // to get the BODY_PARTS index the getIdx function is available
                    for (int i = 0; i < first_object.keypoint_2d.size(); i++) {
                        auto &kp = first_object.keypoint_2d[i];
                        cout << "    " << printBodyParts((BODY_PARTS) i) << " " << kp.x << ", " << kp.y << "\n";
                    }

                    // The BODY_PARTS can be link as bones, using sl::BODY_BONES which gives the BODY_PARTS pair for each
                    cout << " Keypoints 3D \n";
                    for (int i = 0; i < first_object.keypoint.size(); i++) {
                        auto &kp = first_object.keypoint[i];
                        cout << "    " << printBodyParts((BODY_PARTS) i) << " " << kp.x << ", " << kp.y << ", " << kp.z << "\n";
                    }

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