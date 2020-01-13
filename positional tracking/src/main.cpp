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

/*************************************************************************
** This sample demonstrates how to use the ZED for positional tracking  **
** and display camera motion in an OpenGL window. 		                **
**************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std namespace
using namespace std;
using namespace sl;

const int MAX_CHAR = 128;

int main(int argc, char **argv) {

    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.coordinate_units = UNIT_METER;
    initParameters.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;

    if(argc > 1 && string(argv[1]).find(".svo"))
        initParameters.svo_input_filename.set(argv[1]);

    // Open the camera
    ERROR_CODE zed_error = zed.open(initParameters);
    if(zed_error != SUCCESS) {
        cout << zed_error << endl;
        zed.close();
        return 1; // Quit if an error occurred
    }
    
    auto camera_model = zed.getCameraInformation().camera_model;
    GLViewer viewer;
    // Initialize OpenGL viewer
    viewer.init(argc, argv, camera_model);

    // Create text for GUI
    char text_rotation[MAX_CHAR];
    char text_translation[MAX_CHAR];

    // Start motion 
    zed.enableTracking();
    Pose camera_path;
    TRACKING_STATE tracking_state;

    while(viewer.isAvailable()) {
        if(zed.grab() == SUCCESS) {
            // Get the position of the camera in a fixed reference frame (the World Frame)
            tracking_state = zed.getPosition(camera_path, REFERENCE_FRAME_WORLD);

            if(tracking_state == TRACKING_STATE_OK) {
                // Get rotation and translation
                sl::float3 rotation = camera_path.getEulerAngles();
                sl::float3 translation = camera_path.getTranslation();

                // Display translation and rotation (pitch, yaw, roll in OpenGL coordinate system)
                snprintf(text_rotation, MAX_CHAR, "%3.2f; %3.2f; %3.2f", rotation.x, rotation.y, rotation.z);
                snprintf(text_translation, MAX_CHAR, "%3.2f; %3.2f; %3.2f", translation.x, translation.y, translation.z);
            }

            // Update rotation, translation and tracking state values in the OpenGL window
            viewer.updateData(camera_path.pose_data, string(text_translation), string(text_rotation), tracking_state);
        } else
            sleep_ms(1);
    }
    zed.close();
    return 0;
}
