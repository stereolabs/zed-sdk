///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
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
** This sample shows how to capture a real-time 3D reconstruction      **
** of the scene using the Spatial Mapping API. The resulting mesh      **
** is displayed as a wireframe on top of the left image using OpenGL.  **
** Spatial Mapping can be started and stopped with the Space Bar key   **
*************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

int main(int argc, char** argv) {
    Camera zed;
    // Setup configuration parameters for the ZED    
    InitParameters parameters;
    parameters.coordinate_units = UNIT_METER;
    parameters.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // OpenGL coordinates system
    if(argc > 1 && string(argv[1]).find(".svo"))
        parameters.svo_input_filename = argv[1];

    // Open the ZED
    ERROR_CODE zed_error = zed.open(parameters);
    if(zed_error != ERROR_CODE::SUCCESS) {
        cout << zed_error << endl;
        zed.close();
        return -1;
    }

    CameraParameters camera_parameters = zed.getCameraInformation().calibration_parameters.left_cam;
    sl::MODEL camera_model = zed.getCameraInformation().camera_model;
    GLViewer viewer;
    bool error_viewer = viewer.init(argc, argv, camera_parameters, camera_model);
    if(error_viewer) {
        viewer.exit();
        zed.close();
        return -1;
    }

    Mat image; // current left image
    Pose pose; // positional tracking data
    Plane plane; // detected plane 
    Mesh mesh; // plane mesh

    ERROR_CODE find_plane_status;
    TRACKING_STATE tracking_state = TRACKING_STATE_OFF;

    // time stamp of the last mesh request
    chrono::high_resolution_clock::time_point ts_last;

    UserAction user_action;
    user_action.clear();

    // Enable positional tracking before starting spatial mapping
    zed.enableTracking();
    
    RuntimeParameters runtime_parameters;
    runtime_parameters.measure3D_reference_frame = REFERENCE_FRAME_WORLD;
    
    while(viewer.isAvailable()) {
        if(zed.grab(runtime_parameters) == SUCCESS) {
            // Retrieve image in GPU memory
            zed.retrieveImage(image, VIEW_LEFT, MEM_GPU);
            // Update pose data (used for projection of the mesh over the current image)
            tracking_state = zed.getPosition(pose);

            if(tracking_state == sl::TRACKING_STATE_OK) {
                // Compute elapse time since the last call of plane detection
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - ts_last).count();
                // Ask for a mesh update 
                if(user_action.hit)
                    find_plane_status = zed.findPlaneAtHit(user_action.hit_coord, plane);

                //if 500ms have spend since last request
                if((duration > 500) && user_action.press_space) {
                    // Update pose data (used for projection of the mesh over the current image)
                    sl::Transform resetTrackingFloorFrame;
                    find_plane_status = zed.findFloorPlane(plane, resetTrackingFloorFrame);
                    ts_last = std::chrono::high_resolution_clock::now();
                }

                if(find_plane_status == sl::SUCCESS) {
                    mesh = plane.extractMesh();
                    viewer.updateMesh(mesh, plane.type);
                }
            }

            user_action = viewer.updateImageAndState(image, pose.pose_data, tracking_state);
        }
    }

    image.free();
    mesh.clear();

    zed.disableTracking();
    zed.close();
    return 0;
}
