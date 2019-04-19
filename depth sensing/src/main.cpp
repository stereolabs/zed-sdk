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
 ** This sample demonstrates how to capture images and 3D point cloud   **
 ** with the ZED SDK and display the result in an OpenGL window. 	    **
 *************************************************************************/

 // ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

int main(int argc, char **argv) {
    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.depth_mode = DEPTH_MODE_ULTRA;
    initParameters.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    // open SVO if one given as parameter
    if(argc > 1 && string(argv[1]).find(".svo"))
        initParameters.svo_input_filename = argv[1];

    // Open the camera
    ERROR_CODE zed_error = zed.open(initParameters);

    if(zed_error != SUCCESS) {// Quit if an error occurred
        cout << zed_error << endl;
        zed.close();
        return 1;
    }

    Resolution resolution = zed.getResolution();
    CameraParameters camera_parameters = zed.getCameraInformation().calibration_parameters.left_cam;

    // Point cloud viewer
    GLViewer viewer;
    // Initialize point cloud viewer 
    viewer.init(argc, argv, camera_parameters);

    // Allocation of 4 channels of float on GPU
    Mat point_cloud(resolution, MAT_TYPE_32F_C4, MEM_GPU);

    // Main Loop
    while(viewer.isAvailable()) {
        if(zed.grab() == SUCCESS) {
            zed.retrieveMeasure(point_cloud, MEASURE_XYZRGBA, MEM_GPU);
            viewer.updatePointCloud(point_cloud);
        } else sleep_ms(1);
    }
    // free allocated memory before closing the ZED
    point_cloud.free();

    // close the ZED
    zed.close();

    return 0;
}
