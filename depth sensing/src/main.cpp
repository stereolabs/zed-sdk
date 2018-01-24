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
 ** with the ZED SDK and display the result in an OpenGL window. 		    **
 *************************************************************************/

 // Standard includes
#include <stdio.h>
#include <string.h>

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

// Create ZED objects (camera, callback, images)
sl::Camera zed;
sl::Mat point_cloud;
std::thread zed_callback;
int width, height;
bool quit;

// Point cloud viewer
GLViewer viewer;

// Sample functions
void startZED();
void run();
void close();

int main(int argc, char **argv) {

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    if (argc == 2) initParameters.svo_input_filename = argv[1];
    initParameters.camera_resolution = RESOLUTION_HD720;
    initParameters.depth_mode = DEPTH_MODE_PERFORMANCE;
    initParameters.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    initParameters.coordinate_units = UNIT_METER;

    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != SUCCESS) {
        cout << toString(err) << endl;
        zed.close();
        viewer.exit();
        return 1; // Quit if an error occurred
    }

    // Initialize point cloud viewer in half-size
    width = (int) zed.getResolution().width / 2;
    height = (int) zed.getResolution().height / 2;
    viewer.init(width, height);

    // Start the camera thread
    startZED();

    // Set the display callback
    glutCloseFunc(close);
    glutMainLoop();
    return 0;
}

/**
    Launch ZED thread. Using a thread here allows to capture a point cloud and update the GL window concurrently.
 **/
void startZED() {
    quit = false;
    zed_callback = std::thread(run);
}

/**
    This function loops to get image and motion data from the ZED. It is similar to a callback.
    Add your own code here.
 **/
void run() {

    while (!quit) {
        if (zed.grab() == SUCCESS) {
            // Retrieve a colored RGBA point cloud in GPU memory and update GL viewing window
            // width and height specify the total number of columns and rows for the point cloud dataset
            // In this example, we retrieve and display a half size point cloud using width and height parameters
            zed.retrieveMeasure(point_cloud, MEASURE_XYZRGBA, MEM_GPU, width, height);
            viewer.updatePointCloud(point_cloud);

        } else sl::sleep_ms(1);
    }
}

/**
    This function closes the ZED camera, its callback (thread) and the GL viewer
 **/
void close() {
    quit = true;

    // Stop callback
    zed_callback.join();

    // Exit point cloud viewer
    viewer.exit();

    // Free buffer and close the ZED
    point_cloud.free(MEM_GPU);
    zed.close();
}
