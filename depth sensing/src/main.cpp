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


/***************************************************************************************************
 ** This sample demonstrates how to grab images and depth map with the ZED SDK                    **
 ** and apply the result in a 3D view "point cloud style" with OpenGL /freeGLUT                   **
 ** Some of the functions of the ZED SDK are linked with a key press event		                  **
 ***************************************************************************************************/

// Standard includes
#include <stdio.h>
#include <string.h>

// OpenGL includes
#include <GL/glew.h>
#include <GL/freeglut.h> 
#include <GL/gl.h>
#include <GL/glut.h>

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"
#include "SaveDepth.hpp"

//// Using std and sl namespaces
using namespace std;
using namespace sl;

//// Create ZED object (camera, callback, images)
sl::Camera zed;
sl::Mat point_cloud, depth_image;
std::thread zed_callback;
bool exit_;

//// Point Cloud visualizer
GLViewer viewer;

//// Sample functions
void startZED();
void run();
void close();
void printHelp();

int main(int argc, char **argv) {


    // Setup configuration parameters for the ZED
    InitParameters initParameters;
    if (argc == 2) initParameters.svo_input_filename = argv[1];
    initParameters.camera_resolution = sl::RESOLUTION_HD720;
    initParameters.depth_mode = sl::DEPTH_MODE_PERFORMANCE; //need quite a powerful graphic card in QUALITY
    initParameters.coordinate_units = sl::UNIT_METER; // set meter as the OpenGL world will be in meters
    initParameters.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed

    // Open the ZED
    ERROR_CODE err = zed.open(initParameters);
    if (err != SUCCESS) {
        cout << errorCode2str(err) << endl;
        zed.close();
        viewer.exit();
        return 1; // Quit if an error occurred
    }

    // Print help in console
    printHelp();

    // Initialize point cloud viewer
    viewer.init(zed.getResolution());

    //Start ZED 
    startZED();

    /// Set GLUT callback
    glutCloseFunc(close);
    glutMainLoop();
    return 0;
}

/**
 *  This function frees and close the ZED, its callback(thread) and the viewer
 **/
void close() {
    exit_ = true;

    // Stop callback
    zed_callback.join();

    // Exit point cloud viewer
    viewer.exit();

    // free buffer and close ZED
    depth_image.free(MEM_CPU);
    point_cloud.free(MEM_GPU);
    zed.close();
}

/**
 *  This functions start the ZED's thread that grab images and data.
 **/
void startZED() {
    exit_ = false;
    zed_callback = std::thread(run);
}

/**
 *  This function loops to get images and data from the ZED. it can be considered as a callback
 *  You can add your own code here.
 **/
void run() {

    char key = ' ';
    while (!exit_) {
        if (zed.grab() == SUCCESS) {
            // Get depth as a displayable image (8bits) and display it with OpenCV
            zed.retrieveImage(depth_image, sl::VIEW_DEPTH); // For display purpose ONLY. To get real world depth values, use retrieveMeasure(mat, sl::MEASURE_DEPTH)
            cv::imshow("Depth", cv::Mat(depth_image.getHeight(), depth_image.getWidth(), CV_8UC4, depth_image.getPtr<sl::uchar1>(sl::MEM_CPU)));

            // Get XYZRGBA point cloud on GPU and send it to OpenGL
            zed.retrieveMeasure(point_cloud, sl::MEASURE_XYZRGBA, sl::MEM_GPU); // Actual metrics values
            viewer.updatePointCloud(point_cloud);

            // Handles keyboard event
            key = cv::waitKey(15);
            processKeyEvent(zed, key);
        }
        else sl::sleep_ms(1);
    }
}

/**
 * This function displays help in console
 **/
void printHelp() {
    std::cout << " Press 's' to save Side by side images" << std::endl;
    std::cout << " Press 'p' to save Point Cloud" << std::endl;
    std::cout << " Press 'd' to save Depth image" << std::endl;
    std::cout << " Press 'm' to switch Point Cloud format" << std::endl;
    std::cout << " Press 'n' to switch Depth format" << std::endl;
}