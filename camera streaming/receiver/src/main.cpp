///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2019, STEREOLABS.
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

std::vector<std::string> split(const std::string& s, char seperator) {
    std::vector<std::string> output;
    std::string::size_type prev_pos = 0, pos = 0;

    while ((pos = s.find(seperator, pos)) != std::string::npos) {
        std::string substring(s.substr(prev_pos, pos - prev_pos));
        output.push_back(substring);
        prev_pos = ++pos;
    }

    output.push_back(s.substr(prev_pos, pos - prev_pos));
    return output;
}



int main(int argc, char **argv) {
    Camera zed;

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.depth_mode = DEPTH_MODE_PERFORMANCE;
    initParameters.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    initParameters.sdk_verbose = true;

    if (argc > 1) {
        std::vector<std::string> configStream = split(std::string(argv[1]), ':');        
        sl::String ip = sl::String(configStream.at(0).c_str());
        if (configStream.size() == 2) {
            initParameters.input.setFromStream(ip, atoi(configStream.at(1).c_str()));
        } else initParameters.input.setFromStream(ip);
    } else {
        std::cout << "Opening the stream requires the IP of the sender\n Usage : ./ZED_Streaming_Receiver IP:[port]" << std::endl;
        return 1;
    }

    // Open the camera
    ERROR_CODE zed_error = zed.open(initParameters);

    if (zed_error != SUCCESS) {// Quit if an error occurred
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

    int fc = 0;
    // Main Loop
    while (viewer.isAvailable()) {
        if (zed.grab() == SUCCESS) {
            zed.retrieveMeasure(point_cloud, MEASURE_XYZRGBA, MEM_GPU);
            viewer.updatePointCloud(point_cloud); 
            fc++;
        }

        sleep_ms(1);

    }
    // Free allocated memory before closing the ZED
    point_cloud.free();

    // Closing App
    printf("Closing App....\n");
    viewer.exit();
    zed.close();
    return 0;
}
