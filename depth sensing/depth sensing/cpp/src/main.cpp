///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2025, STEREOLABS.
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

/*********************************************************************
 ** This sample demonstrates how to capture a live 3D point cloud   **
 ** with the ZED SDK and display the result in an OpenGL window.    **
 *********************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

// Using std and sl namespaces
using namespace std;
using namespace sl;

std::string parseArgs(int argc, char **argv, sl::InitParameters& param);

int main(int argc, char **argv) {
    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters.sdk_verbose = 1;
    init_parameters.maximum_working_resolution = sl::Resolution(0, 0);
    auto mask_path = parseArgs(argc, argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

    // Load optional region of interest to exclude irrelevant area of the image
    if(!mask_path.empty()) {
        sl::Mat mask_roi;
        auto err = mask_roi.read(mask_path.c_str());
        if(err == sl::ERROR_CODE::SUCCESS)
            zed.setRegionOfInterest(mask_roi, {MODULE::ALL});
        else
            std::cout << "Error loading Region of Interest file: " << err << std::endl;
    }

    auto camera_config = zed.getCameraInformation().camera_configuration;    
    // Automatically set to the optimal resolution
    sl::Resolution res(-1, -1); 
    
    Mat point_cloud;
    zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, res);
    res = point_cloud.getResolution();

    auto stream = zed.getCUDAStream();

    // Point cloud viewer
    GLViewer viewer;
    // Initialize point cloud viewer 
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam, stream, res);
    if (errgl != GLEW_OK) {
        print("Error OpenGL: " + std::string((char*)glewGetErrorString(errgl)));
        return EXIT_FAILURE;
    }

    RuntimeParameters runParameters;
    // Setting the depth confidence parameters
    //runParameters.confidence_threshold = 98;
    //runParameters.texture_confidence_threshold = 100;

    std::cout << "Press on 's' for saving current .ply file" << std::endl;
    // Main Loop
    while (viewer.isAvailable()) {        
        // Check that a new image is successfully acquired
        if (zed.grab(runParameters) == ERROR_CODE::SUCCESS) {
            // retrieve the current 3D coloread point cloud in GPU
            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::GPU, res);
            viewer.updatePointCloud(point_cloud);
            std::cout << "FPS: " << zed.getCurrentFPS() << "\r" << std::flush;
            if(viewer.shouldSaveData()){
                sl::Mat point_cloud_to_save;
                zed.retrieveMeasure(point_cloud_to_save, MEASURE::XYZRGBA);
                auto write_suceed = point_cloud_to_save.write("Pointcloud.ply");
                if(write_suceed == sl::ERROR_CODE::SUCCESS)
                    std::cout << "Current .ply file saving succeed" << std::endl;                
                else
                    std::cout << "Current .ply file saving failed" << std::endl;
            }
        }
    }
    // free allocated memory before closing the ZED
    point_cloud.free();

    // close the ZED
    zed.close();

    return EXIT_SUCCESS;
}

inline int findImageExtension(int argc, char **argv) {
    int arg_idx=-1;
    int arg_idx_search = 0;
    if (argc > 2) arg_idx_search=2;
    else if(argc > 1) arg_idx_search=1;

    if(arg_idx_search > 0 && (string(argv[arg_idx_search]).find(".png") != string::npos || 
        string(argv[arg_idx_search]).find(".jpg") != string::npos))
        arg_idx = arg_idx_search;
    return arg_idx;
}


std::string parseArgs(int argc, char **argv, sl::InitParameters& param) {
    int mask_arg = findImageExtension(argc, argv);
    std::string mask_path;

    if (argc > 1 && string(argv[1]).find(".svo") != string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout << "[Sample] Using SVO File input: " << argv[1] << endl;
    } else if (argc > 1 && string(argv[1]).find(".svo") == string::npos) {
        string arg = string(argv[1]);
        unsigned int a, b, c, d, port;
        if (sscanf(arg.c_str(), "%u.%u.%u.%u:%d", &a, &b, &c, &d, &port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a) + "." + to_string(b) + "." + to_string(c) + "." + to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()), port);
            cout << "[Sample] Using Stream input, IP : " << ip_adress << ", port : " << port << endl;
        } else if (sscanf(arg.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout << "[Sample] Using Stream input, IP : " << argv[1] << endl;
        }else if (arg.find("HD2K") != string::npos) {
            param.camera_resolution = RESOLUTION::HD2K;
            cout << "[Sample] Using Camera in resolution HD2K" << endl;
        }else if (arg.find("HD1200") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1200;
            cout << "[Sample] Using Camera in resolution HD1200" << endl;
        } else if (arg.find("HD1080") != string::npos) {
            param.camera_resolution = RESOLUTION::HD1080;
            cout << "[Sample] Using Camera in resolution HD1080" << endl;
        } else if (arg.find("HD720") != string::npos) {
            param.camera_resolution = RESOLUTION::HD720;
            cout << "[Sample] Using Camera in resolution HD720" << endl;
        }else if (arg.find("SVGA") != string::npos) {
            param.camera_resolution = RESOLUTION::SVGA;
            cout << "[Sample] Using Camera in resolution SVGA" << endl;
        }else if (arg.find("VGA") != string::npos) {
            param.camera_resolution = RESOLUTION::VGA;
            cout << "[Sample] Using Camera in resolution VGA" << endl;
        }
    } 
    
    if (mask_arg > 0) {
        mask_path = string(argv[mask_arg]);
        cout << "[Sample] Using Region of Interest from file : " << mask_path << endl;
    }

    return mask_path;
}
