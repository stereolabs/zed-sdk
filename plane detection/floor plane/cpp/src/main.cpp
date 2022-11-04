///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2022, STEREOLABS.
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
 
void parseArgs(int argc, char **argv,sl::InitParameters& param);

#define AUTO_SEARCH 0

int main(int argc, char** argv) {
    Camera zed;
    // Setup configuration parameters for the ZED    
    InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::PERFORMANCE;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL coordinates system
    parseArgs(argc,argv, init_parameters);

    // Open the camera
    ERROR_CODE zed_open_state = zed.open(init_parameters);
    if (zed_open_state != ERROR_CODE::SUCCESS) {
        std::cout << "Camera Open" << zed_open_state << "\nExit program." << std::endl;;
        return EXIT_FAILURE;
    }

    sl::Resolution low_res(720, 404);
    sl::Mat pointCloud(low_res, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
    sl::Mat image(low_res, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);
    auto camera_infos = zed.getCameraInformation(low_res);

    GLViewer viewer;
    viewer.init(argc, argv, camera_infos.camera_configuration.calibration_parameters.left_cam);
   
    Pose pose; // positional tracking data
    ERROR_CODE find_plane_status = ERROR_CODE::SUCCESS;
    POSITIONAL_TRACKING_STATE tracking_state = POSITIONAL_TRACKING_STATE::SEARCHING_FLOOR_PLANE;

    RuntimeParameters runtime_parameters;
    runtime_parameters.measure3D_reference_frame = REFERENCE_FRAME::WORLD;

#if AUTO_SEARCH
    PositionalTrackingParameters tracking_parameters;
    tracking_parameters.set_floor_as_origin = true;
    zed.enablePositionalTracking(tracking_parameters);
#else
    Plane floor_plane;
    Transform reset_tracking;
#endif
    
    while(viewer.isAvailable()) {
        if(zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {

            zed.retrieveImage(image, VIEW::LEFT, MEM::GPU, low_res);
            viewer.updateImage(image);
#if !AUTO_SEARCH
            if (tracking_state == POSITIONAL_TRACKING_STATE::SEARCHING_FLOOR_PLANE) {
                if (zed.findFloorPlane(floor_plane, reset_tracking) == ERROR_CODE::SUCCESS) {
                    PositionalTrackingParameters tracking_parameters;
                    tracking_parameters.initial_world_transform = reset_tracking;
                    zed.enablePositionalTracking(tracking_parameters);
                    tracking_state = POSITIONAL_TRACKING_STATE::OK;
                    std::cout << "\rFloor Plane found ! Set world reference and start point cloud retrieval" << std::endl;
                }
            }
            else
#endif
            {
                tracking_state = zed.getPosition(pose);

                if (tracking_state == POSITIONAL_TRACKING_STATE::OK) {
                    zed.retrieveMeasure(pointCloud, MEASURE::XYZRGBA, MEM::GPU, low_res);
                    viewer.updateData(pointCloud, pose.pose_data);
                }
                else if (tracking_state == POSITIONAL_TRACKING_STATE::SEARCHING_FLOOR_PLANE)
                    std::cout << "\rFloor Plane not found, mouve around to find it";
            }
        }
    }

    pointCloud.free();
    image.free();
    zed.disablePositionalTracking();
    zed.close();
    return EXIT_SUCCESS;
}

void parseArgs(int argc, char **argv,sl::InitParameters& param)
{
    if (argc > 1 && string(argv[1]).find(".svo")!=string::npos) {
        // SVO input mode
        param.input.setFromSVOFile(argv[1]);
        cout<<"[Sample] Using SVO File input: "<<argv[1]<<endl;
    } else if (argc > 1 && string(argv[1]).find(".svo")==string::npos) {
        string arg = string(argv[1]);
        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
            param.input.setFromStream(sl::String(ip_adress.c_str()),port);
            cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<<endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(sl::String(argv[1]));
            cout<<"[Sample] Using Stream input, IP : "<<argv[1]<<endl;
        }
        else if (arg.find("HD2K")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD2K;
            cout<<"[Sample] Using Camera in resolution HD2K"<<endl;
        } else if (arg.find("HD1080")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD1080;
            cout<<"[Sample] Using Camera in resolution HD1080"<<endl;
        } else if (arg.find("HD720")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::HD720;
            cout<<"[Sample] Using Camera in resolution HD720"<<endl;
        } else if (arg.find("VGA")!=string::npos) {
            param.camera_resolution = sl::RESOLUTION::VGA;
            cout<<"[Sample] Using Camera in resolution VGA"<<endl;
        }
    } else {
        // Default
    }
}
