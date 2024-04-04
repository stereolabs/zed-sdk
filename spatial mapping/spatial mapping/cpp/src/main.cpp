///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024, STEREOLABS.
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

/**********************************************************************************
 ** This sample demonstrates how to capture a live 3D reconstruction of a scene  **
 ** as a fused point cloud and display the result in an OpenGL window.           **
 **********************************************************************************/

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"

#include <opencv2/opencv.hpp>

// Using std and sl namespaces
using namespace std;
using namespace sl;

#define BUILD_MESH 1

void parse_args(int argc, char **argv,InitParameters& param, sl::Mat &roi);

void print(std::string msg_prefix, sl::ERROR_CODE err_code = sl::ERROR_CODE::SUCCESS, std::string msg_suffix = "");

int main(int argc, char **argv) {

    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters init_parameters;
    init_parameters.depth_mode = DEPTH_MODE::NEURAL;
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL's coordinate system is right_handed
    init_parameters.depth_maximum_distance = 8.;

    sl::Mat roi;
    parse_args(argc, argv, init_parameters, roi);

    // Open the camera
    auto returned_state = zed.open(init_parameters);

    if (returned_state != ERROR_CODE::SUCCESS) {// Quit if an error occurred
        print("Open Camera", returned_state, "\nExit program.");
        zed.close();
        return EXIT_FAILURE;
    }

    if(roi.isInit()){
        auto state = zed.setRegionOfInterest(roi, {sl::MODULE::POSITIONAL_TRACKING, sl::MODULE::SPATIAL_MAPPING});
        std::cout<<"Applied ROI "<<state<<"\n";
    }

    /* Print shortcuts*/
    std::cout<<"Shortcuts\n";
    std::cout<<"\t- 'l' to enable/disable current live point cloud display\n";
    std::cout<<"\t- 'w' to switch mesh display from faces to triangles\n";
    std::cout<<"\t- 'd' to switch background color from dark to light\n";

    auto camera_infos = zed.getCameraInformation();

    // Setup and start positional tracking
    Pose pose;
    POSITIONAL_TRACKING_STATE tracking_state = POSITIONAL_TRACKING_STATE::OFF;

    returned_state = zed.enablePositionalTracking();
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Enabling positional tracking failed: ", returned_state);
        zed.close();
        return EXIT_FAILURE;
    }

    // Set spatial mapping parameters
    SpatialMappingParameters spatial_mapping_parameters;
    // Request a Point Cloud
#if BUILD_MESH
    spatial_mapping_parameters.map_type = SpatialMappingParameters::SPATIAL_MAP_TYPE::MESH;
    Mesh map;
#else
    spatial_mapping_parameters.map_type = SpatialMappingParameters::SPATIAL_MAP_TYPE::FUSED_POINT_CLOUD;
    FusedPointCloud map;
#endif
    // Set mapping range, it will set the resolution accordingly (a higher range, a lower resolution)
    spatial_mapping_parameters.set(SpatialMappingParameters::MAPPING_RANGE::SHORT);
    spatial_mapping_parameters.set(SpatialMappingParameters::MAPPING_RESOLUTION::HIGH);
    // Request partial updates only (only the last updated chunks need to be re-draw)
    spatial_mapping_parameters.use_chunk_only = true;
    // Stability counter defines how many times a stable 3D points should be seen before it is integrated into the spatial mapping
    spatial_mapping_parameters.stability_counter = 4;

    // Timestamp of the last fused point cloud requested
    chrono::high_resolution_clock::time_point ts_last; 

    // Setup runtime parameters
    RuntimeParameters runtime_parameters;
    // Use low depth confidence to avoid introducing noise in the constructed model
    runtime_parameters.confidence_threshold = 50;

    auto resolution = camera_infos.camera_configuration.resolution;

    // Define display resolution and check that it fit at least the image resolution
    float image_aspect_ratio = resolution.width / (1.f * resolution.height);
    int requested_low_res_w = min(720, (int)resolution.width);
    sl::Resolution display_resolution(requested_low_res_w, requested_low_res_w / image_aspect_ratio);

    Mat image(display_resolution, MAT_TYPE::U8_C4, sl::MEM::GPU);
    Mat point_cloud(display_resolution, MAT_TYPE::F32_C4, sl::MEM::GPU);
    
    // Point cloud viewer
    GLViewer viewer;

    viewer.init(argc, argv, image, point_cloud, zed.getCUDAStream());
 
    bool request_new_mesh = true;

    bool wait_for_mapping = true;

    sl::Timestamp timestamp_start;
    timestamp_start.data_ns = 0;

    // Start the main loop
    while (viewer.isAvailable()) {
        // Grab a new image
        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS)
        {
            // Retrieve the left image
            zed.retrieveImage(image, VIEW::LEFT, MEM::GPU, display_resolution);
            zed.retrieveMeasure(point_cloud, MEASURE::XYZBGRA, MEM::GPU, display_resolution);
            // Retrieve the camera pose data
            tracking_state = zed.getPosition(pose);
            viewer.updateCameraPose(pose.pose_data, tracking_state);

            if (tracking_state == POSITIONAL_TRACKING_STATE::OK) {
                if(wait_for_mapping){
                    zed.enableSpatialMapping(spatial_mapping_parameters);
                    wait_for_mapping = false;
                }else{
                    if(request_new_mesh){
                        auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - ts_last).count();                    
                        // Ask for a fused point cloud update if 500ms have elapsed since last request
                        if(duration > 100) {
                            // Ask for a point cloud refresh
                            zed.requestSpatialMapAsync();
                            request_new_mesh = false;
                        }
                    }
                    
                    // If the point cloud is ready to be retrieved
                    if(zed.getSpatialMapRequestStatusAsync() == ERROR_CODE::SUCCESS && !request_new_mesh) {
                        zed.retrieveSpatialMapAsync(map);
                        viewer.updateMap(map);
                        request_new_mesh = true;
                        ts_last = chrono::high_resolution_clock::now();
                    }
                }
            }
        }
    }

    // Save generated point cloud
    map.save("MyMap", sl::MESH_FILE_FORMAT::PLY);

    // Free allocated memory before closing the camera
    image.free();
    point_cloud.free();
    // Close the ZED
    zed.close();

    return 0;
}

void parse_args(int argc, char **argv,InitParameters& param, sl::Mat &roi)
{
    if(argc == 1) return;
    for(int id = 1; id < argc; id ++) {
        std::string arg(argv[id]);
        if(arg.find(".svo")!=string::npos) {
            // SVO input mode
            param.input.setFromSVOFile(arg.c_str());
            param.svo_real_time_mode=true;
            cout<<"[Sample] Using SVO File input: "<<arg<<endl;
        }

        unsigned int a,b,c,d,port;
        if (sscanf(arg.c_str(),"%u.%u.%u.%u:%d", &a, &b, &c, &d,&port) == 5) {
            // Stream input mode - IP + port
            string ip_adress = to_string(a)+"."+to_string(b)+"."+to_string(c)+"."+to_string(d);
            param.input.setFromStream(String(ip_adress.c_str()),port);
            cout<<"[Sample] Using Stream input, IP : "<<ip_adress<<", port : "<<port<<endl;
        }
        else  if (sscanf(arg.c_str(),"%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
            // Stream input mode - IP only
            param.input.setFromStream(String(argv[1]));
            cout<<"[Sample] Using Stream input, IP : "<<argv[1]<<endl;
        }
        else if (arg.find("HD2K") != string::npos) {
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
        }else if ((arg.find(".png") != string::npos) || ((arg.find(".jpg") != string::npos))) {
            roi.read(arg.c_str());
            cout << "[Sample] Using Region of intererest from "<< arg << endl;
        }
    }
}

void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    cout <<"[Sample]";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout<<" ";
    cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}
