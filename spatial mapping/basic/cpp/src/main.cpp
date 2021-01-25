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

/*************************************************************************
 ** This sample shows how to capture a real-time 3D reconstruction      **
 ** of the scene using the Spatial Mapping API. The resulting mesh      **
 ** is displayed as a wireframe on top of the left image using OpenGL.  **
 ** Spatial Mapping can be started and stopped with the space bar key   **
 *************************************************************************/

 // ZED includes
#include <sl/Camera.hpp>

 // Sample includes
#include "GLViewer.hpp"

 // Using std and sl namespaces
using namespace std;
using namespace sl;

// set to 0 to create a Fused Point Cloud
#define CREATE_MESH 1

void parseArgs(int argc, char **argv,sl::InitParameters& param);

int main(int argc, char** argv) {
    Camera zed;
    // Setup configuration parameters for the ZED    
    InitParameters init_parameters;
    init_parameters.coordinate_units = UNIT::METER;
    init_parameters.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // OpenGL coordinates system
    parseArgs(argc,argv, init_parameters);

    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }

#if CREATE_MESH
    Mesh map; // Current incremental mesh
#else
    FusedPointCloud map; // Current incremental fused point cloud
#endif

    CameraParameters camera_parameters = zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam;

    GLViewer viewer;
    bool error_viewer = viewer.init(argc, argv, camera_parameters, &map);

    if(error_viewer) {
        viewer.exit();
        zed.close();
        return EXIT_FAILURE;
    }

    Mat image; // Current left image
    Pose pose; // Camera pose tracking data

    SpatialMappingParameters spatial_mapping_parameters;
    POSITIONAL_TRACKING_STATE tracking_state = POSITIONAL_TRACKING_STATE::OFF;
    SPATIAL_MAPPING_STATE mapping_state = SPATIAL_MAPPING_STATE::NOT_ENABLED;
    bool mapping_activated = false; // Indicates if spatial mapping is running or not
    chrono::high_resolution_clock::time_point ts_last; // Timestamp of the last mesh request
    
    // Enable positional tracking before starting spatial mapping
    returned_state = zed.enablePositionalTracking();
    if(returned_state != ERROR_CODE::SUCCESS) {
        print("Enabling positional tracking failed: ", returned_state);
        zed.close();
        return EXIT_FAILURE;
    }

    while(viewer.isAvailable()) {
        if(zed.grab() == ERROR_CODE::SUCCESS) {
            // Retrieve image in GPU memory
            zed.retrieveImage(image, VIEW::LEFT, MEM::GPU);
            // Update pose data (used for projection of the mesh over the current image)
            tracking_state = zed.getPosition(pose);

            if(mapping_activated) {
                mapping_state = zed.getSpatialMappingState();
                // Compute elapsed time since the last call of Camera::requestSpatialMapAsync()
                auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - ts_last).count();
                // Ask for a mesh update if 500ms elapsed since last request
                if((duration > 500) && viewer.chunksUpdated()) {
                    zed.requestSpatialMapAsync();
                    ts_last = chrono::high_resolution_clock::now();
                }

                if(zed.getSpatialMapRequestStatusAsync() == ERROR_CODE::SUCCESS) {
                    zed.retrieveSpatialMapAsync(map);
                    viewer.updateChunks();
                }
            }

            bool change_state = viewer.updateImageAndState(image, pose.pose_data, tracking_state, mapping_state);

            if(change_state) {
                if(!mapping_activated) {
                    Transform init_pose;
                    zed.resetPositionalTracking(init_pose);

                    // Configure Spatial Mapping parameters
					spatial_mapping_parameters.resolution_meter = SpatialMappingParameters::get(SpatialMappingParameters::MAPPING_RESOLUTION::LOW);
                    spatial_mapping_parameters.use_chunk_only = true;
                    spatial_mapping_parameters.save_texture = false;
#if CREATE_MESH
					spatial_mapping_parameters.map_type = SpatialMappingParameters::SPATIAL_MAP_TYPE::MESH;
#else
					spatial_mapping_parameters.map_type = SpatialMappingParameters::SPATIAL_MAP_TYPE::FUSED_POINT_CLOUD;
#endif					
                    // Enable spatial mapping
                    try {
                        zed.enableSpatialMapping(spatial_mapping_parameters);
                        print("Spatial Mapping will output a " + string(toString(spatial_mapping_parameters.map_type).c_str()));
                    } catch(string e) {
                        print("Error enable Spatial Mapping "+ e);
                    }

                    // Clear previous Mesh data
                    map.clear();
                    viewer.clearCurrentMesh();

                    // Start a timer, we retrieve the mesh every XXms.
                    ts_last = chrono::high_resolution_clock::now();

                    mapping_activated = true;
                } else {
                    // Extract the whole mesh
                    zed.extractWholeSpatialMap(map);
#if CREATE_MESH
                    MeshFilterParameters filter_params;
                    filter_params.set(MeshFilterParameters::MESH_FILTER::MEDIUM);
                    // Filter the extracted mesh
                    map.filter(filter_params, true);
					viewer.clearCurrentMesh();

                    // If textures have been saved during spatial mapping, apply them to the mesh
                    if(spatial_mapping_parameters.save_texture)
                        map.applyTexture(MESH_TEXTURE_FORMAT::RGB);
#endif
                    // Save mesh as an OBJ file
                    string saveName = getDir() + "mesh_gen.obj";
                    bool error_save = map.save(saveName.c_str());
                    if(error_save)
                        print("Mesh saved under: " +saveName);
					else
                        print("Failed to save the mesh under: " +saveName);

                    mapping_state = SPATIAL_MAPPING_STATE::NOT_ENABLED;
                    mapping_activated = false;
                }
            }
        }
    }

    image.free();
    map.clear();

    zed.disableSpatialMapping();
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
