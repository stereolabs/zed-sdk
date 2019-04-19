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

#define CREATE_MESH 1

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

    GLViewer viewer;
    bool error_viewer = viewer.init(argc, argv, camera_parameters);

    if(error_viewer) {
        viewer.exit();
        zed.close();
        return -1;
    }

    Mat image; // current left image
    Pose pose; // positional tracking data

#if CREATE_MESH
    Mesh map; // current incemental mesh
#else
    FusedPointCloud map; // current incemental fused point cloud
#endif

    SpatialMappingParameters spatial_mapping_parameters;
    TRACKING_STATE tracking_state = TRACKING_STATE_OFF;
    SPATIAL_MAPPING_STATE mapping_state = SPATIAL_MAPPING_STATE_NOT_ENABLED;
    bool mapping_activated = false; // indicates if the spatial mapping is running or not
    chrono::high_resolution_clock::time_point ts_last; // time stamp of the last mesh request
    
    // Enable positional tracking before starting spatial mapping
    zed.enableTracking();

    while(viewer.isAvailable()) {
        if(zed.grab() == SUCCESS) {
            // Retrieve image in GPU memory
            zed.retrieveImage(image, VIEW_LEFT, MEM_GPU);
            // Update pose data (used for projection of the mesh over the current image)
            tracking_state = zed.getPosition(pose);

            if(mapping_activated) {
                mapping_state = zed.getSpatialMappingState();
                // Compute elapse time since the last call of Camera::requestMeshAsync()
                auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - ts_last).count();
                // Ask for a mesh update if 500ms have spend since last request
                if(duration > 500) {
                    zed.requestSpatialMapAsync();
                    ts_last = chrono::high_resolution_clock::now();
                }

                if(zed.getSpatialMapRequestStatusAsync() == SUCCESS) {
                    zed.retrieveSpatialMapAsync(map);
                    viewer.updateMap(map);
                }
            }

            bool change_state = viewer.updateImageAndState(image, pose.pose_data, tracking_state, mapping_state);

            if(change_state) {
                if(!mapping_activated) {
                    Transform init_pose;
                    zed.resetTracking(init_pose);

                    // Configure Spatial Mapping parameters
                    spatial_mapping_parameters.resolution_meter = SpatialMappingParameters::get(SpatialMappingParameters::MAPPING_RESOLUTION_MEDIUM);
                    spatial_mapping_parameters.use_chunk_only = true;
                    spatial_mapping_parameters.save_texture = false;
#if CREATE_MESH
					spatial_mapping_parameters.map_type = SpatialMappingParameters::SPATIAL_MAP_TYPE_MESH;
#else
					spatial_mapping_parameters.map_type = SpatialMappingParameters::SPATIAL_MAP_TYPE_FUSED_POINT_CLOUD;
#endif					
                    // Enable spatial mapping
                    try {
                        zed.enableSpatialMapping(spatial_mapping_parameters);
						std::cout << "Spatial Mapping will output a " << spatial_mapping_parameters.map_type << "\n";
                    } catch(std::string e) {
                        std::cout <<"Error enable Spatial Mapping "<< e << std::endl;
                    }
                    // Start a timer, we retrieve the mesh every XXms.
                    ts_last = chrono::high_resolution_clock::now();

                    // clear previous Mesh data
                    viewer.clearCurrentMesh();

                    mapping_activated = true;
                } else {
                    // Extract the whole mesh
                    zed.extractWholeSpatialMap(map);
#if CREATE_MESH
                    MeshFilterParameters filter_params;
                    filter_params.set(MeshFilterParameters::MESH_FILTER_MEDIUM);
                    // Filter the extracted mesh
                    map.filter(filter_params, true);

					viewer.clearCurrentMesh();
					viewer.updateMap(map);

                    // If textures have been saved during spatial mapping, apply them to the mesh
                    if(spatial_mapping_parameters.save_texture)
                        map.applyTexture(MESH_TEXTURE_RGB);
#endif
                    //Save as an OBJ file
                    string saveName = getDir() + "mesh_gen.obj";
                    bool error_save = map.save(saveName.c_str());
                    if(error_save)
						cout << ">> Mesh saved under: " << saveName << endl;
					else
						cout << ">> Failed to save the mesh under: " << saveName << endl;

                    mapping_state = SPATIAL_MAPPING_STATE_NOT_ENABLED;
                    mapping_activated = false;
                }
            }
        }
    }

    image.free();
    map.clear();

    zed.disableSpatialMapping();
    zed.disableTracking();
    zed.close();
    return 0;
}
