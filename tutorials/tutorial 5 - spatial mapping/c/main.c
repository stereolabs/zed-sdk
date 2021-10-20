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


#include <sl/c_api/zed_interface.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

    // Create a ZED camera object
	int camera_id = 0;
	sl_create_camera(camera_id);

	struct SL_InitParameters init_param;
	init_param.camera_fps = 30;
	init_param.resolution = SL_RESOLUTION_HD1080;
	init_param.input_type = SL_INPUT_TYPE_USB;
	init_param.camera_device_id = camera_id;
	init_param.camera_image_flip = SL_FLIP_MODE_AUTO; 
	init_param.camera_disable_self_calib = false;
	init_param.enable_image_enhancement = true;
	init_param.svo_real_time_mode = true;
	init_param.depth_mode = SL_DEPTH_MODE_PERFORMANCE;
	init_param.depth_stabilization = true;
	init_param.depth_maximum_distance = 40;
	init_param.depth_minimum_distance = -1;
	init_param.coordinate_unit = SL_UNIT_METER;
	init_param.coordinate_system = SL_COORDINATE_SYSTEM_LEFT_HANDED_Y_UP;
	init_param.sdk_gpu_id = -1;
	init_param.sdk_verbose = false;
	init_param.sensors_required = false;
	init_param.enable_right_side_measure = false;

    // Open the camera
	int state = sl_open_camera(camera_id, &init_param, "", "", 0, "", "", "");

    if (state != 0) {
		printf("Error Open \n");
        return 1;
    }

	//Enable Positional tracking
	struct SL_PositionalTrackingParameters tracking_param;
	tracking_param.enable_area_memory = true;
	tracking_param.enable_imu_fusion = true;
	tracking_param.enable_pose_smothing = false;

	struct SL_Vector3  position;
	position = (struct SL_Vector3) { .x = 0, .y = 0, .z = 0 };
	struct SL_Quaternion  rotation;
	rotation = (struct SL_Quaternion) { .x = 0, .y = 0, .z = 0, .w = 1 };

	tracking_param.initial_world_position = position;
	tracking_param.initial_world_rotation = rotation;
	tracking_param.set_as_static = false;
	tracking_param.set_floor_as_origin = false;

	state = sl_enable_positional_tracking(camera_id, &tracking_param, "");
	if (state != 0) {
		printf("Error Enable Tracking %i , exit program.\n", state);
		return 0;
	}

	struct SL_SpatialMappingParameters mapping_param;
	mapping_param.map_type = SL_SPATIAL_MAP_TYPE_MESH;
	mapping_param.max_memory_usage = 2048;
	mapping_param.range_meter = 0;
	mapping_param.resolution_meter = 0.05f;
	mapping_param.save_texture = true;
	mapping_param.use_chunk_only = true;
	mapping_param.reverse_vertex_order = false;

	sl_enable_spatial_mapping(camera_id, &mapping_param);

	struct SL_RuntimeParameters rt_param;
	rt_param.enable_depth = true;
	rt_param.confidence_threshold = 100;
	rt_param.reference_frame = SL_REFERENCE_FRAME_CAMERA;
	rt_param.sensing_mode = SL_SENSING_MODE_STANDARD;
	rt_param.texture_confidence_threshold = 100;

	int width = sl_get_width(camera_id);
	int height = sl_get_height(camera_id);

	int i = 1;
	// Grab data during 500 frames
	while (i <= 500) {
		// Grab an image
		state = sl_grab(camera_id, &rt_param);
		// A new image is available if grab() returns ERROR_CODE::SUCCESS
		if (state == 0) {

			// In the background, spatial mapping will use newly retrieved images, depth and pose to update the mesh
			enum SL_SPATIAL_MAPPING_STATE map_state = sl_get_spatial_mapping_state(camera_id);

			printf("\r Images captured: %i / 500 || Spatial mapping state : %i \t \n", i, map_state);
			i++;
		}
	}

	printf("Extracting Mesh...\n");
	// Extract the whole mesh.
	sl_extract_whole_spatial_map(camera_id);
	// Filter the mesh

	int *nb_vertices_per_submesh = (int *)malloc(sizeof(int[MAX_SUBMESH]));
	int *nb_triangles_per_submesh = (int *)malloc(sizeof(int[MAX_SUBMESH]));
	int *updated_indices = (int *)malloc(sizeof(int[MAX_SUBMESH]));
	int nb_updated_submeshes = 0;
	int nb_vertices_tot = 0;
	int nb_triangles_tot = 0;

	printf("Filtering Mesh...\n");
	sl_filter_mesh(camera_id, SL_MESH_FILTER_MEDIUM, nb_vertices_per_submesh, nb_triangles_per_submesh, &nb_updated_submeshes, updated_indices, &nb_vertices_tot, &nb_triangles_tot, MAX_SUBMESH);

	// Save the mesh
	printf("Saving Mesh ... \n");
	sl_save_mesh(camera_id, "mesh.obj", SL_MESH_FILE_FORMAT_OBJ);

	free(nb_vertices_per_submesh);
	free(nb_triangles_per_submesh);
	free(updated_indices);

	sl_disable_spatial_mapping(camera_id);
	sl_disable_positional_tracking(camera_id, "");
	sl_close_camera(camera_id);
    return 0;
}
