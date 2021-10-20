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

#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include <sl/c_api/zed_interface.h>

int main(int argc, char **argv) {

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

	struct SL_RuntimeParameters rt_param;
	rt_param.enable_depth = true;
	rt_param.confidence_threshold = 100;
	rt_param.reference_frame = SL_REFERENCE_FRAME_CAMERA;
	rt_param.sensing_mode = SL_SENSING_MODE_STANDARD;
	rt_param.texture_confidence_threshold = 100;

	int width = sl_get_width(camera_id);
	int height = sl_get_height(camera_id);

	//Create image ptr.
	int* image_ptr;
	// Init pointer.
	image_ptr = sl_mat_create_new(width, height, SL_MAT_TYPE_U8_C4, SL_MEM_CPU);
	//Create depth ptr.
	int* depth_ptr;
	// Init pointer.
	depth_ptr = sl_mat_create_new(width, height, SL_MAT_TYPE_F32_C4, SL_MEM_CPU);
	//Create point cloud ptr.
	int* point_cloud_ptr;
	// Init pointer.
	point_cloud_ptr = sl_mat_create_new(width, height, SL_MAT_TYPE_F32_C4, SL_MEM_CPU);

	// Capture 50 frames and stop
	int i = 0;
	while (i < 50) {
		// Grab an image
		state = sl_grab(camera_id, &rt_param);
		// A new image is available if grab() returns ERROR_CODE::SUCCESS
		if (state == 0) {

			// Retrieve left image
			sl_retrieve_image(camera_id, image_ptr, SL_VIEW_LEFT, SL_MEM_CPU, width, height);
			// Retrieve depth map. Depth is aligned on the left image
			sl_retrieve_measure(camera_id, depth_ptr, SL_MEASURE_DEPTH, SL_MEM_CPU, width, height);
			// Retrieve colored point cloud. Point cloud is aligned on the left image.
			sl_retrieve_measure(camera_id, point_cloud_ptr, SL_MEASURE_XYZRGBA, SL_MEM_CPU, width, height);

			// Get and print distance value in mm at the center of the image
			// We measure the distance camera - object using Euclidean distance
			int x = sl_mat_get_width(point_cloud_ptr) / 2;
			int y = sl_mat_get_height(point_cloud_ptr) / 2;

			struct SL_Vector4 point_cloud_value;
			sl_mat_get_value_float4(point_cloud_ptr, x, y, &point_cloud_value, SL_MEM_CPU);

			if (isfinite(point_cloud_value.z)) {
				float distance = sqrt(point_cloud_value.x * point_cloud_value.x + point_cloud_value.y * point_cloud_value.y + point_cloud_value.z * point_cloud_value.z);
				printf("Distance to the Camera at {%f; %f; %f}: %f mm \n", point_cloud_value.x, point_cloud_value.y, point_cloud_value.z, distance);
			}
			else {
				printf("Distance can not be computed at {%i; %i} \n", x, y);
			}

			i++;
		}
	}

	sl_close_camera(camera_id);
    return 0;
}

