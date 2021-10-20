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

	struct SL_RuntimeParameters rt_param;
	rt_param.enable_depth = true;
	rt_param.confidence_threshold = 100;
	rt_param.reference_frame = SL_REFERENCE_FRAME_CAMERA;
	rt_param.sensing_mode = SL_SENSING_MODE_STANDARD;
	rt_param.texture_confidence_threshold = 100;

	int width = sl_get_width(camera_id);
	int height = sl_get_height(camera_id);

	bool zed_has_imu = sl_get_sensors_configuration(camera_id)->gyroscope_parameters.is_available;

	//Create image ptr.
	int* image_ptr;
	// Init pointer.
	image_ptr = sl_mat_create_new(width, height, SL_MAT_TYPE_U8_C4, SL_MEM_CPU);

	// Capture 1000 frames and stop
	int i = 0;
	while (i < 1000) {
		// Grab an image
		state = sl_grab(camera_id, &rt_param);
		// A new image is available if grab() returns ERROR_CODE::SUCCESS
		if (state == 0) {

			struct SL_PoseData pose;
			sl_get_position_data(camera_id, &pose, SL_REFERENCE_FRAME_WORLD);

			struct SL_Vector3  zed_translation = pose.translation;
			struct SL_Quaternion zed_orientation = pose.rotation;

			unsigned long long ts = pose.timestamp;

			printf("Camera Translation: {%f, %f, %f}, Orientation: {%f, %f, %f, %f}, timestamp: %lld \n", zed_translation.x, zed_translation.y, zed_translation.z,
				zed_orientation.x, zed_orientation.y, zed_orientation.z, zed_orientation.w, ts);

			if (zed_has_imu) {

				struct SL_SensorData sensor_data;
				sl_get_sensors_data(camera_id, &sensor_data, SL_TIME_REFERENCE_IMAGE);

				struct SL_Quaternion imu_orientation = sensor_data.imu.orientation;
				struct SL_Vector3 acceleration = sensor_data.imu.linear_acceleration;

				printf("IMU Orientation: {%f, %f, %f, %f}, Acceleration: {%f, %f, %f}", imu_orientation.x, imu_orientation.y, imu_orientation.z, imu_orientation.w,
					acceleration.x, acceleration.y, acceleration.z);
			}

			i++;
		}
	}

	sl_disable_positional_tracking(camera_id, "");
	sl_close_camera(camera_id);
    return 0;
}
