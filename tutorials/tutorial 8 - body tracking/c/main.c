///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2023, STEREOLABS.
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
	init_param.depth_stabilization = 1;
	init_param.depth_maximum_distance = 40;
	init_param.depth_minimum_distance = -1;
	init_param.coordinate_unit = SL_UNIT_METER;
	init_param.coordinate_system = SL_COORDINATE_SYSTEM_LEFT_HANDED_Y_UP;
	init_param.sdk_gpu_id = -1;
	init_param.sdk_verbose = false;
	init_param.sensors_required = false;
	init_param.enable_right_side_measure = false;
	init_param.open_timeout_sec = 5.0f;
	init_param.async_grab_camera_recovery = false;

    // Open the camera
	int state = sl_open_camera(camera_id, &init_param, 0,  "", "", 0, "", "", "");

    if (state != 0) {
		printf("Error Open \n");
        return 1;
    }

	//Enable Positional tracking
	struct SL_PositionalTrackingParameters tracking_param;
	tracking_param.enable_area_memory = true;
	tracking_param.enable_imu_fusion = true;
	tracking_param.enable_pose_smothing = false;
	tracking_param.depth_min_range = -1;

	struct SL_Vector3  position;
	position = (struct SL_Vector3) { .x = 0, .y = 0, .z = 0 };
	struct SL_Quaternion  rotation;
	rotation = (struct SL_Quaternion) { .x = 0, .y = 0, .z = 0, .w = 1 };

	tracking_param.initial_world_position = position;
	tracking_param.initial_world_rotation = rotation;
	tracking_param.set_as_static = false;
	tracking_param.set_floor_as_origin = false;
	tracking_param.set_gravity_as_origin = true;


	state = sl_enable_positional_tracking(camera_id, &tracking_param, "");
	if (state != 0) {
		printf("Error enable tracking \n");
		return 1;
	}

	struct SL_BodyTrackingParameters bt_param;
	bt_param.enable_segmentation = false;
	bt_param.enable_tracking = true;
	bt_param.enable_body_fitting = true;
	bt_param.image_sync = true;
	bt_param.max_range = 40;
	bt_param.detection_model = SL_BODY_TRACKING_MODEL_HUMAN_BODY_MEDIUM;
	bt_param.allow_reduced_precision_inference = false;
	bt_param.body_format = SL_BODY_FORMAT_BODY_70;
	bt_param.body_selection = SL_BODY_KEYPOINTS_SELECTION_FULL;
	bt_param.instance_module_id = 0;

	state = sl_enable_body_tracking(camera_id, &bt_param);
	if (state != 0) {
		printf("Error enable od \n");
		return 1;
	}

	struct SL_ObjectDetectionRuntimeParameters bt_rt_param;
	bt_rt_param.detection_confidence_threshold = 40;

	struct SL_RuntimeParameters rt_param;
	rt_param.enable_depth = true;
	rt_param.confidence_threshold = 95;
	rt_param.reference_frame = SL_REFERENCE_FRAME_CAMERA;
	rt_param.texture_confidence_threshold = 100;
	rt_param.confidence_threshold = 95;
	rt_param.remove_saturated_areas = true;

	struct SL_Bodies bodies;

	// Capture 50 frames and stop
	int nb_detection = 0;
	while (nb_detection < 100) {
		// Grab an image
		state = sl_grab(camera_id, &rt_param);
		// A new image is available if grab() returns ERROR_CODE::SUCCESS
		if (state == 0) {
			sl_retrieve_bodies(camera_id, &bt_rt_param, &bodies, 0);

			if (bodies.is_new == 1) {
				printf("%i Objects detected \n", bodies.nb_bodies);

				if (bodies.nb_bodies > 0) {
					struct SL_BodyData first_body = bodies.body_list[0];

					printf("First object attributes :\n");
					if (bt_param.enable_tracking == true) {
						printf(" Tracking ID: %i tracking state: %i / %i \n", (int)first_body.id, (int)first_body.tracking_state, (int)first_body.action_state);
					}

					printf(" 3D Position: (%f, %f, %f) / Velocity: (%f, %f, %f) \n", first_body.position.x, first_body.position.y, first_body.position.z,
						first_body.velocity.x, first_body.velocity.y, first_body.velocity.z);

					printf(" Bounding Box 2D \n");
					for (int i = 0; i < 4; i++) {
						printf("    (%f,%f) \n", first_body.bounding_box_2d[i].x, first_body.bounding_box_2d[i].y);
					}

					printf(" Bounding Box 3D \n");
					for (int i = 0; i < 4; i++) {
						printf("    (%f,%f,%f) \n", first_body.bounding_box[i].x, first_body.bounding_box[i].y, first_body.bounding_box[i].z);
					}

					printf("Press Enter to Continue");
					while (getchar() != '\n');
				}
				nb_detection++;
			}
		}
	}

	sl_close_camera(camera_id);
    return 0;
}

