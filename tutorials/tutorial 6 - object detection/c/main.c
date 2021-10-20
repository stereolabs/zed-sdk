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

	struct SL_ObjectDetectionParameters objs_param;
	objs_param.enable_body_fitting = false;
	objs_param.enable_mask_output = false;
	objs_param.enable_tracking = true;
	objs_param.image_sync = true;
	objs_param.max_range = 40;
	objs_param.model = SL_DETECTION_MODEL_MULTI_CLASS_BOX_MEDIUM;

	sl_enable_objects_detection(camera_id, &objs_param);

	struct SL_ObjectDetectionRuntimeParameters objs_rt_param;
	objs_rt_param.detection_confidence_threshold = 40;

	struct SL_RuntimeParameters rt_param;
	rt_param.enable_depth = true;
	rt_param.confidence_threshold = 100;
	rt_param.reference_frame = SL_REFERENCE_FRAME_CAMERA;
	rt_param.sensing_mode = SL_SENSING_MODE_STANDARD;
	rt_param.texture_confidence_threshold = 100;

	struct SL_Objects objects;

	// Capture 50 frames and stop
	int nb_detection = 0;
	while (nb_detection < 100) {
		// Grab an image
		state = sl_grab(camera_id, &rt_param);

		// A new image is available if grab() returns ERROR_CODE::SUCCESS
		if (state == 0) {
			sl_retrieve_objects(camera_id, &objs_rt_param, &objects);
			if (objects.is_new == 1) {
				printf("%i Objects detected \n", objects.nb_object);

				if (objects.nb_object > 0) {
					struct SL_ObjectData first_object = objects.object_list[0];

					printf("First object attributes :\n");
					printf("Label '%i' (conf. %f / 100) \n", (int)first_object.label, first_object.confidence);

					if (objs_param.enable_tracking == true) {
						printf(" Tracking ID: %i tracking state: %i / %i \n", (int)first_object.id, (int)first_object.tracking_state, (int)first_object.action_state);
					}

					printf(" 3D Position: (%f, %f, %f) / Velocity: (%f, %f, %f) \n", first_object.position.x, first_object.position.y, first_object.position.z,
						first_object.velocity.x, first_object.velocity.y, first_object.velocity.z);

					printf(" Bounding Box 2D \n");
					for (int i = 0; i < 4; i++) {
						printf("    (%f,%f) \n", first_object.bounding_box_2d[i].x, first_object.bounding_box_2d[i].y);
					}

					printf(" Bounding Box 3D \n");
					for (int i = 0; i < 4; i++) {
						printf("    (%f,%f,%f) \n", first_object.bounding_box[i].x, first_object.bounding_box[i].y, first_object.bounding_box[i].z);
					}

					printf("Press Enter to Continue");
					while (getchar() != '\n');
				}
				nb_detection++;
			}
		}
	}

	sl_disable_positional_tracking(camera_id, "");
	sl_disable_objects_detection(camera_id);
	sl_close_camera(camera_id);
    return 0;
}

