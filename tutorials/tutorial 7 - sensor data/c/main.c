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
#include <math.h>


void printSensorConfiguration(struct SL_SensorParameters* sensor_parameters) {
	if (sensor_parameters->is_available) {
		printf("*****************************\n");
		printf("Sensor Type: %i \n", (int)sensor_parameters->type);
		printf("Max Rate: %f Hz \n", sensor_parameters->sampling_rate);
		printf("Range: [%f, %f] Sensor unit: %i \n", sensor_parameters->range.x, sensor_parameters->range.y, (int)sensor_parameters->sensor_unit);
		printf("Resolution: %f " " Sensor unit: %i \n", sensor_parameters->resolution, sensor_parameters->sensor_unit);
		if (isfinite(sensor_parameters->noise_density)) printf("Noise Density:  %f  Sensor unit: %i /sr(Hz) \n", sensor_parameters->noise_density, sensor_parameters->sensor_unit);
		if (isfinite(sensor_parameters->random_walk)) printf("Random Walk:  %f  Sensor unit: %i /s/sr(Hz) \n", sensor_parameters->random_walk, sensor_parameters->sensor_unit);
	}
}

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

	enum SL_MODEL model = sl_get_camera_model(camera_id);

	if (model == SL_MODEL_ZED) {
		printf("This tutorial does not work with ZED cameras which does not have additional sensors");
		return 0;
	}

	struct SL_SensorData sensor_data;

	int n = 0;

	while (n < 500) {

		// Depending on your camera model, different sensors are available.
		// NOTE: There is no need to acquire images with grab(). getSensorsData runs in a separate internal capture thread.
		if (sl_get_sensors_data(camera_id, &sensor_data, SL_TIME_REFERENCE_CURRENT) == SL_ERROR_CODE_SUCCESS) {

			printf("Sample %i \n", n++);
			printf(" - IMU:\n");
			printf(" \t Orientation: {%f,%f,%f,%f} \n", sensor_data.imu.orientation.x, sensor_data.imu.orientation.y, sensor_data.imu.orientation.z, sensor_data.imu.orientation.w);
			printf(" \t Acceleration: {%f,%f,%f} [m/sec^2] \n", sensor_data.imu.linear_acceleration.x, sensor_data.imu.linear_acceleration.y, sensor_data.imu.linear_acceleration.z);
			printf(" \t Angular Velocity: {%f,%f,%f} [deg/sec] \n", sensor_data.imu.angular_velocity.x, sensor_data.imu.angular_velocity.y, sensor_data.imu.angular_velocity.z);

			printf(" - Magnetometer \n \t Magnetic Field: {%f,%f,%f} [uT] \n", sensor_data.magnetometer.magnetic_field_c.x, sensor_data.magnetometer.magnetic_field_c.y, sensor_data.magnetometer.magnetic_field_c.z);
		
			printf(" - Barometer \n \t Atmospheric pressure: %f [hPa] \n", sensor_data.barometer.pressure);
		}
	}

	sl_close_camera(camera_id);
    return 0;
}
