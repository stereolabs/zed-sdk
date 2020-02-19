///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2020, STEREOLABS.
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


#include <sl/Camera.hpp>

using namespace sl;

int main(int argc, char **argv) {

    // Create a ZED camera object
    Camera zed;

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD720; // Use HD720 video mode (default fps: 60)
    init_params.coordinate_system = COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP; // Use a right-handed Y-up coordinate system
    init_params.coordinate_units = UNIT::METER; // Set units in meters

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        printf("Error opening the camera: %s\n", toString(err).c_str());
        return 1;
    }

    // Check camera model
    MODEL cam_model = zed.getCameraInformation().camera_model;
    if (cam_model == MODEL::ZED) {
        printf("This tutorial only supports ZED-M and ZED2 camera models\n");
        return 1;
    }

    // Get Sensor Data for 2 seconds (800 samples)
    int i = 0;
    Timestamp first_ts = 0, prev_imu_ts = 0, prev_baro_ts = 0, prev_mag_ts = 0;
    SensorsData data;

    while (i < 800) {
        // Get Sensor Data not synced with image frames
        if (zed.getSensorsData(data, TIME_REFERENCE::CURRENT) != ERROR_CODE::SUCCESS) {
            printf("Error retrieving Sensor Data\n");
            return 1;
        }

        Timestamp imu_ts = data.imu.timestamp;

        if (i == 0) 
            first_ts = imu_ts;
        
        // Check if Sensor Data are updated
        if (prev_imu_ts == imu_ts) 
            continue;
        
        prev_imu_ts = imu_ts;

        printf("*** Sample #%d\n", i);

        printf(" * Relative timestamp: %g sec\n", static_cast<double> (data.imu.timestamp - first_ts) / 1e9);

        // Filtered orientation quaternion
        printf(" * IMU Orientation: Ox: %.3f, Oy: %.3f, Oz: %.3f, Ow: %.3f\n",
                data.imu.pose.getOrientation().ox, data.imu.pose.getOrientation().oy,
                data.imu.pose.getOrientation().oz, data.imu.pose.getOrientation().ow);
        // Filtered acceleration
        printf(" * IMU Acceleration [m/sec^2]: x: %.3f, y: %.3f, z: %.3f\n",
                data.imu.linear_acceleration.x, data.imu.linear_acceleration.y, data.imu.linear_acceleration.z);
        // Filtered angular velocities
        printf(" * IMU angular velocities [deg/sec]: x: %.3f, y: %.3f, z: %.3f\n",
                data.imu.angular_velocity.x, data.imu.angular_velocity.y, data.imu.angular_velocity.z);

        if (cam_model == MODEL::ZED2) {

            // IMU temperature
            float imu_temp;
            data.temperature.get(SensorsData::TemperatureData::SENSOR_LOCATION::IMU, imu_temp);
            printf(" * IMU temperature: %g C\n", imu_temp);

            // Check if Magnetometer Data are updated
            Timestamp mag_ts = data.magnetometer.timestamp;
            if (prev_mag_ts != mag_ts) {
                prev_mag_ts = mag_ts;

                // Filtered magnetic fields
                printf(" * Magnetic Fields [uT]: x: %.3f, y: %.3f, z: %.3f\n",
                        data.magnetometer.magnetic_field_calibrated.x, data.magnetometer.magnetic_field_calibrated.y, data.magnetometer.magnetic_field_calibrated.z);
            }

            // Check if Barometer Data are updated
            Timestamp baro_ts = data.barometer.timestamp;
            if (prev_baro_ts != baro_ts) {
                prev_baro_ts = baro_ts;

                // Atmospheric pressure
                printf(" * Atmospheric pressure [hPa]: %g\n", data.barometer.pressure);

                // Barometer temperature
                float baro_temp;
                data.temperature.get(SensorsData::TemperatureData::SENSOR_LOCATION::BAROMETER, baro_temp);
                printf(" * Barometer temperature: %g\n", baro_temp);

                // Camera temperatures
                float left_temp, right_temp;
                data.temperature.get(SensorsData::TemperatureData::SENSOR_LOCATION::ONBOARD_LEFT, left_temp);
                data.temperature.get(SensorsData::TemperatureData::SENSOR_LOCATION::ONBOARD_RIGHT, right_temp);
                printf(" * Camera left temperature: %g C\n", left_temp);
                printf(" * Camera right temperature: %g C\n", right_temp);
            }
        }

        i++;
    }

    // Close camera
    zed.close();
    return 0;
}
