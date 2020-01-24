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
    init_params.sensors_required = true;
    
    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << "Error " << err << ", exit program.\n";
        return -1;
    }

    // Enable positional tracking with default parameters
    PositionalTrackingParameters tracking_parameters;
    err = zed.enablePositionalTracking(tracking_parameters);
    if (err != ERROR_CODE::SUCCESS)
        return -1;

    // Track the camera position during 1000 frames
    int i = 0;
    Pose zed_pose;

    // Check if the camera is a ZED M and therefore if an IMU is available
    bool zed_mini = (zed.getCameraInformation().camera_model == MODEL::ZED_M);
    SensorsData sensor_data;

    while (i < 1000) {
        if (zed.grab() == ERROR_CODE::SUCCESS) {

            // Get the pose of the left eye of the camera with reference to the world frame
            zed.getPosition(zed_pose, REFERENCE_FRAME::WORLD); 

            // get the translation information
            auto zed_translation = zed_pose.getTranslation();

            // Display the translation and timestamp
            printf("\nTranslation: Tx: %.3f, Ty: %.3f, Tz: %.3f, Timestamp: %llu\n", zed_translation.tx,
                    zed_translation.ty, zed_translation.tz, (long long unsigned int) zed_pose.timestamp.getNanoseconds());
            
            // get the orientation information
            auto zed_orientation = zed_pose.getOrientation();

            // Display the orientation quaternion
            printf("Orientation: Ox: %.3f, Oy: %.3f, Oz: %.3f, Ow: %.3f\n", zed_orientation.ox,
                   zed_orientation.oy, zed_orientation.oz, zed_orientation.ow);
                        
            if (zed_mini) { // Display IMU data

                 // Get IMU data
                zed.getSensorsData(sensor_data, TIME_REFERENCE::IMAGE);

                auto imu_orientation = sensor_data.imu.pose.getOrientation();

                // Filtered orientation quaternion
                printf("IMU Orientation: Ox: %.3f, Oy: %.3f, Oz: %.3f, Ow: %.3f\n", imu_orientation.ox,
                       imu_orientation.oy, imu_orientation.oz, imu_orientation.ow);

                // Raw acceleration
                printf("IMU Acceleration: x: %.3f, y: %.3f, z: %.3f\n", sensor_data.imu.linear_acceleration.x,
                        sensor_data.imu.linear_acceleration.y, sensor_data.imu.linear_acceleration.z);
            }

            i++;
        }
    }

    // Disable positional tracking and close the camera
    zed.disablePositionalTracking();
    zed.close();
    return 0;
}
