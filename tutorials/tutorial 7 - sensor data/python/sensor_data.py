########################################################################
#
# Copyright (c) 2020, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import pyzed.sl as sl
import cv2
import numpy as np

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units = sl.UNIT.METER

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    cam_model = zed.get_camera_information().camera_model
    if cam_model == sl.MODEL.ZED :
        print("This tutorial only supports ZED-M and ZED2 camera models")
        exit(1)

    # Get Sensor Data for 2 seconds (800 samples)
    i = 0
    data = sl.SensorsData()
    first_ts = sl.Timestamp()
    prev_imu_ts = sl.Timestamp()
    prev_baro_ts = sl.Timestamp()
    prev_mag_ts = sl.Timestamp()
    while i < 800 :
        # Get Sensor Data not synced with image frames
        if zed.get_sensors_data(data, sl.TIME_REFERENCE.CURRENT) != sl.ERROR_CODE.SUCCESS :
            print("Error retrieving Sensor Data")
            break

        imu_ts = data.get_imu_data().timestamp

        if i == 0 :
            first_ts = imu_ts

        # Check if Sensors Data are updated
        if prev_imu_ts.data_ns == imu_ts.data_ns :
            continue

        prev_imu_ts = imu_ts

        print("*** Sample #"+str(i))

        seconds = data.get_imu_data().timestamp.get_seconds() - first_ts.get_seconds()
        print(" * Relative timestamp: "+str(seconds)+" sec")

        # Filtered orientation quaternion
        zed_imu = data.get_imu_data()

        #Display the IMU acceleratoin
        acceleration = [0,0,0]
        zed_imu.get_linear_acceleration(acceleration)
        ax = round(acceleration[0], 3)
        ay = round(acceleration[1], 3)
        az = round(acceleration[2], 3)
        print("IMU Acceleration: Ax: {0}, Ay: {1}, Az {2}\n".format(ax, ay, az))

        #Display the IMU angular velocity
        a_velocity = [0,0,0]
        zed_imu.get_angular_velocity(a_velocity)
        vx = round(a_velocity[0], 3)
        vy = round(a_velocity[1], 3)
        vz = round(a_velocity[2], 3)
        print("IMU Angular Velocity: Vx: {0}, Vy: {1}, Vz {2}\n".format(vx, vy, vz))

        # Display the IMU orientation quaternion
        zed_imu_pose = sl.Transform()
        ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3)
        oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
        oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
        ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)
        print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

        if cam_model == sl.MODEL.ZED2 :

            # IMU temperature
            location = sl.SENSOR_LOCATION.IMU
            temp = data.get_temperature_data().get(location)
            if temp != -1:
               print(" *  IMU temperature: "+str(temp)+"C")

            # Check if Magnetometer Data are updated
            mag_ts = data.get_magnetometer_data().timestamp
            if (prev_mag_ts.data_ns != mag_ts.data_ns) :
                prev_mag_ts = mag_ts
                mx = round(data.get_magnetometer_data().get_magnetic_field_calibrated()[0])
                my = round(data.get_magnetometer_data().get_magnetic_field_calibrated()[1])
                mz = round(data.get_magnetometer_data().get_magnetic_field_calibrated()[2])
                print(" * Magnetic Fields [uT]: x: {0}, y: {1}, z: {2}".format(mx, my, mz))

            baro_ts = data.get_barometer_data().timestamp
            if (prev_baro_ts.data_ns != baro_ts.data_ns) :
                prev_baro_ts = baro_ts

                # Atmospheric pressure
                print(" * Atmospheric pressure [hPa]: "+str(data.get_barometer_data().pressure))

                # Barometer temperature
                location = sl.SENSOR_LOCATION.BAROMETER
                baro_temp = data.get_temperature_data().get(location)
                if baro_temp != -1:
                    print(" * Barometer temperature: "+str(temp)+"C")

                # Camera temperatures
                location_left = sl.SENSOR_LOCATION.ONBOARD_LEFT
                location_right = sl.SENSOR_LOCATION.ONBOARD_RIGHT

                left_temp = data.get_temperature_data().get(location_left)
                right_temp = data.get_temperature_data().get(location_right)
                print(" * Camera left temperature: "+str(left_temp))
                print(" * Camera right temperature: "+str(right_temp))

        i = i+1

    zed.close()
    return 0

if __name__ == "__main__":
    main()

