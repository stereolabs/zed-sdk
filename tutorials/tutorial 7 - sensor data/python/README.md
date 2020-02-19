# Tutorial 7: Getting Sensors Data

This tutorial shows how to use retrieve sensor data from ZED Mini and ZED2. It will loop until 800 data samples are grabbed, printing the updated values on console.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 7 64bits or later, Ubuntu 16.04/18.04
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED SDK Python API](https://github.com/stereolabs/zed-python-api)

# Code overview
# Create a ZED camera object

```python
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
```

# Capture data
The sensor data can be retrieved in two ways: synchronized or not synchronized to the image frames<br/>
In this tutorial, we grab 800 not synchronized data, equal to 2 seconds.

```python
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

    [...]
```

## Process data

To be sure that data are updated, since they are not synchronized to camera frames, we must check that the
timestamp as changed from the previous retrieved values. We use the timestamp of the IMU sensor as main value
since it's the sensor that runs at higher frequency. <br/>

```python
# Check if Sensors Data are updated
if prev_imu_ts.data_ns == imu_ts.data_ns :
    continue

prev_imu_ts = imu_ts
```

If the data are updated we can extract and use them:

```python
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
```

If we are using a ZED2 we have more sensor data to be acquired and elaborated:

```python
        if cam_model == sl.MODEL.ZED2 :
```

IMU Temperature:
```python
            # IMU temperature
            location = sl.SENSOR_LOCATION.IMU
            temp = data.get_temperature_data().get(location)
            if temp != -1:
               print(" *  IMU temperature: "+str(temp)+"C")

```

Magnetic fields:
```python
            # Check if Magnetometer Data are updated
            mag_ts = data.get_magnetometer_data().timestamp
            if (prev_mag_ts.data_ns != mag_ts.data_ns) :
                prev_mag_ts = mag_ts
                mx = round(data.get_magnetometer_data().get_magnetic_field_calibrated()[0])
                my = round(data.get_magnetometer_data().get_magnetic_field_calibrated()[1])
                mz = round(data.get_magnetometer_data().get_magnetic_field_calibrated()[2])
                print(" * Magnetic Fields [uT]: x: {0}, y: {1}, z: {2}".format(mx, my, mz))
```

Barometer data:
```python
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
```

## Close camera and exit

Once the data are extracted, don't forget to close the camera before exiting the program.<br/>

```python
# Close camera
zed.close()
return 0
```

And this is it!<br/>

You can now get all the sensor data from ZED-M and ZED2 cameras.

