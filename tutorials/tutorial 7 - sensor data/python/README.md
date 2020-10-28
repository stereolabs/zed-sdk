# Tutorial 7: Getting Sensors Data

This tutorial shows how to use retrieve sensor data from ZED Mini and ZED2. It will loop until 100 data samples are grabbed, printing the updated values on console.<br/>
We assume that you have followed previous tutorials.

## Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED SDK Python API](https://www.stereolabs.com/docs/app-development/python/install/)

## Code overview

### Create a ZED camera object

```python
    # Create a Camera object
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)
```

### Device Informations

We first display some informations about the sensors

```python
    # Display camera information (model,S/N, fw version)
    print("Camera Model: " + str(cam_model))
    print("Serial Number: " + str(info.serial_number))
    print("Camera Firmware: " + str(info.camera_configuration.firmware_version))
    print("Sensors Firmware: " + str(info.sensors_configuration.firmware_version))

    # Display sensors parameters (imu,barometer,magnetometer)
    printSensorParameters(info.sensors_configuration.accelerometer_parameters) # accelerometer configuration
    printSensorParameters(info.sensors_configuration.gyroscope_parameters) # gyroscope configuration
    printSensorParameters(info.sensors_configuration.magnetometer_parameters) # magnetometer configuration
    printSensorParameters(info.sensors_configuration.barometer_parameters) # barometer configuration
```

### Capture data

The sensor data can be retrieved in two ways: synchronized or not synchronized to the image frames<br/>
In this tutorial, we grab 100 not synchronized data

```python
    # Used to store the sensors timestamp to know if the sensors_data is a new one or not
    ts_handler = TimestampHandler()

    # Get Sensor Data for 5 seconds
    i = 0
    sensors_data = sl.SensorsData()

    while i < 100 :
        # retrieve the current sensors sensors_data
        # Depending on your Camera model or its firmware, differents sensors are presents.
        # They do not run at the same rate: Therefore, to do not miss samples we iterate as fast as we can and compare timestamp to know when a sensors_data is a new one
        # NOTE: There is no need to acquire images with grab() function. Sensors sensors_data are running in a separated internal capture thread.
        if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS :

            [...]
```

### Process data

To be sure that data are updated, since they are not synchronized to camera frames, we must check that the
timestamp as changed from the previous retrieved values. We use the timestamp of the IMU sensor as main value
since it's the sensor that runs at higher frequency. We defined a small utility class that will handle this. <br/>

```python
# Check if the data has been updated since the last time
# IMU is the sensor with the highest rate
if ts_handler.is_new(sensors_data.get_imu_data()):
```

If the data are updated we can extract and use them:

```python
print(" - IMU:")
# Filtered orientation quaternion
quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
print(" \t Orientation: [ Ox: {0}, Oy: {1}, Oz {2}, Ow: {3} ]".format(quaternion[0], quaternion[1], quaternion[2], quaternion[3]))

# linear acceleration
linear_acceleration = sensors_data.get_imu_data().get_linear_acceleration()
print(" \t Acceleration: [ {0} {1} {2} ] [m/sec^2]".format(linear_acceleration[0], linear_acceleration[1], linear_acceleration[2]))

# angular velocities
angular_velocity = sensors_data.get_imu_data().get_angular_velocity()
print(" \t Angular Velocities: [ {0} {1} {2} ] [deg/sec]".format(angular_velocity[0], angular_velocity[1], angular_velocity[2]))

# Check if Magnetometer data has been updated (not the same frequency than IMU)
if ts_handler.is_new(sensors_data.get_magnetometer_data()):
    magnetic_field_calibrated = sensors_data.get_magnetometer_data().get_magnetic_field_calibrated()
    print(" - Magnetometer\n \t Magnetic Field: [ {0} {1} {2} ] [uT]".format(magnetic_field_calibrated[0], magnetic_field_calibrated[1], magnetic_field_calibrated[2]))

# Check if Barometer data has been updated
if ts_handler.is_new(sensors_data.get_barometer_data()):
    magnetic_field_calibrated = sensors_data.get_barometer_data().pressure
    print(" - Barometer\n \t Atmospheric pressure: {0} [hPa]".format(sensors_data.get_barometer_data().pressure))
```

If we are using a ZED2 we have more sensor data to be acquired.

IMU:

```python
    # Filtered orientation quaternion
    quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
    print(" \t Orientation: [ Ox: {0}, Oy: {1}, Oz {2}, Ow: {3} ]".format(quaternion[0], quaternion[1], quaternion[2], quaternion[3]))

    # linear acceleration
    linear_acceleration = sensors_data.get_imu_data().get_linear_acceleration()
    print(" \t Acceleration: [ {0} {1} {2} ] [m/sec^2]".format(linear_acceleration[0], linear_acceleration[1], linear_acceleration[2]))

    # angular velocities
    angular_velocity = sensors_data.get_imu_data().get_angular_velocity()
    print(" \t Angular Velocities: [ {0} {1} {2} ] [deg/sec]".format(angular_velocity[0], angular_velocity[1], angular_velocity[2]))
```

Magnetic fields:

```python
    # Check if Magnetometer data has been updated (not the same frequency than IMU)
    if ts_handler.is_new(sensors_data.get_magnetometer_data()):
        magnetic_field_calibrated = sensors_data.get_magnetometer_data().get_magnetic_field_calibrated()
        print(" - Magnetometer\n \t Magnetic Field: [ {0} {1} {2} ] [uT]".format(magnetic_field_calibrated[0], magnetic_field_calibrated[1], magnetic_field_calibrated[2]))
```

Barometer data:

```python
    # Check if Barometer data has been updated 
    if ts_handler.is_new(sensors_data.get_barometer_data()):
        magnetic_field_calibrated = sensors_data.get_barometer_data().pressure
        print(" - Barometer\n \t Atmospheric pressure: {0} [hPa]".format(sensors_data.get_barometer_data().pressure))
```

### Close camera and exit

Once the data are extracted, don't forget to close the camera before exiting the program.<br/>

```python
# Close camera
zed.close()
return 0
```

And this is it!<br/>

You can now get all the sensor data from ZED-M and ZED2 cameras.

