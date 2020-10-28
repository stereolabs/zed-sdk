# Tutorial 4: Positional tracking with the ZED

This tutorial shows how to use the ZED as a positional tracker. The program will loop until 1000 position are grabbed.
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED SDK Python API](https://www.stereolabs.com/docs/app-development/python/install/)

# Code overview
## Create a camera

As in previous tutorials, we create, configure and open the ZED. 

```python
# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 video mode (default fps: 60)
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # Use a right-handed Y-up coordinate system
init_params.coordinate_units = sl.UNIT.METER # Set units in meters

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS):
    exit(-1)
```

## Enable positional tracking

Once the camera is opened, we must enable the positional tracking module in order to get the position and orientation of the ZED.

```python
# Enable positional tracking with default parameters
tracking_parameters = sl.PositionalTrackingParameters()
err = zed.enablePositionalTracking(tracking_parameters)
if (err != sl.ERROR_CODE.SUCCESS):
    exit(-1)
```

In the above example, we leave the default tracking parameters. For the list of available parameters, check the online documentation.

## Capture pose data

Now that the ZED is opened and the positional tracking enabled, we create a loop to grab and retrieve the camera position.

The camera position is given by the class sl::Pose. This class contains the translation and orientation of the camera, as well as image timestamp and tracking confidence (quality).<br/>
A pose is always linked to a reference frame. The SDK provides two reference frame : REFERENCE_FRAME.WORLD and REFERENCE_FRAME.CAMERA.<br/> It is not the purpose of this tutorial to go into the details of these reference frame. Read the documentation for more information.<br/>
In the example, we get the device position in the World Frame.

```python
# Track the camera position during 1000 frames
i = 0
zed_pose = sl.Pose()
while (i < 1000) :
    if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
        zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD) # Get the pose of the left eye of the camera with reference to the world frame

        # Display the translation and timestamp
        py_translation = sl.Translation()
        tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
        ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
        tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
        print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}\n".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))

        # Display the orientation quaternion
        py_orientation = sl.Orientation()
        ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
        oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
        oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
        ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
        print("Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

	i = i+1
    

```

### Inertial Data

If a ZED Mini or a ZED2 is open, we can have access to the inertial data from the integrated IMU

```python
bool zed_mini = (zed.getCameraInformation().camera_model == sl.MODEL.ZED_M);
bool zed_two = (zed.getCameraInformation().camera_model == sl.MODEL.ZED2);
```

First, we test that the opened camera is a ZED Mini, then, we display some useful IMU data, such as the quaternion and the linear acceleration.

```python
if (zed_mini or zed_two) : # Display IMU data

    # Get IMU data
    zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
    zed_imu = zed_sensors.get_imu_data()

    # Display the IMU orientation quaternion
    zed_imu_pose = sl.Transform()
    ox = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[0], 3)
    oy = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[1], 3)
    oz = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[2], 3)
    ow = round(zed_imu.get_pose(zed_imu_pose).get_orientation().get()[3], 3)
    print("IMU Orientation: Ox: {0}, Oy: {1}, Oz {2}, Ow: {3}\n".format(ox, oy, oz, ow))

    #Display the IMU acceleratoin
    acceleration = [0,0,0]
    zed_imu.get_linear_acceleration(acceleration)
    ax = round(acceleration[0], 3)
    ay = round(acceleration[1], 3)
    az = round(acceleration[2], 3)
    print("IMU Acceleration: Ax: {0}, Ay: {1}, Az {2}\n".format(ax, ay, az))


}
```

This will loop until the ZED has been tracked during 1000 frames. We display the camera translation (in meters) in the console window and close the camera before exiting the application.

```python
# Disable positional tracking and close the camera
zed.disable_positional_tracking()
zed.close()
return 0
```

You can now use the ZED as an inside-out positional tracker. You can now read the next tutorial to learn how to use the Spatial Mapping.

