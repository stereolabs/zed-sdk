# Tutorial 1: Hello ZED

This tutorial simply shows how to configure and open the ZED, then print its serial number and then close the camera. This is the most basic step and a good start for using the ZED SDK.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- [ZED SDK Python API](https://www.stereolabs.com/docs/app-development/python/install/)

# Code overview
The ZED API provides low-level access to camera control and configuration. To use the ZED in your application, you will need to create and open a Camera object. The API can be used with two different video inputs: the ZED live video (Live mode) or video files recorded in SVO format with the ZED API (Playback mode).

## Camera Configuration
To configure the camera, create a Camera object and specify your `InitParameters`. Initial parameters let you adjust camera resolution, FPS, depth sensing parameters and more. These parameters can only be set before opening the camera and cannot be changed while the camera is in use.

```python
# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080 
init_params.camera_fps = 30 
```

`InitParameters` contains a configuration by default. To get the list of available parameters, see [API](https://www.stereolabs.com/developers/documentation/API/classsl_1_1InitParameters.html) documentation.   

Once initial configuration is done, open the camera.

```python
# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) :
    exit(-1)
```

You can set the following initial parameters:
* Camera configuration parameters, using the `camera_*` entries (resolution, image flip...).
* SDK configuration parameters, using the `sdk_*` entries (verbosity, GPU device used...).
* Depth configuration parameters, using the `depth_*` entries (depth mode, minimum distance...).
* Coordinate frames configuration parameters, using the `coordinate_*` entries (coordinate system, coordinate units...).
* SVO parameters to use Stereolabs video files with the ZED SDK (filename, real-time mode...)


### Getting Camera Information
Camera parameters such as focal length, field of view or stereo calibration can be retrieved for each eye and resolution:

- Focal length: fx, fy.
- Principal points: cx, cy.
- Lens distortion: k1, k2.
- Horizontal and vertical field of view.
- Stereo calibration: rotation and translation between left and right eye.

Those values are available in `CalibrationParameters`. They can be accessed using `get_camera_information()`.

In this tutorial, we simplfy retrieve the serial number of the camera:

```python
# Get camera information (serial number)
zed_serial = zed.get_camera_information().serial_number
print("Hello! This is my serial number: ", zed_serial)
```

In the console window, you should now see the serial number of the camera (also available on a sticker on the ZED USB cable).

<i> Note: </i>`CameraInformation` also contains the firmware version of the ZED, as well as calibration parameters.

To close the camera properly, use zed.close() and exit the program.

```
# Close the camera
zed.close()
return 0
```

