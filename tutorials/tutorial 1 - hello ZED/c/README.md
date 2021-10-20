# Tutorial 1: Hello ZED

This tutorial simply shows how to configure and open the ZED, then print its serial number and then close the camera. This is the most basic step and a good start for using the ZED SDK.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- ZED SDK C API

# Code overview
The ZED API provides low-level access to camera control and configuration. To use the ZED in your application, you will need to create and open a Camera object. The API can be used with two different video inputs: the ZED live video (Live mode) or video files recorded in SVO format with the ZED API (Playback mode).

## Camera Configuration
To configure the camera, create a Camera object and specify your `SL_InitParameters`. Initial parameters let you adjust camera resolution, FPS, depth sensing parameters and more. These parameters can only be set before opening the camera and cannot be changed while the camera is in use.

```c
//Create a ZED camera object
int camera_id = 0;
sl_create_camera(camera_id);

//Set configuration parameters
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
```
Once initial configuration is done, open the camera.

```c
# Open the camera
int state = sl_open_camera(camera_id, &init_param, "", "", "", "", "", "");

  if (state != 0) {
  printf("Error Open \n");
      return 1;
  }
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

In this tutorial, we simply retrieve the serial number of the camera:

```c
// Get camera information (serial number)
int sn = sl_get_zed_serial(camera_id);
printf("Hello! This is my serial number: %d\n", sn);
```

In the console window, you should now see the serial number of the camera (also available on a sticker on the ZED USB cable).

<i> Note: </i>`CameraInformation` also contains the firmware version of the ZED, as well as calibration parameters.

To close the camera properly, use zed.close() and exit the program.

```c
# Close the camera
sl_close_camera(camera_id);
  return 0;
```
