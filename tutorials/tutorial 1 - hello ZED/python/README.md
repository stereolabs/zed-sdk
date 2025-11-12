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
if (err > sl.ERROR_CODE.SUCCESS) :
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

# Testing the ZED SDK Python GPU support with CuPy

CuPy is a NumPy/SciPy-compatible Array Library for GPU-accelerated Computing with Python (see https://cupy.dev/) and the ZED SDK Python API support getting data in its format.

If you want to run high performance python scripts using CuPy and GPU retrieval, you can first start by instally CuPy (see https://cupy.dev/) and then running the script `hello_zed_gpu.py`. The script will:
- Open a connected ZED camera.
- Retrieve an image.
- Run some operations and benchmark on the retrieved image.
- Display on the terminal the results of the tests.

Without deep-diving into the script content, you can just look at its output to validate everything is fine with your setup.

For example, on an Orin NX16 with a ZED-X
``` bash
> python hello_zed_gpu.py
✅ CuPy detected - GPU acceleration available
   CuPy version: 13.5.1
   CUDA version: 12080
ZED SDK CuPy Integration Test
========================================
Opening ZED camera...
[2025-07-31 12:54:15 UTC][ZED][INFO] Logging level INFO
[2025-07-31 12:54:16 UTC][ZED][INFO] Using GMSL input... Switched to default resolution HD1200
[2025-07-31 12:54:19 UTC][ZED][INFO] [Init]  Camera FW version: 2001
[2025-07-31 12:54:19 UTC][ZED][INFO] [Init]  Video mode: HD1200@30
[2025-07-31 12:54:19 UTC][ZED][INFO] [Init]  Serial Number: S/N 48922857
[2025-07-31 12:54:19 UTC][ZED][INFO] [Init]  Depth mode: NEURAL
ZED camera opened successfully.
Retrieving image data...
Retrieved image on GPU: 1920x1200

🧪 Testing GPU image processing (basic grayscale conversion)...
   Input image: (1200, 1920, 4)
   Processed image: (1200, 1920)
✅ GPU processing test passed!
========================================

💾 Testing memory allocation strategies...
   CPU allocation: (480, 640, 4), float32
   GPU allocation: (480, 640, 4), float32
   CPU->GPU transfer: (480, 640, 4)
   GPU->CPU transfer: (480, 640, 4)
✅ Memory allocation test passed!
========================================

🔍 Testing GPU memory usage...
   Initial GPU memory usage: 0.0 MB
   After allocation: 15.3 MB
   After cleanup: 0.0 MB
✅ GPU memory test passed!
========================================

🔬 Testing data integrity...
   Data integrity verified: (2, 2, 4)
✅ Data integrity test passed!
========================================

⚡ Running performance benchmark...
   Benchmark image size: 1920x1200
   CPU processing (10 iterations): 538.658 milliseconds
   GPU processing (10 iterations): 93.961 milliseconds
   Speedup: 5.7x
🚀 GPU processing is faster!
========================================

🎉 All tests completed!
   Your system is ready for GPU-accelerated ZED processing with the Python API!
```

Similarly, on a computer equipped with a NVIDIA GeForce RTX 4060 Ti and with a ZED2:
``` bash
> python hello_zed_gpu.py
✅ CuPy detected - GPU acceleration available
   CuPy version: 13.4.1
   CUDA version: 11080
ZED SDK CuPy Integration Test
========================================
Opening ZED camera...
[2025-07-30 15:38:50 UTC][ZED][INFO] Logging level INFO
[2025-07-30 15:38:51 UTC][ZED][INFO] Using USB input... Switched to default resolution HD720
[2025-07-30 15:38:52 UTC][ZED][INFO] [Init]  Camera successfully opened.
[2025-07-30 15:38:52 UTC][ZED][INFO] [Init]  Camera FW version: 1523
[2025-07-30 15:38:52 UTC][ZED][INFO] [Init]  Video mode: HD720@30
[2025-07-30 15:38:52 UTC][ZED][INFO] [Init]  Serial Number: S/N 24046162
[2025-07-30 15:38:52 UTC][ZED][INFO] [Init]  Depth mode: NEURAL
ZED camera opened successfully.
Retrieving image data...
Retrieved image on GPU: 1280x720

🧪 Testing GPU image processing (basic grayscale conversion)...
   Input image: (720, 1280, 4)
   Processed image: (720, 1280)
✅ GPU processing test passed!
========================================

💾 Testing memory allocation strategies...
   CPU allocation: (480, 640, 4), float32
   GPU allocation: (480, 640, 4), float32
   CPU->GPU transfer: (480, 640, 4)
   GPU->CPU transfer: (480, 640, 4)
✅ Memory allocation test passed!
========================================

🔍 Testing GPU memory usage...
   Initial GPU memory usage: 0.0 MB
   After allocation: 15.3 MB
   After cleanup: 0.0 MB
✅ GPU memory test passed!
========================================

🔬 Testing data integrity...
   Data integrity verified: (2, 2, 4)
✅ Data integrity test passed!
========================================

⚡ Running performance benchmark...
   Benchmark image size: 1280x720
   CPU processing (10 iterations): 97.423 milliseconds
   GPU processing (10 iterations): 1.970 milliseconds
   Speedup: 49.4x
🚀 GPU processing is faster!
========================================

🎉 All tests completed!
   Your system is ready for GPU-accelerated ZED processing with the Python API!
```
