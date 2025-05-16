# Tutorial 1: Hello ZED

This tutorial simply shows how to configure and open the ZED, then print its serial number and then close the camera. This is the most basic step and a good start for using the ZED C# wrapper.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).
- Make sure to build the .NET wrapper located in Stereolabs.zed folder at the root of the repository. Once built and installed, you should have the C# dll Stereolabs.zed.dll and the C interface DLL loaded by the C# dll in the ZED SDK bin directory (C:/Program Files(x86)/ZED SDK/bin)

## Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

# Code overview
The ZED C# API provides low-level access to camera control and configuration. To use the ZED in your application, you will need to create and open a Camera object. The API can be used with different video inputs: the ZED live video (Live mode) or video files recorded in SVO format with the ZED API (Playback mode) or a Streaming input (Streaming mode)

## Camera Configuration
To configure the camera, create a Camera object and specify your `InitParameters`. Initial parameters let you adjust camera resolution, FPS, depth sensing parameters and more. These parameters can only be set before opening the camera and cannot be changed while the camera is in use.

```csharp
 InitParameters init_params = new InitParameters();
 init_params.resolution = RESOLUTION.HD1080;
 init_params.cameraFPS = 30;
 Camera zedCamera = new Camera();

 // Open the camera
 ERROR_CODE err = zedCamera.Open(ref init_params);
 if (err != ERROR_CODE.SUCCESS)
	Environment.Exit(-1);
```
 

## Getting Camera Information

Camera parameters such as focal length, field of view or stereo calibration can be retrieved for each eye and resolution:

- Focal length: fx, fy.
- Principal points: cx, cy.
- Lens distortion: k1, k2.
- Horizontal and vertical field of view.
- Stereo calibration: rotation and translation between left and right eye.

Those values are available in `CalibrationParameters` structure. To retrieve those parameters, call 
`zedCamera.GetCalibrationParameters(true/false);`

Information such as serial number are available through dedicated functions : 
`zedCamera.GetZEDSerialNumber();`
`zedCamera.GetSensorsFirmwareVersion();`
