# Tutorial 4: Positional tracking with the ZED

This tutorial shows how to use the ZED as a positional tracker. The program will loop until 1000 position are grabbed.
We assume that you have followed previous tutorials.

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
## Enable positional tracking

Once the camera is opened, we must enable the positional tracking module in order to get the position and orientation of the ZED.

```csharp
Quaternion quat = Quaternion.Identity;
Vector3 vec = Vector3.Zero;

err = zedCamera.EnablePositionalTracking(ref quat, ref vec);
if (err != ERROR_CODE.SUCCESS)
    Environment.Exit(-1);
```

## Capture pose data

The camera position is given by the class `sl.Pose`. This struct contains the translation and orientation of the camera, as well as image timestamp and tracking confidence (pose_confidence).

A pose is always linked to a reference frame. The SDK provides two reference frame : `REFERENCE_FRAME.WORLD` and `REFERENCE_FRAME.CAMERA`.

It is not the purpose of this tutorial to go into the details of these reference frame. Read the documentation for more information.

In the example, we get the device position in the World Frame.

### Inertial Data

If a ZED Mini or a ZED 2 is open, we can have access to the inertial data from the integrated IMU