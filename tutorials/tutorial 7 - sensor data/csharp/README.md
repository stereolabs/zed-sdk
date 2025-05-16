# Tutorial 5: Getting sensor data from ZED Mini and ZED2

This tutorial shows how to use retrieve sensor data from ZED Mini and ZED2. It will loop until 800 data samples are grabbed, printing the updated values on console.<br/>
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

## Capture data

The sensor data can be retrieved in two ways: synchronized (`TIME_REFERENCE.IMAGE`) or not synchronized (`TIME_REFERENCE.CURRENT`) to the image frames.

## Process data

To be sure that data are updated, since they are not synchronized to camera frames, we must check that the timestamp as changed from the previous retrieved values. We use the timestamp of the IMU sensor as main value since it's the sensor that runs at higher frequency.

If we are using a ZED2 we have more sensor data to be acquired and elaborated:

- IMU Temperature
- Magnetic fields
- Barometer data