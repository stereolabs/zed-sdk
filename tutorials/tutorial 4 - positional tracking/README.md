# Tutorial 4: Positional tracking with the ZED

This tutorial shows how to use the ZED as a positional tracker. The program will loop until 1000 position are grabbed.
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

# Code overview

## Enable positional tracking

Once the camera is opened, we must enable the positional tracking module in order to get the position and orientation of the ZED.

## Capture pose data

The camera position is given by the class `sl::Pose`. This class contains the translation and orientation of the camera, as well as image timestamp and tracking confidence (quality).

A pose is always linked to a reference frame. The SDK provides two reference frame : `REFERENCE_FRAME::WORLD` and `REFERENCE_FRAME::CAMERA`.

It is not the purpose of this tutorial to go into the details of these reference frame. Read the documentation for more information.

In the example, we get the device position in the World Frame.

### Inertial Data

If a ZED Mini or a ZED 2 is open, we can have access to the inertial data from the integrated IMU