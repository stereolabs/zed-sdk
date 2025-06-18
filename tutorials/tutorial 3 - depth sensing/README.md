# Tutorial 3: Depth sensing with the ZED

This tutorial shows how to get the depth from the ZED SDK. The program will loop until 50 frames are grabbed.
We assume that you have followed previous tutorials (opening the ZED and image capture).

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

# Code overview

## Create a camera

As in other tutorials, we create, configure and open the ZED.
We set the ZED in HD720 mode at 60fps and enable depth in NEURAL mode. The ZED SDK provides different depth modes: NEURAL_PLUS, NEURAL, NEURAL_LIGHT. For more information, see online documentation.

## Capture data

Now that the ZED is opened, we can capture images and depth. Retrieving the depth map is as simple as retrieving an image:
* We create a Mat to store the depth map.
* We call retrieveMeasure() to get the depth map.
* We call retrieveMeasure() to get the point cloud.
