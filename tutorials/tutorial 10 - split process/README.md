# Tutorial 10: Split Process with the ZED

This tutorial shows how to get an image then the depth from the ZED SDK. The program will loop until 50 frames are grabbed.
We assume that you have followed previous tutorials (opening the ZED and image capture).

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

# Code overview

## Create a camera

As in other tutorials, we create, configure and open the ZED.
We set the ZED in HD720 mode at 60fps and enable depth in NEURAL mode. The ZED SDK provides different depth modes: NEURAL_PLUS, NEURAL, NEURAL_LIGHT. For more information, see online documentation.

## Capture data

In previous tutorials, you have seen that images and depth can be captured using the grab method (See Tutorial on Depth Sensing).

It is also possible to first capture the image and then depth. To do so, you can use the read method and then the grab method.
