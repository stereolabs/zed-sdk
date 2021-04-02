# Tutorial 8: Human Body tracking with the ZED 2

This tutorial shows how to use the human body module with the ZED 2.<br/>
We assume that you have followed previous tutorials.

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

# Code overview

## Create a camera

As in previous tutorials, we create, configure and open the ZED 2. Please note that the ZED 1 is **not** compatible with the object detection module.

This module uses the GPU to perform deep neural networks computations. On platforms with limited amount of memory such as jetson Nano, it's advise to disable the GUI to improve the performances and avoid memory overflow.


## Enable Object detection

We will define the object detection parameters. Notice that the object tracking needs the positional tracking to be able to track the objects in the world reference frame.

Then we can start the module, it will load the model. This operation can take a few seconds. The first time the module is used, the model will be optimized for the hardware and will take more time. This operation is done only once.

## Capture data

The object confidence threshold can be adjusted at runtime to select only the revelant skeleton depending on the scene complexity, and person distances. 

A confidence of 20 is better suited for outdoor, long range environment while indoor closer range scene will need a confidence closer to 50 or more to avoid false positive.

Since the parameters have been set to `image_sync`, for each `grab` call, the image will be fed into the AI module and will output the detections for each frames.

## Disable modules and exit

Once the program is over the modules can be disabled and the camera closed. This step is optional since the `zed.close()` will take care of disabling all the modules. This function is also called automatically by the destructor if necessary.