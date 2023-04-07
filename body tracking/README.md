# ZED SDK Body Tracking Samples

This repository contains samples demonstrating how to use the ZED stereoscopic camera's body tracking features using the ZED SDK. The samples show how to fuse the data from several cameras, export the data in JSON format, and how to integrate the SDK with external tools like Unity and Unreal.

## Requirements

    ZED Camera
    ZED SDK
    CUDA 10.2 or later
    Python 3.6 or later (for Python samples)
    C++11 or later (for C++ samples)
    Unity or Unreal Engine (for integration samples)

## Installation

    Install the ZED SDK following the instructions on the official website.
    Clone this repository to your local machine.
    For Python samples, install the required Python packages by running pip install -r requirements.txt in the root directory of the repository.
    Build the C++ samples using the CMake build system. Refer to the individual sample README files for build instructions.

## Samples
### Overview
This samples demonstrates how to build simple body tracking app. It provide :
- Single camera body tracking
- Whole hand **fingertracking**
- A simple 3D display

### Fusion
This sample demonstrate how to fuse the Body tracking data from several camera to track an entire space with a much higher quality. That can be done on one single machine, or on a network.
The camera must be calibrated first with ZED 360.
If your camera are distributed over a local network, you'll need to use ZED Hub. [Subscribe for free.](https://hub.stereolabs.com)

### Export
This sample shows you how to export the data into a JSON format. you can adapt the code to fit your needs.

### Integrations
This folder contains links to other repositories that provide Body Tracking integrations examples and tutorials with Unreal, Unity, Livelink.