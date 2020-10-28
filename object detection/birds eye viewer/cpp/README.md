# Stereolabs ZED - 3D Object Detection

This sample shows how to detect and track objects in space and display it in an OpenGL window. It demonstrates how to:

- Detect and track objects in the scene using the API
- Display the 2D detection on the image, including the object mask
- Display the bird-view of the tracked objects trajectory relative to the camera
- Display the point cloud with the 3D bounding box of the detected objects

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build the program

#### Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the Visual Studio `Win64` solution
- Open the resulting solution and change configuration to `Release`
- Build solution

#### Build for Linux

Open a terminal in the sample directory and execute the following command:

    mkdir build
    cd build
    cmake ..
    make

## Run the program

- Navigate to the build directory and launch the executable file
- Or open a terminal in the build directory and run the sample :

        ./ZED_Object_Detection

You can optionally provide an SVO file path (recorded stereo video of the ZED 2).

*NOTE:* A ZED 2 is required to run use this module.
