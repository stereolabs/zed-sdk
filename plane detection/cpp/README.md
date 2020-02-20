# Stereolabs ZED - Plane detection

This sample shows how to retrieves the floor plane from the scene and the plane at a specific image coordinate with the ZED SDK.

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

        ./ZED_Plane_Detection

## Features

This sample shows how to retrieves the floor plane from the scene with the ZED SDK. By clicking on the image the corresponding plane will be also retrieved.

### Keyboard shortcuts

This table lists keyboard shortcuts that you can use in the sample application.

Parameter             | Description                   |   Hotkey
---------------------|------------------------------------|-------------------------------------------------
Plane detection | detect currents floor plane | (space bar)
Exit         | Quit the application             | 'q'
