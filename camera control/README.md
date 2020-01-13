# Stereolabs ZED - Camera Control

This sample shows how to capture images with the ZED SDK and adjust camera settings.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))
- OpenCV

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

- Navigate to the build directory and launch the executable
- Or open a terminal in the build directory and run the sample :

      ./ZED_Camera_Control

## Features

This sample demonstrates how to captures images and change camera settings with the ZED SDK.
The following parameters can be changed:

  - Exposure
  - Gain
  - Saturation
  - Hue
  - Contrast
  - Brightness


### Keyboard shortcuts

This table lists keyboard shortcuts that you can use in the sample application.

Parameter             | Description                   |   Hotkey
---------------------|------------------------------------|-------------------------------------------------
Switch settings | Toggle between camera settings | 's'
Increase settings value | Increase current settings value | '+'
Decrease settings value | Decrease current settings value | '-'
Reset | Reset all parameters to default values | 'r'
Exit         | Quit the application             | 'q'
