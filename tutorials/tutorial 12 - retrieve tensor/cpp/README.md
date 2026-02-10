# Tutorial 12: Retrieve Tensor

This tutorial shows how to use the `Camera::retrieveTensor` function to get a pre-processed tensor from the ZED camera, suitable for deep learning inference.

## Getting started

- First, download the latest version of the ZED SDK on [stereolabs.com](https://www.stereolabs.com).
- For more information, read the ZED [API documentation](https://www.stereolabs.com/developers/documentation/API/).

### Prerequisites

- Windows 10, Ubuntu LTS, L4T
- [ZED SDK](https://www.stereolabs.com/developers/) and its dependencies ([CUDA](https://developer.nvidia.com/cuda-downloads))

## Build the program

Download the sample and follow the instructions below: [More](https://www.stereolabs.com/docs/getting-started/application-development/)

#### Build for Windows

- Create a "build" folder in the source folder
- Open cmake-gui and select the source and build folders
- Generate the solution
- Open the resulting solution and build it

#### Build for Linux/Jetson

Open a terminal in the sample directory and execute the following command:

    mkdir build
    cd build
    cmake ..
    make
