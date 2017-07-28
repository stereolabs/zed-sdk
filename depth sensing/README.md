# Stereolabs ZED - Depth Sensing

This sample captures a 3D point cloud and display it in an OpenGL window. It shows how to:
- Get a 3D point cloud with the API.
- Display point cloud in OpenGL.
- Use a thread to capture the point cloud and update the GL window simultaneously.

To retrieve a depth map of the scene, see [Depth Sensing](https://github.com/stereolabs/zed-examples/tree/master/tutorials) tutorial.

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

        ./ZED_Depth_Sensing
