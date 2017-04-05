# Stereolabs ZED - Spatial Mapping

This sample demonstrates how to get a mesh with the ZED.
We show the mesh on a wireframe overlay display on top of the image.
This sample demonstrates how to use the asynchronous function of the  ZED mapping API for live preview, 
and the synchronous function to extract, filter and save a mesh in a obj file  

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

        ./ZED_Spatial_Mapping