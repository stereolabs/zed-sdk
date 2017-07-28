# Stereolabs ZED - Spatial Mapping

This sample shows how to capture a real-time 3D map of the scene with the ZED API. It demonstrates how to:
- Start spatial mapping to capture the surrounding area.
- Display the live reconstruction as a wireframe mesh over the image using OpenGL.
- Process and save the mesh in OBJ format.
- Use the lower-level API to asynchronously capture, update and display the mesh.
- Access and retrieve chunks of the entire mesh.

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
