# Stereolabs ZED - SVO Recording utilities

This sample shows how to record video in Stereolabs SVO format.
SVO video files can be played with the ZED API and used with its different modules.

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

Open a terminal in build directory and execute the following command:

     ./ZED_SVO_Recording [arg]

Example :

     ./ZED_SVO_Recording "./mysvo.svo"
