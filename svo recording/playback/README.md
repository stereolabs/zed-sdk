# Stereolabs ZED - SVO Playback utilities

This sample demonstrates how to read a SVO video file.
We use OpenCV to display the video.

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