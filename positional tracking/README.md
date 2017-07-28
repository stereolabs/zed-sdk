# Stereolabs ZED - Positional Tracking

This sample shows how to track camera motion in space and display it in an OpenGL window. It demonstrates how to:
- Get position and orientation of the device using the API
- Select your coordinate system, frames and units
- Tranform pose data at the center of the camera
- Write pose data and timestamps in a CSV file
- Display camera motion in an OpenGL window

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

        ./ZED_Motion_Tracking

You can optionally provide an SVO file path (recorded stereo video of the ZED).

*NOTE:* ZED camera tracking is based on stereo vision only. Quick and sudden camera movements can be difficult to track if the image is too blurry or there is no visual information in the scene. To improve tracking performance, we recommend using the ZED in HD720 mode at 60fps.
