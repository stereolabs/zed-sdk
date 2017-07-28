# Stereolabs ZED - SVO Recording utilities

This sample demonstrates how to read a SVO file and convert it into an AVI file (LEFT + RIGHT) or (LEFT + DEPTH).

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

     ./ZED_SVO_Recording "./mysvo.svo" "./mysvo_converted.avi"
     ./ZED_SVO_Recording "./mysvo.svo" "./mysvo_with_depth.avi" 1

### Launch options
You should provide at least two arguments :
  - 1st : SVO file path (input)
  - 2nd : AVI file path (output)
  - 3rd : (optional) (bool) to export colored representation of the depth instead of the right image

## Troubleshooting

If you want to tweak the video file option in the sample code (for example recording a mp4 file), you may have to recompile OpenCV with the FFmpeg option (WITH_FFMPEG).
