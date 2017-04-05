# Stereolabs ZED - SVO Recording utilities

This sample demonstrates how to use the recording capabilities of the SDK. The SVO file is used to simulate a ZED.
This sample allow the recording of such a file and also the decoding. The decoding provides a way to convert the file into a standard vidoe file or into a sequence of images.

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

     ./ZED_SVO_Recording [option] [arg]

Example :

     ./ZED_SVO_Recording -f="./mysvo.svo" -v="./mysvo_converted.avi"  -z

**NOTE :** Make sure you put a '=' between the option and the argument.


### Launch options

Option                    |               Descriptions             |                 Available Arguments                 
 -----------------------------------------|----------------------------------------|-----------------------------------------------------
 --help                | Display help message.                   |                                                  
 -f, --filename      | SVO input filename                          | Path to an input SVO file    
 -v, --video      | Name of the output file, Left+Disparity with -z option, Left+Right otherwise                           |      filename with ".avi" extension for a video file, if no extension is given a sequence of images is then recorded (png file)
 -z, --disparity          | Compute disparity      |   

## Troubleshooting

If you want to tweak the video file option in the sample code (for example recording a mp4 file), you may have to recompile OpenCV with the FFmpeg option (WITH_FFMPEG).
