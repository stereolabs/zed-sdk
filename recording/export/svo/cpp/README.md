# ZED SDK - SVO Export

This sample demonstrates how to read a SVO file and convert it into an AVI file (LEFT + RIGHT) or (LEFT + DEPTH_VIEW).

It can also convert a SVO in the following png image sequences: LEFT+RIGHT, LEFT+DEPTH_VIEW, and LEFT+DEPTH_16Bit.

## Getting Started
 - Get the latest [ZED SDK](https://www.stereolabs.com/developers/release/)
 - Check the [Documentation](https://www.stereolabs.com/docs/)

## Build the program
 - Build for [Windows](https://www.stereolabs.com/docs/app-development/cpp/windows/)
 - Build for [Linux/Jetson](https://www.stereolabs.com/docs/app-development/cpp/linux/)
 
## Run the program
- Navigate to the build directory and launch the executable
- Or open a terminal in the build directory and run the sample :

      ./ZED_SVO_Export  svo_file.svo

### Features
```
Usage:

ZED_SVO_Export A B C

Please use the following parameters from the command line:
 A - SVO file path (input) : "path/to/file.svo"
 B - AVI file path (output) or image sequence folder(output) : "path/to/output/file.avi" or "path/to/output/folder/"
 C - Export mode:  0=Export LEFT+RIGHT AVI.
				   1=Export LEFT+DEPTH_VIEW AVI.
				   2=Export LEFT+RIGHT image sequence.
				   3=Export LEFT+DEPTH_VIEW image sequence.
				   4=Export LEFT+DEPTH_16Bit image sequence.
 A and B need to end with '/' or '\'

Examples:
  (AVI LEFT+RIGHT)              ZED_SVO_Export "path/to/file.svo" "path/to/output/file.avi" 0
  (AVI LEFT+DEPTH)              ZED_SVO_Export "path/to/file.svo" "path/to/output/file.avi" 1
  (SEQUENCE LEFT+RIGHT)         ZED_SVO_Export "path/to/file.svo" "path/to/output/folder/" 2
  (SEQUENCE LEFT+DEPTH)         ZED_SVO_Export "path/to/file.svo" "path/to/output/folder/" 3
  (SEQUENCE LEFT+DEPTH_16Bit)   ZED_SVO_Export "path/to/file.svo" "path/to/output/folder/" 4
```

## Troubleshooting

If you want to tweak the video file option in the sample code (for example recording a mp4 file), you may have to recompile OpenCV with the FFmpeg option (WITH_FFMPEG).

## Support
If you need assistance go to our Community site at https://community.stereolabs.com/